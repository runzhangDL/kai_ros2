"""Walking deployment: the standing policy, the walking policy, and one bus.

This replaces ``policy_node`` for walking runs. It publishes on exactly the
topics ``servo_node`` already subscribes to, so the executor -- and every
safety rule in it -- is unchanged and unaware that there are now two policies.

Why both policies live in one process with the executor
-------------------------------------------------------
Two reasons, and the second is the one that matters.

The serial bus has a single owner. Standing and walking cannot be two
processes taking turns at it, because the handover would have a window with
nobody driving the servos.

More importantly, **Ctrl-C must not be a process kill**. The requirement is
that interrupting the walk hands the robot back to the standing policy rather
than dropping it, and a robot whose control process has exited is a robot on
the floor. So this module installs its own SIGINT handler: the first Ctrl-C
asks the sequencer to stop and the node keeps running until the robot is
standing again; only a second Ctrl-C exits. That is only sound if the thing
that is still running also owns the motors, which is why ``main()`` spins the
executor node and the policy node in a single-threaded executor here rather
than launching them separately.

One consequence worth knowing: the IMU driver is a separate process, so a
Ctrl-C delivered to a whole ``ros2 launch`` process group kills it while this
node is still trying to balance. Run the IMU from its own terminal
(``imu.launch.py``) for any run where the handback matters -- see
``walk.launch.py``, whose ``imu`` argument defaults to false for this reason.

The observation
---------------
``mjx_walk_free_env`` appends five entries to the standing policy's 48, so one
builder feeds both: the walking policy gets all 53, the standing policy gets
``obs[:48]`` of the same vector. They are computed once, from the same sensor
sample, every cycle -- including while the other policy is driving, because
whichever one is not in control still needs a coherent ``last_action`` history
for the moment it takes over.
"""

from __future__ import annotations

import signal
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from .gait import GaitConfig, GaitError, GaitMode, GaitSequencer
from .imu import ImuConditioner, ImuError
from .policy import Policy, PolicyError

_SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class WalkPolicyNode(Node):
    """Runs both policies and sequences them. Owns no hardware."""

    def __init__(self) -> None:
        super().__init__("humanoid_policy")

        p = self.declare_parameter
        p("bundle", "")                 # standing policy
        p("walk_bundle", "")            # walking policy
        p("imu_topic", "/imu/data")
        p("accel_units", "auto")
        p("gyro_units", "rad/s")
        p("gravity_source", "complementary")
        p("complementary_tau", 1.0)
        p("warn_ood_sigma", 8.0)
        # gait sequencing
        p("crouch_s", 2.0)
        p("settle_s", 0.75)
        p("walk_duration_s", 5.0)
        p("recover_s", 2.0)
        p("crouch_style", "ramp")
        p("recover_style", "via_crouch")
        p("cmd_vx", 0.085)
        p("cmd_wz", 0.0)
        p("stop_phase", -1.0)           # <0 means "no phase preference"
        p("max_stop_wait_s", 1.5)
        # Walking is refused above this tilt, and an in-progress walk is
        # abandoned above it. Deliberately far below servo_node's 40 deg FALL
        # threshold: that one cuts torque, which is the right response to a
        # fall but the wrong one to a wobble. This gets the standing policy
        # back on the robot while it can still recover.
        p("walk_abort_tilt_deg", 22.0)
        p("arm_tilt_deg", 12.0)
        # The safest thing this node can do on a robot whose walking policy is
        # not yet trusted: run the whole sequence -- crouch, hold, handback,
        # Ctrl-C -- with the walking policy's authority forced to zero, so the
        # robot never does anything but squat and stand up again.
        p("crouch_only", False)
        # Per-joint offset added to the walking policy's nominal pose, degrees,
        # in the bundle's joint order.
        #
        # This exists because the model and the robot disagree about where the
        # centre of mass is. Measured 2026-08-14: held open loop in the crouch,
        # every joint tracked its command to within 1.4 deg with ZERO drift,
        # and the robot still rotated backward onto its heels, 6.1 deg growing
        # to 14.8. Nothing mechanical -- the pose itself does not balance.
        #
        # Trimming ankle pitch pitches the torso relative to the feet and moves
        # the CoM fore/aft. Use it with crouch_only to find the trim that makes
        # the real robot stand upright in the crouch; that trim then belongs in
        # NOMINAL_POSE_DEG and in training, not here. Left in the deployment it
        # would offset every observation the policy sees from what it trained
        # on, which is why it warns when set outside crouch_only.
        p("nominal_trim_deg", [0.0] * 13)
        # The ergonomic form of the same thing, and the only one usually
        # needed: ankle pitch is what pitches the torso relative to the feet.
        # Applied MIRRORED (+d to the left ankle, -d to the right) because the
        # two joints have opposite sign conventions, so this is one physical
        # direction of lean rather than a twist.
        p("ankle_pitch_trim_deg", 0.0)
        # Scale the whole nominal pose toward the standing pose. 1.0 = the
        # trained crouch, 0.6 = a crouch 40% shallower, 0.0 = straight legs.
        #
        # Squatting with the torso vertical sends the hips BACKWARD as well as
        # down, which is what walks the centre of mass toward the heels -- so a
        # shallower crouch keeps it further forward. The pose's hip + knee +
        # ankle sum to ~0 per leg, which is what keeps the torso vertical and
        # the feet flat, and scaling is linear so it preserves that.
        #
        # Same caveat as the trim: this changes the pose the policy's residual
        # is measured about, so it is a way to FIND a stable crouch under
        # crouch_only, not a way to fly one.
        p("crouch_scale", 1.0)

        stand_bundle = self.get_parameter("bundle").value
        walk_bundle = self.get_parameter("walk_bundle").value
        if not stand_bundle or not walk_bundle:
            raise SystemExit(
                "both 'bundle' (standing) and 'walk_bundle' (walking) are "
                "required")
        try:
            self.stand = Policy(stand_bundle)
            self.walk = Policy(walk_bundle)
        except PolicyError as exc:
            raise SystemExit(f"cannot load a policy: {exc}") from exc

        self._check_bundles()
        self.get_logger().info("standing " + self.stand.describe())
        self.get_logger().info("walking  " + self.walk.describe())

        try:
            self.imu = ImuConditioner(
                accel_units=self.get_parameter("accel_units").value,
                gyro_units=self.get_parameter("gyro_units").value,
                gravity_source=self.get_parameter("gravity_source").value,
                complementary_tau=float(
                    self.get_parameter("complementary_tau").value),
            )
        except ImuError as exc:
            raise SystemExit(f"IMU configuration rejected: {exc}") from exc

        stop_phase = float(self.get_parameter("stop_phase").value)
        gait = self.walk.gait
        try:
            config = GaitConfig(
                control_dt=self.walk.control_dt,
                crouch_s=float(self.get_parameter("crouch_s").value),
                settle_s=float(self.get_parameter("settle_s").value),
                walk_duration_s=float(self.get_parameter("walk_duration_s").value),
                recover_s=float(self.get_parameter("recover_s").value),
                crouch_style=str(self.get_parameter("crouch_style").value),
                recover_style=str(self.get_parameter("recover_style").value),
                phase_increment=float(gait["phase_increment_per_cycle"]),
                cmd_vx=float(self.get_parameter("cmd_vx").value),
                cmd_wz=float(self.get_parameter("cmd_wz").value),
                cmd_vx_range=tuple(gait["cmd_vx_range"]),
                cmd_wz_range=tuple(gait["cmd_wz_range"]),
                stop_phase=None if stop_phase < 0.0 else stop_phase,
                max_stop_wait_s=float(self.get_parameter("max_stop_wait_s").value),
            )
            self.seq = GaitSequencer(config, self.walk.default_pose)
        except (GaitError, KeyError) as exc:
            raise SystemExit(f"gait configuration rejected: {exc}") from exc
        self.get_logger().info(self.seq.describe())

        trim = list(self.get_parameter("nominal_trim_deg").value or [])
        if not trim:
            trim = [0.0] * self.walk.nu
        ankle = float(self.get_parameter("ankle_pitch_trim_deg").value)
        if abs(ankle) > 1e-9:
            names = list(self.walk.joint_names)
            trim[names.index("left_ankle_pitch_joint")] += ankle
            trim[names.index("right_ankle_pitch_joint")] -= ankle
        scale = float(self.get_parameter("crouch_scale").value)
        if trim and len(trim) != self._nu_expected(self.walk):
            raise SystemExit(
                f"nominal_trim_deg has {len(trim)} entries, expected "
                f"{self.walk.nu} (one per joint, in the bundle's order)")
        self.crouch_only = bool(self.get_parameter("crouch_only").value)
        if self.crouch_only:
            self.get_logger().warning(
                "crouch_only is set: the walking policy will be evaluated and "
                "published for inspection but given ZERO authority. The robot "
                "will crouch, hold, and stand back up. Use this to verify the "
                "sequence, the handback and Ctrl-C before trusting the gait.")

        if abs(scale - 1.0) > 1e-9:
            self.seq.nominal = self.seq.nominal * scale
            self.get_logger().warning(
                f"crouch SCALED to {scale:.2f} of the trained pose: "
                + " ".join(f"{np.degrees(v):+.1f}" for v in self.seq.nominal))
        if trim and np.any(np.abs(trim) > 1e-9):
            self.seq.nominal = self.seq.nominal + np.radians(trim)
            self.get_logger().warning(
                "nominal pose TRIMMED by (deg): "
                + " ".join(f"{v:+.1f}" for v in trim))
            if not self.crouch_only:
                self.get_logger().error(
                    "  ...and crouch_only is FALSE. The walking policy's "
                    "observations will be offset from the pose it was trained "
                    "about by exactly this trim. Use a trim to FIND the "
                    "balancing pose under crouch_only, then put it in "
                    "NOMINAL_POSE_DEG and retrain -- do not fly it.")

        self._nu = self.stand.nu
        self._obs = np.zeros(self.walk.obs_size)
        self._stand_action = np.zeros(self._nu)
        self._walk_action = np.zeros(self._nu)
        self._imu_msg: Imu | None = None
        self._last_imu_time: float | None = None
        self._units_resolved = False
        self._warned_ood = False
        self._tilt_deg = 0.0
        self._cycles = 0

        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self._on_imu, _SENSOR_QOS)
        self.create_subscription(
            JointState, "/humanoid/joint_states", self._on_joints, _SENSOR_QOS)
        self._cmd_pub = self.create_publisher(JointState, "~/joint_command", _SENSOR_QOS)
        self._obs_pub = self.create_publisher(
            Float64MultiArray, "~/observation", _SENSOR_QOS)
        self._act_pub = self.create_publisher(Float64MultiArray, "~/action", _SENSOR_QOS)
        self._gait_pub = self.create_publisher(String, "~/gait", _LATCHED)

        self.create_service(Trigger, "~/walk", self._srv_walk)
        self.create_service(Trigger, "~/stop", self._srv_stop)

        self.get_logger().info(
            "gait node up, standing. Arm the executor, let it settle, then "
            "call:  ros2 service call /humanoid_policy/walk std_srvs/srv/Trigger")
        self._publish_gait()

    @staticmethod
    def _nu_expected(policy):
        return policy.nu

    # -- startup checks ----------------------------------------------------

    def _check_bundles(self) -> None:
        """Refuse any pair of bundles that cannot describe the same robot.

        Every one of these has the same failure mode if it slips through --
        commands land on the wrong joint, or the wrong number of them -- and
        none of them is visible in the robot's behaviour until it moves.
        """
        problems = []
        if list(self.stand.joint_names) != list(self.walk.joint_names):
            problems.append(
                f"joint order differs:\n    stand {self.stand.joint_names}\n"
                f"    walk  {self.walk.joint_names}")
        if self.stand.nu != self.walk.nu:
            problems.append(f"joint count {self.stand.nu} vs {self.walk.nu}")
        if abs(self.stand.control_dt - self.walk.control_dt) > 1e-6:
            problems.append(
                f"control rate {1 / self.stand.control_dt:.2f} Hz vs "
                f"{1 / self.walk.control_dt:.2f} Hz -- one gait clock cannot "
                "serve both")
        if self.walk.layout.stand_size != self.stand.obs_size:
            problems.append(
                f"the walking policy's shared prefix is "
                f"{self.walk.layout.stand_size} entries but the standing "
                f"policy takes {self.stand.obs_size}; they are not the "
                "observation superset this node assumes")
        if not self.walk.gait:
            problems.append(
                "the walking bundle carries no gait metadata; re-export it "
                "with tools/export_policy.py --sidecar <checkpoint>.pkl.json")
        if np.max(np.abs(self.stand.default_pose)) > 1e-9:
            problems.append("the standing policy's nominal pose is not 0 rad")
        if problems:
            raise SystemExit("the two bundles disagree:\n  " + "\n  ".join(problems))

    # -- callbacks ---------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        self._imu_msg = msg

    def _on_joints(self, msg: JointState) -> None:
        """One control cycle, driven by the executor's joint_states publish."""
        if self._imu_msg is None:
            return
        try:
            index = [msg.name.index(name) for name in self.stand.joint_names]
        except ValueError as exc:
            self.get_logger().error(f"joint_states is missing a joint: {exc}",
                                    throttle_duration_sec=5.0)
            return
        if len(msg.position) < len(msg.name) or len(msg.velocity) < len(msg.name):
            self.get_logger().error("joint_states lacks position/velocity",
                                    throttle_duration_sec=5.0)
            return

        sample = self._condition_imu()
        if sample is None:
            return
        self._tilt_deg = sample.tilt_deg

        obs = self._obs
        obs[0:3] = sample.accel
        obs[3:6] = sample.gyro
        obs[6:9] = sample.gravity
        obs[9:9 + self._nu] = np.asarray(msg.position, dtype=np.float64)[index]
        obs[9 + self._nu:9 + 2 * self._nu] = np.asarray(
            msg.velocity, dtype=np.float64)[index]
        action_slice = self.stand.layout.action

        # Both policies are evaluated every cycle, each shown its own action
        # history. The one that is not driving is kept warm so that when it is
        # handed the robot, its last_action means what it meant in training.
        obs[action_slice] = self._stand_action
        self._stand_action = self.stand.act(obs[:self.stand.obs_size])
        stand_target = self.stand.action_to_targets(self._stand_action)

        walk_target = None
        if self.seq.walking:
            obs[action_slice] = self._walk_action
            obs[self.walk.layout.tail] = np.concatenate(
                [self.seq.phase_features(), self.seq.command()])
            self._walk_action = self.walk.act(obs)
            walk_target = self.walk.action_to_targets(self._walk_action)
            if self.crouch_only:
                # Evaluated and published so it can be inspected against the
                # measured pose, but never commanded.
                walk_target = self.seq.nominal.copy()

        self._guard_tilt()
        target, status = self.seq.step(stand_target, walk_target)
        if status.walk_started:
            # The trainer's reset() hands the policy last_action = 0 alongside
            # phase = 0, and 'cold start' means both. Zeroing it here rather
            # than carrying a shadow history in is what makes the handover the
            # condition that was measured at 100%.
            self._walk_action = np.zeros(self._nu)
        if status.walk_finished:
            self.get_logger().info(
                f"back under the standing policy ({self.seq.last_exit})")

        stamp = self.get_clock().now().to_msg()
        command = JointState()
        command.header.stamp = stamp
        command.name = list(self.stand.joint_names)
        command.position = [float(v) for v in target]
        self._cmd_pub.publish(command)
        self._obs_pub.publish(Float64MultiArray(
            data=[float(v) for v in obs[:self.stand.obs_size]]))
        self._act_pub.publish(Float64MultiArray(
            data=[float(v) for v in (self._walk_action if self.seq.walking
                                     else self._stand_action)]))

        self._check_ood(obs)
        self._cycles += 1
        if self._cycles % 13 == 0 or status.walk_started or status.walk_finished:
            self._publish_gait(status)

    def _condition_imu(self):
        message = self._imu_msg
        raw_accel = np.array([message.linear_acceleration.x,
                              message.linear_acceleration.y,
                              message.linear_acceleration.z])
        raw_gyro = np.array([message.angular_velocity.x,
                             message.angular_velocity.y,
                             message.angular_velocity.z])
        quat = (message.orientation.x, message.orientation.y,
                message.orientation.z, message.orientation.w)
        if not self._units_resolved:
            try:
                units = self.imu.resolve_accel_units(raw_accel)
            except ImuError as exc:
                self.get_logger().warning(f"accel unit detection: {exc}",
                                          throttle_duration_sec=2.0)
                return None
            self._units_resolved = True
            self.get_logger().info(f"accelerometer resolved as {units}")

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self._last_imu_time is None else max(0.0, now - self._last_imu_time)
        self._last_imu_time = now
        try:
            return self.imu.condition(raw_accel, raw_gyro, dt, quat)
        except ImuError as exc:
            self.get_logger().error(f"IMU conditioning failed: {exc}",
                                    throttle_duration_sec=2.0)
            return None

    def _guard_tilt(self) -> None:
        limit = float(self.get_parameter("walk_abort_tilt_deg").value)
        if self._tilt_deg > limit and self.seq.mode is not GaitMode.STAND:
            if not self.seq.stop_pending:
                self.get_logger().warning(
                    f"tilt {self._tilt_deg:.1f} deg over the {limit:.0f} deg "
                    "walking limit -- handing back to the standing policy")
            self.seq.request_stop(f"tilt {self._tilt_deg:.1f} deg")

    def _check_ood(self, obs) -> None:
        # Only while the walking policy is driving: at other times obs[48:53]
        # holds no meaningful clock or command, and scoring those five entries
        # against the walking normaliser would report a distribution error that
        # is really just an unset tail.
        if self._warned_ood or not self.seq.walking:
            return
        limit = float(self.get_parameter("warn_ood_sigma").value)
        offenders = self.walk.out_of_distribution(obs, limit)
        if offenders:
            self._warned_ood = True
            summary = ", ".join(f"{label}={z:+.1f}s"
                                for _, label, _, z in offenders[:6])
            self.get_logger().warning(
                f"observation far outside the walking policy's training "
                f"distribution: {summary}")

    # -- services ----------------------------------------------------------

    def _srv_walk(self, request, response):
        del request
        limit = float(self.get_parameter("arm_tilt_deg").value)
        if self._imu_msg is None:
            response.success, response.message = False, "no IMU data yet"
            return response
        if self._tilt_deg > limit:
            response.success = False
            response.message = (f"robot is {self._tilt_deg:.1f} deg from "
                                f"upright; must be within {limit:.0f} to walk")
            return response
        ok, message = self.seq.request_walk()
        if ok:
            self.get_logger().info(f"WALK requested: {message}")
        response.success, response.message = ok, message
        return response

    def _srv_stop(self, request, response):
        del request
        ok, message = self.seq.request_stop("operator")
        response.success, response.message = ok, message
        return response

    # -- outward state -----------------------------------------------------

    def request_stop(self, reason: str) -> bool:
        ok, message = self.seq.request_stop(reason)
        self.get_logger().warning(f"stop requested ({reason}): {message}")
        return ok

    @property
    def idle(self) -> bool:
        return self.seq.mode is GaitMode.STAND

    def _publish_gait(self, status=None) -> None:
        parts = [f"mode={self.seq.mode.value}", f"tilt={self._tilt_deg:.1f}deg"]
        if status is not None:
            parts.append(f"phase={status.phase:.2f}")
            parts.append(f"walk_authority={status.walk_authority:.2f}")
            if status.remaining_s > 0.0:
                parts.append(f"remaining={status.remaining_s:.1f}s")
            if status.stop_reason:
                parts.append(f"stopping={status.stop_reason}")
        if self.crouch_only:
            parts.append("CROUCH_ONLY")
        self._gait_pub.publish(String(data="  ".join(parts)))


def main(argv=None) -> None:
    """Spin the executor and the gait node together, under one SIGINT owner."""
    from .servo_node import ServoNode

    rclpy.init(args=argv)
    servo = gait = None
    try:
        servo = ServoNode()
        gait = WalkPolicyNode()
    except SystemExit as exc:
        print(f"walking node failed to start: {exc}")
        for node in (servo, gait):
            if node is not None:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise

    executor = SingleThreadedExecutor()
    executor.add_node(servo)
    executor.add_node(gait)

    # How long the first Ctrl-C may spend getting back to standing before this
    # gives up and exits anyway. The sequencer only advances when joint_states
    # arrives, so a dead bus would otherwise leave the operator's Ctrl-C
    # waiting for a mode change that can never come.
    grace = (gait.seq.cfg.crouch_s + gait.seq.cfg.recover_s
             + gait.seq.cfg.max_stop_wait_s + 3.0)
    state = {"interrupts": 0, "deadline": None}

    def on_sigint(signum, frame):
        """First Ctrl-C stops the walk; the second one gives up and exits.

        rclpy installs a handler at init() that tears the context down. That is
        the correct behaviour for a node that owns nothing and the wrong one
        here: shutting down mid-stride drops the robot. So this replaces it,
        and the loop below is what actually ends the process, once the
        sequencer reports the robot standing again.
        """
        del signum, frame
        state["interrupts"] += 1
        if state["interrupts"] == 1 and not gait.idle:
            state["deadline"] = time.monotonic() + grace
            gait.request_stop("Ctrl-C")
            gait.get_logger().warning(
                "Ctrl-C: returning to the standing policy. The node stays up "
                "and the robot stays powered -- press Ctrl-C again to exit "
                "and drop torque, holding the robot when you do.")
        else:
            gait.get_logger().warning("second Ctrl-C: shutting down")
            state["interrupts"] = 99

    signal.signal(signal.SIGINT, on_sigint)

    try:
        while rclpy.ok() and state["interrupts"] < 99:
            executor.spin_once(timeout_sec=0.1)
            if state["interrupts"] != 1:
                continue
            if gait.idle:
                gait.get_logger().info(
                    "standing again; exiting. Hold the robot -- torque is "
                    "about to come off.")
                break
            if time.monotonic() > state["deadline"]:
                gait.get_logger().error(
                    f"still {gait.seq.mode.value} {grace:.0f}s after Ctrl-C -- "
                    "the control loop is not advancing (a dead bus stops "
                    "joint_states, and with it the sequencer). Exiting and "
                    "dropping torque; hold the robot.")
                break
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        executor.shutdown()
        servo.shutdown()
        for node in (gait, servo):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
