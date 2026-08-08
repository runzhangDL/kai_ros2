"""Hardware executor: the only thing in the system that may move a motor.

Owns the serial bus, the calibration, and every safety decision. Runs the
control loop:

    read all 13 positions (one sync read)
      -> counts to radians via the calibration
      -> publish /humanoid/joint_states
      -> run the safety monitors
      -> shape the most recent policy command (clamp, ramp, rate limit)
      -> one sync write of goal positions

Deliberate design choices
-------------------------
*It refuses to start on an uncalibrated robot.* Counts only mean radians
because of the calibration, so without it every command is meaningless.
``require_calibration`` fails closed.

*It never trusts the policy node.* Commands arrive by topic, are re-ordered by
joint name, clamped to the calibrated envelope, and expire. A crashed policy
node produces a stale command, which is a fault, not a frozen limb.

*It watches the IMU itself.* Fall detection does not route through the policy
node, so a hung policy cannot suppress it.

*Arming is explicit and gated.* Torque stays off until a human calls the arm
service AND every preflight check passes -- including whether the observation
looks statistically like training data, which is what catches a swapped IMU
axis before it becomes a fall.

*FAULT is terminal.* Torque off, latched. Automatic recovery after a fall is
how a robot destroys itself in a stand-up/fall-over loop.

Start here with ``dry_run:=true``: the full loop runs, everything is published
and logged, and not a single byte is written to a servo.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from humanoid_calibration.guard import NotCalibrated, require_calibration

from .imu import ImuConditioner, ImuError
from .joint_map import JointMap, JointMapError
from .policy import Policy, PolicyError
from .safety import SafetyConfig, SafetyState, SafetySupervisor
from .servo_bus import ServoBus, ServoBusError

_SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ServoNode(Node):
    def __init__(self) -> None:
        super().__init__("humanoid_servo")

        p = self.declare_parameter
        p("bundle", "")
        p("port", "/dev/ttyTHS1")
        p("baudrate", 250000)
        p("timeout_ms", 30)
        # 0.0 means "run at whatever rate the bundle was trained at", which is
        # the only value that cannot drift away from training. Override only to
        # deliberately run slower.
        p("control_rate_hz", 0.0)
        p("dry_run", True)
        p("shutdown_on_fault", False)
        p("velocity_filter_tau", 0.0)
        p("servo_goal_speed", 2000)
        p("servo_acc", 30)
        p("health_check_period_cycles", 10)
        p("imu_topic", "/imu/data")
        p("accel_units", "auto")
        p("gyro_units", "rad/s")
        p("gravity_source", "complementary")
        p("limit_margin_deg", 2.0)
        # Optional explicit model-joint -> calibration-joint overrides, as
        # "model_name:calibration_name" strings. Only needed when the automatic
        # suffix matching cannot resolve a name; the resolved mapping is always
        # logged so it can be checked by eye.
        p("joint_name_aliases", [""])
        # safety
        p("ramp_s", 3.0)
        p("max_joint_rate", 4.0)
        p("ramp_joint_rate", 1.0)
        p("fall_tilt_deg", 40.0)
        p("arm_tilt_deg", 12.0)
        p("arm_pose_tolerance_deg", 5.0)
        p("command_timeout_s", 0.20)
        p("ood_sigma", 6.0)
        p("max_temperature_c", 65)
        p("min_voltage_v", 9.5)
        p("stall_error_deg", 15.0)
        p("stall_persist_s", 2.0)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        requested_hz = float(self.get_parameter("control_rate_hz").value)

        # --- 1. the robot must be calibrated -----------------------------
        try:
            calibrations = require_calibration(self, allow_override=False)
        except NotCalibrated as exc:
            raise SystemExit(f"refusing to start: {exc}") from exc

        # --- 2. load the policy bundle (for joint order and limits) ------
        bundle = self.get_parameter("bundle").value
        if not bundle:
            raise SystemExit("parameter 'bundle' is required")
        try:
            self.policy = Policy(bundle)
        except PolicyError as exc:
            raise SystemExit(f"cannot load policy bundle: {exc}") from exc

        if requested_hz <= 0.0:
            self.control_dt = self.policy.control_dt
            self.get_logger().info(
                f"control rate taken from the bundle: "
                f"{1 / self.control_dt:.1f} Hz (as trained)")
        else:
            self.control_dt = 1.0 / requested_hz
            if abs(self.control_dt - self.policy.control_dt) > 1e-4 * self.policy.control_dt:
                self.get_logger().warning(
                    f"running at {requested_hz:.1f} Hz but the policy was trained "
                    f"at {1 / self.policy.control_dt:.1f} Hz. It will still run, but "
                    "one policy step no longer means one physical timestep, and "
                    "the joint velocities it sees are scaled wrong. Set "
                    "control_rate_hz to 0.0 to follow the bundle."
                )

        # --- 3. map policy joints onto calibrated servos -----------------
        margin = np.radians(float(self.get_parameter("limit_margin_deg").value))
        aliases = {}
        for entry in self.get_parameter("joint_name_aliases").value or []:
            if entry and ":" in entry:
                model_name, _, calib_name = entry.partition(":")
                aliases[model_name.strip()] = calib_name.strip()
        try:
            self.map = JointMap(
                joint_names=self.policy.joint_names,
                calibrations=calibrations,
                xml_lower=self.policy.xml_lower,
                xml_upper=self.policy.xml_upper,
                action_scale=self.policy.action_scale,
                limit_margin_rad=margin,
                default_pose=self.policy.default_pose,
                aliases=aliases,
            )
        except JointMapError as exc:
            raise SystemExit(f"joint mapping failed:\n{exc}") from exc
        self.get_logger().info("joint map:\n" + self.map.describe())
        if self.map.snapped:
            self.get_logger().info(
                "these joints' measured travel missed the standing pose by a "
                "fraction of a degree and was widened to include it: "
                + ", ".join(f"{name} ({gap:.2f} deg)"
                            for name, gap in self.map.snapped)
            )
        if self.map.unmargined:
            self.get_logger().warning(
                "these joints stand on (or within the safety margin of) a hard "
                "limit, so there is no cushion on that side: "
                + ", ".join(self.map.unmargined)
            )

        seam = self.map.seam_violations()
        if seam:
            detail = "\n  ".join(v.describe() for v in seam)
            raise SystemExit(
                "refusing to start -- these joints would have to be commanded "
                f"across the encoder wrap:\n  {detail}\n"
                "Commanding across the 0/4095 boundary makes the servo take the "
                "long way round, nearly a full revolution at speed. Re-zero the "
                "joint away from the seam (remount the horn) and recalibrate."
            )

        # --- 4. safety supervisor ----------------------------------------
        self.safety = SafetySupervisor(
            self.map,
            SafetyConfig(
                ramp_s=float(self.get_parameter("ramp_s").value),
                max_joint_rate=float(self.get_parameter("max_joint_rate").value),
                ramp_joint_rate=float(self.get_parameter("ramp_joint_rate").value),
                fall_tilt_deg=float(self.get_parameter("fall_tilt_deg").value),
                arm_tilt_deg=float(self.get_parameter("arm_tilt_deg").value),
                arm_pose_tolerance_rad=np.radians(
                    float(self.get_parameter("arm_pose_tolerance_deg").value)),
                command_timeout_s=float(self.get_parameter("command_timeout_s").value),
                ood_sigma=float(self.get_parameter("ood_sigma").value),
                max_temperature_c=int(self.get_parameter("max_temperature_c").value),
                min_voltage_v=float(self.get_parameter("min_voltage_v").value),
                stall_error_rad=np.radians(
                    float(self.get_parameter("stall_error_deg").value)),
                stall_persist_s=float(self.get_parameter("stall_persist_s").value),
            ),
        )

        # --- 5. IMU (independent of the policy node) ----------------------
        try:
            self.imu = ImuConditioner(
                accel_units=self.get_parameter("accel_units").value,
                gyro_units=self.get_parameter("gyro_units").value,
                gravity_source=self.get_parameter("gravity_source").value,
            )
        except ImuError as exc:
            raise SystemExit(f"IMU configuration rejected: {exc}") from exc

        # --- 6. the bus ---------------------------------------------------
        try:
            self.bus = ServoBus(
                port=self.get_parameter("port").value,
                baudrate=int(self.get_parameter("baudrate").value),
                timeout_ms=int(self.get_parameter("timeout_ms").value),
            )
        except ServoBusError as exc:
            raise SystemExit(f"cannot open servo bus: {exc}") from exc

        self.servo_ids = [int(i) for i in self.map.servo_ids]
        silent = [i for i in self.servo_ids if not self.bus.ping(i)]
        if silent:
            self.bus.close()
            raise SystemExit(f"these servos did not answer: {silent}")
        self.get_logger().info(f"all {len(self.servo_ids)} servos responding")

        # --- state --------------------------------------------------------
        self._positions = np.zeros(self.map.size)
        self._velocities = np.zeros(self.map.size)
        self._have_positions = False
        self._last_read_time: float | None = None
        self._command: np.ndarray | None = None
        self._command_time: float | None = None
        self._observation: np.ndarray | None = None
        self._imu_sample = None
        self._last_imu_time: float | None = None
        self._units_resolved = False
        self._cycle = 0
        self._torque_on = False
        self._fault_announced = False
        self._temperatures: dict[str, int | None] = {}
        self._voltages: dict[str, float | None] = {}

        # --- ROS interfaces -------------------------------------------------
        self._state_pub = self.create_publisher(
            JointState, "/humanoid/joint_states", _SENSOR_QOS)
        # What would actually be written, after clamping, ramping and rate
        # limiting -- all 13 joints at full precision. The log line truncates to
        # six and rounds to 0.1 deg, which is not enough to check by eye before
        # energising, and not recordable with `ros2 bag`.
        self._command_pub = self.create_publisher(
            JointState, "~/command", _SENSOR_QOS)
        self._status_pub = self.create_publisher(String, "~/status", _LATCHED)
        self.create_subscription(
            JointState, "/humanoid_policy/joint_command", self._on_command, _SENSOR_QOS)
        self.create_subscription(
            Float64MultiArray, "/humanoid_policy/observation",
            self._on_observation, _SENSOR_QOS)
        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self._on_imu, _SENSOR_QOS)

        self.create_service(Trigger, "~/arm", self._srv_arm)
        self.create_service(Trigger, "~/disarm", self._srv_disarm)
        self.create_service(Trigger, "~/estop", self._srv_estop)

        self.create_timer(self.control_dt, self._control_cycle)

        banner = "DRY RUN -- no servo will be written to" if self.dry_run else \
                 "LIVE -- servos will move once armed"
        self.get_logger().info(
            f"executor up at {1 / self.control_dt:.1f} Hz. {banner}. "
            f"Torque is OFF; call ~/arm when the robot is held upright."
        )
        self._publish_status()

    # -- subscriptions -----------------------------------------------------

    def _on_command(self, msg: JointState) -> None:
        try:
            index = [msg.name.index(name) for name in self.map.names]
        except ValueError:
            self.get_logger().error("joint_command is missing joints; ignoring",
                                    throttle_duration_sec=5.0)
            return
        self._command = np.asarray(msg.position, dtype=np.float64)[index]
        self._command_time = self.get_clock().now().nanoseconds * 1e-9

    def _on_observation(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == self.policy.obs_size:
            self._observation = np.asarray(msg.data, dtype=np.float64)

    def _on_imu(self, msg: Imu) -> None:
        raw_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                              msg.linear_acceleration.z])
        raw_gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                             msg.angular_velocity.z])
        if not self._units_resolved:
            try:
                self.imu.resolve_accel_units(raw_accel)
            except ImuError:
                return
            self._units_resolved = True
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self._last_imu_time is None else max(0.0, now - self._last_imu_time)
        self._last_imu_time = now
        try:
            self._imu_sample = self.imu.condition(raw_accel, raw_gyro, dt)
        except ImuError:
            self._imu_sample = None

    # -- control loop ------------------------------------------------------

    def _control_cycle(self) -> None:
        self._cycle += 1

        raw = self.bus.read_positions(self.servo_ids)
        read_ok = len(raw) == len(self.servo_ids)
        if read_ok:
            counts = np.array([raw[i] for i in self.servo_ids], dtype=np.float64)
            positions = self.map.counts_to_rad(counts)
            now = self.get_clock().now().nanoseconds * 1e-9
            # Divide by the interval that actually elapsed, not the nominal
            # period. The bus takes ~24 ms with a long tail, so timer callbacks
            # jitter; using the nominal dt would scale every velocity by the
            # jitter ratio, and qvel is a third of the observation.
            elapsed = self.control_dt
            if self._last_read_time is not None:
                measured = now - self._last_read_time
                if 0.2 * self.control_dt < measured < 5.0 * self.control_dt:
                    elapsed = measured
            if self._have_positions:
                tau = float(self.get_parameter("velocity_filter_tau").value)
                raw_velocity = (positions - self._positions) / elapsed
                if tau > 0.0:
                    alpha = elapsed / (tau + elapsed)
                    self._velocities += alpha * (raw_velocity - self._velocities)
                else:
                    # No extra filter by default: one encoder count over one
                    # 40 ms cycle is 0.038 rad/s, which is 1-5% of the training
                    # sigma on every qvel channel. Filtering that adds lag the
                    # policy never trained with and buys nothing.
                    self._velocities = raw_velocity
            self._positions = positions
            self._last_read_time = now
            self._have_positions = True
            self._publish_joint_states()

        if self._imu_sample is None:
            if self._cycle % 50 == 0:
                self.get_logger().warning("no usable IMU data yet")
            return

        age = (0.0 if self._command_time is None
               else self.get_clock().now().nanoseconds * 1e-9 - self._command_time)
        if self._command_time is None:
            age = float("inf") if self.safety.state is SafetyState.RUNNING else 0.0

        if self._cycle % max(1, int(self.get_parameter(
                "health_check_period_cycles").value)) == 0:
            self._sample_health()

        fault = self.safety.check_health(
            self._imu_sample, age, read_ok, self._temperatures, self._voltages)
        if fault is None and read_ok:
            fault = self.safety.check_tracking(self._positions, self.control_dt)
        if fault is not None:
            self._enter_fault()
            return

        if self.safety.state in (SafetyState.RAMPING, SafetyState.RUNNING):
            if self._command is None:
                return
            command, status = self.safety.shape(self._command, self.control_dt)
            if command is None:
                return
            if status.clamped_joints and self._cycle % 25 == 0:
                self.get_logger().warning(
                    "clamped to safe envelope: " + ", ".join(status.clamped_joints))
            self._write(command)

        if self._cycle % 25 == 0:
            self._publish_status()

    def _write(self, command_rad: np.ndarray) -> None:
        counts = self.map.rad_to_counts(command_rad)

        shaped = JointState()
        shaped.header.stamp = self.get_clock().now().to_msg()
        shaped.name = list(self.map.names)
        shaped.position = [float(v) for v in command_rad]
        self._command_pub.publish(shaped)

        if self.dry_run:
            if self._cycle % 50 == 0:
                self.get_logger().info(
                    "[dry run] would command "
                    + " ".join(f"{np.degrees(v):+.1f}" for v in command_rad[:6])
                    + " ... deg")
            return
        try:
            self.bus.write_goal_positions(self.servo_ids, counts)
        except ServoBusError as exc:
            self.safety.trip("WRITE_REJECTED", str(exc))
            self._enter_fault()

    def _sample_health(self) -> None:
        """Round-robin one servo per call so telemetry never stalls the loop."""
        index = (self._cycle // 10) % len(self.servo_ids)
        servo_id = self.servo_ids[index]
        name = self.map.names[index]

        temperature = self.bus.read_temperature(servo_id)
        limit = int(self.get_parameter("max_temperature_c").value)
        if temperature is not None and temperature > limit:
            temperature = self._confirm_temperature(servo_id, name, temperature, limit)
        if temperature is not None:
            self._temperatures[name] = temperature

        voltage = self.bus.read_voltage(servo_id)
        if voltage is not None:
            self._voltages[name] = voltage

    def _confirm_temperature(self, servo_id, name, first, limit):
        """Re-read before believing an over-limit temperature.

        A single corrupted frame that happens to pass its checksum would
        otherwise stop the robot dead, and each servo is only polled every few
        seconds, so one bad byte costs a whole run. Meanwhile a servo that is
        genuinely cooking will still be cooking three milliseconds later -- so
        confirming costs nothing real and rules out the bus.

        Returns the temperature to record, or None to discard the sample.
        """
        retries = [self.bus.read_temperature(servo_id) for _ in range(3)]
        answered = [t for t in retries if t is not None]
        if answered and sum(t > limit for t in answered) >= 2:
            return max(answered)
        self.get_logger().warning(
            f"discarding an implausible temperature for {name}: {first} C "
            f"(limit {limit}); immediate re-reads were {retries}. This is a bus "
            "glitch, not a hot servo -- but check it by hand if it repeats: "
            f"python3 tools/sts_tool.py --baud 250000 reg {servo_id} 63"
        )
        return None

    # -- publishing --------------------------------------------------------

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.map.names)
        msg.position = [float(v) for v in self._positions]
        msg.velocity = [float(v) for v in self._velocities]
        self._state_pub.publish(msg)

    def _publish_status(self) -> None:
        parts = [f"state={self.safety.state.value}"]
        if self.safety.fault is not None:
            parts.append(f"fault={self.safety.fault}")
        if self._imu_sample is not None:
            parts.append(f"tilt={self._imu_sample.tilt_deg:.1f}deg")
        if self._temperatures:
            parts.append(f"maxtemp={max(v for v in self._temperatures.values()):.0f}C")
        if self.dry_run:
            parts.append("DRY_RUN")
        self._status_pub.publish(String(data="  ".join(parts)))

    # -- state transitions -------------------------------------------------

    def _enter_fault(self) -> None:
        if self._fault_announced:
            return
        self._fault_announced = True
        self.get_logger().fatal(
            f"\n{'=' * 60}\n  SAFETY FAULT: {self.safety.fault}\n"
            f"  Torque OFF. No further commands will be sent.\n"
            f"  This is latched -- restart the node to run again.\n{'=' * 60}")
        self._set_torque(False)
        self._publish_status()
        if bool(self.get_parameter("shutdown_on_fault").value):
            self.get_logger().fatal("shutdown_on_fault set; stopping the node")
            raise SystemExit(1)

    def _set_torque(self, enabled: bool) -> None:
        if self.dry_run:
            self.get_logger().info(f"[dry run] would set torque {'ON' if enabled else 'OFF'}")
            self._torque_on = False
            return
        try:
            self.bus.set_torque(self.servo_ids, enabled)
            self._torque_on = enabled
        except ServoBusError as exc:
            self.get_logger().error(f"torque write failed: {exc}")

    # -- services ----------------------------------------------------------

    def _srv_arm(self, request, response):
        del request
        if not self._have_positions:
            response.success, response.message = False, "no joint readings yet"
            return response
        if self._imu_sample is None:
            response.success, response.message = False, "no IMU data yet"
            return response
        if self._observation is None:
            response.success, response.message = (
                False,
                "no observation from the policy node -- is humanoid_policy running?",
            )
            return response

        problems = self.safety.arm(
            self._positions, self._imu_sample, self._observation, self.policy)
        if problems:
            text = "arm refused:\n  " + "\n  ".join(problems)
            self.get_logger().error(text)
            response.success, response.message = False, text
            return response

        for note in self.safety.arm_notes:
            self.get_logger().warning(note)

        if not self.dry_run:
            self.bus.set_speed_and_acc(
                self.servo_ids,
                int(self.get_parameter("servo_goal_speed").value),
                int(self.get_parameter("servo_acc").value))
            # Park the goal at where the joints already are, so enabling torque
            # cannot snap anything to a stale target from a previous run.
            self.bus.write_goal_positions(
                self.servo_ids, self.map.rad_to_counts(self._positions))
        self._set_torque(True)

        self.get_logger().info(
            f"ARMED -- ramping from the current pose over "
            f"{self.get_parameter('ramp_s').value:.1f}s")
        self._publish_status()
        response.success = True
        response.message = "armed; ramping"
        return response

    def _srv_disarm(self, request, response):
        del request
        self.safety.disarm()
        self._set_torque(False)
        self._publish_status()
        response.success, response.message = True, "disarmed, torque off"
        return response

    def _srv_estop(self, request, response):
        del request
        self.safety.trip("ESTOP", "operator requested emergency stop")
        self._enter_fault()
        response.success, response.message = True, "emergency stop latched"
        return response

    # -- shutdown ----------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self._set_torque(False)
        finally:
            self.bus.close()
            self.get_logger().info("bus closed, torque off")


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = ServoNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code not in (None, 0, 1):
            print(exc)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
