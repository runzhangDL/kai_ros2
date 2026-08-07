"""Inference node: sensors in, joint position targets out.

This node owns no hardware. It subscribes to the IMU and to the joint states
published by ``servo_node``, assembles the 48-element observation the policy
was trained on, runs the network, and publishes position targets in radians.

It is driven by ``/joint_states`` rather than by its own timer, so the loop is
sensor -> compute -> command with one control period of latency rather than
two. It does *no* safety enforcement: everything it publishes is a request,
and ``servo_node`` is free to clamp, ramp, rate-limit or ignore it. Keeping the
authority on the far side of the boundary means a hung or crashed policy node
cannot drive the robot -- the command simply goes stale and the executor
faults.

Published for the executor and for debugging:
  ~/joint_command   sensor_msgs/JointState  targets, rad, named
  ~/observation     std_msgs/Float64MultiArray  the exact 48-vector fed in
  ~/action          std_msgs/Float64MultiArray  raw tanh action, [-1, 1]
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from .imu import ImuConditioner, ImuError
from .policy import Policy, PolicyError

_SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("humanoid_policy")

        self.declare_parameter("bundle", "")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("accel_units", "auto")
        self.declare_parameter("gyro_units", "rad/s")
        self.declare_parameter("gravity_source", "complementary")
        self.declare_parameter("complementary_tau", 1.0)
        self.declare_parameter("imu_rotation", list(np.eye(3).ravel()))
        self.declare_parameter("use_default_rotation", True)
        self.declare_parameter("warn_ood_sigma", 8.0)

        bundle = self.get_parameter("bundle").value
        if not bundle:
            raise SystemExit(
                "parameter 'bundle' is required: the exported policy_bundle.npz"
            )
        try:
            self.policy = Policy(bundle)
        except PolicyError as exc:
            raise SystemExit(f"cannot load policy: {exc}") from exc
        self.get_logger().info(self.policy.describe())

        rotation = None
        if not self.get_parameter("use_default_rotation").value:
            rotation = np.asarray(
                self.get_parameter("imu_rotation").value, dtype=float
            ).reshape(3, 3)
        try:
            self.imu = ImuConditioner(
                rotation=rotation,
                accel_units=self.get_parameter("accel_units").value,
                gyro_units=self.get_parameter("gyro_units").value,
                gravity_source=self.get_parameter("gravity_source").value,
                complementary_tau=float(self.get_parameter("complementary_tau").value),
            )
        except ImuError as exc:
            raise SystemExit(f"IMU configuration rejected: {exc}") from exc

        self._warn_sigma = float(self.get_parameter("warn_ood_sigma").value)
        self._nu = self.policy.nu
        self._layout = self.policy.layout
        self._last_action = np.zeros(self._nu)
        self._imu_msg: Imu | None = None
        self._last_imu_time: float | None = None
        self._units_resolved = False
        self._warned_once = False

        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self._on_imu, _SENSOR_QOS
        )
        self.create_subscription(
            JointState, "/humanoid/joint_states", self._on_joints, _SENSOR_QOS
        )
        self._cmd_pub = self.create_publisher(
            JointState, "~/joint_command", _SENSOR_QOS
        )
        self._obs_pub = self.create_publisher(
            Float64MultiArray, "~/observation", _SENSOR_QOS
        )
        self._act_pub = self.create_publisher(
            Float64MultiArray, "~/action", _SENSOR_QOS
        )

        self.get_logger().info("policy node up; waiting for IMU and joint states")

    # -- callbacks ---------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        self._imu_msg = msg

    def _on_joints(self, msg: JointState) -> None:
        """Drive one inference step. Fires at the executor's control rate."""
        if self._imu_msg is None:
            return

        # Reorder by name: never trust positional agreement between nodes.
        try:
            index = [msg.name.index(name) for name in self.policy.joint_names]
        except ValueError as exc:
            self.get_logger().error(
                f"joint_states is missing a joint the policy drives: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        if len(msg.position) < len(msg.name) or len(msg.velocity) < len(msg.name):
            self.get_logger().error("joint_states lacks position/velocity",
                                    throttle_duration_sec=5.0)
            return
        qpos = np.asarray(msg.position, dtype=np.float64)[index]
        qvel = np.asarray(msg.velocity, dtype=np.float64)[index]

        imu_msg = self._imu_msg
        raw_accel = np.array([imu_msg.linear_acceleration.x,
                              imu_msg.linear_acceleration.y,
                              imu_msg.linear_acceleration.z])
        raw_gyro = np.array([imu_msg.angular_velocity.x,
                             imu_msg.angular_velocity.y,
                             imu_msg.angular_velocity.z])
        quat = (imu_msg.orientation.x, imu_msg.orientation.y,
                imu_msg.orientation.z, imu_msg.orientation.w)

        if not self._units_resolved:
            try:
                units = self.imu.resolve_accel_units(raw_accel)
            except ImuError as exc:
                self.get_logger().warning(f"accel unit detection: {exc}",
                                          throttle_duration_sec=2.0)
                return
            self._units_resolved = True
            self.get_logger().info(
                f"accelerometer resolved as {units}"
                + ("  (converting to m/s^2)" if units == "g" else "")
            )

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self._last_imu_time is None else max(0.0, now - self._last_imu_time)
        self._last_imu_time = now

        try:
            sample = self.imu.condition(raw_accel, raw_gyro, dt, quat)
        except ImuError as exc:
            self.get_logger().error(f"IMU conditioning failed: {exc}",
                                    throttle_duration_sec=2.0)
            return

        obs = np.empty(self._layout.size, dtype=np.float64)
        obs[self._layout.accel] = sample.accel
        obs[self._layout.gyro] = sample.gyro
        obs[self._layout.gravity] = sample.gravity
        obs[self._layout.qpos] = qpos
        obs[self._layout.qvel] = qvel
        obs[self._layout.action] = self._last_action

        action = self.policy.act(obs)
        targets = self.policy.action_to_targets(action)
        self._last_action = action

        offenders = self.policy.out_of_distribution(obs, self._warn_sigma)
        if offenders and not self._warned_once:
            self._warned_once = True
            summary = ", ".join(f"{label}={z:+.1f}s" for _, label, _, z in offenders[:6])
            self.get_logger().warning(
                f"observation is far outside the training distribution: {summary}. "
                "Check IMU axes and units before arming."
            )

        stamp = self.get_clock().now().to_msg()
        command = JointState()
        command.header.stamp = stamp
        command.name = list(self.policy.joint_names)
        command.position = [float(v) for v in targets]
        self._cmd_pub.publish(command)

        self._obs_pub.publish(Float64MultiArray(data=[float(v) for v in obs]))
        self._act_pub.publish(Float64MultiArray(data=[float(v) for v in action]))


def main(argv=None) -> None:
    rclpy.init(args=argv)
    try:
        node = PolicyNode()
    except SystemExit as exc:
        print(f"policy node failed to start: {exc}")
        rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
