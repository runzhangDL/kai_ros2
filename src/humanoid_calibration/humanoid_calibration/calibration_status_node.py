"""Publishes the robot's calibration state so any node can observe it.

Topics (both ``TRANSIENT_LOCAL`` + ``RELIABLE``, depth 1, so a node that
starts late still receives the last value immediately):

    /calibration_status/ok      std_msgs/Bool    -- the one-bit answer
    /calibration_status/status  std_msgs/String  -- human-readable detail

Service:

    /calibration_status/check   std_srvs/Trigger -- re-read the files, re-publish

Those are the default node name's resolved names; the declarations below use
``~/`` so a remapped node name moves them with it.

The file is re-validated periodically, so deleting the calibration or editing
``joint_limits.yaml`` on a running robot is noticed within one period.

This node is for *observability*. It is not the enforcement mechanism -- a node
that must not run uncalibrated should call
``humanoid_calibration.guard.require_calibration()`` at start-up, which fails
closed instead of waiting for a topic that may never arrive.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .calibration_store import CalibrationStoreError
from .guard import check
from .joint_config import ConfigError

_LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class CalibrationStatusNode(Node):
    def __init__(self) -> None:
        super().__init__("calibration_status")

        self.declare_parameter("config_path", "")
        self.declare_parameter("store_path", "")
        self.declare_parameter("period_s", 5.0)

        self._config_path = self.get_parameter("config_path").value or None
        self._store_path = self.get_parameter("store_path").value or None

        self._ok_pub = self.create_publisher(Bool, "~/ok", _LATCHED)
        self._status_pub = self.create_publisher(String, "~/status", _LATCHED)
        self._service = self.create_service(Trigger, "~/check", self._on_check)

        self._last_ok: bool | None = None
        self._publish()
        period = float(self.get_parameter("period_s").value)
        self._timer = self.create_timer(max(0.5, period), self._publish)

    def _evaluate(self) -> tuple[bool, str]:
        try:
            report, config, store = check(self._config_path, self._store_path)
        except (ConfigError, CalibrationStoreError) as exc:
            return False, f"calibration could not be verified: {exc}"
        if report.ok:
            return True, (
                f"calibrated: {len(config.joints)} joints, "
                f"updated {store.updated_at} ({store.path})"
            )
        return False, report.reason()

    def _publish(self) -> None:
        ok, detail = self._evaluate()
        self._ok_pub.publish(Bool(data=ok))
        self._status_pub.publish(String(data=detail))

        # Log only on transitions, so a healthy robot stays quiet.
        if ok != self._last_ok:
            (self.get_logger().info if ok else self.get_logger().error)(detail)
            self._last_ok = ok

    def _on_check(self, request, response):
        del request
        self._publish()
        response.success, response.message = self._evaluate()
        return response


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = CalibrationStatusNode()
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
