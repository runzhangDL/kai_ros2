"""The gate every motion node must pass through before it runs.

ROS 2 has no built-in notion of "this robot is calibrated", so we define one:
a validated calibration file (see :mod:`calibration_store`) is the single
source of truth, and this module is the only thing anyone needs to call.

There are three enforcement points, in order of how early they catch a
mistake:

1. **Launch time** -- ``calibration_gate()`` from :mod:`launch_guard` aborts
   the whole launch before a single node starts. Cheapest failure.
2. **Node start-up** -- ``require_calibration(node)`` logs a fatal message and
   raises :class:`NotCalibrated`, so a node started by hand still refuses.
3. **Runtime** -- the ``calibration_status`` node publishes a latched
   ``/calibration_status/ok`` for anything that wants to watch it.

Typical use in a node::

    from humanoid_calibration.guard import require_calibration

    class WalkingNode(Node):
        def __init__(self):
            super().__init__('walking')
            self.calib = require_calibration(self)   # raises if not calibrated
            ...

``require_calibration`` returns the per-joint calibration, so a node that
needs the numbers gets them from the same call that enforces the check --
there is no way to skip the check and still get the data.
"""

from __future__ import annotations

import argparse
import os
import sys

from .calibration_store import (
    CalibrationStore,
    CalibrationStoreError,
    JointCalibration,
    ValidationReport,
)
from .joint_config import ConfigError, RobotConfig, load_config

#: Escape hatch for bench work. Set to 1/true/yes to downgrade the hard failure
#: to a warning. Never set this on a robot that is going to stand up.
OVERRIDE_ENV = "HUMANOID_ALLOW_UNCALIBRATED"


class NotCalibrated(RuntimeError):
    """Raised when a node is started on an uncalibrated robot."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.reason())


def override_enabled() -> bool:
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def check(
    config_path: str | None = None, store_path: str | None = None
) -> tuple[ValidationReport, RobotConfig | None, CalibrationStore]:
    """Validate the stored calibration against the joint config.

    Never raises for the "not calibrated" case -- that is reported in the
    returned :class:`ValidationReport`. Only a broken/unreadable config or
    calibration file raises.
    """
    config = load_config(config_path)
    store = CalibrationStore.load(store_path)
    return store.validate(config), config, store


def is_calibrated(config_path: str | None = None, store_path: str | None = None) -> bool:
    try:
        report, _, _ = check(config_path, store_path)
    except (ConfigError, CalibrationStoreError):
        return False
    return report.ok


def _banner(report: ValidationReport) -> str:
    return (
        "\n"
        "  ============================================================\n"
        "   ROBOT IS NOT CALIBRATED -- refusing to start\n"
        "  ============================================================\n"
        f"   {report.reason()}\n"
        "\n"
        "   Run the one-time calibration first:\n"
        "       ros2 run humanoid_calibration calibrate\n"
        "\n"
        "   Check status at any time with:\n"
        "       ros2 run humanoid_calibration check_calibration\n"
        "  ============================================================\n"
    )


def require_calibration(
    node=None,
    config_path: str | None = None,
    store_path: str | None = None,
    allow_override: bool = True,
) -> dict[str, JointCalibration]:
    """Assert the robot is calibrated and return the per-joint calibration.

    :param node: optional ``rclpy`` node, used only for logging.
    :param allow_override: whether ``$HUMANOID_ALLOW_UNCALIBRATED`` may
        downgrade the failure to a warning. Pass ``False`` from anything that
        actually drives the legs.
    :raises NotCalibrated: if the robot is not calibrated.
    """
    log_error = node.get_logger().fatal if node is not None else (
        lambda message: print(message, file=sys.stderr)
    )
    log_warn = node.get_logger().warning if node is not None else (
        lambda message: print(message, file=sys.stderr)
    )
    log_info = node.get_logger().info if node is not None else print

    try:
        report, config, store = check(config_path, store_path)
    except (ConfigError, CalibrationStoreError) as exc:
        log_error(f"calibration check failed: {exc}")
        raise

    if report.ok:
        assert config is not None
        log_info(
            f"calibration ok: {len(config.joints)} joints, "
            f"recorded {store.updated_at} ({store.path})"
        )
        return dict(store.joints)

    if allow_override and override_enabled():
        log_warn(
            f"{OVERRIDE_ENV} is set -- continuing on an UNCALIBRATED robot. "
            f"{report.reason()}"
        )
        return dict(store.joints)

    log_error(_banner(report))
    raise NotCalibrated(report)


def load_calibration(
    config_path: str | None = None, store_path: str | None = None
) -> dict[str, JointCalibration]:
    """Return the calibration without any logging. Raises if invalid."""
    report, _, store = check(config_path, store_path)
    if not report.ok:
        raise NotCalibrated(report)
    return dict(store.joints)


# -- `ros2 run humanoid_calibration check_calibration` --------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_calibration",
        description="Exit 0 if the robot is calibrated, 1 if it is not. "
        "Suitable for scripts, systemd units and launch preconditions.",
    )
    parser.add_argument("--config", help="path to joint_limits.yaml")
    parser.add_argument("--store", help="path to the calibration file")
    parser.add_argument("-q", "--quiet", action="store_true", help="print nothing")
    args = parser.parse_args(argv)

    try:
        report, config, store = check(args.config, args.store)
    except (ConfigError, CalibrationStoreError) as exc:
        if not args.quiet:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.quiet:
        return 0 if report.ok else 1

    from .calibrate_cli import print_status  # local import: keeps startup light
    from . import term_ui as ui

    assert config is not None
    print_status(config, store, ui.supports_color())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
