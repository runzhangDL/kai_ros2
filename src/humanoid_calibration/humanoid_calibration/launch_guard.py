"""Launch-file precondition: abort a launch on an uncalibrated robot.

Put this at the top of any launch file that starts motion nodes::

    from humanoid_calibration.launch_guard import calibration_gate

    def generate_launch_description():
        return LaunchDescription([
            calibration_gate(),          # <- nothing below starts if this fails
            Node(package='humanoid_walk', executable='walking', ...),
        ])

Because the gate is an ``OpaqueFunction``, it runs during launch description
evaluation -- before any process is spawned -- so a failure means no node ever
comes up, rather than a half-started stack that has to be torn down.
"""

from __future__ import annotations

from launch.actions import OpaqueFunction

from .calibration_store import CalibrationStoreError
from .guard import OVERRIDE_ENV, check, override_enabled
from .joint_config import ConfigError


class LaunchBlocked(RuntimeError):
    """Raised to abort a launch when the robot is not calibrated."""


def _gate(context, config_path: str | None, store_path: str | None, strict: bool):
    del context  # unused; required by the OpaqueFunction signature
    try:
        report, config, store = check(config_path, store_path)
    except (ConfigError, CalibrationStoreError) as exc:
        raise LaunchBlocked(f"cannot verify calibration: {exc}") from exc

    if report.ok:
        print(
            f"[calibration_gate] ok -- {len(config.joints)} joints calibrated "
            f"({store.path}, updated {store.updated_at})"
        )
        return None

    if not strict and override_enabled():
        print(
            f"[calibration_gate] WARNING: {OVERRIDE_ENV} is set, launching an "
            f"UNCALIBRATED robot. {report.reason()}"
        )
        return None

    raise LaunchBlocked(
        f"\n\n  ROBOT IS NOT CALIBRATED -- launch aborted.\n"
        f"  {report.reason()}\n\n"
        f"  Run:  ros2 run humanoid_calibration calibrate\n"
    )


def calibration_gate(
    config_path: str | None = None,
    store_path: str | None = None,
    strict: bool = True,
) -> OpaqueFunction:
    """Return a launch action that aborts the launch if calibration is invalid.

    :param strict: when True (the default) the ``$HUMANOID_ALLOW_UNCALIBRATED``
        escape hatch is ignored. Only relax it for bench/bring-up launch files
        that do not move the legs.
    """
    return OpaqueFunction(
        function=lambda context: _gate(context, config_path, store_path, strict)
    )
