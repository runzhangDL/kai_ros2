"""One-time motor calibration for the 13-DoF humanoid, and the gate that keeps
uncalibrated robots from running.

Only lightweight, dependency-free names are re-exported here. ``rclpy`` and
``launch`` are deliberately *not* imported at package level so that the
calibration CLI runs on a plain terminal without a sourced ROS environment;
import :mod:`humanoid_calibration.calibration_status_node` or
:mod:`humanoid_calibration.launch_guard` directly when you need those.
"""

from .calibration_store import (
    CalibrationStore,
    CalibrationStoreError,
    JointCalibration,
    ValidationReport,
    default_store_path,
)
from .guard import NotCalibrated, check, is_calibrated, load_calibration, require_calibration
from .joint_config import ConfigError, JointConfig, RobotConfig, load_config

__all__ = [
    "CalibrationStore",
    "CalibrationStoreError",
    "ConfigError",
    "JointCalibration",
    "JointConfig",
    "NotCalibrated",
    "RobotConfig",
    "ValidationReport",
    "check",
    "default_store_path",
    "is_calibrated",
    "load_calibration",
    "load_config",
    "require_calibration",
]
