"""Loading and validation of ``joint_limits.yaml``.

This module is deliberately free of any ``rclpy`` import so that the
calibration CLI can run on a bare terminal (or a dev laptop) without a
sourced ROS environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG_ENV = "HUMANOID_JOINT_CONFIG"
_CONFIG_BASENAME = "joint_limits.yaml"


class ConfigError(Exception):
    """Raised when the joint configuration is missing or malformed."""


@dataclass(frozen=True)
class BusConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000
    timeout_ms: int = 30
    retries: int = 2


@dataclass(frozen=True)
class ServoConfig:
    model: str = "STS3215"
    counts_per_rev: int = 4096
    present_position_addr: int = 56
    little_endian: bool = True
    signed_position: bool = False


@dataclass(frozen=True)
class CalibrationTuning:
    reach_tolerance_deg: float = 1.0
    dwell_s: float = 0.20
    refresh_hz: float = 20.0
    preflight_reads: int = 20
    min_read_success: float = 1.0


@dataclass(frozen=True)
class JointConfig:
    name: str
    id: int
    min_deg: float
    max_deg: float
    direction: int = 1

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ConfigError(
                f"joint '{self.name}': direction must be 1 or -1, got {self.direction}"
            )
        if self.min_deg >= self.max_deg:
            raise ConfigError(
                f"joint '{self.name}': min_deg ({self.min_deg}) must be < "
                f"max_deg ({self.max_deg})"
            )
        if not 0 <= self.id <= 253:
            raise ConfigError(f"joint '{self.name}': servo id {self.id} out of range 0..253")


@dataclass(frozen=True)
class RobotConfig:
    bus: BusConfig
    servo: ServoConfig
    calibration: CalibrationTuning
    joints: list[JointConfig]
    source_path: str = ""
    #: Fingerprint over everything a calibration depends on. Stored alongside
    #: the calibration so a later edit of the limits can be detected.
    fingerprint: str = field(default="", compare=False)

    def joint(self, name: str) -> JointConfig:
        for j in self.joints:
            if j.name == name:
                return j
        raise ConfigError(f"unknown joint '{name}'")

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]


def _fingerprint(servo: ServoConfig, joints: list[JointConfig]) -> str:
    """Hash the fields a stored calibration is only valid against.

    Bus settings (port, baud) are intentionally excluded: swapping the USB
    adapter does not invalidate a calibration. Limits, ids, direction and the
    encoder resolution do.
    """
    payload = {
        "counts_per_rev": servo.counts_per_rev,
        "joints": [
            {
                "name": j.name,
                "id": j.id,
                "min_deg": round(j.min_deg, 6),
                "max_deg": round(j.max_deg, 6),
                "direction": j.direction,
            }
            for j in joints
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def default_config_path() -> str:
    """Resolve the joint config path.

    Order of precedence:

    1. ``$HUMANOID_JOINT_CONFIG``
    2. the installed package share directory (normal on-robot case)
    3. ``<repo>/src/humanoid_calibration/config`` (running from a source tree)
    """
    env = os.environ.get(DEFAULT_CONFIG_ENV)
    if env:
        return os.path.expanduser(env)

    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("humanoid_calibration")
        candidate = os.path.join(share, "config", _CONFIG_BASENAME)
        if os.path.isfile(candidate):
            return candidate
    except Exception:  # noqa: BLE001 - ament not available off-robot
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "config", _CONFIG_BASENAME))


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' section must be a mapping")
    return value


def load_config(path: str | None = None) -> RobotConfig:
    """Parse and validate the joint configuration."""
    path = path or default_config_path()
    if not os.path.isfile(path):
        raise ConfigError(f"joint config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    bus = BusConfig(**_section(raw, "bus"))
    servo = ServoConfig(**_section(raw, "servo"))
    tuning = CalibrationTuning(**_section(raw, "calibration"))
    defaults = _section(raw, "defaults")

    raw_joints = raw.get("joints")
    if not isinstance(raw_joints, list) or not raw_joints:
        raise ConfigError(f"{path}: 'joints' must be a non-empty list")

    joints: list[JointConfig] = []
    seen_names: set[str] = set()
    seen_ids: dict[int, str] = {}
    for index, entry in enumerate(raw_joints):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: joints[{index}] must be a mapping")
        merged = {**defaults, **entry}
        try:
            joint = JointConfig(
                name=str(merged["name"]),
                id=int(merged["id"]),
                min_deg=float(merged["min_deg"]),
                max_deg=float(merged["max_deg"]),
                direction=int(merged.get("direction", 1)),
            )
        except KeyError as exc:
            raise ConfigError(f"{path}: joints[{index}] missing key {exc}") from exc

        if joint.name in seen_names:
            raise ConfigError(f"{path}: duplicate joint name '{joint.name}'")
        if joint.id in seen_ids:
            raise ConfigError(
                f"{path}: servo id {joint.id} used by both '{seen_ids[joint.id]}' "
                f"and '{joint.name}'"
            )
        seen_names.add(joint.name)
        seen_ids[joint.id] = joint.name
        joints.append(joint)

    return RobotConfig(
        bus=bus,
        servo=servo,
        calibration=tuning,
        joints=joints,
        source_path=os.path.abspath(path),
        fingerprint=_fingerprint(servo, joints),
    )
