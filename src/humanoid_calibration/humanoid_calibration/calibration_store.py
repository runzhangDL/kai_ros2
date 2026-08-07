"""Persistent storage for the robot's motor calibration.

Where the file lives
--------------------
Calibration is *machine state*, not source code: it belongs to this physical
robot and must survive rebuilds, re-flashes of the workspace and package
reinstalls. So it is written outside the ROS install space, to

    $HUMANOID_CALIBRATION_FILE                        (explicit override)
    $XDG_CONFIG_HOME/humanoid_robot/calibration.yaml  (if XDG is set)
    ~/.config/humanoid_robot/calibration.yaml         (default)

Writes are atomic (temp file + fsync + rename) so that yanking power midway
through a save can never leave a half-written calibration behind -- the file
is either the old one or the new one.

Every joint records the raw encoder counts observed at its zero and at both
limits, plus the configuration those numbers were captured against. A
fingerprint of the joint configuration is stored with them; if somebody edits
the limits in ``joint_limits.yaml`` afterwards, consumers can detect that the
calibration no longer matches and refuse to run.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import yaml

from .joint_config import RobotConfig

STORE_ENV = "HUMANOID_CALIBRATION_FILE"
STORE_VERSION = 1
_APP_DIR = "humanoid_robot"
_STORE_BASENAME = "calibration.yaml"


class CalibrationStoreError(Exception):
    """Raised when the calibration file exists but cannot be used."""


def default_store_path() -> str:
    env = os.environ.get(STORE_ENV)
    if env:
        return os.path.expanduser(env)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, _APP_DIR, _STORE_BASENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class JointCalibration:
    """Everything needed to convert this joint's counts into radians."""

    name: str
    servo_id: int
    zero_raw: int
    min_raw: int
    max_raw: int
    min_deg: float
    max_deg: float
    direction: int
    counts_per_rev: int
    calibrated_at: str

    # -- conversions used by consumer nodes --------------------------------

    def raw_to_deg(self, raw: float) -> float:
        """Joint angle in degrees for a raw encoder reading.

        The count difference is wrapped into +/- half a turn so a joint whose
        zero sits near the encoder's 0/4095 seam still reads continuously.
        Valid for any joint whose travel is under 180 deg either side of zero,
        which is every joint on this robot.
        """
        half = self.counts_per_rev / 2.0
        delta = (raw - self.zero_raw) % self.counts_per_rev
        if delta > half:
            delta -= self.counts_per_rev
        return self.direction * delta * 360.0 / self.counts_per_rev

    def deg_to_raw(self, degrees: float) -> float:
        raw = self.zero_raw + self.direction * degrees * self.counts_per_rev / 360.0
        return raw % self.counts_per_rev

    def clamp_deg(self, degrees: float) -> float:
        """Clamp a commanded angle to this joint's calibrated travel."""
        return max(self.min_deg, min(self.max_deg, degrees))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JointCalibration":
        try:
            return cls(
                name=str(data["name"]),
                servo_id=int(data["servo_id"]),
                zero_raw=int(data["zero_raw"]),
                min_raw=int(data["min_raw"]),
                max_raw=int(data["max_raw"]),
                min_deg=float(data["min_deg"]),
                max_deg=float(data["max_deg"]),
                direction=int(data["direction"]),
                counts_per_rev=int(data["counts_per_rev"]),
                calibrated_at=str(data.get("calibrated_at", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationStoreError(f"malformed joint entry: {exc}") from exc


@dataclass
class ValidationReport:
    """Outcome of checking a stored calibration against the live config."""

    path: str
    exists: bool = False
    version_ok: bool = False
    fingerprint_ok: bool = False
    missing: list[str] = None  # type: ignore[assignment]
    stale: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.missing = self.missing or []
        self.stale = self.stale or []

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and self.version_ok
            and self.fingerprint_ok
            and not self.missing
            and not self.stale
        )

    def reason(self) -> str:
        """A single operator-readable sentence explaining a failure."""
        if self.ok:
            return "calibration is valid"
        if not self.exists:
            return (
                f"no calibration file at {self.path} -- "
                "run: ros2 run humanoid_calibration calibrate"
            )
        if not self.version_ok:
            return f"{self.path} was written by an incompatible version -- re-run calibration"
        if self.missing:
            joints = ", ".join(self.missing)
            return f"these joints have never been calibrated: {joints}"
        if not self.fingerprint_ok or self.stale:
            joints = ", ".join(self.stale) if self.stale else "all joints"
            return (
                f"joint_limits.yaml changed since calibration ({joints}) -- "
                "re-run calibration for the affected joints"
            )
        return "calibration is invalid"


class CalibrationStore:
    """Read/modify/write access to the calibration file."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_store_path()
        self.version: int = STORE_VERSION
        self.fingerprint: str = ""
        self.config_path: str = ""
        self.created_at: str = ""
        self.updated_at: str = ""
        self.joints: dict[str, JointCalibration] = {}

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "CalibrationStore":
        """Load the store, returning an empty one if the file does not exist."""
        store = cls(path)
        if not os.path.isfile(store.path):
            return store
        try:
            with open(store.path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise CalibrationStoreError(f"cannot read {store.path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CalibrationStoreError(f"{store.path}: top level must be a mapping")

        store.version = int(raw.get("version", 0))
        store.fingerprint = str(raw.get("config_fingerprint", ""))
        store.config_path = str(raw.get("config_path", ""))
        store.created_at = str(raw.get("created_at", ""))
        store.updated_at = str(raw.get("updated_at", ""))
        for name, entry in (raw.get("joints") or {}).items():
            store.joints[str(name)] = JointCalibration.from_dict(entry)
        return store

    def save(self) -> None:
        """Atomically write the store to disk."""
        self.updated_at = utc_now()
        if not self.created_at:
            self.created_at = self.updated_at

        document = {
            "version": self.version,
            "config_fingerprint": self.fingerprint,
            "config_path": self.config_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "joints": {
                name: self.joints[name].to_dict() for name in sorted(self.joints)
            },
        }
        header = (
            "# Humanoid motor calibration -- generated by "
            "`ros2 run humanoid_calibration calibrate`.\n"
            "# Do not edit by hand. Delete this file to force a full recalibration.\n"
        )

        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".calibration-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(header)
                yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Make the rename itself durable.
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    # -- mutation ----------------------------------------------------------

    def record(self, calibration: JointCalibration, fingerprint: str, config_path: str) -> None:
        self.joints[calibration.name] = calibration
        self.fingerprint = fingerprint
        self.config_path = config_path
        self.version = STORE_VERSION

    def clear(self, names: list[str] | None = None) -> None:
        if names is None:
            self.joints.clear()
        else:
            for name in names:
                self.joints.pop(name, None)

    # -- validation --------------------------------------------------------

    def validate(self, config: RobotConfig) -> ValidationReport:
        report = ValidationReport(path=self.path)
        report.exists = os.path.isfile(self.path)
        if not report.exists:
            report.missing = list(config.joint_names)
            return report

        report.version_ok = self.version == STORE_VERSION
        report.fingerprint_ok = self.fingerprint == config.fingerprint

        for joint in config.joints:
            stored = self.joints.get(joint.name)
            if stored is None:
                report.missing.append(joint.name)
                continue
            # Per-joint staleness: only flag joints whose own numbers moved.
            if (
                stored.servo_id != joint.id
                or stored.direction != joint.direction
                or abs(stored.min_deg - joint.min_deg) > 1e-6
                or abs(stored.max_deg - joint.max_deg) > 1e-6
                or stored.counts_per_rev != config.servo.counts_per_rev
            ):
                report.stale.append(joint.name)
        return report

    def is_calibrated(self, config: RobotConfig) -> bool:
        return self.validate(config).ok
