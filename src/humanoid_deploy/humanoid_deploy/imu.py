"""Turn the real IMU's output into the three vectors the policy expects.

The policy consumes, in body frame (x forward, y left, z up):
    accel        m/s^2, proper acceleration -- (0, 0, +9.81) at rest
    gyro         rad/s
    gravity      unit vector along world -z -- (0, 0, -1) when upright

Mounting
--------
This robot's IMU is rotated: standing at the zero pose, its +Z points
backward, +Y points up, +X points right. With body frame x-forward/y-left/z-up
that gives

    imu +X (right)    -> body -Y
    imu +Y (up)       -> body +Z
    imu +Z (backward) -> body -X

    v_body = R @ v_imu,    R = [[ 0, 0, -1],
                                [-1, 0,  0],
                                [ 0, 1,  0]]

det(R) = +1, so it is a proper rotation, and at rest it maps the measured
(0, +9.81, 0) to (0, 0, +9.81). Both checks are unit-tested.

Units
-----
The wit_ros2_imu driver scales acceleration by ``16.0/32768``, which yields
**g, not m/s^2**, despite publishing into a field defined as m/s^2. It scales
the gyro by ``2000.0/32768`` and then calls ``np.radians``, so angular velocity
**is** already rad/s. Both are configurable here, and ``detect_accel_units``
resolves it from data rather than trust.

Gravity direction
-----------------
Three sources, in decreasing robustness:

``complementary`` (default)
    Propagate the gravity estimate with the gyro and correct it slowly toward
    the accelerometer. This is the only option that stays correct *during a
    push*, which is precisely when the policy needs it: a 12 N shove on this
    2.26 kg robot is 5.3 m/s^2, which tilts the raw accelerometer vector by
    ~28 deg. Against a training std of 0.02-0.06 on the gravity elements, that
    is a 10-20 sigma excursion -- garbage exactly when it matters.
``accel``
    Straight from the accelerometer. Exact at rest, wrong under acceleration.
    Useful for bring-up and as a cross-check.
``quaternion``
    From the IMU's own fused orientation. Faithful in principle, but this
    device's fusion assumes its own +Z is up, and here +Z is horizontal --
    a permanent ~90 deg attitude, near the singularity of its euler
    representation. Available, not the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: imu frame -> body frame for this robot's mounting.
R_IMU_TO_BODY = np.array(
    [[0.0, 0.0, -1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]]
)

STANDARD_GRAVITY = 9.80665


class ImuError(Exception):
    """Raised when IMU data cannot be interpreted safely."""


def check_rotation(matrix: np.ndarray, tol: float = 1e-6) -> None:
    """Reject anything that is not a proper rotation (R^T R = I, det = +1)."""
    if matrix.shape != (3, 3):
        raise ImuError(f"rotation must be 3x3, got {matrix.shape}")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=tol):
        raise ImuError("IMU rotation is not orthonormal")
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > tol:
        raise ImuError(
            f"IMU rotation has determinant {det:+.6f}; +1 is required. "
            "A determinant of -1 means the axis mapping includes a reflection, "
            "which would silently mirror the robot's left and right."
        )


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS (x, y, z, w) quaternion -> rotation matrix mapping body -> world."""
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm < 1e-9:
        raise ImuError("received a zero-norm orientation quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def detect_accel_units(magnitude: float) -> str:
    """Classify a resting |accel| as 'm/s^2' or 'g'. Raises if it is neither.

    Called once at start-up while the robot is held still, so a driver that
    reports g cannot quietly halve the policy's sense of which way is down.
    """
    if 8.5 <= magnitude <= 11.0:
        return "m/s^2"
    if 0.85 <= magnitude <= 1.15:
        return "g"
    raise ImuError(
        f"resting |accel| = {magnitude:.3f}, which is neither ~9.81 (m/s^2) "
        "nor ~1.0 (g). The IMU is moving, mis-scaled, or not reporting."
    )


@dataclass
class ImuSample:
    """One conditioned IMU reading, in body frame and policy units."""

    accel: np.ndarray  # m/s^2
    gyro: np.ndarray  # rad/s
    gravity: np.ndarray  # unit vector, (0,0,-1) upright

    @property
    def tilt_cos(self) -> float:
        """cos(angle from upright). 1.0 = perfectly upright, 0 = on its side."""
        return float(-self.gravity[2])

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.tilt_cos, -1.0, 1.0))))


class ImuConditioner:
    """Applies mounting rotation, unit conversion and gravity estimation."""

    def __init__(
        self,
        rotation: np.ndarray | None = None,
        accel_units: str = "auto",
        gyro_units: str = "rad/s",
        gravity_source: str = "complementary",
        complementary_tau: float = 1.0,
    ) -> None:
        self.rotation = R_IMU_TO_BODY if rotation is None else np.asarray(rotation, float)
        check_rotation(self.rotation)

        if accel_units not in ("auto", "m/s^2", "g"):
            raise ImuError(f"unknown accel_units {accel_units!r}")
        if gyro_units not in ("rad/s", "deg/s"):
            raise ImuError(f"unknown gyro_units {gyro_units!r}")
        if gravity_source not in ("complementary", "accel", "quaternion"):
            raise ImuError(f"unknown gravity_source {gravity_source!r}")

        self.accel_units = accel_units
        self.gyro_units = gyro_units
        self.gravity_source = gravity_source
        self.complementary_tau = float(complementary_tau)

        self._gravity: np.ndarray | None = None

    # -- unit handling -----------------------------------------------------

    def resolve_accel_units(self, raw_accel: np.ndarray) -> str:
        """Fix 'auto' against a resting sample. Returns the resolved units."""
        if self.accel_units == "auto":
            self.accel_units = detect_accel_units(float(np.linalg.norm(raw_accel)))
        return self.accel_units

    def _to_body(self, vector) -> np.ndarray:
        return self.rotation @ np.asarray(vector, dtype=np.float64)

    def _accel_body(self, raw_accel) -> np.ndarray:
        accel = self._to_body(raw_accel)
        if self.accel_units == "g":
            accel = accel * STANDARD_GRAVITY
        elif self.accel_units == "auto":
            raise ImuError("accel units not resolved; call resolve_accel_units first")
        return accel

    def _gyro_body(self, raw_gyro) -> np.ndarray:
        gyro = self._to_body(raw_gyro)
        if self.gyro_units == "deg/s":
            gyro = np.radians(gyro)
        return gyro

    # -- gravity -----------------------------------------------------------

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            raise ImuError("cannot normalize a zero-length gravity vector")
        return vector / norm

    def gravity_from_accel(self, accel_body: np.ndarray) -> np.ndarray:
        """At rest an accelerometer reads +g *up*, so gravity is its negation."""
        return self._normalize(-accel_body)

    def gravity_from_quaternion(self, quat_xyzw) -> np.ndarray:
        x, y, z, w = quat_xyzw
        r_world_imu = quat_to_matrix(x, y, z, w)
        gravity_imu = r_world_imu.T @ np.array([0.0, 0.0, -1.0])
        return self._normalize(self.rotation @ gravity_imu)

    def _advance_complementary(
        self, accel_body: np.ndarray, gyro_body: np.ndarray, dt: float
    ) -> np.ndarray:
        measured = self.gravity_from_accel(accel_body)
        if self._gravity is None or dt <= 0.0:
            self._gravity = measured
            return self._gravity
        # A world-fixed vector seen from a rotating body obeys dv/dt = -w x v.
        predicted = self._gravity - np.cross(gyro_body, self._gravity) * dt
        alpha = float(np.clip(dt / max(self.complementary_tau, 1e-3), 0.0, 1.0))
        self._gravity = self._normalize((1.0 - alpha) * predicted + alpha * measured)
        return self._gravity

    def reset(self) -> None:
        self._gravity = None

    # -- public ------------------------------------------------------------

    def condition(self, raw_accel, raw_gyro, dt: float, quat_xyzw=None) -> ImuSample:
        accel = self._accel_body(raw_accel)
        gyro = self._gyro_body(raw_gyro)

        if self.gravity_source == "accel":
            gravity = self.gravity_from_accel(accel)
        elif self.gravity_source == "quaternion":
            if quat_xyzw is None:
                raise ImuError("gravity_source='quaternion' but no orientation supplied")
            gravity = self.gravity_from_quaternion(quat_xyzw)
        else:
            gravity = self._advance_complementary(accel, gyro, dt)

        return ImuSample(accel=accel, gyro=gyro, gravity=gravity)

    def cross_check(self, raw_accel, quat_xyzw, tolerance_deg: float = 15.0) -> float:
        """Angle between accel-derived and quaternion-derived gravity, degrees.

        Run once at rest. They should agree closely; a large disagreement means
        the mounting rotation is wrong, or the device's fusion is not usable
        with this orientation. Returns the angle so the caller can decide.
        """
        from_accel = self.gravity_from_accel(self._accel_body(raw_accel))
        from_quat = self.gravity_from_quaternion(quat_xyzw)
        cos = float(np.clip(np.dot(from_accel, from_quat), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))
