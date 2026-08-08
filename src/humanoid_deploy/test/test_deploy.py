"""Tests for the deployment logic that must not be wrong.

Everything here runs without ROS, without jax and without hardware. The
emphasis is on the failure modes that would damage the robot: wrong axes,
wrong units, wrong joint order, commanding across the encoder seam, and a
safety supervisor that lets go when it should latch.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "humanoid_calibration"))

from humanoid_calibration.calibration_store import JointCalibration  # noqa: E402

from humanoid_deploy.imu import (  # noqa: E402
    R_IMU_TO_BODY,
    ImuConditioner,
    ImuError,
    check_rotation,
    detect_accel_units,
    quat_to_matrix,
)
from humanoid_deploy.joint_map import (  # noqa: E402
    JointMap,
    JointMapError,
    resolve_joint_names,
)
from humanoid_deploy.policy import ObservationLayout, swish  # noqa: E402
from humanoid_deploy.safety import (  # noqa: E402
    SafetyConfig,
    SafetyState,
    SafetySupervisor,
)
from humanoid_deploy.servo_bus import (  # noqa: E402
    ServoBusError,
    build_sync_read,
    build_sync_write,
    parse_sync_read,
)

G = 9.80665


# =====================================================================
# IMU
# =====================================================================


def test_mounting_rotation_is_a_proper_rotation():
    """A determinant of -1 would mirror left and right without any other sign."""
    check_rotation(R_IMU_TO_BODY)
    assert np.linalg.det(R_IMU_TO_BODY) == pytest.approx(1.0)


def test_reflection_is_rejected():
    mirrored = R_IMU_TO_BODY.copy()
    mirrored[:, 0] *= -1  # det -> -1
    with pytest.raises(ImuError, match="determinant"):
        check_rotation(mirrored)


def test_mounting_maps_the_documented_axes():
    """imu +X->body -Y, +Y->body +Z, +Z->body -X."""
    assert np.allclose(R_IMU_TO_BODY @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(R_IMU_TO_BODY @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(R_IMU_TO_BODY @ [0, 0, 1], [-1, 0, 0])


def test_resting_robot_gives_the_training_gravity_vector():
    """The whole point: standing still must produce gravity = (0, 0, -1).

    With this mounting the IMU's +Y points up, so at rest it reads +g on its
    own Y axis.
    """
    conditioner = ImuConditioner(accel_units="m/s^2", gravity_source="accel")
    sample = conditioner.condition([0.0, G, 0.0], [0.0, 0.0, 0.0], dt=0.02)
    assert np.allclose(sample.accel, [0, 0, G], atol=1e-6)
    assert np.allclose(sample.gravity, [0, 0, -1], atol=1e-6)
    assert sample.tilt_cos == pytest.approx(1.0)
    assert sample.tilt_deg == pytest.approx(0.0, abs=1e-4)


def test_accel_in_g_is_converted():
    conditioner = ImuConditioner(accel_units="g", gravity_source="accel")
    sample = conditioner.condition([0.0, 1.0, 0.0], [0, 0, 0], dt=0.02)
    assert np.allclose(sample.accel, [0, 0, G], atol=1e-6)


def test_accel_unit_detection():
    assert detect_accel_units(9.81) == "m/s^2"
    assert detect_accel_units(1.0) == "g"
    with pytest.raises(ImuError):
        detect_accel_units(4.0)


def test_auto_units_resolve_from_a_resting_sample():
    conditioner = ImuConditioner(accel_units="auto", gravity_source="accel")
    assert conditioner.resolve_accel_units(np.array([0.0, 1.0, 0.0])) == "g"
    sample = conditioner.condition([0.0, 1.0, 0.0], [0, 0, 0], dt=0.02)
    assert np.allclose(sample.accel, [0, 0, G], atol=1e-6)


def test_gyro_deg_per_second_is_converted():
    conditioner = ImuConditioner(accel_units="m/s^2", gyro_units="deg/s",
                                 gravity_source="accel")
    # 90 deg/s about the IMU's +Y, which is body +Z.
    sample = conditioner.condition([0, G, 0], [0.0, 90.0, 0.0], dt=0.02)
    assert np.allclose(sample.gyro, [0, 0, np.pi / 2], atol=1e-6)


def test_tilt_is_reported_for_a_leaning_robot():
    conditioner = ImuConditioner(accel_units="m/s^2", gravity_source="accel")
    # 30 deg lean: gravity tilts in body x.
    accel_body = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))]) * G
    accel_imu = R_IMU_TO_BODY.T @ accel_body
    sample = conditioner.condition(accel_imu, [0, 0, 0], dt=0.02)
    assert sample.tilt_deg == pytest.approx(30.0, abs=1e-3)


def test_complementary_filter_rejects_a_push():
    """The reason it exists: a shove corrupts the raw accelerometer direction.

    12 N on 2.26 kg is 5.3 m/s^2, which swings the raw gravity estimate by
    ~28 deg. The filter should barely move over a short push.
    """
    conditioner = ImuConditioner(accel_units="m/s^2",
                                 gravity_source="complementary",
                                 complementary_tau=1.0)
    upright = R_IMU_TO_BODY.T @ np.array([0.0, 0.0, G])
    for _ in range(200):  # settle
        conditioner.condition(upright, [0, 0, 0], dt=0.02)

    pushed_body = np.array([5.3, 0.0, G])
    pushed = R_IMU_TO_BODY.T @ pushed_body
    for _ in range(10):  # 0.2 s of shove, no rotation yet
        sample = conditioner.condition(pushed, [0, 0, 0], dt=0.02)

    naive = ImuConditioner(accel_units="m/s^2", gravity_source="accel")
    naive_sample = naive.condition(pushed, [0, 0, 0], dt=0.02)

    assert naive_sample.tilt_deg > 25.0        # raw accel is badly fooled
    assert sample.tilt_deg < 6.0               # filtered barely moves


def test_complementary_filter_tracks_a_real_lean():
    conditioner = ImuConditioner(accel_units="m/s^2",
                                 gravity_source="complementary",
                                 complementary_tau=0.5)
    leaned_body = np.array([np.sin(np.radians(20)), 0, np.cos(np.radians(20))]) * G
    leaned = R_IMU_TO_BODY.T @ leaned_body
    for _ in range(500):
        sample = conditioner.condition(leaned, [0, 0, 0], dt=0.02)
    assert sample.tilt_deg == pytest.approx(20.0, abs=0.5)


def test_quaternion_gravity_matches_accel_when_level():
    conditioner = ImuConditioner(accel_units="m/s^2", gravity_source="quaternion")
    # +90 deg about world X is the attitude that puts the IMU's +Y along world
    # up, which is how this robot's IMU sits when standing.
    angle = np.pi / 2
    quat = (np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2))
    gravity = conditioner.gravity_from_quaternion(quat)
    assert np.allclose(gravity, [0, 0, -1], atol=1e-6)


def test_quaternion_gravity_flips_if_the_convention_is_wrong():
    """Guards the sign: the opposite attitude must give +1, not -1.

    A silent inversion here would tell the policy the robot is upside down.
    It is caught at arm time too -- grav_z has a training std of 0.02, so +1
    lands ~99 sigma out -- but the sign is pinned here as well.
    """
    conditioner = ImuConditioner(accel_units="m/s^2", gravity_source="quaternion")
    angle = -np.pi / 2
    quat = (np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2))
    assert np.allclose(conditioner.gravity_from_quaternion(quat), [0, 0, 1], atol=1e-6)


def test_quat_to_matrix_rejects_zero_norm():
    with pytest.raises(ImuError):
        quat_to_matrix(0.0, 0.0, 0.0, 0.0)


# =====================================================================
# Joint map
# =====================================================================

CPR = 4096
PER_RAD = CPR / (2 * np.pi)


def make_cal(name, servo_id, zero=2048, lo_deg=-40.0, hi_deg=25.0, direction=1):
    return JointCalibration(
        name=name, servo_id=servo_id, zero_raw=zero,
        min_raw=int(zero + direction * np.radians(lo_deg) * PER_RAD) % CPR,
        max_raw=int(zero + direction * np.radians(hi_deg) * PER_RAD) % CPR,
        min_deg=lo_deg, max_deg=hi_deg, direction=direction,
        counts_per_rev=CPR, calibrated_at="",
    )


def make_map(names=("a", "b"), zeros=(2048, 2048), directions=(1, 1),
             lo=(-40.0, -40.0), hi=(25.0, 25.0), action_scale=0.4, margin=0.0):
    cals = {
        name: make_cal(name, i + 1, zeros[i], lo[i], hi[i], directions[i])
        for i, name in enumerate(names)
    }
    return JointMap(
        joint_names=list(names), calibrations=cals,
        xml_lower=[np.radians(v) for v in lo],
        xml_upper=[np.radians(v) for v in hi],
        action_scale=action_scale, limit_margin_rad=margin,
    )


def test_map_is_built_by_name_not_position():
    """Calibration order and model order differ; the mapping must not care."""
    cals = {"knee": make_cal("knee", 7), "ankle": make_cal("ankle", 3)}
    jm = JointMap(["ankle", "knee"], cals, [-1.0, -1.0], [1.0, 1.0], 0.4)
    assert jm.names == ["ankle", "knee"]
    assert list(jm.servo_ids) == [3, 7]   # follows the model's order, not the dict's


def test_missing_calibration_is_fatal():
    with pytest.raises(JointMapError, match="no calibrated joint"):
        JointMap(["a", "ghost"], {"a": make_cal("a", 1)}, [-1, -1], [1, 1], 0.4)


def test_uncommanded_calibrated_joint_is_fatal():
    cals = {"a": make_cal("a", 1), "spare": make_cal("spare", 2)}
    with pytest.raises(JointMapError, match="not driven by the model"):
        JointMap(["a"], cals, [-1.0], [1.0], 0.4)


# -- name resolution ------------------------------------------------------

# The real robot's two naming conventions and, critically, two ORDERS: the
# MuJoCo model enumerates the left leg top-down while the calibration walks it
# bottom-up. The right leg happens to agree. Nothing may be positional.
MODEL_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "waist_yaw_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
]
CALIB_ORDER = [
    "left_ankle_pitch", "left_ankle_roll", "left_knee_pitch",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "waist_yaw",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_roll", "right_ankle_pitch",
]


def test_joint_suffix_is_resolved():
    mapping, notes = resolve_joint_names(MODEL_ORDER, CALIB_ORDER)
    assert mapping["left_hip_pitch_joint"] == "left_hip_pitch"
    assert len(mapping) == 13
    assert len(notes) == 13          # every one needed the suffix stripped


def test_resolution_survives_the_reversed_left_leg():
    """Model order != calibration order; the mapping must follow names."""
    cals = {name: make_cal(name, i + 1) for i, name in enumerate(CALIB_ORDER)}
    jm = JointMap(MODEL_ORDER, cals, [-1.0] * 13, [1.0] * 13, 0.4)
    # left_hip_pitch is index 5 in the calibration -> servo id 6, but index 0
    # in the model.
    assert jm.names[0] == "left_hip_pitch_joint"
    assert jm.servo_ids[0] == 6
    assert jm.names[5] == "left_ankle_pitch_joint"
    assert jm.servo_ids[5] == 1
    # ...while the right leg, which does agree, keeps its ids in order.
    assert list(jm.servo_ids[7:]) == [8, 9, 10, 11, 12, 13]


def test_exact_names_need_no_rewriting():
    mapping, notes = resolve_joint_names(["a", "b"], ["b", "a"])
    assert mapping == {"a": "a", "b": "b"}
    assert notes == []


def test_explicit_alias_wins():
    mapping, notes = resolve_joint_names(["hip_j"], ["hip_left"],
                                         aliases={"hip_j": "hip_left"})
    assert mapping == {"hip_j": "hip_left"}
    assert "alias" in notes[0]


def test_ambiguous_name_is_refused_rather_than_guessed():
    with pytest.raises(JointMapError, match="matches several"):
        resolve_joint_names(["knee_joint"], ["knee", "knee_jnt"])


def test_many_to_one_resolution_is_refused():
    with pytest.raises(JointMapError, match="matched more than once"):
        resolve_joint_names(["knee", "knee_joint"], ["knee", "spare"])


def test_duplicate_servo_id_is_fatal():
    cals = {"a": make_cal("a", 5), "b": make_cal("b", 5)}
    with pytest.raises(JointMapError, match="claimed by both"):
        JointMap(["a", "b"], cals, [-1, -1], [1, 1], 0.4)


def test_counts_and_radians_round_trip():
    jm = make_map()
    for degrees in (-30.0, -5.0, 0.0, 10.0, 20.0):
        radians = np.radians([degrees, degrees])
        counts = jm.rad_to_counts(radians)
        assert np.allclose(jm.counts_to_rad(counts), radians, atol=2e-3)


def test_counts_to_rad_takes_the_short_way_round_the_seam():
    jm = make_map(zeros=(10, 10))
    # 20 counts below a zero of 10 wraps to 4086; must read as negative.
    assert jm.counts_to_rad([4086, 4086])[0] < 0
    assert jm.counts_to_rad([4086, 4086])[0] == pytest.approx(-20 / PER_RAD, abs=1e-6)


def test_direction_inverts_the_mapping():
    jm = make_map(directions=(-1, -1))
    counts = jm.rad_to_counts([0.1, 0.1])
    assert counts[0] < 2048          # negative direction -> fewer counts
    assert jm.counts_to_rad(counts)[0] == pytest.approx(0.1, abs=2e-3)


def test_safe_envelope_is_the_intersection():
    """Calibration says +/-40/25; the model says +/-10. Take the tighter."""
    cals = {"a": make_cal("a", 1, lo_deg=-40.0, hi_deg=25.0)}
    jm = JointMap(["a"], cals, [np.radians(-10.0)], [np.radians(10.0)], 0.4)
    assert jm.safe_lower[0] == pytest.approx(np.radians(-10.0))
    assert jm.safe_upper[0] == pytest.approx(np.radians(10.0))


def test_margin_never_excludes_the_nominal_pose():
    """A knee that cannot hyper-extend stands *at* its limit.

    left_knee_pitch is [-90, 0] with the standing pose at 0. A naive symmetric
    margin would make the envelope [-88, -2], which excludes the very pose the
    policy targets -- the robot could then never arm.
    """
    cals = {"knee": make_cal("knee", 1, lo_deg=-90.0, hi_deg=0.0)}
    jm = JointMap(["knee"], cals, [np.radians(-90.0)], [0.0], 0.4,
                  limit_margin_rad=np.radians(2.0))
    assert jm.safe_upper[0] == pytest.approx(0.0)             # not -2 deg
    assert jm.safe_lower[0] == pytest.approx(np.radians(-88.0))
    assert jm.clamp([0.0])[0] == pytest.approx(0.0)
    assert jm.unmargined == ["knee"]


def test_margin_applies_normally_when_the_pose_is_interior():
    cals = {"a": make_cal("a", 1, lo_deg=-40.0, hi_deg=25.0)}
    jm = JointMap(["a"], cals, [np.radians(-40.0)], [np.radians(25.0)], 0.4,
                  limit_margin_rad=np.radians(2.0))
    assert jm.safe_lower[0] == pytest.approx(np.radians(-38.0))
    assert jm.safe_upper[0] == pytest.approx(np.radians(23.0))
    assert jm.unmargined == []


def test_nominal_pose_outside_the_hard_range_is_fatal():
    """Calibration and model disagreeing about where zero is must not pass."""
    cals = {"a": make_cal("a", 1, lo_deg=10.0, hi_deg=40.0)}
    with pytest.raises(JointMapError, match="nominal pose lies outside"):
        JointMap(["a"], cals, [np.radians(10.0)], [np.radians(40.0)], 0.4)


def test_zero_width_range_is_fatal():
    """A joint with no travel at all cannot be commanded; refuse rather than
    emit a constant. (Non-overlapping ranges are caught earlier, by the
    nominal-pose check.)"""
    cals = {"a": make_cal("a", 1, lo_deg=0.0, hi_deg=0.0)}
    with pytest.raises(JointMapError, match="empty safe range"):
        JointMap(["a"], cals, [0.0], [0.0], 0.4)


def test_command_window_is_bounded_by_action_scale():
    jm = make_map(lo=(-90.0, -90.0), hi=(90.0, 90.0), action_scale=0.4)
    lo, hi = jm.command_window()
    assert lo[0] == pytest.approx(-0.4)
    assert hi[0] == pytest.approx(0.4)


def test_seam_violation_is_detected_when_zero_sits_near_the_wrap():
    """A zero at count 100 with a +/-0.4 rad window reaches below 0 -> unsafe."""
    jm = make_map(zeros=(100, 2048), lo=(-90.0, -90.0), hi=(90.0, 90.0))
    violations = jm.seam_violations()
    assert [v.name for v in violations] == ["a"]
    assert "crosses the" in violations[0].describe()


def test_no_seam_violation_when_zero_is_central():
    assert make_map(zeros=(2048, 2048)).seam_violations() == []


def test_seam_check_accounts_for_a_narrow_envelope():
    """A tight joint near the seam is fine: it can never be told to cross it."""
    jm = make_map(zeros=(100, 2048), lo=(-5.0, -5.0), hi=(5.0, 5.0))
    assert jm.seam_violations() == []


def test_clamp_bounds_every_joint():
    jm = make_map()
    clamped = jm.clamp([10.0, -10.0])
    assert clamped[0] == pytest.approx(jm.safe_upper[0])
    assert clamped[1] == pytest.approx(jm.safe_lower[1])


def test_commanded_counts_are_always_in_range():
    jm = make_map(zeros=(4000, 100), lo=(-20.0, -20.0), hi=(20.0, 20.0))
    counts = jm.rad_to_counts(jm.clamp([5.0, -5.0]))
    assert np.all(counts >= 0) and np.all(counts < CPR)


# =====================================================================
# Safety supervisor
# =====================================================================


class FakeImu:
    def __init__(self, tilt_deg=0.0, accel_magnitude=G):
        self.tilt_deg = tilt_deg
        self.accel = np.array([0.0, 0.0, accel_magnitude])

    @property
    def tilt_cos(self):
        return float(np.cos(np.radians(self.tilt_deg)))


class FakePolicy:
    def __init__(self, offenders=()):
        self._offenders = list(offenders)

    def out_of_distribution(self, obs, limit):
        return self._offenders


def make_supervisor(**overrides):
    jm = make_map(lo=(-90.0, -90.0), hi=(90.0, 90.0))
    cfg = SafetyConfig(**overrides)
    return SafetySupervisor(jm, cfg), jm


def test_starts_disarmed():
    sup, _ = make_supervisor()
    assert sup.state is SafetyState.DISARMED
    assert sup.shape([0.1, 0.1], 0.02)[0] is None


def test_arm_succeeds_when_everything_is_healthy():
    sup, _ = make_supervisor()
    assert sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy()) == []
    assert sup.state is SafetyState.RAMPING


def test_arm_refuses_when_not_upright():
    sup, _ = make_supervisor(arm_tilt_deg=10.0)
    problems = sup.arm([0.0, 0.0], FakeImu(tilt_deg=30.0), np.zeros(48), FakePolicy())
    assert any("upright" in p for p in problems)
    assert sup.state is SafetyState.DISARMED


def test_arm_refuses_on_out_of_distribution_observation():
    """The check that catches a swapped IMU axis before it becomes a fall."""
    sup, _ = make_supervisor()
    offenders = [(8, "grav_z", 0.99, 98.0)]
    problems = sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy(offenders))
    assert any("sigma" in p for p in problems)
    assert any("grav_z" in p for p in problems)
    assert sup.state is SafetyState.DISARMED


def test_arm_refuses_when_a_joint_starts_outside_its_envelope():
    sup, jm = make_supervisor()
    problems = sup.arm([5.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert any("outside its safe range" in p for p in problems)


def test_arm_refuses_on_a_seam_violation():
    jm = make_map(zeros=(100, 2048), lo=(-90.0, -90.0), hi=(90.0, 90.0))
    sup = SafetySupervisor(jm, SafetyConfig())
    problems = sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert any("encoder seam" in p for p in problems)


def test_ramp_starts_at_the_measured_pose_not_the_target():
    """Arming must never snap the robot to the policy's command."""
    sup, _ = make_supervisor(ramp_s=1.0, ramp_joint_rate=100.0)
    start = [0.5, -0.5]
    sup.arm(start, FakeImu(), np.zeros(48), FakePolicy())
    command, status = sup.shape([0.0, 0.0], 0.02)
    assert status.state is SafetyState.RAMPING
    assert command[0] == pytest.approx(0.5, abs=0.02)   # barely moved
    assert command[1] == pytest.approx(-0.5, abs=0.02)


def test_ramp_completes_and_reaches_the_target():
    sup, _ = make_supervisor(ramp_s=1.0, ramp_joint_rate=100.0, max_joint_rate=100.0)
    sup.arm([0.5, -0.5], FakeImu(), np.zeros(48), FakePolicy())
    for _ in range(100):
        command, status = sup.shape([0.0, 0.0], 0.02)
    assert status.state is SafetyState.RUNNING
    assert np.allclose(command, [0.0, 0.0], atol=1e-6)


def test_rate_limit_caps_a_violent_command():
    sup, _ = make_supervisor(ramp_s=0.0, max_joint_rate=2.0)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    sup.shape([0.0, 0.0], 0.02)          # finish the (zero-length) ramp
    command, status = sup.shape([1.5, -1.5], 0.02)
    assert abs(command[0]) <= 2.0 * 0.02 + 1e-9
    assert set(status.rate_limited_joints) == {"a", "b"}


def test_commands_are_always_inside_the_envelope():
    sup, jm = make_supervisor(ramp_s=0.0, max_joint_rate=1e6)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    for target in ([100.0, -100.0], [-100.0, 100.0], [0.0, 0.0]):
        command, _ = sup.shape(target, 0.02)
        assert np.all(command >= jm.safe_lower - 1e-9)
        assert np.all(command <= jm.safe_upper + 1e-9)


def test_sustained_tilt_trips_a_fall():
    sup, _ = make_supervisor(fall_tilt_deg=40.0, fall_persist_cycles=3)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert sup.check_health(FakeImu(tilt_deg=50.0), 0.0, True) is None
    assert sup.check_health(FakeImu(tilt_deg=50.0), 0.0, True) is None
    fault = sup.check_health(FakeImu(tilt_deg=50.0), 0.0, True)
    assert fault.code == "FALL"
    assert sup.state is SafetyState.FAULT


def test_a_brief_tilt_spike_does_not_trip():
    sup, _ = make_supervisor(fall_tilt_deg=40.0, fall_persist_cycles=3)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    sup.check_health(FakeImu(tilt_deg=50.0), 0.0, True)
    sup.check_health(FakeImu(tilt_deg=5.0), 0.0, True)     # recovered
    sup.check_health(FakeImu(tilt_deg=50.0), 0.0, True)
    assert sup.state is not SafetyState.FAULT


def test_freefall_trips():
    sup, _ = make_supervisor(freefall_persist_cycles=2)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    sup.check_health(FakeImu(accel_magnitude=1.0), 0.0, True)
    assert sup.check_health(FakeImu(accel_magnitude=1.0), 0.0, True).code == "FREEFALL"


def test_lost_bus_trips():
    sup, _ = make_supervisor(max_consecutive_read_failures=3)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    for _ in range(2):
        assert sup.check_health(FakeImu(), 0.0, read_ok=False) is None
    assert sup.check_health(FakeImu(), 0.0, read_ok=False).code == "BUS_LOST"


def test_stale_command_trips_only_once_running():
    sup, _ = make_supervisor(ramp_s=0.0, command_timeout_s=0.1)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert sup.check_health(FakeImu(), 5.0, True) is None      # still ramping
    sup.shape([0.0, 0.0], 0.02)                                 # -> RUNNING
    assert sup.check_health(FakeImu(), 5.0, True).code == "COMMAND_STALE"


def test_overtemperature_trips():
    sup, _ = make_supervisor(max_temperature_c=60)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    fault = sup.check_health(FakeImu(), 0.0, True, temperatures={"a": 75})
    assert fault.code == "OVERTEMP"


def _run_to_running(sup, at=(0.0, 0.0)):
    """Arm and finish the ramp, leaving the last target at ``at``."""
    sup.arm(list(at), FakeImu(), np.zeros(48), FakePolicy())
    for _ in range(10):
        sup.shape(list(at), 0.04)
    assert sup.state is SafetyState.RUNNING
    return sup


def test_a_joint_that_never_reaches_its_command_trips_stall():
    sup, _ = make_supervisor(ramp_s=0.0, stall_error_rad=0.26, stall_persist_s=0.2)
    _run_to_running(sup)
    # commanded to 0, but the joint is stuck 30 deg away
    stuck = [np.radians(30.0), 0.0]
    assert sup.check_tracking(stuck, 0.04) is None      # 0.04 s of 0.2
    assert sup.check_tracking(stuck, 0.04) is None
    assert sup.check_tracking(stuck, 0.04) is None
    assert sup.check_tracking(stuck, 0.04) is None
    fault = sup.check_tracking(stuck, 0.04)
    assert fault.code == "STALL"
    assert "a" in fault.detail


def test_stall_timer_resets_when_the_joint_catches_up():
    """A slow joint is not a stalled one; only a sustained gap counts."""
    sup, _ = make_supervisor(ramp_s=0.0, stall_error_rad=0.26, stall_persist_s=0.2)
    _run_to_running(sup)
    for _ in range(4):
        assert sup.check_tracking([np.radians(30.0), 0.0], 0.04) is None
    assert sup.check_tracking([0.0, 0.0], 0.04) is None          # caught up
    for _ in range(4):
        assert sup.check_tracking([np.radians(30.0), 0.0], 0.04) is None
    assert not sup.faulted


def test_normal_tracking_error_never_trips_stall():
    """MuJoCo puts steady-state error under 0.4 deg and peaks at 2.5 deg."""
    sup, _ = make_supervisor(ramp_s=0.0)
    _run_to_running(sup)
    for _ in range(500):                                  # 20 s at 25 Hz
        assert sup.check_tracking([np.radians(2.5), np.radians(-2.5)], 0.04) is None
    assert not sup.faulted


def test_stall_is_not_checked_while_ramping():
    """During the ramp the command is deliberately away from the robot."""
    sup, _ = make_supervisor(ramp_s=10.0, stall_error_rad=0.26, stall_persist_s=0.2)
    sup.arm([1.0, 1.0], FakeImu(), np.zeros(48), FakePolicy())
    for _ in range(50):
        sup.shape([0.0, 0.0], 0.04)
        assert sup.check_tracking([1.0, 1.0], 0.04) is None
    assert sup.state is SafetyState.RAMPING


def test_fault_is_latched_and_stops_all_commands():
    sup, _ = make_supervisor(fall_persist_cycles=1)
    sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    sup.check_health(FakeImu(tilt_deg=80.0), 0.0, True)
    assert sup.faulted
    assert sup.shape([0.0, 0.0], 0.02)[0] is None
    # and it cannot be cleared by re-arming
    assert sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert sup.state is SafetyState.FAULT


def test_first_fault_cause_is_the_one_kept():
    sup, _ = make_supervisor()
    sup.trip("FALL", "first")
    sup.trip("OVERTEMP", "second")
    assert sup.fault.code == "FALL"


# =====================================================================
# Servo protocol
# =====================================================================


def test_sync_read_packet_layout():
    packet = build_sync_read([1, 2, 3], 56, 2)
    assert packet[:3] == b"\xff\xff\xfe"
    assert packet[3] == 3 + 4          # LEN = n + 4
    assert packet[4] == 0x82
    assert packet[5:7] == bytes([56, 2])
    assert packet[7:10] == bytes([1, 2, 3])


def test_sync_write_packet_layout():
    packet = build_sync_write([1, 2], 42, [b"\x00\x08", b"\xff\x07"])
    assert packet[:3] == b"\xff\xff\xfe"
    assert packet[4] == 0x83
    assert packet[5:7] == bytes([42, 2])
    assert packet[7:10] == bytes([1, 0x00, 0x08])
    assert packet[10:13] == bytes([2, 0xFF, 0x07])


def test_eeprom_writes_are_refused():
    """The guarantee: deployment cannot re-id a servo or change its baud."""
    with pytest.raises(ServoBusError, match="RAM control block"):
        build_sync_write([1], 5, [b"\x02"])        # servo ID
    with pytest.raises(ServoBusError, match="RAM control block"):
        build_sync_write([1], 6, [b"\x00"])        # baud rate
    with pytest.raises(ServoBusError, match="RAM control block"):
        build_sync_write([1], 55, [b"\x00"])       # EEPROM lock


def test_ram_writes_are_allowed():
    for addr, width in ((40, 1), (41, 1), (42, 2), (46, 2)):
        build_sync_write([1], addr, [bytes(width)])


def test_write_spanning_out_of_ram_is_refused():
    with pytest.raises(ServoBusError):
        build_sync_write([1], 54, [b"\x00\x00"])   # 54..55 crosses the boundary


def status_packet(servo_id, params):
    body = bytes([servo_id, len(params) + 2, 0]) + bytes(params)
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def test_sync_read_replies_are_parsed():
    stream = b"".join(status_packet(i, [i, 0]) for i in (1, 2, 3))
    frames = parse_sync_read(stream, [1, 2, 3], 2)
    assert set(frames) == {1, 2, 3}
    assert frames[2] == b"\x02\x00"


def test_sync_read_skips_the_echoed_instruction():
    """Same echo hazard as a single read, on a bus that always echoes."""
    sent = build_sync_read([1, 2], 56, 2)
    stream = sent + status_packet(1, [0x00, 0x08]) + status_packet(2, [0xFF, 0x07])
    frames = parse_sync_read(stream, [1, 2], 2, sent)
    assert frames[1] == b"\x00\x08"
    assert frames[2] == b"\xff\x07"


def test_sync_read_reports_a_missing_servo_rather_than_guessing():
    stream = status_packet(1, [0, 0]) + status_packet(3, [0, 0])
    frames = parse_sync_read(stream, [1, 2, 3], 2)
    assert set(frames) == {1, 3}


def test_sync_read_rejects_a_corrupt_reply():
    good = status_packet(1, [0, 8])
    bad = bytearray(status_packet(2, [0, 8]))
    bad[-1] ^= 0xFF
    frames = parse_sync_read(good + bytes(bad), [1, 2], 2)
    assert set(frames) == {1}


# =====================================================================
# Policy plumbing
# =====================================================================


def test_observation_layout_matches_the_env():
    layout = ObservationLayout(13)
    assert layout.size == 48
    assert (layout.accel.start, layout.gyro.start, layout.gravity.start) == (0, 3, 6)
    assert (layout.qpos.start, layout.qpos.stop) == (9, 22)
    assert (layout.qvel.start, layout.qvel.stop) == (22, 35)
    # last_action must be last: eval scripts read obs[-nu:]
    assert (layout.action.start, layout.action.stop) == (35, 48)


def test_layout_labels_are_useful_in_diagnostics():
    layout = ObservationLayout(13)
    assert layout.label(8) == "grav_z"
    assert layout.label(2) == "accel_z"
    assert layout.label(35) == "act_0"


def test_swish_matches_the_definition():
    x = np.array([-2.0, 0.0, 1.5])
    assert np.allclose(swish(x), x / (1 + np.exp(-x)))
    assert swish(np.zeros(1))[0] == 0.0


def test_seam_check_covers_the_whole_travel_not_just_the_command_window():
    """Arming starts from wherever the robot is, so the ramp can cross a seam
    the policy's own command window never touches."""
    # zero at 875, +/-78 deg of travel = +/-887 counts -> dips below 0,
    # but +/-0.4 rad of command (+/-261 counts) stays inside.
    jm = make_map(names=("a",), zeros=(875,), directions=(1,),
                  lo=(-78.0,), hi=(78.0,), action_scale=0.4)
    violations = jm.seam_violations()
    assert [v.scope for v in violations] == ["travel"]
    assert "ramp crosses the wrap" in violations[0].describe()


def test_a_command_window_crossing_is_reported_once_at_the_tighter_scope():
    jm = make_map(names=("a",), zeros=(10,), directions=(1,),
                  lo=(-78.0,), hi=(78.0,), action_scale=0.4)
    violations = jm.seam_violations()
    assert [v.scope for v in violations] == ["command"]


def test_a_centred_zero_has_no_seam_violation_at_either_scope():
    """After re-centring on 2048 the same joint is clean."""
    jm = make_map(names=("a",), zeros=(2048,), directions=(1,),
                  lo=(-88.0,), hi=(88.0,), action_scale=0.4)
    assert jm.seam_violations() == []


def test_a_sweep_that_misses_the_nominal_pose_by_a_hair_is_absorbed():
    """Two encoder counts of measurement noise must not ground the robot.

    The zero is set deliberately (operator aligns the link and presses a key);
    the min/max are swept to within a degree. When the sweep endpoint lands
    just the wrong side of the zero, the sweep is what is imprecise.
    """
    jm = make_map(names=("a",), zeros=(2026,), directions=(1,),
                  lo=(0.18,), hi=(89.12,), action_scale=0.4)
    assert jm.safe_lower[0] == pytest.approx(0.0)          # nominal is included
    assert [name for name, _ in jm.snapped] == ["a"]
    assert jm.snapped[0][1] == pytest.approx(0.18, abs=0.01)


def test_a_real_zero_disagreement_is_still_fatal():
    """A flipped direction or a zero taken in the wrong pose is tens of degrees."""
    with pytest.raises(JointMapError, match="nominal pose"):
        make_map(names=("a",), zeros=(2048,), directions=(1,),
                 lo=(20.0,), hi=(89.0,), action_scale=0.4)


def test_nothing_is_snapped_when_the_ranges_already_agree():
    jm = make_map(names=("a",), zeros=(2048,), directions=(1,),
                  lo=(-40.0,), hi=(40.0,), action_scale=0.4)
    assert jm.snapped == []


# =====================================================================
# Bus transactions against a fake half-duplex port
# =====================================================================


class FakePort:
    """A single-wire bus that echoes our transmit, then answers.

    The echo is the whole point. This robot's circuit loops TX onto the same
    wire RXD listens to, so every reply arrives behind a copy of the question.
    A reader that mistakes the echo for the answer, or that stops reading once
    it has counted the echo's bytes, looks perfectly fine until it meets real
    hardware -- which is exactly what happened.
    """

    def __init__(self, responder, echo=True):
        self._responder = responder
        self._echo = echo
        self.buf = bytearray()

    def reset_input_buffer(self):
        self.buf.clear()

    def reset_output_buffer(self):
        pass

    def write(self, packet):
        if self._echo:
            self.buf += packet
        self.buf += self._responder(bytes(packet))
        return len(packet)

    def flush(self):
        pass

    @property
    def in_waiting(self):
        return len(self.buf)

    def read(self, n=1):
        chunk = bytes(self.buf[:n])
        del self.buf[:n]
        return chunk

    def close(self):
        pass


def _frame(servo_id, params=b""):
    body = bytes([servo_id, len(params) + 2, 0]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def make_bus(responder, echo=True, retries=0):
    bus = JointlessServoBus.__new__(JointlessServoBus)
    bus.port_name = "fake"
    bus.baudrate = 250000
    bus._reply_timeout = 0.05
    bus._retries = retries
    bus._port = FakePort(responder, echo=echo)
    return bus


from humanoid_deploy.servo_bus import ServoBus as JointlessServoBus  # noqa: E402


def test_ping_succeeds_behind_its_own_echo():
    """The bug that stopped the robot: a ping reply is 6 bytes and so is its
    echo, so a byte-counting reader finished on the echo alone and every servo
    looked dead on a perfectly good bus."""
    bus = make_bus(lambda packet: _frame(packet[2]))
    assert bus.ping(1) is True
    assert bus.ping(13) is True


def test_ping_fails_when_nothing_answers():
    bus = make_bus(lambda packet: b"")
    assert bus.ping(1) is False


def test_ping_is_not_fooled_by_the_echo_alone():
    """Echo but no reply must read as absent, not present."""
    bus = make_bus(lambda packet: b"", echo=True)
    assert bus.ping(7) is False


def test_sync_read_collects_every_servo_behind_the_echo():
    ids = list(range(1, 14))
    positions = {i: 2000 + i for i in ids}

    def responder(packet):
        return b"".join(
            _frame(i, bytes([positions[i] & 0xFF, positions[i] >> 8])) for i in ids
        )

    bus = make_bus(responder)
    got = bus.read_positions(ids)
    assert got == positions, f"missing {sorted(set(ids) - set(got))}"


def test_sync_read_returns_what_it_got_when_a_servo_is_silent():
    ids = list(range(1, 14))
    answering = [i for i in ids if i != 9]

    def responder(packet):
        return b"".join(_frame(i, bytes([0x00, 0x08])) for i in answering)

    bus = make_bus(responder)
    got = bus.read_positions(ids)
    assert sorted(got) == answering        # partial, and the caller sees it
    assert 9 not in got


def test_read_one_returns_the_reply_not_the_echo():
    """A READ instruction packet and a status reply have the same structure,
    so a parser that does not cut the echo reads back the address it asked
    about and calls it data. Here the echo's payload bytes are (62, 2) -- which
    would decode as 6.2 V and 2 C if mistaken for the reply."""
    bus = make_bus(lambda packet: _frame(packet[2], bytes([123, 57])))
    assert bus.read_temperature(3) == 57
    bus = make_bus(lambda packet: _frame(packet[2], bytes([123, 57])))
    assert bus.read_voltage(3) == pytest.approx(12.3)


def make_knee_supervisor(**overrides):
    """A joint shaped like the real left knee: model range [-90, 0], so the
    standing pose sits exactly on the upper limit and the physical joint can
    rest a little past it."""
    jm = make_map(names=("knee", "b"), lo=(-90.0, -90.0), hi=(0.0, 90.0))
    cfg = SafetyConfig(**overrides)
    return SafetySupervisor(jm, cfg), jm


def test_arm_tolerates_a_joint_resting_just_outside_its_envelope():
    """The real left knee hyperextends past the model's zero, so standing on it
    the leg rests outside the envelope and no posing can fix that. The ramp
    pulls it back in; refusing would mean never arming."""
    sup, jm = make_knee_supervisor(arm_pose_tolerance_rad=np.radians(5.0))
    assert jm.safe_upper[0] == pytest.approx(0.0)
    assert sup.arm([np.radians(2.4), 0.0], FakeImu(), np.zeros(48), FakePolicy()) == []
    assert sup.state is SafetyState.RAMPING
    assert any("knee" in note and "outside" in note for note in sup.arm_notes)


def test_a_joint_far_outside_the_envelope_still_refuses():
    sup, _ = make_knee_supervisor(arm_pose_tolerance_rad=np.radians(5.0))
    problems = sup.arm([np.radians(40.0), 0.0], FakeImu(), np.zeros(48), FakePolicy())
    assert any("outside its safe range" in p for p in problems)
    assert sup.state is SafetyState.DISARMED


def test_the_first_command_pulls_an_outside_joint_back_in():
    """Tolerating the start only makes sense if the command is still clamped."""
    sup, jm = make_knee_supervisor(arm_pose_tolerance_rad=np.radians(5.0),
                                   ramp_s=3.0, ramp_joint_rate=1.0)
    start = np.radians(2.4)
    sup.arm([start, 0.0], FakeImu(), np.zeros(48), FakePolicy())
    command, _status = sup.shape([0.0, 0.0], 0.04)
    assert command[0] <= jm.safe_upper[0] + 1e-9      # inside the envelope
    assert command[0] < start                          # and moving inward


def test_no_note_when_every_joint_starts_inside():
    sup, _ = make_supervisor()
    assert sup.arm([0.0, 0.0], FakeImu(), np.zeros(48), FakePolicy()) == []
    assert sup.arm_notes == []


def test_health_is_one_transaction_so_volts_cannot_pose_as_degrees():
    """Voltage (62) and temperature (63) have byte-identical 1-byte replies --
    the STS reply never echoes the address it answered. Read separately, a
    14.0 V supply and a 140 C servo are indistinguishable downstream. Read as
    the adjacent pair, the reply is len=4 and unambiguous."""
    seen = []

    def responder(packet):
        seen.append((packet[5], packet[6]))       # (addr, count)
        return _frame(packet[2], bytes([123, 41]))  # 12.3 V, 41 C

    bus = make_bus(responder)
    volts, celsius = bus.read_health(9)
    assert volts == pytest.approx(12.3)
    assert celsius == 41
    assert seen == [(62, 2)], "must be a single 2-byte read at address 62"


def test_health_returns_none_pair_when_the_servo_is_silent():
    bus = make_bus(lambda packet: b"")
    assert bus.read_health(4) == (None, None)
