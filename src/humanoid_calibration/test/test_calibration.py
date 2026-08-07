"""Unit tests for the parts of calibration that must not be wrong.

These cover config validation, the persistent store round-trip, encoder
unwrapping and the staleness rules -- everything that does not need a servo.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid_calibration.calibration_store import (  # noqa: E402
    CalibrationStore,
    JointCalibration,
    utc_now,
)
from humanoid_calibration.feetech_bus import (  # noqa: E402
    AngleTracker,
    build_packet,
    checksum,
    parse_status_frame,
)
from humanoid_calibration.joint_config import ConfigError, load_config  # noqa: E402

BASE_CONFIG = {
    "bus": {"port": "/dev/null", "baudrate": 250000},
    "servo": {"counts_per_rev": 4096},
    "calibration": {"reach_tolerance_deg": 1.0, "dwell_s": 0.1},
    "defaults": {"min_deg": -90.0, "max_deg": 90.0, "direction": 1},
    "joints": [
        {"name": "left_ankle_pitch", "id": 1, "min_deg": -40.0, "max_deg": 25.0},
        {"name": "left_knee_pitch", "id": 3, "min_deg": 0.0, "max_deg": 120.0},
    ],
}


def write_config(tmp_path, **overrides):
    document = {**BASE_CONFIG, **overrides}
    path = tmp_path / "joint_limits.yaml"
    path.write_text(yaml.safe_dump(document))
    return str(path)


# -- config ---------------------------------------------------------------


def test_loads_joints_in_file_order(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.joint_names == ["left_ankle_pitch", "left_knee_pitch"]
    assert config.joint("left_knee_pitch").id == 3


def test_defaults_are_merged_per_joint(tmp_path):
    path = write_config(tmp_path, joints=[{"name": "waist_yaw", "id": 7}])
    joint = load_config(path).joints[0]
    assert (joint.min_deg, joint.max_deg, joint.direction) == (-90.0, 90.0, 1)


def test_duplicate_servo_id_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        joints=[{"name": "a", "id": 1}, {"name": "b", "id": 1}],
    )
    with pytest.raises(ConfigError, match="servo id 1"):
        load_config(path)


def test_inverted_limits_are_rejected(tmp_path):
    inverted = [{"name": "a", "id": 1, "min_deg": 10.0, "max_deg": -10.0}]
    path = write_config(tmp_path, joints=inverted)
    with pytest.raises(ConfigError, match="must be <"):
        load_config(path)


def test_bad_direction_is_rejected(tmp_path):
    path = write_config(tmp_path, joints=[{"name": "a", "id": 1, "direction": 0}])
    with pytest.raises(ConfigError, match="direction"):
        load_config(path)


def test_fingerprint_ignores_bus_but_tracks_limits(tmp_path):
    baseline = load_config(write_config(tmp_path)).fingerprint

    moved_port = dict(BASE_CONFIG, bus={"port": "/dev/ttyUSB9", "baudrate": 500000})
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(moved_port))
    assert load_config(str(path)).fingerprint == baseline

    changed_limit = dict(BASE_CONFIG)
    changed_limit["joints"] = [dict(j) for j in BASE_CONFIG["joints"]]
    changed_limit["joints"][0]["max_deg"] = 30.0
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(changed_limit))
    assert load_config(str(path)).fingerprint != baseline


# -- store ----------------------------------------------------------------


def make_joint(name="left_ankle_pitch", servo_id=1, **kwargs):
    defaults = dict(
        name=name,
        servo_id=servo_id,
        zero_raw=2048,
        min_raw=1593,
        max_raw=2332,
        min_deg=-40.0,
        max_deg=25.0,
        direction=1,
        counts_per_rev=4096,
        calibrated_at=utc_now(),
    )
    defaults.update(kwargs)
    return JointCalibration(**defaults)


def test_store_round_trips(tmp_path):
    path = str(tmp_path / "calibration.yaml")
    store = CalibrationStore(path)
    store.record(make_joint(), "fp123", "/tmp/joint_limits.yaml")
    store.save()

    reloaded = CalibrationStore.load(path)
    assert reloaded.fingerprint == "fp123"
    assert reloaded.joints["left_ankle_pitch"].zero_raw == 2048
    assert reloaded.created_at and reloaded.updated_at


def test_missing_store_loads_empty_not_error(tmp_path):
    store = CalibrationStore.load(str(tmp_path / "nope.yaml"))
    assert store.joints == {}


def test_save_is_atomic_leaving_no_temp_files(tmp_path):
    path = str(tmp_path / "calibration.yaml")
    store = CalibrationStore(path)
    store.record(make_joint(), "fp", "cfg")
    store.save()
    store.save()
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []


# -- validation -----------------------------------------------------------


def calibrated_store(tmp_path, config):
    store = CalibrationStore(str(tmp_path / "calibration.yaml"))
    for joint in config.joints:
        store.record(
            make_joint(
                name=joint.name,
                servo_id=joint.id,
                min_deg=joint.min_deg,
                max_deg=joint.max_deg,
                direction=joint.direction,
            ),
            config.fingerprint,
            config.source_path,
        )
    store.save()
    return store


def test_fully_calibrated_robot_validates(tmp_path):
    config = load_config(write_config(tmp_path))
    store = calibrated_store(tmp_path, config)
    assert CalibrationStore.load(store.path).validate(config).ok


def test_absent_file_blocks_and_lists_every_joint(tmp_path):
    config = load_config(write_config(tmp_path))
    report = CalibrationStore.load(str(tmp_path / "absent.yaml")).validate(config)
    assert not report.ok
    assert report.missing == config.joint_names
    assert "run" in report.reason()


def test_partial_calibration_blocks(tmp_path):
    config = load_config(write_config(tmp_path))
    store = CalibrationStore(str(tmp_path / "calibration.yaml"))
    store.record(
        make_joint(name="left_ankle_pitch", servo_id=1), config.fingerprint, ""
    )
    store.save()
    report = CalibrationStore.load(store.path).validate(config)
    assert not report.ok
    assert report.missing == ["left_knee_pitch"]


def test_editing_a_limit_marks_only_that_joint_stale(tmp_path):
    config = load_config(write_config(tmp_path))
    store = calibrated_store(tmp_path, config)

    edited = [dict(j) for j in BASE_CONFIG["joints"]]
    edited[0]["max_deg"] = 30.0
    new_config = load_config(write_config(tmp_path, joints=edited))

    report = CalibrationStore.load(store.path).validate(new_config)
    assert not report.ok
    assert report.stale == ["left_ankle_pitch"]
    assert not report.fingerprint_ok


def test_changing_a_servo_id_marks_the_joint_stale(tmp_path):
    config = load_config(write_config(tmp_path))
    store = calibrated_store(tmp_path, config)

    edited = [dict(j) for j in BASE_CONFIG["joints"]]
    edited[1]["id"] = 4
    new_config = load_config(write_config(tmp_path, joints=edited))
    assert CalibrationStore.load(store.path).validate(new_config).stale == ["left_knee_pitch"]


# -- conversions ----------------------------------------------------------


def test_raw_to_deg_is_signed_around_zero():
    joint = make_joint(zero_raw=2048)
    assert joint.raw_to_deg(2048) == pytest.approx(0.0)
    assert joint.raw_to_deg(2048 + 1024) == pytest.approx(90.0)
    assert joint.raw_to_deg(2048 - 1024) == pytest.approx(-90.0)


def test_raw_to_deg_wraps_across_the_encoder_seam():
    """A joint zeroed near 0/4095 must still read continuously."""
    joint = make_joint(zero_raw=10)
    assert joint.raw_to_deg(4086) == pytest.approx(-20 * 360.0 / 4096)
    assert joint.raw_to_deg(110) == pytest.approx(100 * 360.0 / 4096)


def test_direction_inverts_the_sign():
    joint = make_joint(zero_raw=2048, direction=-1)
    assert joint.raw_to_deg(2048 + 1024) == pytest.approx(-90.0)


def test_deg_to_raw_inverts_raw_to_deg():
    joint = make_joint(zero_raw=300, direction=-1)
    for raw in (0, 300, 1000, 4000):
        assert joint.raw_to_deg(joint.deg_to_raw(joint.raw_to_deg(raw))) == pytest.approx(
            joint.raw_to_deg(raw), abs=1e-6
        )


def test_clamp_respects_calibrated_travel():
    joint = make_joint(min_deg=-40.0, max_deg=25.0)
    assert joint.clamp_deg(-100.0) == -40.0
    assert joint.clamp_deg(100.0) == 25.0
    assert joint.clamp_deg(10.0) == 10.0


# -- unwrapping -----------------------------------------------------------


def test_tracker_accumulates_across_the_seam_upward():
    tracker = AngleTracker(4096)
    tracker.update(4090)
    assert tracker.update(6) == pytest.approx(4102)  # +12 counts, not -4084


def test_tracker_accumulates_across_the_seam_downward():
    tracker = AngleTracker(4096)
    tracker.update(6)
    assert tracker.update(4090) == pytest.approx(-6)


def test_tracker_follows_a_long_sweep_without_drift():
    tracker = AngleTracker(4096)
    tracker.update(0)
    for step in range(1, 200):
        tracker.update((step * 30) % 4096)
    assert tracker.unwrapped == pytest.approx(199 * 30)


def test_tracker_reset_rebaselines():
    tracker = AngleTracker(4096)
    tracker.update(100)
    tracker.reset()
    assert tracker.update(4000) == pytest.approx(4000)


# -- wire protocol --------------------------------------------------------


def test_read_packet_matches_the_feetech_spec():
    """Read Present_Position (addr 56, 2 bytes) from servo 1."""
    assert build_packet(1, 0x02, bytes([56, 2])).hex() == "ffff0104023802be"


def test_ping_packet_matches_the_feetech_spec():
    assert build_packet(1, 0x01, b"").hex() == "ffff0102 01fb".replace(" ", "")


def test_checksum_is_the_inverted_sum():
    assert checksum(bytes([1, 4, 2, 56, 2])) == 0xBE


def status_packet(servo_id, params, error=0):
    body = bytes([servo_id, len(params) + 2, error]) + bytes(params)
    return b"\xff\xff" + body + bytes([checksum(body)])


def test_parses_a_clean_status_packet():
    assert parse_status_frame(status_packet(1, [0x00, 0x08]), 1, 2) == b"\x00\x08"


def test_skips_an_echoed_instruction_packet():
    """Half-duplex adapters echo what we sent; the reply follows it."""
    sent = build_packet(1, 0x02, bytes([56, 2]))
    buf = sent + status_packet(1, [0x11, 0x02])
    assert parse_status_frame(buf, 1, 2, sent) == b"\x11\x02"


def test_an_echo_is_never_mistaken_for_a_reply():
    """The failure this guards against: a READ echo looks exactly like a reply.

    Both have id 1, LEN 4 and a valid checksum, so without cutting the bytes we
    sent, the parser returns the register address (0x38, 0x02 -> 568 counts) as
    a position.
    """
    sent = build_packet(1, 0x02, bytes([56, 2]))
    assert parse_status_frame(sent, 1, 2) == b"\x38\x02"  # the trap
    assert parse_status_frame(sent, 1, 2, sent) is None  # disarmed


def test_read_raw_ignores_an_echoing_adapter():
    sent = build_packet(1, 0x02, bytes([56, 2]))
    reader = make_reader([sent + status_packet(1, [0x00, 0x08])])
    assert reader.read_raw(1) == 2048  # not 568


def test_skips_leading_garbage_without_an_echo():
    buf = b"\x00\xa5\xff" + status_packet(1, [0x11, 0x02])
    assert parse_status_frame(buf, 1, 2) == b"\x11\x02"


def test_rejects_a_corrupt_checksum():
    packet = bytearray(status_packet(1, [0x00, 0x08]))
    packet[-1] ^= 0xFF
    assert parse_status_frame(bytes(packet), 1, 2) is None


def test_rejects_a_reply_from_the_wrong_servo():
    assert parse_status_frame(status_packet(7, [0x00, 0x08]), 1, 2) is None


def test_rejects_a_truncated_packet():
    assert parse_status_frame(status_packet(1, [0x00, 0x08])[:-2], 1, 2) is None


class FakePort:
    """Enough of a pyserial Serial to exercise FeetechReader.read_raw.

    Each write pops one canned byte stream, which models what actually comes
    back on this robot's single-wire bus: our own echoed packet followed by the
    servo's reply.
    """

    def __init__(self, streams):
        self._streams = list(streams)
        self._buf = b""
        self.written = []

    def reset_input_buffer(self):
        self._buf = b""

    def reset_output_buffer(self):
        pass

    def flush(self):
        pass

    def write(self, data):
        self.written.append(data)
        self._buf += self._streams.pop(0) if self._streams else b""

    def read(self, count):
        chunk, self._buf = self._buf[:count], self._buf[count:]
        return chunk

    @property
    def in_waiting(self):
        return len(self._buf)

    def close(self):
        pass


READ_POS = build_packet(1, 0x02, bytes([56, 2]))  # what read_raw(1) transmits


def echoed(params, servo_id=1, sent=READ_POS):
    """The full RX stream for one transaction: TX echo, then the reply."""
    return sent + status_packet(servo_id, params)


def make_reader(streams, **servo_kwargs):
    from humanoid_calibration.feetech_bus import FeetechReader
    from humanoid_calibration.joint_config import ServoConfig

    reader = FeetechReader.__new__(FeetechReader)  # bypass opening a real port
    reader._servo = ServoConfig(**servo_kwargs)
    reader._retries = 1
    reader._reply_timeout = 0.01
    reader.baudrate = 250000
    reader.port_name = "fake"
    reader._port = FakePort(streams)
    return reader


def test_read_raw_decodes_little_endian_counts():
    reader = make_reader([echoed([0x00, 0x08])])  # 0x0800 = 2048
    assert reader.read_raw(1) == 2048


def test_read_raw_decodes_big_endian_when_configured():
    reader = make_reader([echoed([0x08, 0x00])], little_endian=False)
    assert reader.read_raw(1) == 2048


def test_read_raw_survives_a_partial_echo_without_a_retry():
    """Bus turnaround eats the echo's first byte.

    The echo drain then over-reads into the reply. Because the drained bytes
    are handed back to the parser rather than dropped, the reply still resolves
    on this attempt.
    """
    reader = make_reader([READ_POS[1:] + status_packet(1, [0x00, 0x08])])
    assert reader.read_raw(1) == 2048
    assert len(reader._port.written) == 1  # no retry was needed


def test_read_raw_ignores_a_late_echo_not_yet_drained():
    """Echo still in flight when the drain budget expires; parser handles it."""
    reader = make_reader([echoed([0x00, 0x08])])
    reader.baudrate = 1  # makes the drain budget expire before reading anything
    assert reader.read_raw(1) == 2048


def test_read_raw_retries_then_gives_up():
    reader = make_reader([b"", b""])
    assert reader.read_raw(1) is None
    assert len(reader._port.written) == 2  # one retry, as configured


def test_read_raw_recovers_on_the_retry():
    reader = make_reader([b"\x00\x01garbage", echoed([0xFF, 0x0F])])
    assert reader.read_raw(1) == 4095


def test_read_raw_never_returns_the_echoed_register_address():
    """The whole-stack version of the echo trap: 0x38,0x02 would be 568."""
    reader = make_reader([READ_POS])  # echo only, servo never replied
    assert reader.read_raw(1) is None


def test_read_raw_wraps_counts_into_one_turn():
    reader = make_reader([echoed([0x00, 0x10])])  # 4096 -> 0
    assert reader.read_raw(1) == 0


def test_read_raw_honours_multi_turn_sign_bit():
    reader = make_reader([echoed([0x64, 0x80])], signed_position=True)
    assert reader.read_raw(1) == -100


# -- pre-flight bus check -------------------------------------------------


class ScriptedReader:
    """Answers read_raw with a fixed success rate."""

    def __init__(self, fail_every=0):
        self.fail_every = fail_every
        self.count = 0

    def read_raw(self, servo_id):
        self.count += 1
        if self.fail_every and self.count % self.fail_every == 0:
            return None
        return 2048

    def read_health(self, servo_id):
        return 12.0, 30

    def ping(self, servo_id):
        return True

    def close(self):
        pass


def test_preflight_passes_a_clean_bus(tmp_path):
    from humanoid_calibration.calibrate_cli import preflight

    config = load_config(write_config(tmp_path))
    assert preflight(config, ScriptedReader(), color=False) == []


def test_preflight_rejects_a_flaky_bus(tmp_path):
    """One dropped read in ten must block calibration, not just warn.

    This is the failure that silently corrupts limits: reads mostly work, so
    the operator sees a plausible live angle, but the sample captured at the
    moment a limit is crossed may be stale.
    """
    from humanoid_calibration.calibrate_cli import preflight

    config = load_config(write_config(tmp_path))
    bad = preflight(config, ScriptedReader(fail_every=10), color=False)
    assert [j.name for j in bad] == config.joint_names


def test_preflight_rejects_a_silent_bus(tmp_path):
    from humanoid_calibration.calibrate_cli import preflight

    config = load_config(write_config(tmp_path))
    bad = preflight(config, ScriptedReader(fail_every=1), color=False)
    assert [j.name for j in bad] == config.joint_names


def test_preflight_threshold_is_configurable(tmp_path):
    from humanoid_calibration.calibrate_cli import preflight

    tuning = {"preflight_reads": 20, "min_read_success": 0.5}
    config = load_config(write_config(tmp_path, calibration=tuning))
    assert preflight(config, ScriptedReader(fail_every=10), color=False) == []
