"""Read/write Feetech STS3215 bus client for deployment.

Relationship to ``humanoid_calibration.feetech_bus``
---------------------------------------------------
That module is deliberately read-only and must stay that way, so this one does
not extend it -- it imports only its pure packet helpers (``build_packet``,
``parse_status_frame``, ``checksum``) and its hard-won bus behaviour: the
Tegra first-open baud workaround, the warm-up probe, and the fact that this
robot's single-wire circuit echoes every transmit back onto RXD.

Writes are confined to RAM
--------------------------
Every write goes through :func:`_check_writable`, which rejects any address
outside the STS3215's RAM control block (40..54: Torque_Enable, Acc,
Goal_Position, Goal_Time, Goal_Speed). Servo id, baud rate, angle limits and
the EEPROM lock all live below 40 and are therefore unreachable from this
module. A deployment bug can move the robot badly; it cannot re-id the servos
or corrupt their configuration.

Bus efficiency
--------------
Reading 13 servos one at a time costs 13 round trips. SYNC READ (0x82) asks
all of them in one packet and they answer back to back, which roughly halves
the airtime and is what makes a usable control rate possible at all. Baud is a
parameter, not a constant here -- the bus was re-qualified at 500000 and the
achievable control rate must be re-measured with bus_benchmark after any change.
SYNC WRITE (0x83) sends all 13 goal positions in a single packet with no
replies at all.
"""

from __future__ import annotations

import time

import numpy as np

try:  # pragma: no cover - present on the robot
    import serial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore[assignment]

from humanoid_calibration.feetech_bus import (
    build_packet,
    checksum,
    parse_status_frame,
)

_HEADER = b"\xff\xff"
_INST_PING = 0x01
_INST_READ = 0x02
_INST_WRITE = 0x03
_INST_SYNC_READ = 0x82
_INST_SYNC_WRITE = 0x83
_BROADCAST_ID = 0xFE

# STS3215 RAM control block. Everything below 40 is EEPROM (id, baud, angle
# limits, lock); 56 and up is read-only telemetry.
ADDR_TORQUE_ENABLE = 40
ADDR_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_TIME = 44
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

_WRITABLE_LOW, _WRITABLE_HIGH = 40, 54

_PORT_SETTLE_S = 0.1
_ECHO_MARGIN_S = 0.003


class ServoBusError(Exception):
    """Raised when the bus cannot be opened or a write is rejected."""


def _check_writable(addr: int, length: int) -> None:
    if addr < _WRITABLE_LOW or addr + length - 1 > _WRITABLE_HIGH:
        raise ServoBusError(
            f"refusing to write {length} byte(s) at address {addr}: outside the "
            f"RAM control block {_WRITABLE_LOW}..{_WRITABLE_HIGH}. EEPROM "
            "(servo id, baud, angle limits, lock) is not writable from the "
            "deployment path."
        )


def build_sync_read(ids, addr: int, count: int) -> bytes:
    params = bytes([addr & 0xFF, count & 0xFF]) + bytes(int(i) & 0xFF for i in ids)
    return build_packet(_BROADCAST_ID, _INST_SYNC_READ, params)


def build_sync_write(ids, addr: int, payloads) -> bytes:
    """One packet setting ``addr`` on every id. ``payloads`` are equal-length."""
    payloads = [bytes(p) for p in payloads]
    if not payloads:
        raise ServoBusError("sync write with no payloads")
    width = len(payloads[0])
    if any(len(p) != width for p in payloads):
        raise ServoBusError("sync write payloads must all be the same length")
    if len(payloads) != len(list(ids)):
        raise ServoBusError("sync write id/payload count mismatch")
    _check_writable(addr, width)

    params = bytearray([addr & 0xFF, width & 0xFF])
    for servo_id, payload in zip(ids, payloads):
        params.append(int(servo_id) & 0xFF)
        params.extend(payload)
    return build_packet(_BROADCAST_ID, _INST_SYNC_WRITE, bytes(params))


def parse_sync_read(buf: bytes, ids, count: int, sent: bytes = b"") -> dict[int, bytes]:
    """Pull one status frame per id out of a back-to-back reply stream.

    Same echo hazard as a single read, and worse: the sync-read instruction we
    sent is itself a valid-looking frame. Cut it first, then scan.
    """
    if sent:
        echo = buf.find(sent)
        if echo != -1:
            buf = buf[echo + len(sent):]

    out: dict[int, bytes] = {}
    wanted = {int(i) for i in ids}
    total = 6 + count
    start = buf.find(_HEADER)
    while start != -1:
        frame = buf[start:]
        if len(frame) >= total:
            resp_id, resp_len = frame[2], frame[3]
            if resp_id in wanted and resp_id not in out and resp_len == count + 2:
                body = frame[2: 4 + resp_len - 1]
                if checksum(body) == frame[4 + resp_len - 1]:
                    out[resp_id] = frame[5: 5 + count]
                    start = buf.find(_HEADER, start + total)
                    continue
        start = buf.find(_HEADER, start + 1)
    return out


class ServoBus:
    """Half-duplex Feetech bus with sync read and sync write."""

    def __init__(self, port: str, baudrate: int, timeout_ms: int = 30,
                 retries: int = 1) -> None:
        if serial is None:
            raise ServoBusError("pyserial is not installed (pip3 install pyserial)")
        self.port_name = port
        self.baudrate = int(baudrate)
        self._reply_timeout = timeout_ms / 1000.0
        self._retries = max(0, int(retries))
        try:
            self._port = self._open(port, baudrate)
        except Exception as exc:  # noqa: BLE001
            raise ServoBusError(f"cannot open {port} @ {baudrate}: {exc}") from exc
        self._warmup()

    def _open(self, port, baudrate):
        settings = dict(
            port=port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            timeout=self._reply_timeout, write_timeout=self._reply_timeout,
        )
        # Tegra HS-UARTs frequently ignore the baud divisor on the first open.
        probe = serial.Serial(**settings)
        probe.close()
        time.sleep(_PORT_SETTLE_S)
        return serial.Serial(**settings)

    def _warmup(self) -> None:
        time.sleep(_PORT_SETTLE_S)
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
            self._port.write(build_packet(0xFD, _INST_PING))
            self._port.flush()
            time.sleep(0.05)
            self._port.reset_input_buffer()
        except Exception:  # noqa: BLE001
            pass

    # -- low level ---------------------------------------------------------

    def _drain_echo(self, nbytes: int) -> bytes:
        budget = (nbytes * 10.0) / max(1, self.baudrate) + _ECHO_MARGIN_S
        deadline = time.monotonic() + budget
        out = bytearray()
        while len(out) < nbytes and time.monotonic() < deadline:
            waiting = self._port.in_waiting
            if waiting:
                out += self._port.read(min(waiting, nbytes - len(out)))
            else:
                time.sleep(0.0002)
        return bytes(out)

    def _collect_until(self, parse, prefix: bytes, timeout: float):
        """Accumulate bytes until ``parse`` succeeds, or the deadline passes.

        Terminating on a byte *count* is wrong here, and it failed on hardware:
        the prefix is our own transmit, already drained off RXD, so counting it
        toward the reply meant a 6-byte ping was "complete" the instant its
        6-byte echo arrived -- before the servo had said anything. Every ping
        returned False on a working bus.

        A sync read has the same disease more subtly: its replies are only
        nominally ``n*(6+count)`` bytes, so one stray byte on the line shifts
        the total and silently truncates the last frames.

        Parsing is the only honest termination condition, and it is what
        ``humanoid_calibration.feetech_bus`` has always done. It also returns
        as soon as the data is there rather than waiting out a fixed window,
        which matters at a 24 ms cycle.
        """
        buf = bytearray(prefix)
        result = parse(bytes(buf))
        if result is not None:
            return result
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waiting = self._port.in_waiting
            chunk = self._port.read(waiting) if waiting else self._port.read(1)
            if not chunk:
                continue
            buf += chunk
            result = parse(bytes(buf))
            if result is not None:
                return result
        return None

    # -- reads -------------------------------------------------------------

    def sync_read(self, ids, addr: int, count: int) -> dict[int, bytes]:
        """Read ``count`` bytes at ``addr`` from every id in one transaction."""
        ids = [int(i) for i in ids]
        packet = build_sync_read(ids, addr, count)
        expected = len(ids) * (6 + count)
        # Airtime for all the replies, plus per-servo turnaround, plus slack.
        window = (expected * 10.0) / self.baudrate + 0.0004 * len(ids) + self._reply_timeout

        frames: dict[int, bytes] = {}

        def parse(buf):
            # Keep the best partial result: a servo that answered late is still
            # better than nothing, and the caller reports a short read as a
            # failed cycle rather than pretending it is fresh.
            nonlocal frames
            found = parse_sync_read(buf, ids, count, packet)
            if len(found) > len(frames):
                frames = found
            return found if len(found) == len(ids) else None

        for attempt in range(self._retries + 1):
            try:
                self._port.reset_input_buffer()
                self._port.write(packet)
                self._port.flush()
                echo = self._drain_echo(len(packet))
                complete = self._collect_until(parse, echo, window)
            except Exception:  # noqa: BLE001 - a flaky bus must not kill the loop
                complete = None
            if complete is not None:
                return complete
            if attempt < self._retries:
                time.sleep(0.001)
        return frames

    def read_positions(self, ids) -> dict[int, int]:
        """Present position in counts, per id. Missing ids simply do not appear."""
        raw = self.sync_read(ids, ADDR_PRESENT_POSITION, 2)
        return {i: (data[0] | (data[1] << 8)) & 0x0FFF for i, data in raw.items()}

    def _read_one(self, servo_id: int, addr: int, count: int) -> bytes | None:
        packet = build_packet(servo_id, _INST_READ, bytes([addr, count]))
        try:
            self._port.reset_input_buffer()
            self._port.write(packet)
            self._port.flush()
            echo = self._drain_echo(len(packet))
            return self._collect_until(
                lambda buf: parse_status_frame(buf, servo_id, count, packet),
                echo, self._reply_timeout)
        except Exception:  # noqa: BLE001
            return None

    def read_health(self, servo_id: int) -> tuple[float | None, int | None]:
        """``(volts, celsius)`` from ONE transaction. Voltage is 62, temp 63.

        Deliberately not two 1-byte reads. The STS reply does not echo the
        address it answered, so a 1-byte voltage reply and a 1-byte temperature
        reply are byte-identical in structure -- id, len=3, err, value, sum.
        Nothing downstream can tell them apart, so any cross-talk between the
        two silently reports one as the other, and a 14.0 V supply becomes a
        140 C servo. Reading the adjacent pair together makes the reply len=4,
        distinct from every 1-byte read on the bus, and halves the traffic.
        """
        data = self._read_one(servo_id, ADDR_PRESENT_VOLTAGE, 2)
        if data is None or len(data) != 2:
            return None, None
        return data[0] / 10.0, data[1]

    def read_temperature(self, servo_id: int) -> int | None:
        return self.read_health(servo_id)[1]

    def read_voltage(self, servo_id: int) -> float | None:
        return self.read_health(servo_id)[0]

    def ping(self, servo_id: int) -> bool:
        packet = build_packet(servo_id, _INST_PING)
        try:
            self._port.reset_input_buffer()
            self._port.write(packet)
            self._port.flush()
            echo = self._drain_echo(len(packet))
            # A ping reply carries no parameter bytes, so a successful parse
            # returns b"" -- falsy but not None. Test against None, not truth.
            return self._collect_until(
                lambda buf: parse_status_frame(buf, servo_id, 0, packet),
                echo, self._reply_timeout) is not None
        except Exception:  # noqa: BLE001
            return False

    # -- writes (RAM only) -------------------------------------------------

    def _write_broadcast(self, packet: bytes) -> None:
        """Broadcast writes get no reply; just clear our own echo."""
        self._port.reset_input_buffer()
        self._port.write(packet)
        self._port.flush()
        self._drain_echo(len(packet))

    def write_goal_positions(self, ids, counts) -> None:
        """Sync-write Goal_Position. The hot path -- one packet, no replies."""
        counts = np.asarray(counts).astype(np.int64)
        if np.any(counts < 0) or np.any(counts > 4095):
            raise ServoBusError(f"goal position out of 0..4095: {counts.tolist()}")
        payloads = [bytes([int(c) & 0xFF, (int(c) >> 8) & 0xFF]) for c in counts]
        self._write_broadcast(build_sync_write(ids, ADDR_GOAL_POSITION, payloads))

    def set_torque(self, ids, enabled: bool) -> None:
        value = bytes([1 if enabled else 0])
        self._write_broadcast(
            build_sync_write(ids, ADDR_TORQUE_ENABLE, [value] * len(list(ids)))
        )

    def set_speed_and_acc(self, ids, speed: int, acc: int) -> None:
        """Set the servos' own speed cap and acceleration once, at arm time.

        Writing these every cycle would waste bus time; they are RAM registers
        that persist until power-down. Speed 0 means 'as fast as possible',
        which is not what we want on a legged robot.
        """
        ids = [int(i) for i in ids]
        speed = int(np.clip(speed, 0, 4095))
        acc = int(np.clip(acc, 0, 255))
        self._write_broadcast(
            build_sync_write(ids, ADDR_ACC, [bytes([acc])] * len(ids))
        )
        payload = bytes([speed & 0xFF, (speed >> 8) & 0xFF])
        self._write_broadcast(
            build_sync_write(ids, ADDR_GOAL_SPEED, [payload] * len(ids))
        )

    def close(self) -> None:
        try:
            self._port.close()
        except Exception:  # noqa: BLE001
            pass
