"""Strictly read-only client for a Feetech STS/SMS servo bus (STS3215).

Safety note
-----------
This module implements *only* the ``READ`` instruction (0x02) of the Feetech
half-duplex protocol. There is no write, no reg-write, no action, no reset and
no sync-write path anywhere in this file, by design: the calibration procedure
must never be able to move a servo or mutate its EEPROM. If you ever need to
write to a servo, do it from a different module so this guarantee stays easy
to audit.

Because nothing is written, torque is left exactly as the servos powered up.
To back-drive the joints by hand, release torque first with the bring-up tool::

    python3 sts_tool.py --port /dev/ttyTHS1 --baud 500000 torque <id> 0

Wire format
-----------
Instruction packet::

    0xFF 0xFF  ID  LEN  INST  PARAM...  CHECKSUM

``LEN`` is ``len(params) + 2`` and ``CHECKSUM`` is
``(~(ID + LEN + INST + sum(params))) & 0xFF``.

Status packet::

    0xFF 0xFF  ID  LEN  ERR  PARAM...  CHECKSUM

This robot's bus
----------------
Three behaviours of the Jetson Orin Nano 40-pin wiring (74LVC1G125 open-drain TX
+ pull-up, with RXD tapped off the same single wire) drive the design below.
They were established empirically with ``sts_tool.py`` on the real robot, and
none of them are optional:

1. **Every transmit is echoed back on RXD.** The circuit loops TX onto the wire
   RXD listens to, so the reply is always preceded by our own packet. This is
   not an edge case here, it is what happens on every single transaction --
   see :func:`parse_status_frame` for why an unhandled echo produces a
   plausible but completely wrong position.
2. **Tegra HS-UARTs often ignore the baud divisor on the first open().** The
   port comes up at some default rate and every byte is malformed. Opening,
   closing and reopening makes the setting stick.
3. **The first transaction after opening is routinely lost** while the line
   settles and stray bytes sit in the FIFO.
"""

from __future__ import annotations

import time
from typing import Protocol

try:  # pragma: no cover - exercised on the robot, not in unit tests
    import serial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore[assignment]

from .joint_config import BusConfig, ServoConfig

_HEADER = b"\xff\xff"
_INST_READ = 0x02
_INST_PING = 0x01

#: Id used for the warm-up probe. Nothing on this robot answers to it, and the
#: broadcast id (0xFE) is deliberately never used anywhere in this package.
_UNUSED_ID = 0xFD

#: Settle time around opening a Tegra UART, and after it, before the first read.
_PORT_SETTLE_S = 0.1

#: Slack added to the echo's computed airtime before we stop waiting for it.
_ECHO_MARGIN_S = 0.003

#: STS3215 status registers used for the operator readout. All read-only.
ADDR_PRESENT_VOLTAGE = 62  # 1 byte, units of 0.1 V
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, degrees C


class BusError(Exception):
    """Raised when the serial port itself cannot be used."""


class PositionSource(Protocol):
    """Minimal interface the calibration loop needs from a servo bus."""

    def read_raw(self, servo_id: int) -> int | None: ...

    def ping(self, servo_id: int) -> bool: ...

    def close(self) -> None: ...


def checksum(payload: bytes) -> int:
    """Feetech checksum over everything between the header and the checksum."""
    return (~sum(payload)) & 0xFF


def build_packet(servo_id: int, inst: int, params: bytes = b"") -> bytes:
    """Assemble an instruction packet.

    Reading Present_Position (addr 56, 2 bytes) from servo 1 gives
    ``ff ff 01 04 02 38 02 be``.
    """
    body = bytes([servo_id, len(params) + 2, inst]) + params
    return _HEADER + body + bytes([checksum(body)])


def parse_status_frame(
    buf: bytes, servo_id: int, expect: int, sent: bytes = b""
) -> bytes | None:
    """Extract ``expect`` parameter bytes from a status packet inside ``buf``.

    ``sent`` must be the instruction packet that produced this reply. Passing it
    is not an optimisation, it is required for correctness on any adapter that
    echoes: a READ of N bytes and its status reply have *identical* structure
    (same id, same LEN of N+2, both with valid checksums), so an echo is
    indistinguishable from a real reply by inspection alone. Read Present
    Position and a naive parser would happily return the echoed register
    address 0x38,0x02 as a position of 568 counts -- a plausible-looking,
    completely wrong number, which for a calibration tool is the worst possible
    failure mode. Cutting the exact bytes we sent removes the ambiguity.

    Beyond that, a noisy bus can prepend garbage, so the remainder is scanned
    at every 0xFF 0xFF header for the first frame with the right id, the right
    length and a valid checksum. Returns None if no such frame is present.
    """
    if sent:
        echo = buf.find(sent)
        if echo != -1:
            buf = buf[echo + len(sent) :]

    total = 6 + expect
    start = buf.find(_HEADER)
    while start != -1:
        frame = buf[start:]
        if len(frame) >= total:
            resp_id, resp_len = frame[2], frame[3]
            if resp_id == servo_id and resp_len == expect + 2:
                body = frame[2 : 4 + resp_len - 1]
                if checksum(body) == frame[4 + resp_len - 1]:
                    return frame[5 : 5 + expect]
        # Advance by 1, not 2: a stray 0xFF before a real header makes the
        # candidates overlap, and stepping by 2 would skip the real one.
        start = buf.find(_HEADER, start + 1)
    return None


class FeetechReader:
    """Read-only Feetech bus client."""

    def __init__(self, bus: BusConfig, servo: ServoConfig) -> None:
        if serial is None:
            raise BusError(
                "pyserial is not installed. Install it with:  pip3 install pyserial"
            )
        self._servo = servo
        self._retries = max(0, int(bus.retries))
        self._reply_timeout = bus.timeout_ms / 1000.0
        self.port_name = bus.port
        self.baudrate = bus.baudrate

        try:
            self._port = self._open(bus)
        except Exception as exc:  # noqa: BLE001 - surface any pyserial failure
            raise BusError(f"cannot open {bus.port} at {bus.baudrate} baud: {exc}") from exc
        self._warmup()

    def _open(self, bus: BusConfig):
        """Open the port, working around the Tegra first-open baud bug.

        On the Jetson's HS-UARTs the baud divisor frequently is not applied by
        the first open(), leaving the port at some default rate where every
        byte comes back malformed. Opening, closing and reopening makes it
        stick. Harmless on a USB adapter, so it is done unconditionally.
        """
        settings = dict(
            port=bus.port,
            baudrate=bus.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._reply_timeout,
            write_timeout=self._reply_timeout,
        )
        probe = serial.Serial(**settings)
        probe.close()
        time.sleep(_PORT_SETTLE_S)
        return serial.Serial(**settings)

    def _warmup(self) -> None:
        """Burn one throwaway transaction so real reads start from a clean bus.

        The first transaction after opening is routinely lost while the line
        settles and stray bytes sit in the FIFO. The probe is addressed to an id
        nothing uses, so even if it somehow reached a servo it could only
        provoke a reply, never a write.
        """
        time.sleep(_PORT_SETTLE_S)
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
            self._port.write(build_packet(_UNUSED_ID, _INST_PING))
            self._port.flush()
            time.sleep(0.05)
            self._port.reset_input_buffer()
        except Exception:  # noqa: BLE001 - warm-up failure is not fatal
            pass

    # -- low level ---------------------------------------------------------

    def _transact(self, servo_id: int, params: bytes, inst: int, expect: int) -> bytes | None:
        """Send one instruction and return ``expect`` response parameter bytes."""
        packet = build_packet(servo_id, inst, params)
        for attempt in range(self._retries + 1):
            try:
                self._port.reset_input_buffer()
                self._port.write(packet)
                self._port.flush()
                echo = self._drain_echo(len(packet))
                response = self._read_status(servo_id, expect, packet, echo)
            except Exception:  # noqa: BLE001 - a flaky bus must not kill the tool
                response = None
            if response is not None:
                return response
            if attempt < self._retries:
                time.sleep(0.002)
        return None

    def _drain_echo(self, nbytes: int) -> bytes:
        """Read back our own transmit, looped onto RXD by the half-duplex circuit.

        Deliberately *not* ``read(nbytes)``: if bus turnaround swallows a byte
        of the echo, a blocking read would sit for the whole port timeout and
        consume part of the servo's reply as filler. Instead we budget only the
        echo's airtime plus a small margin and stop as soon as we have enough
        bytes.

        The bytes are *returned rather than discarded*, and the caller feeds
        them back into the parser. That makes this purely a latency
        optimisation with no correctness risk: if turnaround dropped a byte and
        we therefore grabbed the first byte of the real reply, that byte is
        still in the buffer and the reply still parses on this attempt instead
        of costing a retry.
        """
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

    def _read_status(
        self, servo_id: int, expect: int, sent: bytes, prefix: bytes = b""
    ) -> bytes | None:
        """Accumulate bytes until a valid reply appears, or the deadline passes."""
        buf = bytearray(prefix)
        frame = parse_status_frame(bytes(buf), servo_id, expect, sent)
        if frame is not None:
            return frame

        deadline = time.monotonic() + self._reply_timeout
        while time.monotonic() < deadline:
            waiting = self._port.in_waiting
            chunk = self._port.read(waiting) if waiting else self._port.read(1)
            if not chunk:
                continue
            buf += chunk
            frame = parse_status_frame(bytes(buf), servo_id, expect, sent)
            if frame is not None:
                return frame
        return None

    def _read_bytes(self, servo_id: int, addr: int, count: int) -> bytes | None:
        return self._transact(
            servo_id, bytes([addr & 0xFF, count & 0xFF]), _INST_READ, count
        )

    # -- public API --------------------------------------------------------

    def ping(self, servo_id: int) -> bool:
        """Return True if the servo answers on the bus."""
        return self._transact(servo_id, b"", _INST_PING, 0) is not None

    def read_raw(self, servo_id: int) -> int | None:
        """Present position in encoder counts, or None if the read failed."""
        data = self._read_bytes(servo_id, self._servo.present_position_addr, 2)
        if data is None or len(data) != 2:
            return None
        if self._servo.little_endian:
            value = data[0] | (data[1] << 8)
        else:
            value = (data[0] << 8) | data[1]
        if self._servo.signed_position:
            # Multi-turn mode: bit 15 is a sign bit, not part of the magnitude.
            if value & 0x8000:
                return -(value & 0x7FFF)
            return value
        return value % self._servo.counts_per_rev

    def read_health(self, servo_id: int) -> tuple[float | None, int | None]:
        """Return ``(volts, celsius)``; either element may be None on failure."""
        volts: float | None = None
        celsius: int | None = None
        data = self._read_bytes(servo_id, ADDR_PRESENT_VOLTAGE, 1)
        if data:
            volts = data[0] / 10.0
        data = self._read_bytes(servo_id, ADDR_PRESENT_TEMPERATURE, 1)
        if data:
            celsius = data[0]
        return volts, celsius

    def close(self) -> None:
        try:
            self._port.close()
        except Exception:  # noqa: BLE001
            pass


class MockReader:
    """Keyboard-driven stand-in for a real bus, for testing off-robot.

    ``calibrate --mock`` uses this so the whole flow can be walked through
    without hardware. Arrow keys / ``j`` / ``k`` nudge the simulated encoder.
    """

    def __init__(self, servo: ServoConfig, start: int = 2048) -> None:
        self._servo = servo
        self._positions: dict[int, int] = {}
        self._start = start

    def nudge(self, servo_id: int, counts: int) -> None:
        current = self._positions.get(servo_id, self._start)
        self._positions[servo_id] = (current + counts) % self._servo.counts_per_rev

    def ping(self, servo_id: int) -> bool:
        return True

    def read_raw(self, servo_id: int) -> int | None:
        return self._positions.setdefault(servo_id, self._start)

    def read_health(self, servo_id: int) -> tuple[float | None, int | None]:
        return 12.0, 30

    def close(self) -> None:
        pass


class AngleTracker:
    """Turns wrapping encoder counts into a continuous joint angle.

    A single-turn STS3215 reports 0..4095 and wraps. If the operator drags a
    joint across that seam mid-calibration the raw value jumps by ~4096 counts;
    without unwrapping the recorded limit would be nonsense. This accumulates
    deltas instead, assuming successive samples are less than half a turn apart
    (true at 30 Hz for any hand-driven motion).
    """

    def __init__(self, counts_per_rev: int) -> None:
        self._cpr = counts_per_rev
        self._unwrapped: float | None = None

    def reset(self) -> None:
        self._unwrapped = None

    def update(self, raw: int) -> float:
        if self._unwrapped is None:
            self._unwrapped = float(raw)
            return self._unwrapped
        previous_raw = self._unwrapped % self._cpr
        delta = raw - previous_raw
        if delta > self._cpr / 2:
            delta -= self._cpr
        elif delta < -self._cpr / 2:
            delta += self._cpr
        self._unwrapped += delta
        return self._unwrapped

    @property
    def unwrapped(self) -> float | None:
        return self._unwrapped


def counts_to_deg(counts: float, counts_per_rev: int) -> float:
    return counts * 360.0 / counts_per_rev


def deg_to_counts(degrees: float, counts_per_rev: int) -> float:
    return degrees * counts_per_rev / 360.0
