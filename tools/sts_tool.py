#!/usr/bin/env python3
"""
sts_tool.py -- minimal Feetech STS3215 bus tool for single-wire half-duplex adapters
(74LVC1G125 open-drain TX + pull-up + resistor tap to RXD, e.g. Jetson 40-pin designs).

Key difference vs the stock FTServo_Python SDK: this tool expects to hear its own
transmit echoed back on RXD (your circuit loops TX onto the bus that RXD listens to)
and explicitly reads/discards it, so replies are never mis-parsed.

Usage examples:
  python3 sts_tool.py scan
  python3 sts_tool.py scan --bauds 1000000,500000,115200 --full
  python3 sts_tool.py ping 1
  python3 sts_tool.py set-id 1 2                # connect ONE servo only!
  python3 sts_tool.py set-baud 1 115200
  python3 sts_tool.py move 1 2048 --speed 1000 --acc 50
  python3 sts_tool.py pos 1
  python3 sts_tool.py torque 1 0
  python3 sts_tool.py sweep 1,2                 # exercise several servos together

Default port is /dev/ttyTHS1 (Jetson Orin Nano 40-pin, pins 8/10).
Override with --port, e.g. --port /dev/ttyUSB0. Default baud 1000000 (STS factory).
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing: pip3 install pyserial")

# ---- STS3215 register map (EEPROM: torque off + unlock before writing) ----
REG_MODEL = 3          # 2B
REG_ID = 5             # 1B EEPROM
REG_BAUD = 6           # 1B EEPROM (index, see BAUD_INDEX)
REG_TORQUE = 40        # 1B RAM 0/1
REG_ACC = 41           # 1B RAM
REG_GOAL_POS = 42      # 2B RAM little-endian, 0..4095
REG_GOAL_SPEED = 46    # 2B RAM
REG_LOCK = 55          # 1B 0=unlock EEPROM, 1=lock
REG_PRESENT_POS = 56   # 2B
REG_VOLTAGE = 62       # 1B (x0.1 V)
REG_TEMP = 63          # 1B (degC)
REG_MOVING = 66        # 1B

BAUD_INDEX = {1000000: 0, 500000: 1, 250000: 2, 128000: 3,
              115200: 4, 76800: 5, 57600: 6, 38400: 7}

PING, READ, WRITE = 0x01, 0x02, 0x03
BROADCAST = 0xFE  # deliberately NOT used by this tool


class Bus:
    def __init__(self, port, baud, timeout=0.06, verbose=False):
        self.verbose = verbose
        # Tegra HS-UARTs frequently do not apply the baud divisor on the FIRST
        # open() -- the port ends up at some default rate and every byte is
        # malformed. Opening, closing, and reopening makes the setting stick.
        s = serial.Serial(port, baud, timeout=timeout)
        s.close()
        time.sleep(0.1)
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self._warmup()

    def _warmup(self):
        # The first transaction after opening a Jetson UART is routinely lost:
        # the line settles and stray bytes sit in the FIFO. Burn one throwaway
        # packet and flush, so real commands start from a clean state.
        time.sleep(0.15)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        try:
            self._tx(0xFD, PING)   # unused id, nothing should answer
            time.sleep(0.05)
        except Exception:
            pass
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def reopen(self, baud):
        port = self.ser.port
        self.ser.close()
        time.sleep(0.1)
        self.ser = serial.Serial(port, baud, timeout=self.ser.timeout)
        self._warmup()

    # ---- low level ----
    def _tx(self, sid, instr, params=b""):
        pkt = bytes([0xFF, 0xFF, sid, len(params) + 2, instr]) + bytes(params)
        pkt += bytes([(~sum(pkt[2:])) & 0xFF])
        self._last_tx = pkt
        self.ser.reset_input_buffer()
        self.ser.write(pkt)
        self.ser.flush()
        # The half-duplex circuit loops our own TX back into RXD. Bus turnaround
        # can insert or drop a byte, so we do NOT assume the echo is exactly
        # len(pkt) bytes -- _rx also skips any frame identical to what we sent.
        echo = self._drain_echo(len(pkt))
        if self.verbose:
            print(f"  tx  {pkt.hex(' ')}")
            print(f"  echo{' MISSING/PARTIAL!' if echo != pkt else ''} {echo.hex(' ')}")
        if len(echo) == 0:
            print("  [warn] no TX echo on RXD -- check buffer OE wiring / RXD tap")
        return echo == pkt

    def _drain_echo(self, nbytes):
        # Read back our own transmit WITHOUT a blocking read(N): if the bus
        # turnaround drops a byte, read(N) would sit for the full port timeout
        # and swallow the servo's reply. Budget only the echo's airtime plus a
        # small margin, and stop as soon as we have enough bytes.
        budget = (nbytes * 10.0) / self.ser.baudrate + 0.003
        end = time.time() + budget
        out = bytearray()
        while len(out) < nbytes and time.time() < end:
            n = self.ser.in_waiting
            if n:
                out += self.ser.read(min(n, nbytes - len(out)))
            else:
                time.sleep(0.0002)
        return bytes(out)

    def _rx(self, deadline=None):
        end = time.time() + (deadline if deadline is not None else 0.08)
        buf = bytearray()
        while time.time() < end:
            chunk = self.ser.read(1)
            if not chunk:
                continue
            buf += chunk
            while True:
                i = buf.find(b"\xff\xff")
                if i < 0:
                    if len(buf) > 1:
                        del buf[:-1]
                    break
                if len(buf) < i + 4:
                    break
                length = buf[i + 3]
                if not (2 <= length <= 250):
                    del buf[: i + 2]
                    continue
                need = i + 4 + length
                if len(buf) < need:
                    break
                body = buf[i + 2 : need]
                if (sum(body)) & 0xFF == 0xFF:
                    frame = bytes(buf[i:need])
                    if frame == getattr(self, "_last_tx", None):
                        # our own echo arrived intact -- drop it, keep looking
                        del buf[:need]
                        continue
                    sid, err = buf[i + 2], buf[i + 4]
                    params = bytes(buf[i + 5 : need - 1])
                    if self.verbose:
                        print(f"  rx  {frame.hex(' ')}")
                    return sid, err, params
                del buf[: i + 2]
        return None

    # ---- protocol ops ----
    def ping(self, sid, deadline=0.1, tries=1):
        for _ in range(tries):
            self._tx(sid, PING)
            r = self._rx(deadline)
            if r and r[0] == sid:
                return r
            time.sleep(0.01)
        return None

    def read(self, sid, addr, n, tries=3):
        for _ in range(tries):
            self._tx(sid, READ, [addr, n])
            r = self._rx()
            if r and r[0] == sid and len(r[2]) == n:
                return r[2]
        return None

    def write(self, sid, addr, data, expect_reply=True):
        self._tx(sid, WRITE, [addr] + list(data))
        if not expect_reply:
            return True
        r = self._rx()
        ok = r is not None and r[1] == 0
        if r and r[1] != 0:
            print(f"  [servo error] id={r[0]} err=0x{r[1]:02x}")
        return ok

    def write1(self, sid, addr, v, **kw):
        return self.write(sid, addr, [v & 0xFF], **kw)

    def write2(self, sid, addr, v, **kw):  # STS = little-endian
        return self.write(sid, addr, [v & 0xFF, (v >> 8) & 0xFF], **kw)

    def read2(self, sid, addr, tries=3):
        d = self.read(sid, addr, 2, tries=tries)
        return None if d is None else d[0] | (d[1] << 8)


# ---- commands ----

def cmd_scan(bus, args):
    if args.bauds:
        bauds = [int(b) for b in args.bauds.split(",")]
    else:
        # No explicit list: try the rate given by --baud first, then every other
        # rate the STS firmware supports, so a lost servo always turns up.
        bauds = [args.baud] + [b for b in sorted(BAUD_INDEX, reverse=True)
                               if b != args.baud]
    ids = range(0, 254) if args.full else range(0, args.max_id + 1)
    found_any = False
    for baud in bauds:
        bus.reopen(baud)
        print(f"-- scanning at {baud} baud (ids {ids.start}..{ids.stop-1}) --")
        for sid in ids:
            r = bus.ping(sid, deadline=0.03, tries=2)
            if r:
                found_any = True
                volt = bus.read(sid, REG_VOLTAGE, 1)
                temp = bus.read(sid, REG_TEMP, 1)
                model = bus.read2(sid, REG_MODEL)
                print(f"  FOUND id={sid}  model={model}  "
                      f"V={volt[0]/10 if volt else '?'}  T={temp[0] if temp else '?'}C")
    if not found_any:
        print("no servos answered. checks: 12V power on? echo present on TX "
              "(run with -v)? try --full and more --bauds.")


def cmd_ping(bus, args):
    r = bus.ping(args.id, deadline=0.1, tries=5)
    if r:
        print(f"id={args.id} alive, err=0x{r[1]:02x}")
    else:
        print(f"id={args.id}: no reply")


def cmd_set_id(bus, args):
    print("!! make sure ONLY this one servo is on the bus !!")
    old, new = args.old, args.new
    if not bus.ping(old, tries=10):
        sys.exit(f"id {old} not responding; aborting")
    bus.write1(old, REG_TORQUE, 0)
    bus.write1(old, REG_LOCK, 0)
    bus.write1(old, REG_ID, new, expect_reply=False)
    time.sleep(0.05)
    bus.write1(new, REG_LOCK, 1)
    time.sleep(0.05)
    if bus.ping(new, tries=10):
        print(f"ok: servo now answers as id {new}")
    else:
        print("verify failed -- rescan to find where it ended up: sts_tool.py scan")


def cmd_set_baud(bus, args):
    target = args.new_baud
    if target not in BAUD_INDEX:
        sys.exit(f"unsupported baud; pick one of {sorted(BAUD_INDEX)}")
    sid = args.id
    print(f"talking to id {sid} at {bus.ser.baudrate} -> switching it to {target}")
    if not bus.ping(sid, tries=10):
        sys.exit(f"id {sid} not responding at current baud; aborting")
    bus.write1(sid, REG_TORQUE, 0)
    bus.write1(sid, REG_LOCK, 0)
    bus.write1(sid, REG_BAUD, BAUD_INDEX[target], expect_reply=False)
    time.sleep(0.1)
    bus.reopen(target)
    bus.write1(sid, REG_LOCK, 1)
    time.sleep(0.05)
    if bus.ping(sid, tries=10):
        print(f"ok: id {sid} now on {target} baud "
              f"(remember to pass --baud {target} from now on)")
    else:
        print("no reply at new baud -- rescan across bauds: "
              "sts_tool.py scan --bauds 1000000,500000,250000,128000,115200")


def cmd_move(bus, args):
    sid = args.id
    bus.write1(sid, REG_TORQUE, 1)
    bus.write1(sid, REG_ACC, args.acc)
    pos, spd = args.pos, args.speed
    bus.write(sid, REG_GOAL_POS,
              [pos & 0xFF, pos >> 8, 0, 0, spd & 0xFF, spd >> 8])
    time.sleep(0.2)
    for _ in range(25):
        p = bus.read2(sid, REG_PRESENT_POS)
        m = bus.read(sid, REG_MOVING, 1)
        if p is not None:
            print(f"  pos={p}")
        if m is not None and m[0] == 0:
            break
        time.sleep(0.1)


def cmd_pos(bus, args):
    p = bus.read2(args.id, REG_PRESENT_POS)
    print(f"id={args.id} pos={p}")


def cmd_reg(bus, args):
    """Read any register, raw. Diagnostic only -- this never writes.

    Useful when a ROS node reports something implausible and you need to know
    whether the servo really said it. Temperature is 63, voltage 62, present
    position 56.
    """
    ids = [int(x) for x in args.ids.split(",")]
    for sid in ids:
        values = []
        for _ in range(args.count):
            data = bus.read(sid, args.addr, args.size)
            if data is None:
                values.append(None)
            elif args.size == 2:
                values.append(data[0] | (data[1] << 8))
            else:
                values.append(data[0])
        good = [v for v in values if v is not None]
        spread = f"{min(good)}..{max(good)}" if good else "no reply"
        print(f"  id{sid:>3}  addr {args.addr:>3}  reads {values}  range {spread}")
    print("\n(disagreeing reads mean a flaky bus, not a changing register)")


def cmd_torque(bus, args):
    bus.write1(args.id, REG_TORQUE, args.on)
    print(f"id={args.id} torque={'on' if args.on else 'off'}")


def cmd_sweep(bus, args):
    ids = [int(x) for x in args.ids.split(",")]
    for sid in ids:
        if not bus.ping(sid, tries=6):
            sys.exit(f"id {sid} not responding -- fix that before sweeping")
        bus.write1(sid, REG_TORQUE, 1)
        bus.write1(sid, REG_ACC, 50)
    targets = [1024, 3072]
    for cycle in range(args.cycles):
        tgt = targets[cycle % 2]
        for sid in ids:
            bus.write(sid, REG_GOAL_POS,
                      [tgt & 0xFF, tgt >> 8, 0, 0, 0xE8, 0x03])  # speed 1000
        time.sleep(1.2)
        report = []
        for sid in ids:
            p = bus.read2(sid, REG_PRESENT_POS)
            report.append(f"id{sid}={p}")
        print(f"target {tgt}: " + "  ".join(report))
    print("sweep done -- every listed id should have tracked both targets")


def cmd_bustest(bus, args):
    """Hammer the bus and report packet loss -- use this to qualify a baud rate
    against your REAL cable lengths and motor count before committing to it."""
    ids = [int(x) for x in args.ids.split(",")]
    per = {sid: [0, 0] for sid in ids}   # [ok, fail]
    for _ in range(args.count):
        for sid in ids:
            if bus.read2(sid, REG_PRESENT_POS, tries=1) is not None:
                per[sid][0] += 1
            else:
                per[sid][1] += 1
    print(f"-- {args.count} reads per servo at {bus.ser.baudrate} baud --")
    worst = 100.0
    for sid in ids:
        ok, fail = per[sid]
        rate = 100.0 * ok / max(1, ok + fail)
        worst = min(worst, rate)
        print(f"  id{sid}: {ok} ok / {fail} fail  = {rate:.1f}% success")
    if worst == 100.0:
        print("PERFECT -- this baud is safe for this cabling. Try the next rate up.")
    elif worst >= 99.0:
        print("marginal: mostly fine but dropping packets. Prefer a slower rate.")
    else:
        print("UNRELIABLE at this baud/cable length -- step down to a slower rate.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan"); s.add_argument("--bauds", default=None)
    s.add_argument("--max-id", type=int, default=30)
    s.add_argument("--full", action="store_true"); s.set_defaults(f=cmd_scan)

    s = sub.add_parser("ping"); s.add_argument("id", type=int); s.set_defaults(f=cmd_ping)

    s = sub.add_parser("set-id"); s.add_argument("old", type=int)
    s.add_argument("new", type=int); s.set_defaults(f=cmd_set_id)

    s = sub.add_parser("set-baud"); s.add_argument("id", type=int)
    s.add_argument("new_baud", type=int)   # NOT "baud": would clobber --baud
    s.set_defaults(f=cmd_set_baud)

    s = sub.add_parser("move"); s.add_argument("id", type=int)
    s.add_argument("pos", type=int); s.add_argument("--speed", type=int, default=1000)
    s.add_argument("--acc", type=int, default=50); s.set_defaults(f=cmd_move)

    s = sub.add_parser("pos"); s.add_argument("id", type=int); s.set_defaults(f=cmd_pos)

    s = sub.add_parser("reg", help="read a register repeatedly (read-only)")
    s.add_argument("ids"); s.add_argument("addr", type=int)
    s.add_argument("--size", type=int, default=1, choices=[1, 2])
    s.add_argument("--count", type=int, default=5); s.set_defaults(f=cmd_reg)

    s = sub.add_parser("torque"); s.add_argument("id", type=int)
    s.add_argument("on", type=int, choices=[0, 1]); s.set_defaults(f=cmd_torque)

    s = sub.add_parser("sweep"); s.add_argument("ids")
    s.add_argument("--cycles", type=int, default=6); s.set_defaults(f=cmd_sweep)

    s = sub.add_parser("bustest"); s.add_argument("ids")
    s.add_argument("--count", type=int, default=200); s.set_defaults(f=cmd_bustest)

    args = ap.parse_args()
    bus = Bus(args.port, args.baud, verbose=args.verbose)
    try:
        args.f(bus, args)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
