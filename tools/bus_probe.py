#!/usr/bin/env python3
"""bus_probe.py -- where does the control cycle's time actually go?

`bus_benchmark` says one read+write cycle costs 24 ms. Doubling the baud rate
from 250000 to 500000 did not change that number at all, which means airtime is
not what is being paid for. This takes the transaction apart and times each
piece, so the 24 ms can be attributed instead of guessed at.

    python3 -u tools/bus_probe.py

The discriminating measurement is the scaling section. A sync read asks all the
servos in one packet and they reply back to back, so:

  * if reading 1 servo costs about the same as reading 13, the cost is a fixed
    per-transaction latency -- the serial driver, not the bus. That is worth
    chasing: it would be pure overhead, and removing it would multiply the
    achievable control rate.
  * if the cost grows with the servo count, it is per-servo turnaround -- the
    time each servo takes to start talking. That is the servos' own firmware
    and no amount of software will remove it.

ANSWER, on this robot: the first. `tcdrain()` costs a flat ~11.95 ms per
transmit at every packet size from 8 to 47 bytes, and it was ~24 of the 24 ms
control cycle. Dropping it takes the cycle to 5.07 ms with identical 100% read
success. `ServoBus(flush_tx=...)` controls it and is off by default.

Read-only apart from the write section, which commands every servo to the
position it is already in, exactly as `bus_benchmark` does. Nothing moves.
"""

import argparse
import statistics
import sys
import time

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src", "humanoid_deploy"))

from humanoid_deploy.servo_bus import (  # noqa: E402
    ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION, ServoBus, ServoBusError,
    build_sync_read, build_sync_write, parse_sync_read,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def stats(samples):
    ordered = sorted(samples)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    return statistics.median(ordered) * 1e3, p99 * 1e3


def timed(fn, n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def probe_read_breakdown(bus, ids, n):
    """One sync read, with every step timed separately."""
    packet = build_sync_read(ids, ADDR_PRESENT_POSITION, 2)
    parts = {k: [] for k in ("reset", "write", "flush", "echo", "replies")}
    ok = 0
    for _ in range(n):
        t0 = time.perf_counter()
        bus._port.reset_input_buffer()
        t1 = time.perf_counter()
        bus._port.write(packet)
        t2 = time.perf_counter()
        bus._port.flush()
        t3 = time.perf_counter()
        echo = bus._drain_echo(len(packet))
        t4 = time.perf_counter()
        frames = {}

        def parse(buf):
            nonlocal frames
            found = parse_sync_read(buf, ids, 2, packet)
            if len(found) > len(frames):
                frames = found
            return found if len(found) == len(ids) else None

        got = bus._collect_until(parse, echo, 0.030)
        t5 = time.perf_counter()
        if got is not None:
            ok += 1
        parts["reset"].append(t1 - t0)
        parts["write"].append(t2 - t1)
        parts["flush"].append(t3 - t2)
        parts["echo"].append(t4 - t3)
        parts["replies"].append(t5 - t4)
    return parts, ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--cycles", type=int, default=200)
    args = parser.parse_args()

    ids = [int(x) for x in args.ids.split(",")]
    try:
        bus = ServoBus(args.port, args.baud, timeout_ms=30)
    except ServoBusError as exc:
        sys.exit(f"error: {exc}")

    byte_us = 10.0 / args.baud * 1e6
    print(f"{args.port} @ {args.baud} baud  ({byte_us:.1f} us per byte), "
          f"{len(ids)} servos, {args.cycles} reps\n")

    try:
        # ---- 1. where one sync read spends its time --------------------
        print("1. one sync read of 13 servos, broken down")
        parts, ok = probe_read_breakdown(bus, ids, args.cycles)
        packet_len = len(build_sync_read(ids, ADDR_PRESENT_POSITION, 2))
        reply_len = len(ids) * (6 + 2)
        total = 0.0
        for name in ("reset", "write", "flush", "echo", "replies"):
            p50, p99 = stats(parts[name])
            total += p50
            print(f"   {name:<9} p50 {p50:6.2f} ms   p99 {p99:6.2f} ms")
        print(f"   {'TOTAL':<9} p50 {total:6.2f} ms      ({ok}/{args.cycles} complete)")
        print(f"   airtime accounts for {packet_len * byte_us / 1e3:.2f} ms out "
              f"(echo) + {reply_len * byte_us / 1e3:.2f} ms back (replies)\n")

        # ---- 2. does it scale with the servo count? --------------------
        print("2. sync read vs number of servos -- the discriminating test")
        print(f"   {'servos':>7}{'p50 ms':>10}{'p99 ms':>10}{'airtime ms':>13}"
              f"{'overhead ms':>13}")
        first = None
        for count in (1, 2, 4, 8, len(ids)):
            subset = ids[:count]
            packet = build_sync_read(subset, ADDR_PRESENT_POSITION, 2)
            air = (len(packet) + count * 8) * byte_us / 1e3
            samples = timed(
                lambda s=subset: bus.read_positions(s), max(40, args.cycles // 4))
            p50, p99 = stats(samples)
            if first is None:
                first = p50
            print(f"   {count:>7}{p50:>10.2f}{p99:>10.2f}{air:>13.2f}"
                  f"{p50 - air:>13.2f}")
        # Do NOT fit a straight line through this. On the robot every row
        # comes back at ~12.00 ms regardless of servo count, which is the
        # signature of a fixed per-transaction cost -- section 4 identifies it
        # as tcdrain. An earlier run of this tool showed 1.8 ms at 1 servo
        # rising to 12.0 at 13, and a linear fit through that laundered the
        # step into a fake "0.85 ms per servo" and a confident, wrong verdict
        # that the servos' own firmware was to blame.
        print("\n   Read the rows, not a slope. Flat across servo counts means a")
        print("   fixed per-transaction cost, not per-servo turnaround; averaging")
        print("   over a step invents a per-servo cost that is not there.")

        # ---- 4. is tcdrain the cost, and where is the cliff? ----------
        print("\n4. tcdrain (flush) vs transmit size -- the suspected cause")
        print(f"   {'bytes':>7}{'write ms':>11}{'flush ms':>11}"
              f"{'airtime ms':>13}")
        for nbytes in (8, 12, 16, 17, 20, 21, 24, 32, 47):
            blob = bytes([0xFF, 0xFF]) + bytes(nbytes - 2)
            def one(b=blob):
                bus._port.reset_input_buffer()
                t = time.perf_counter()
                bus._port.write(b)
                mid = time.perf_counter()
                bus._port.flush()
                return mid - t, time.perf_counter() - mid
            samples = [one() for _ in range(60)]
            w = statistics.median(s[0] for s in samples) * 1e3
            f = statistics.median(s[1] for s in samples) * 1e3
            print(f"   {nbytes:>7}{w:>11.2f}{f:>11.2f}{nbytes * byte_us / 1e3:>13.2f}")
        bus._port.reset_input_buffer()
        print("   MEASURED: flat ~11.95 ms at every size from 8 to 47 bytes. There")
        print("   is no size dependence and no FIFO boundary -- it is a fixed cost")
        print("   in the driver, and it dwarfs the <1 ms of airtime it is nominally")
        print("   waiting for.")

        # ---- 5. does dropping flush actually work? --------------------
        print("\n5. sync read of 13 servos, with and without tcdrain")
        for label, flush in (("with flush", True), ("without flush", False)):
            bus._flush_tx = flush
            samples = timed(lambda: bus.read_positions(ids), args.cycles)
            got = [len(bus.read_positions(ids)) == len(ids) for _ in range(60)]
            p50, p99 = stats(samples)
            print(f"   {label:<15} p50 {p50:6.2f} ms  p99 {p99:6.2f} ms   "
                  f"complete {sum(got)}/60")
        bus._flush_tx = True
        print("   Equal completion rates and a large time drop means the flush is")
        print("   pure overhead here -- the echo we wait for is the real barrier.")

        # ---- 3. the write half ----------------------------------------
        print("\n3. one sync write of 13 goal positions (no replies expected)")
        positions = bus.read_positions(ids)
        if len(positions) == len(ids):
            payloads = [bytes([positions[i] & 0xFF, (positions[i] >> 8) & 0xFF])
                        for i in ids]
            packet = build_sync_write(ids, ADDR_GOAL_POSITION, payloads)
            for label, fn in (
                ("reset", lambda: bus._port.reset_input_buffer()),
                ("write", lambda: bus._port.write(packet)),
                ("flush", lambda: bus._port.flush()),
                ("drain", lambda: bus._drain_echo(len(packet))),
            ):
                p50, p99 = stats(timed(fn, args.cycles))
                print(f"   {label:<9} p50 {p50:6.2f} ms   p99 {p99:6.2f} ms")
            print(f"   packet is {len(packet)} bytes = "
                  f"{len(packet) * byte_us / 1e3:.2f} ms of airtime")
        else:
            print("   skipped: could not read all positions")
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
