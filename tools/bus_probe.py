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
        p50_all, _ = stats(timed(lambda: bus.read_positions(ids), 40))
        per_servo = (p50_all - first) / (len(ids) - 1)
        print(f"\n   1 servo {first:.2f} ms -> {len(ids)} servos {p50_all:.2f} ms")
        print(f"   marginal cost per extra servo: {per_servo:.2f} ms")
        if per_servo < 0.15:
            print("   -> FIXED per-transaction latency dominates. This is the "
                  "serial\n      driver, not the servos, and it is worth chasing.")
        else:
            print("   -> per-servo turnaround dominates. This is the servos' own\n"
                  "      firmware; software cannot remove it.")

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
