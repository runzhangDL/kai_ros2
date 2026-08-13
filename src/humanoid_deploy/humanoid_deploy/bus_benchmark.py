"""Measure the control rate this robot's bus can actually sustain.

The policy was trained at 125 Hz (8 ms per step). Whether that is reachable is
a property of the wiring, not of the policy, so measure rather than assume::

    ros2 run humanoid_deploy bus_benchmark --ros-args -p port:=/dev/ttyTHS1

Each cycle does exactly what the executor does -- one sync read of all 13
positions plus one sync write of all 13 goals -- and reports the distribution
of cycle times plus the read success rate. The write is a *no-op write*: it
commands each servo to the position it is already at, so nothing moves. Run it
with torque off anyway.

What to do with the answer: set ``control_rate_hz`` to something with real
headroom over the p99 (a good rule is 70% of the achievable maximum). If that
lands far below 125 Hz, retraining with ``frame_skip`` matched to the real rate
will beat trying to force the bus.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from .servo_bus import ServoBus, ServoBusError


# Piping into `tee` or a file switches Python's stdout from line-buffered to
# BLOCK-buffered, so a long-running scan prints nothing at all until it exits
# or fills 4 KB -- which reads exactly like a hang, and cost a bring-up session.
# Line buffering costs nothing here and keeps `| tee` honest.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure sustainable control rate.")
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--timeout-ms", type=int, default=30)
    parser.add_argument("--no-write", action="store_true",
                        help="measure reads only")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    ids = [int(x) for x in args.ids.split(",")]
    try:
        bus = ServoBus(args.port, args.baud, timeout_ms=args.timeout_ms)
    except ServoBusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{args.port} @ {args.baud}, {len(ids)} servos, {args.cycles} cycles")
    print("read = sync read of present position; write = sync write of goal position")
    print("(the write commands each servo to where it already is -- nothing moves)\n")

    read_times, write_times, cycle_times = [], [], []
    read_failures = 0

    for _ in range(args.cycles):
        cycle_start = time.perf_counter()

        t0 = time.perf_counter()
        positions = bus.read_positions(ids)
        read_times.append(time.perf_counter() - t0)
        if len(positions) != len(ids):
            read_failures += 1

        if not args.no_write and len(positions) == len(ids):
            t0 = time.perf_counter()
            bus.write_goal_positions(ids, [positions[i] for i in ids])
            write_times.append(time.perf_counter() - t0)

        cycle_times.append(time.perf_counter() - cycle_start)

    bus.close()

    def report(label, samples):
        if not samples:
            return
        ordered = sorted(samples)
        p50 = statistics.median(ordered)
        p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        print(f"  {label:<8} p50 {p50 * 1000:6.2f} ms   p99 {p99 * 1000:6.2f} ms   "
              f"max {ordered[-1] * 1000:6.2f} ms")

    print("timing:")
    report("read", read_times)
    report("write", write_times)
    report("cycle", cycle_times)

    success = 100.0 * (args.cycles - read_failures) / args.cycles
    print(f"\nread success: {success:.2f}%  ({read_failures} failed cycles)")

    if not cycle_times:
        return 1
    ordered = sorted(cycle_times)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    ceiling = 1.0 / p99
    recommended = 5.0 * int(0.7 * ceiling / 5.0)  # round down to a multiple of 5
    print(f"sustainable ceiling (1/p99): {ceiling:.1f} Hz")
    print(f"recommended control_rate_hz: {max(10.0, recommended):.0f} Hz "
          "(~70% of ceiling, leaving headroom for retries and jitter)")

    if success < 99.9:
        print("\nWARNING: reads are dropping. Fix that before deploying -- a missed "
              "read means the policy acts on a stale observation.")
    if ceiling < 125.0:
        print(f"\nNOTE: 125 Hz (the training rate) is not reachable on this bus. "
              f"Either raise the baud rate and re-qualify it with "
              f"`sts_tool.py bustest`, or retrain with frame_skip set so the "
              f"simulated control rate matches ~{max(10.0, recommended):.0f} Hz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
