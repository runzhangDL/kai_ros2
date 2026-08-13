#!/usr/bin/env python3
"""
servo_step.py -- measure what a joint can ACTUALLY do, loaded, on this robot.

The datasheet speed and the Goal_Speed register are both upper bounds. What a
walking policy needs is the real one: how fast the joint slews when it is
carrying the robot, and how far behind its command it runs. Standing forgives a
lagging servo; walking does not, because a foot that arrives late lands
somewhere the policy did not plan.

This commands one joint through a step, samples position as fast as the bus
allows, and reports peak speed, rise time and steady-state error.

    python3 tools/servo_step.py --id 6                    # 15 deg, speed 3000
    python3 tools/servo_step.py --id 6 --amplitude 30
    python3 tools/servo_step.py --id 6 --speed 2000       # compare settings

THIS MOVES A MOTOR. One joint at a time, never more. The step is clamped to the
joint's calibrated travel, so it cannot drive into a hard stop, and torque is
released at the end.

Run it twice per joint -- once with the leg hanging free, once bearing weight.
The difference between those two numbers is the load penalty, and it is the
number to design a gait around.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_tool import Bus, REG_ACC, REG_GOAL_POS, REG_PRESENT_POS, REG_TORQUE  # noqa: E402

CPR = 4096
DEG = CPR / 360.0


# Piping into `tee` or a file switches Python's stdout from line-buffered to
# BLOCK-buffered, so a long-running scan prints nothing at all until it exits
# or fills 4 KB -- which reads exactly like a hang, and cost a bring-up session.
# Line buffering costs nothing here and keeps `| tee` honest.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def calibration_for(servo_id):
    """Calibrated travel for this servo, so the step cannot hit a hard stop."""
    import yaml
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    path = os.environ.get("HUMANOID_CALIBRATION_FILE") or os.path.join(
        base, "humanoid_robot", "calibration.yaml")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    for name, entry in (document.get("joints") or {}).items():
        if int(entry["servo_id"]) == servo_id:
            lo, hi = sorted((int(entry["min_raw"]), int(entry["max_raw"])))
            return name, lo, hi
    return None


def sample(bus, servo_id, seconds):
    """Read position as fast as the bus allows. Returns [(t, counts)]."""
    out = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        value = bus.read2(servo_id, REG_PRESENT_POS, tries=1)
        if value is not None:
            out.append((time.perf_counter(), value))
    return out


def analyse(trace, start, target, ceiling_deg_s):
    """Speed, acceleration, rise time and residual error -- all in degrees.

    Peak speed is NOT max(), because this bus occasionally returns a corrupted
    frame that still passes its checksum. Differentiating one bad position
    sample invents an arbitrarily large velocity -- it produced a 491 deg/s
    reading on a servo whose datasheet no-load speed is 270. Samples implying
    motion faster than the servo can physically turn are discarded, and the
    quoted peak is the 95th percentile of what survives.

    The acceleration estimate does not depend on differentiating at all. For an
    acceleration-limited (triangular) move, 90% of the distance is covered at
    t90 = 1.553*sqrt(d/a), so a = d*(1.553/t90)^2 -- a single robust number
    from one timing measurement. Because peak speed is then sqrt(a*d), it grows
    with the size of the move: a joint that manages 73 deg/s over 15 degrees
    will manage far more over 40.
    """
    if len(trace) < 5:
        return None
    t0 = trace[0][0]
    span = target - start
    if abs(span) < 1:
        return None

    speeds, dropped = [], 0
    for (ta, pa), (tb, pb) in zip(trace, trace[1:]):
        dt = tb - ta
        if dt <= 1e-4:
            continue
        v = abs(pb - pa) / dt / DEG
        if v > ceiling_deg_s:
            dropped += 1
            continue
        speeds.append(v)
    speeds.sort()
    peak = speeds[int(0.95 * (len(speeds) - 1))] if speeds else 0.0

    rise = None
    for t, p in trace:
        if abs(p - start) >= 0.9 * abs(span):
            rise = t - t0
            break

    accel = implied = None
    if rise and rise > 1e-3:
        accel = abs(span) * (1.553 / rise) ** 2        # counts/s^2
        implied = (accel * abs(span)) ** 0.5 / DEG     # deg/s

    final = (trace[-1][1] - target) / DEG
    reached = 100.0 * (trace[-1][1] - start) / span
    rate = len(trace) / (trace[-1][0] - t0)
    return peak, rise, final, rate, reached, accel, implied, dropped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--id", type=int, required=True, help="ONE servo id")
    parser.add_argument("--amplitude", type=float, default=15.0,
                        help="step size in degrees (default 15)")
    parser.add_argument("--speed", type=int, default=3000,
                        help="Goal_Speed in counts/s (0 = unlimited)")
    # MUST match deploy.yaml's servo_acc, which is 100. The old default of 30
    # is 3000 counts/s^2 = 4.6 rad/s^2, and that is BELOW what every joint on
    # this robot can produce -- so a step run at 30 measures the register, not
    # the motor, and reports an acceleration limit roughly 4x too low. It
    # silently invalidated a whole characterisation round.
    parser.add_argument("--acc", type=int, default=100)
    parser.add_argument("--settle", type=float, default=1.5,
                        help="seconds to record after the step")
    parser.add_argument("--no-load-speed", type=float, default=280.0,
                        help="datasheet no-load speed in deg/s; samples implying "
                             "anything faster are corrupted reads (STS3215 is "
                             "0.222 s/60 deg = 270 deg/s)")
    args = parser.parse_args()

    bus = Bus(args.port, args.baud)
    try:
        if not bus.ping(args.id, tries=6):
            sys.exit(f"id {args.id} did not answer")
        start = bus.read2(args.id, REG_PRESENT_POS)
        if start is None:
            sys.exit("could not read the starting position")

        limits = calibration_for(args.id)
        step = args.amplitude * DEG
        target = int(round(start + step))
        if limits is not None:
            name, lo, hi = limits
            clamped = max(lo, min(hi, target))
            if clamped != target:
                print(f"clamping the step to {name}'s calibrated travel "
                      f"[{lo}, {hi}]: {target} -> {clamped}")
                target = clamped
            print(f"joint {name} (id {args.id}), calibrated travel [{lo}, {hi}]")
        else:
            print("no calibration file found -- stepping without a travel limit; "
                  "keep the amplitude small")

        if abs(target - start) < 10:
            sys.exit(f"start {start} is already at the travel limit; "
                     "move the joint to the middle of its range first")

        print(f"step {start} -> {target} counts "
              f"({(target - start) / DEG:+.1f} deg) at goal speed {args.speed}\n")

        bus.write1(args.id, REG_TORQUE, 1)
        bus.write1(args.id, REG_ACC, args.acc)
        bus.write(args.id, REG_GOAL_POS,
                  [start & 0xFF, start >> 8, 0, 0,
                   args.speed & 0xFF, (args.speed >> 8) & 0xFF])
        time.sleep(0.3)

        bus.write(args.id, REG_GOAL_POS,
                  [target & 0xFF, target >> 8, 0, 0,
                   args.speed & 0xFF, (args.speed >> 8) & 0xFF])
        trace = sample(bus, args.id, args.settle)

        result = analyse(trace, start, target, args.no_load_speed)
        if result is None:
            print("not enough samples -- is the bus healthy?")
        else:
            peak, rise, final, rate, reached, accel, implied, dropped = result
            print(f"  sampled at        {rate:.0f} Hz ({len(trace)} reads"
                  + (f", {dropped} impossible ones discarded)" if dropped else ")"))
            print(f"  peak speed        {peak:.0f} deg/s "
                  f"({peak / 360 * 60:.0f} rpm)   [95th pct]")
            print(f"  goal speed cap    {args.speed / DEG:.0f} deg/s")
            print(f"  step completed    {reached:.0f}%")
            when = f"{rise * 1000:.0f} ms" if rise is not None \
                else "n/a -- stalled short of 90%"
            print(f"  90% rise time     {when}")
            print(f"  final error       {final:+.2f} deg "
                  f"(residual load/kp error; this does not close)")
            if accel is not None:
                print(f"\n  acceleration      {accel / DEG:.0f} deg/s^2 "
                      f"= {accel / DEG * 3.14159 / 180:.1f} rad/s^2")
                print(f"  implied peak      {implied:.0f} deg/s for this "
                      f"{abs(target-start)/DEG:.0f} deg move")
                print(f"  This is the number to design a gait around. Peak speed "
                      f"scales as sqrt(a*distance),\n  so a 30 deg swing would "
                      f"reach {(accel * 30 * DEG) ** 0.5 / DEG:.0f} deg/s "
                      f"and take {2 * (30 * DEG / accel) ** 0.5 * 1000:.0f} ms.")
            if peak < 0.5 * args.speed / DEG:
                print("\n  NOTE: peak is well under the goal-speed cap, so "
                      "acceleration or torque is the limit, not the register.")

        bus.write(args.id, REG_GOAL_POS,
                  [start & 0xFF, start >> 8, 0, 0, 0xB8, 0x0B])
        time.sleep(1.0)
    finally:
        try:
            bus.write1(args.id, REG_TORQUE, 0)
        finally:
            bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
