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


def analyse(trace, start, target):
    """Peak speed, 90% rise time, final error -- all in degrees."""
    if len(trace) < 5:
        return None
    t0 = trace[0][0]
    span = target - start
    if abs(span) < 1:
        return None

    speeds = []
    for (ta, pa), (tb, pb) in zip(trace, trace[1:]):
        dt = tb - ta
        if dt > 1e-4:
            speeds.append(abs(pb - pa) / dt / DEG)
    peak = max(speeds) if speeds else 0.0

    rise = None
    for t, p in trace:
        if abs(p - start) >= 0.9 * abs(span):
            rise = t - t0
            break
    final = (trace[-1][1] - target) / DEG
    # How much of the commanded step it actually completed. A servo that stops
    # short is not slow, it is out of torque -- with a proportional position
    # loop the residual error is load/kp, and it never closes.
    reached = 100.0 * (trace[-1][1] - start) / span
    return peak, rise, final, len(trace) / (trace[-1][0] - t0), reached


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument("--id", type=int, required=True, help="ONE servo id")
    parser.add_argument("--amplitude", type=float, default=15.0,
                        help="step size in degrees (default 15)")
    parser.add_argument("--speed", type=int, default=3000,
                        help="Goal_Speed in counts/s (0 = unlimited)")
    parser.add_argument("--acc", type=int, default=30)
    parser.add_argument("--settle", type=float, default=1.5,
                        help="seconds to record after the step")
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

        result = analyse(trace, start, target)
        if result is None:
            print("not enough samples -- is the bus healthy?")
        else:
            peak, rise, final, rate, reached = result
            print(f"  sampled at        {rate:.0f} Hz ({len(trace)} reads)")
            print(f"  peak speed        {peak:.0f} deg/s "
                  f"({peak / 360 * 60:.0f} rpm)")
            print(f"  goal speed cap    {args.speed / DEG:.0f} deg/s")
            print(f"  step completed    {reached:.0f}%")
            when = f"{rise * 1000:.0f} ms" if rise is not None \
                else "n/a -- stalled short of 90%"
            print(f"  90% rise time     {when}")
            print(f"  final error       {final:+.2f} deg "
                  f"(residual load/kp error; this does not close)")
            print(f"\n  at 25 Hz one control cycle is 40 ms, so this joint can "
                  f"move about {peak * 0.04:.1f} deg per cycle.")
            if peak < 0.5 * args.speed / DEG:
                print("  NOTE: peak speed is less than half the goal-speed cap, "
                      "so load or torque is the limit here, not the register.")

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
