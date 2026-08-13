#!/usr/bin/env python3
"""servo_trace.py -- raw step-response trace, for measuring TRANSPORT DELAY.

Copy this into tools/ on the robot, next to servo_step.py. It uses the same bus
helper and the same calibration guard, and it moves ONE joint at a time.

servo_step.py already measures peak speed, rise time and acceleration well.
What it does not report is DEAD TIME: how long after the write the joint first
moves at all. That is the parameter the training side is most sensitive to --
in simulation, removing it is the difference between a policy that learns to
walk and one that plateaus, while the rate and acceleration limits barely
matter. It is also the parameter the RMS-based fit in HARDWARE_ACTUATOR_MODEL.md
can determine least well, because a pure delay and a rate limit produce similar
RMS from completely different waveforms.

So: command a step, sample position as fast as the bus allows, and write the
whole (t, position) trace out. Dead time is then read straight off -- the time
between the write and the first sample that has moved past the noise floor. No
fitting, and no trade-off against the other two parameters.

    python3 tools/servo_trace.py --id 6  --amplitude 20 --save knee_free.npz
    python3 tools/servo_trace.py --id 3  --amplitude 25 --save knee_load.npz

THIS MOVES A MOTOR. The step is clamped to the joint's calibrated travel, and
torque is released at the end.

Run each joint TWICE -- once hanging free, once bearing the robot's weight.
Everything in HARDWARE_ACTUATOR_MODEL.md was measured suspended, and the
walking policy needs the loaded numbers for the stance leg.
"""

import argparse
import os
import sys
import time

import numpy as np

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyTHS1")
    p.add_argument("--baud", type=int, default=500000)
    p.add_argument("--id", type=int, required=True, help="ONE servo id")
    p.add_argument("--amplitude", type=float, default=20.0, help="degrees")
    p.add_argument("--speed", type=int, default=3000, help="goal speed register")
    p.add_argument("--acc", type=int, default=100,
                   help="servo acceleration register. MUST match what the "
                        "deployment node uses, or the trace describes a "
                        "different servo than the one that runs the policy")
    p.add_argument("--settle", type=float, default=1.5, help="s before the step")
    p.add_argument("--record", type=float, default=1.5, help="s after the step")
    p.add_argument("--save", metavar="FILE.npz", default=None)
    args = p.parse_args()

    cal = calibration_for(args.id)
    if cal is None:
        print(f"no calibration entry for servo {args.id}; refusing to move it")
        return 1
    name, lo, hi = cal

    bus = Bus(args.port, args.baud)
    bus.write1(args.id, REG_ACC, args.acc)
    bus.write1(args.id, REG_TORQUE, 1)
    time.sleep(0.2)

    start = bus.read2(args.id, REG_PRESENT_POS)
    if start is None:
        print("could not read start position"); return 1

    target = int(round(start + args.amplitude * DEG))
    target = max(lo, min(hi, target))
    if abs(target - start) < 2 * DEG:
        print(f"clamped step is only {(target-start)/DEG:.1f} deg -- "
              f"joint is near its travel limit, pick the other direction")
        bus.write1(args.id, REG_TORQUE, 0)
        return 1

    print(f"joint {name} (id {args.id}), travel [{lo}, {hi}], acc register {args.acc}")
    print(f"step {start} -> {target} counts ({(target-start)/DEG:+.1f} deg)")

    # Baseline: sample BEFORE the step, so the noise floor is measured rather
    # than assumed. Dead time is meaningless without it -- position readings
    # jitter by a count or two even when the joint is perfectly still.
    pre = []
    t_end = time.perf_counter() + args.settle
    while time.perf_counter() < t_end:
        v = bus.read2(args.id, REG_PRESENT_POS, tries=1)
        if v is not None:
            pre.append((time.perf_counter(), v))

    # The write timestamp is t=0. Everything downstream is measured from here.
    t_write = time.perf_counter()
    bus.write2(args.id, REG_GOAL_POS, target)

    post = []
    t_end = time.perf_counter() + args.record
    while time.perf_counter() < t_end:
        v = bus.read2(args.id, REG_PRESENT_POS, tries=1)
        if v is not None:
            post.append((time.perf_counter(), v))

    bus.write1(args.id, REG_TORQUE, 0)

    t = np.array([x[0] for x in post]) - t_write
    pos = np.array([x[1] for x in post], dtype=float)
    pre_pos = np.array([x[1] for x in pre], dtype=float)

    noise = max(pre_pos.std() * 3.0, 2.0) if len(pre_pos) > 5 else 2.0
    moved = np.where(np.abs(pos - start) > noise)[0]
    dead = t[moved[0]] if len(moved) else float("nan")
    rate = len(post) / (t[-1] - t[0]) if len(post) > 1 else 0.0

    print(f"\n  sampled at        {rate:.0f} Hz ({len(post)} reads, "
          f"{len(pre)} baseline)")
    print(f"  noise floor       {noise:.1f} counts ({noise/DEG:.2f} deg)")
    print(f"  DEAD TIME         {dead*1000:.0f} ms  "
          f"= {dead/0.04:.1f} cycles at 25 Hz")
    print(f"     (the fit in HARDWARE_ACTUATOR_MODEL.md says 3-4 cycles.")
    print(f"      ~1 cycle of that is its own write-then-read-same-cycle")
    print(f"      structure, which this measurement does NOT have, so expect")
    print(f"      this to come in about a cycle lower than the fitted value.)")

    if args.save:
        np.savez(args.save, t=t, pos_counts=pos, start=start, target=target,
                 pre_t=np.array([x[0] for x in pre]) - t_write, pre_pos=pre_pos,
                 joint=name, servo_id=args.id, acc_register=args.acc,
                 goal_speed=args.speed, deg_per_count=1.0 / DEG)
        print(f"\n  wrote {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
