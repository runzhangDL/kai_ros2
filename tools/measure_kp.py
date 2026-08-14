#!/usr/bin/env python3
"""measure_kp.py -- what is the position servo's actual stiffness?

Everything the training side assumes about this robot's compliance rests on
inference, and the inference has now been wrong in both directions. This
measures kp with nothing but the robot's own leg.

The method
----------
A ``position`` actuator at rest satisfies ``tau = kp * (target - present)``.
Swing one hip through its range with the robot hanging and the leg's own weight
supplies a known, smoothly varying torque -- +-1.15 N.m at +-75 deg, passing
through zero when the leg hangs straight down. Read the steady-state error at
each angle and fit a line:

    error = tau / kp + offset

The SLOPE gives kp. The INTERCEPT absorbs friction and deadband, which is why
this works where a single reading does not: stiction was worth 0.18-1.67 deg on
the joints measured on 2026-08-13, comparable to the whole signal at kp 50. A
constant offset moves the intercept and leaves the slope alone.

    python3 -u tools/measure_kp.py

Every angle is approached from the same direction, so the friction offset is
consistent across the sweep rather than flipping sign halfway.

THE ROBOT MUST BE SUSPENDED, hanging vertically, with at least 30 cm of clear
space in front of and behind it -- the leg swings through 150 degrees. Nothing
to hang, nothing to hold, no interaction once it starts.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_tool import Bus, REG_ACC, REG_GOAL_POS, REG_PRESENT_POS, REG_TORQUE  # noqa: E402

CPR = 4096
DEG = CPR / 360.0

#: Gravity hold torque at left_hip_pitch (id 6) with the robot suspended and
#: every other joint at zero, computed from robot.xml. Degrees -> N.m. The
#: robot has no mujoco, so this table is baked in; regenerate it if the model's
#: masses change.
HIP_PITCH_TAU = {
    -75: -1.145, -50: -0.906, -25: -0.497, 0: 0.005,
    25: 0.506, 50: 0.912, 75: 1.148,
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def calibration_for(servo_id):
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
            return (name, int(entry["zero_raw"]), int(entry["direction"]), lo, hi)
    return None


def hold_and_read(bus, servo_id, raw_target, speed, settle_s, samples=30):
    bus.write(servo_id, REG_GOAL_POS,
              [raw_target & 0xFF, raw_target >> 8, 0, 0,
               speed & 0xFF, (speed >> 8) & 0xFF])
    time.sleep(settle_s)
    seen = []
    while len(seen) < samples:
        v = bus.read2(servo_id, REG_PRESENT_POS, tries=3)
        if v is not None:
            seen.append(v)
        time.sleep(0.01)
    seen.sort()
    return seen[len(seen) // 2]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--id", type=int, default=6, help="6 = left_hip_pitch")
    parser.add_argument("--speed", type=int, default=800)
    parser.add_argument("--acc", type=int, default=100)
    parser.add_argument("--settle", type=float, default=2.5)
    args = parser.parse_args()

    cal = calibration_for(args.id)
    if cal is None:
        sys.exit("no calibration found; this needs the joint's zero and direction")
    name, zero, direction, lo, hi = cal

    angles = sorted(HIP_PITCH_TAU)          # ascending: one approach direction
    plan = []
    for a in angles:
        raw = int(round(zero + direction * a * DEG))
        if lo <= raw <= hi:
            plan.append((a, raw))
    if len(plan) < 4:
        sys.exit(f"{name}'s calibrated travel [{lo}, {hi}] only admits "
                 f"{len(plan)} of the {len(angles)} sweep angles; too few to fit")

    print(f"joint {name} (id {args.id}), sweeping "
          f"{plan[0][0]:+.0f}..{plan[-1][0]:+.0f} deg in {len(plan)} steps")
    print("The leg's own weight is the load -- nothing to attach.\n")
    print("SAFETY")
    print("  * SUSPEND the robot, hanging vertically, feet well clear.")
    print("  * The leg swings through ~150 deg fore and aft. Clear 30 cm each")
    print("    way, and keep hands out of its arc.")
    print("  * Do not touch or lift the robot once it starts -- handling it is")
    print("    exactly what invalidates this measurement.")
    print("  * Torque is released at the end.\n")
    input("press ENTER when it is hanging clear, or Ctrl-C to abort... ")

    bus = Bus(args.port, args.baud)
    rows = []
    try:
        if not bus.ping(args.id, tries=6):
            sys.exit(f"id {args.id} did not answer")
        bus.write1(args.id, REG_TORQUE, 1)
        bus.write1(args.id, REG_ACC, args.acc)
        # Start below the sweep so every point is approached the same way.
        hold_and_read(bus, args.id, plan[0][1], args.speed, args.settle + 1.5)

        print(f"\n  {'angle':>7}{'tau':>9}{'target':>9}{'present':>9}{'error':>9}")
        for a, raw in plan:
            present = hold_and_read(bus, args.id, raw, args.speed, args.settle)
            err_deg = direction * (present - raw) / DEG   # +ve = lagging behind
            rows.append((HIP_PITCH_TAU[a], err_deg))
            print(f"  {a:>6}d{HIP_PITCH_TAU[a]:>9.3f}{raw:>9}{present:>9}"
                  f"{err_deg:>8.2f}d")
    finally:
        try:
            bus.write1(args.id, REG_TORQUE, 0)
            print("\ntorque released")
        finally:
            bus.close()

    if len(rows) < 4:
        print("not enough points to fit")
        return 1
    taus = [t for t, _ in rows]
    errs = [e for _, e in rows]
    n = len(rows)
    mt, me = sum(taus) / n, sum(errs) / n
    sxx = sum((t - mt) ** 2 for t in taus)
    sxy = sum((t - mt) * (e - me) for t, e in rows)
    if abs(sxx) < 1e-12 or abs(sxy) < 1e-12:
        print("degenerate fit -- the torque did not vary, or the joint did not move")
        return 1
    slope_deg_per_nm = sxy / sxx                       # deg per N.m
    slope = slope_deg_per_nm * 3.14159265 / 180.0      # rad per N.m
    kp = 1.0 / abs(slope)
    resid = [e - (me + slope_deg_per_nm * (t - mt)) for t, e in rows]
    rms = (sum(r * r for r in resid) / n) ** 0.5

    print(f"\n  fit: error = tau/kp + offset over {n} points")
    print(f"    slope     {slope_deg_per_nm:+.3f} deg per N.m   -> kp = {kp:.1f} N.m/rad")
    print(f"    offset    {me - slope_deg_per_nm * mt:+.2f} deg   "
          f"(friction and deadband; not part of kp)")
    print(f"    residual  {rms:.2f} deg rms")
    print(f"\n  the XML models kp = 50  ->  kp_scale = {kp / 50:.2f}")
    if rms > 1.5:
        print("  RESIDUAL IS HIGH -- the points do not lie on a line. Something")
        print("  moved, or the joint hit a stop. Re-run before trusting kp.")
    elif kp > 32:
        print("  -> near the XML. Training should use kp_scale ~1.0.")
    elif kp > 16:
        print("  -> moderately soft.")
    else:
        print("  -> soft, consistent with the 6.3 deg open-loop crouch lean")
        print("     measured by crouch_only. Training's 0.15-0.45 span is right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
