#!/usr/bin/env python3
"""measure_kp.py -- what is the position servo's actual stiffness?

Everything the training side assumes about this robot's compliance rests on
inference, and one of those inferences was already wrong: a crouch leaning
5-6.5 deg was read as "the servos are 5x softer than the model", when the pose
being commanded was clamped 2 deg short at one ankle and the robot was simply
starting to fall. This measures kp instead of arguing about it.

The method
----------
A ``position`` actuator at rest satisfies ``tau = kp * (target - present)``, so
kp is torque divided by steady-state error. Measuring that directly does not
work on this robot: at the torques available the error is around a degree, and
friction and deadband alone were worth 0.18-1.67 deg on the joints stepped on
2026-08-13. The stiction swamps the signal.

So this measures a DIFFERENCE. Hold one joint at one angle, read it, add a
known mass at a known lever arm, read it again. The pose is identical and the
joint is approached from the same side both times, so friction and deadband are
the same in both readings and cancel:

    kp = (m * g * r) / (present_loaded - present_unloaded)

    python3 -u tools/measure_kp.py --id 6 --angle 80 --mass-g 200 --lever 0.445

THIS MOVES A MOTOR, and it deliberately loads it near its torque limit. Read
the safety notes printed at startup before running it.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_tool import Bus, REG_ACC, REG_GOAL_POS, REG_PRESENT_POS, REG_TORQUE  # noqa: E402

CPR = 4096
DEG = CPR / 360.0
G = 9.80665

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def calibration_for(servo_id):
    """(name, zero_raw, direction, lo_raw, hi_raw) for this servo, or None."""
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


def settle_read(bus, servo_id, seconds=1.5, samples=40):
    """Median position after letting the joint come to rest."""
    time.sleep(seconds)
    values = []
    while len(values) < samples:
        v = bus.read2(servo_id, REG_PRESENT_POS, tries=3)
        if v is not None:
            values.append(v)
        time.sleep(0.01)
    values.sort()
    return values[len(values) // 2], values[0], values[-1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--id", type=int, default=6,
                        help="servo to test (6 = left_hip_pitch)")
    parser.add_argument("--angle", type=float, default=80.0,
                        help="joint angle to hold, degrees, in calibrated sign")
    parser.add_argument("--mass-g", type=float, default=200.0,
                        help="mass you will hang, in grams")
    parser.add_argument("--lever", type=float, default=0.445,
                        help="metres from the joint axis to where it hangs, "
                             "measured HORIZONTALLY")
    parser.add_argument("--speed", type=int, default=1200)
    parser.add_argument("--acc", type=int, default=100)
    args = parser.parse_args()

    cal = calibration_for(args.id)
    if cal is None:
        sys.exit("no calibration found; this needs the joint's zero and direction")
    name, zero, direction, lo, hi = cal
    target = int(round(zero + direction * args.angle * DEG))
    if not lo <= target <= hi:
        sys.exit(f"{args.angle:+.0f} deg is raw {target}, outside {name}'s "
                 f"calibrated travel [{lo}, {hi}]. Pick a smaller --angle.")

    tau_extra = args.mass_g / 1000.0 * G * args.lever
    print(f"joint {name} (id {args.id}), holding {args.angle:+.0f} deg "
          f"(raw {target})")
    print(f"added torque will be {args.mass_g:.0f} g x {args.lever:.3f} m = "
          f"{tau_extra:.3f} N.m")
    print(f"expected deflection: {tau_extra / 50 * 180 / 3.14159:.2f} deg at "
          f"kp 50, {tau_extra / 10 * 180 / 3.14159:.2f} deg at kp 10\n")
    print("SAFETY")
    print("  * The robot must be SUSPENDED, feet clear, so the only load on this")
    print("    joint is the limb and the mass you add.")
    print("  * This joint will be near its torque limit (2.35 N.m). Do not add")
    print("    more mass than asked for -- a saturated servo reads as infinite")
    print("    compliance and the measurement becomes meaningless.")
    print("  * Hang the mass gently. Dropping it on shock-loads the gearbox.")
    print("  * Keep clear of the limb; it is holding a load under power.\n")
    input("press ENTER when the robot is hanging clear, or Ctrl-C to abort... ")

    bus = Bus(args.port, args.baud)
    try:
        if not bus.ping(args.id, tries=6):
            sys.exit(f"id {args.id} did not answer")
        bus.write1(args.id, REG_TORQUE, 1)
        bus.write1(args.id, REG_ACC, args.acc)
        bus.write(args.id, REG_GOAL_POS,
                  [target & 0xFF, target >> 8, 0, 0,
                   args.speed & 0xFF, (args.speed >> 8) & 0xFF])
        print(f"\nmoving to {args.angle:+.0f} deg ...")
        unloaded, u_lo, u_hi = settle_read(bus, args.id, seconds=3.0)
        print(f"  unloaded: present {unloaded}  (spread {u_hi - u_lo} counts)"
              f"  error {abs(unloaded - target) / DEG:.2f} deg")

        print(f"\n>>> Now hang {args.mass_g:.0f} g at {args.lever:.3f} m from the "
              f"joint axis.")
        print(">>> Same limb, hanging straight down, gently.")
        input("    press ENTER once it is on and still... ")
        loaded, l_lo, l_hi = settle_read(bus, args.id, seconds=3.0)
        print(f"  loaded:   present {loaded}  (spread {l_hi - l_lo} counts)"
              f"  error {abs(loaded - target) / DEG:.2f} deg")

        delta_counts = abs(loaded - unloaded)
        delta_rad = delta_counts / DEG * 3.14159265 / 180.0
        print(f"\n  deflection under load: {delta_counts} counts = "
              f"{delta_counts / DEG:.2f} deg")
        if delta_counts < 3:
            print("\n  TOO SMALL TO TRUST -- under 3 counts is the encoder's own")
            print("  resolution. Either the servo is very stiff, or the mass is")
            print("  not actually loading this joint. Check the lever arm is")
            print("  horizontal and the mass is beyond the joint, not before it.")
        else:
            kp = tau_extra / delta_rad
            print(f"\n  kp = {tau_extra:.3f} N.m / {delta_rad:.4f} rad = "
                  f"{kp:.1f} N.m/rad")
            print(f"  the XML models kp = 50; kp_scale = {kp / 50:.2f}")
            if kp > 35:
                print("  -> consistent with the XML. Training's 0.60-1.40 span is right.")
            elif kp > 18:
                print("  -> softer than the XML but nowhere near the retracted kp 10.")
            else:
                print("  -> genuinely soft. Tell me: the training kp span needs to move.")
        print("\n>>> Remove the mass before anything else.")
        input("    press ENTER once it is off... ")
    finally:
        try:
            bus.write1(args.id, REG_TORQUE, 0)
            print("torque released")
        finally:
            bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
