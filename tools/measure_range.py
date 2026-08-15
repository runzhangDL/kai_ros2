"""Measure a joint's real mechanical range by hand, with the torque OFF.

WHY THIS EXISTS
---------------
Every one of the 27 entries in humanoid_calibration/config/joint_limits.yaml is
marked ``# PLACEHOLDER``. They came from the CAD model, not from the robot. Two
of them are load-bearing for walking and both look wrong:

    left_hip_roll    -90 .. 0     right_hip_roll    0 .. +90

One-sided. The nominal walking pose puts hip roll at exactly 0.0 deg, which is
the boundary, so the policy has ZERO authority in one direction and 90 deg in
the other. Measured in simulation over a 10 s walk, the trained policy used
4.6 deg of left hip roll and exactly 0.0 deg of right -- it never moved it at
all, because it cannot. Lateral weight transfer over the stance foot is what
unloads a foot so it can swing, and that is the motion this constraint forbids.

A knee being one-sided is correct -- knees bend one way. A hip roll being
one-sided is not obviously correct, and nobody has checked.

SAFETY
------
**This tool never writes a position.** It writes torque=0 once per joint, which
makes the joint limp, and then only reads. Nothing is commanded to move. The
robot must be supported or lying down before you start, because a limp leg
falls.

The 0/4095 seam: the encoder wraps, and a joint whose travel crosses the seam
would otherwise read a range of "0 .. 4095". Readings are unwrapped relative to
the calibrated zero, so a joint sitting near the seam still reports sensibly.

USAGE
-----
    python3 -u tools/measure_range.py 5,9          # both hip rolls
    python3 -u tools/measure_range.py 1,13         # both ankle pitches

Move the joint slowly by hand through its whole travel, both directions, until
it stops against something solid. Stop when the printed min/max stop growing.
Ctrl-C to finish and print the summary.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_tool import Bus, REG_PRESENT_POS, REG_TORQUE  # noqa: E402

CPR = 4096
DEG = CPR / 360.0

# The live display ends in a carriage return with no newline, so it rewrites
# one line in place. Flush explicitly rather than relying on the buffering mode
# -- under `python3 -u` either mode happens to work, but that is luck.
#
# A static line here means the joint is not moving. It does NOT mean the tool
# is hung, which is the reading it invites; hence the torque readback below and
# the zero-travel warning at the end, so the two cases are told apart.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(write_through=True)


def load_calibration():
    """servo_id -> (name, zero_raw, direction). Empty dict if none."""
    import yaml
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    path = os.environ.get("HUMANOID_CALIBRATION_FILE") or os.path.join(
        base, "humanoid_robot", "calibration.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    out = {}
    for name, entry in (document.get("joints") or {}).items():
        out[int(entry["servo_id"])] = (
            name, int(entry["zero_raw"]), int(entry["direction"]))
    return out


def unwrapped_deg(raw, zero, direction):
    """Signed degrees from the calibrated zero, across the 0/4095 seam.

    The servo counts 0..4095 clockwise and wraps. Taking the difference modulo
    the full turn and folding it into +/-half a turn gives the short way round,
    which is the only interpretation that makes sense for a joint that cannot
    physically spin.
    """
    delta = (raw - zero) % CPR
    if delta > CPR // 2:
        delta -= CPR
    return direction * delta / DEG


def main():
    parser = argparse.ArgumentParser(
        description="measure real joint travel by hand, torque off, read-only",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", help="comma-separated servo ids, e.g. 5,9")
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    ids = [int(x) for x in args.ids.split(",")]
    cal = load_calibration()
    missing = [i for i in ids if i not in cal]
    if missing:
        sys.exit(f"no calibration for id(s) {missing}; run the calibration "
                 f"first -- without a zero, degrees are meaningless")

    bus = Bus(args.port, args.baud)
    print(f"port {args.port} @ {args.baud}\n")

    print("Making these joints LIMP. Support the robot before continuing.")
    for sid in ids:
        name, _, _ = cal[sid]
        print(f"  id{sid:>3}  {name}")
    try:
        input("\npress Enter when the robot is supported, Ctrl-C to abort: ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted, nothing was changed")
        return 1

    for sid in ids:
        bus.write1(sid, REG_TORQUE, 0)
    # Read it back. A joint that silently stayed energised would look exactly
    # like a joint nobody moved -- both report zero travel.
    for sid in ids:
        name, _, _ = cal[sid]
        data = bus.read(sid, REG_TORQUE, 1)
        if data is None:
            print(f"  WARNING id{sid} {name}: no reply reading torque back")
        elif data[0] != 0:
            print(f"  WARNING id{sid} {name}: torque still reads {data[0]}, "
                  f"the joint is NOT limp -- do not force it")
        else:
            print(f"  id{sid:>3} {name}: limp, confirmed")
    print("\nNow move each joint SLOWLY by hand to both ends of its travel.")
    print("The line below updates live -- if it does not move when you move")
    print("the joint, stop and say so. Ctrl-C when min/max stop growing.\n")

    lo = {sid: None for sid in ids}
    hi = {sid: None for sid in ids}
    period = 1.0 / args.hz
    last_beat = time.time()
    try:
        while True:
            line = []
            for sid in ids:
                data = bus.read(sid, REG_PRESENT_POS, 2)
                if data is None:
                    line.append(f"id{sid}: --")
                    continue
                raw = data[0] | (data[1] << 8)
                name, zero, direction = cal[sid]
                angle = unwrapped_deg(raw, zero, direction)
                lo[sid] = angle if lo[sid] is None else min(lo[sid], angle)
                hi[sid] = angle if hi[sid] is None else max(hi[sid], angle)
                line.append(f"id{sid} {angle:+7.1f}  [{lo[sid]:+7.1f} .. "
                            f"{hi[sid]:+7.1f}]")
            print("   ".join(line) + "        \r", end="", flush=True)
            # A scrolling record every 2 s, so there is still visible progress
            # if the terminal or a pipe swallows the carriage return.
            now = time.time()
            if now - last_beat >= 2.0:
                print()
                last_beat = now
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n")

    print(f"{'joint':<24}{'id':>4}{'measured travel (deg)':>30}")
    for sid in ids:
        name, _, _ = cal[sid]
        if lo[sid] is None:
            print(f"{name:<24}{sid:>4}{'no replies':>26}")
            continue
        span = hi[sid] - lo[sid]
        print(f"{name:<24}{sid:>4}"
              f"{f'{lo[sid]:+.1f} .. {hi[sid]:+.1f}  (span {span:.1f})':>26}")
    stuck = [sid for sid in ids
             if lo[sid] is not None and hi[sid] - lo[sid] < 1.0]
    if stuck:
        names = ", ".join(cal[sid][0] for sid in stuck)
        print(f"\n*** {names} did not move at all. That is not a measurement.")
        print("    Either the joint was never moved by hand, or it is seized,")
        print("    or the encoder is not reporting. Re-run and move it while")
        print("    watching the live line.")
    print("\nTorque is still OFF -- the joints are limp. Keep the robot "
          "supported.\nNothing was written except torque=0; no position was "
          "ever commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
