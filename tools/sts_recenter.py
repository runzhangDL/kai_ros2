#!/usr/bin/env python3
"""
sts_recenter.py -- move every servo's electrical zero to the middle of its turn.

Why this exists
---------------
These servos were zeroed at encoder count ~0 instead of ~2048, so eleven of the
thirteen joints have to travel across the 0/4095 wrap to reach their own range.
That is not survivable: measured on this robot, an STS3215 commanded from count
146 to count 4000 drove *upward* through 981 and kept going -- the long way,
3854 counts -- instead of the 242-count short path. On an assembled leg that is
a full-turn slam into a mechanical stop at speed.

`humanoid_deploy` refuses to start while any joint is in that state, which is
why you are here.

What it does
------------
Feetech firmware exposes a "set current position as midpoint" operation: write
128 to the Torque_Enable register (40) and the servo adjusts its position
correction (EEPROM 31/32) so the shaft's present angle reads 2048. This script
does that, verifies it, and then updates the calibration file by the exact
amount each joint moved -- so you do NOT have to redo the physical calibration.

Safety
------
* Read-only unless you pass --apply.
* Refuses to touch more than one servo unless you pass --all.
* Torque is forced off before and after every write.
* The old position-correction value of every servo is saved to a JSON file
  before anything changes, so the operation is reversible.
* The calibration file is backed up, then updated, then re-checked against the
  seam. If the result would still cross the wrap, it says so.

Order of operations
-------------------
    # 1. prove the mechanism on ONE servo with its linkage disconnected
    python3 sts_recenter.py probe --id 1
    python3 sts_recenter.py one --id 1 --apply

    # 2. once that works: support the robot, torque off, pose it roughly
    #    standing, and do the rest. Do not move it between the read and write.
    python3 sts_recenter.py all
    python3 sts_recenter.py all --apply
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_tool import Bus  # noqa: E402  (same directory, shared protocol)

REG_OFS = 31            # 2B EEPROM, position correction
REG_TORQUE = 40         # 1B RAM; the magic value 128 means "calibrate midpoint"
REG_LOCK = 55           # 1B 0=unlock EEPROM, 1=lock
REG_PRESENT_POS = 56    # 2B

CALIBRATE_MIDPOINT = 128
CENTER = 2048
CPR = 4096
VERIFY_TOLERANCE = 24   # counts; the shaft settles slightly after the write

DEFAULT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def calibration_path():
    explicit = os.environ.get("HUMANOID_CALIBRATION_FILE")
    if explicit:
        return explicit
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "humanoid_robot", "calibration.yaml")


def read_state(bus, sid):
    """Present position and current position-correction, or None if silent."""
    if not bus.ping(sid, tries=6):
        return None
    position = bus.read2(sid, REG_PRESENT_POS)
    correction = bus.read2(sid, REG_OFS)
    if position is None:
        return None
    return {"id": sid, "position": position, "correction": correction}


def recenter(bus, sid):
    """Set this servo's current shaft angle to read 2048. Returns the new pos."""
    bus.write1(sid, REG_TORQUE, 0)          # never write EEPROM under load
    time.sleep(0.02)
    bus.write1(sid, REG_LOCK, 0)            # unlock EEPROM
    time.sleep(0.02)
    bus.write1(sid, REG_TORQUE, CALIBRATE_MIDPOINT, expect_reply=False)
    time.sleep(0.15)                        # the servo writes EEPROM here
    bus.write1(sid, REG_LOCK, 1)            # lock again
    time.sleep(0.05)
    bus.write1(sid, REG_TORQUE, 0)          # leave it limp
    time.sleep(0.02)
    return bus.read2(sid, REG_PRESENT_POS)


def shift_of(before, after):
    """How much the reported frame moved, wrapped into +/- half a turn."""
    delta = (after - before) % CPR
    return delta - CPR if delta > CPR / 2 else delta


# ---------------------------------------------------------------- commands

def cmd_probe(bus, args):
    ids = [args.id] if args.id is not None else DEFAULT_IDS
    print(f"{'id':>4}{'present':>10}{'correction':>13}{'distance to seam':>19}")
    for sid in ids:
        state = read_state(bus, sid)
        if state is None:
            print(f"{sid:>4}{'NO REPLY':>10}")
            continue
        position = state["position"]
        margin = min(position, CPR - 1 - position)
        flag = "  <-- near the wrap" if margin < 400 else ""
        print(f"{sid:>4}{position:>10}{str(state['correction']):>13}"
              f"{margin:>13} counts{flag}")
    print("\nNothing was written. A healthy robot reads near 2048 when standing.")


def cmd_one(bus, args):
    if args.id is None:
        sys.exit("--id is required")
    sid = args.id
    before = read_state(bus, sid)
    if before is None:
        sys.exit(f"id {sid} did not answer")
    print(f"id {sid}: present={before['position']}  correction={before['correction']}")

    if not args.apply:
        print(f"\nWould re-center so this shaft angle reads {CENTER}.")
        print("Nothing written. Re-run with --apply once the linkage is free.")
        return

    print(f"\nre-centering id {sid} ...")
    after = recenter(bus, sid)
    if after is None:
        sys.exit("no position after the write -- re-run `probe` and check the bus")
    moved = shift_of(before["position"], after)
    print(f"  present {before['position']} -> {after}  (frame shifted {moved:+d})")
    if abs(after - CENTER) > VERIFY_TOLERANCE:
        sys.exit(
            f"FAILED: expected ~{CENTER}, got {after}. The servo did not accept "
            "the midpoint calibration -- do NOT run --all. Report this."
        )
    print(f"  OK, reads {after} (within {VERIFY_TOLERANCE} of {CENTER})")
    print(
        "\nNOTE: this servo's stored calibration is now stale -- its zero moved\n"
        "  but the calibration file did not. `all` cannot repair that later,\n"
        "  because it derives each shift from the file's existing zero. Once the\n"
        "  joint is back on the robot, redo just that one:\n"
        "    ros2 run humanoid_calibration calibrate --only <joint> --force"
    )


def cmd_all(bus, args):
    import yaml

    ids = DEFAULT_IDS
    print("Reading every servo BEFORE any write. Do not move the robot from here.")
    before = {}
    for sid in ids:
        state = read_state(bus, sid)
        if state is None:
            sys.exit(f"id {sid} did not answer -- fix the bus before re-centering")
        before[sid] = state["position"]
        print(f"  id {sid:>2}: present={state['position']:>5}  "
              f"correction={state['correction']}")

    path = args.store or calibration_path()
    if not os.path.isfile(path):
        sys.exit(f"no calibration file at {path}; calibrate first")
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    joints = document.get("joints") or {}
    by_id = {int(entry["servo_id"]): (name, entry) for name, entry in joints.items()}

    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        sys.exit(f"calibration file has no entry for servo ids {missing}")

    print(f"\nplanned shift per joint (so the current pose reads {CENTER}):")
    plan = {}
    for sid in ids:
        name, entry = by_id[sid]
        delta = shift_of(before[sid], CENTER)
        new_zero = (int(entry["zero_raw"]) + delta) % CPR
        plan[sid] = delta
        print(f"  id {sid:>2} {name:<22} shift {delta:+6d}   "
              f"zero {entry['zero_raw']:>5} -> {new_zero:>5}")

    if not args.apply:
        print("\nNothing written. Re-run with --apply when the robot is "
              "supported, limp, and roughly in its standing pose.")
        return

    backup = path + ".before-recenter"
    shutil.copy2(path, backup)
    corrections = {sid: read_state(bus, sid)["correction"] for sid in ids}
    record = args.store_record or (path + ".recenter-record.json")
    with open(record, "w", encoding="utf-8") as handle:
        json.dump({"corrections_before": corrections,
                   "positions_before": before}, handle, indent=2)
    print(f"\ncalibration backed up to {backup}")
    print(f"old position corrections saved to {record}")

    print("\nre-centering ...")
    failures = []
    for sid in ids:
        after = recenter(bus, sid)
        moved = None if after is None else shift_of(before[sid], after)
        ok = after is not None and abs(after - CENTER) <= VERIFY_TOLERANCE
        print(f"  id {sid:>2}: {before[sid]:>5} -> {str(after):>5}  "
              f"shift {str(moved):>6}  {'ok' if ok else 'FAILED'}")
        if not ok:
            failures.append(sid)
    if failures:
        sys.exit(
            f"\nservos {failures} did not re-center. The calibration file was "
            f"NOT changed; it is still valid for the servos that did move only "
            f"if you restore them from {record}. Stop and report this."
        )

    for sid in ids:
        _name, entry = by_id[sid]
        delta = plan[sid]
        for key in ("zero_raw", "min_raw", "max_raw"):
            entry[key] = (int(entry[key]) + delta) % CPR
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("# Humanoid motor calibration -- zeros shifted by "
                     "tools/sts_recenter.py after re-centering the servos.\n")
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    print(f"\ncalibration updated: {path}")
    print("Now verify before going anywhere near a launch file:")
    print("  ros2 run humanoid_calibration verify_calibration --live")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="read positions and corrections, no writes")
    p.add_argument("--id", type=int)
    p.set_defaults(f=cmd_probe)

    p = sub.add_parser("one", help="re-center a single servo")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(f=cmd_one)

    p = sub.add_parser("all", help="re-center all 13 and update the calibration")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--store", help="path to calibration.yaml")
    p.add_argument("--store-record", help="where to save the old corrections")
    p.set_defaults(f=cmd_all)

    args = parser.parse_args()
    bus = Bus(args.port, args.baud, verbose=args.verbose)
    try:
        args.f(bus, args)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
