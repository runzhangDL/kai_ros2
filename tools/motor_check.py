#!/usr/bin/env python3
"""
motor_check.py -- is every servo alive, cool, and free to move?

Read-only by default. Reports, for each servo: whether it answers, its model
number, whether torque is engaged, its position, voltage and temperature.

    python3 tools/motor_check.py
    python3 tools/motor_check.py --release        # force torque off first
    python3 tools/motor_check.py --backdrive 20   # 20 s: move joints by hand

`--release` is the only thing here that writes, and it only ever writes 0 to
Torque_Enable (RAM 40). It cannot touch EEPROM.

Diagnosing a joint that will not move by hand
---------------------------------------------
Three different faults look identical from the outside, and the `torque` column
plus a back-drive test separates them:

  torque=ON            -> not a fault. Something left it engaged; --release fixes
                          it. A servo holding position is *supposed* to resist.
  torque=off, no motion -> mechanically jammed, or the gearbox is stripped. The
                          encoder still reads, so the servo is electrically fine.
  no reply at all       -> wiring, or the servo is dead.

Back-drive mode samples every servo continuously while you move each joint by
hand. A joint you moved that reports a range of ~0 counts is stuck; one that
reports hundreds is healthy.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sts_tool import Bus, REG_MODEL, REG_PRESENT_POS, REG_TORQUE, REG_VOLTAGE  # noqa: E402

DEFAULT_IDS = "1,2,3,4,5,6,7,8,9,10,11,12,13"
DEG = 4096 / 360.0


def survey(bus, ids, release):
    print(f"{'id':>4}{'model':>8}{'torque':>9}{'pos':>7}{'volts':>8}"
          f"{'temp':>7}   note")
    rows = []
    for sid in ids:
        if not bus.ping(sid, tries=6):
            print(f"{sid:>4}{'--':>8}{'--':>9}{'--':>7}{'--':>8}{'--':>7}"
                  f"   NO REPLY -- wiring, power, or a dead servo")
            rows.append((sid, None))
            continue
        if release:
            bus.write1(sid, REG_TORQUE, 0)
            time.sleep(0.01)
        model = bus.read2(sid, REG_MODEL)
        torque = bus.read(sid, REG_TORQUE, 1)
        pos = bus.read2(sid, REG_PRESENT_POS)
        # Voltage and temperature are adjacent; one 2-byte read cannot confuse
        # them for each other the way two 1-byte reads can.
        health = bus.read(sid, REG_VOLTAGE, 2)
        volts = health[0] / 10.0 if health else None
        temp = health[1] if health else None
        on = torque[0] if torque else None

        notes = []
        if on:
            notes.append("TORQUE ON -- it will resist by hand")
        if temp is not None and temp > 55:
            notes.append(f"HOT")
        if volts is not None and volts < 10.0:
            notes.append("LOW VOLTAGE")
        print(f"{sid:>4}{str(model):>8}{('ON' if on else 'off'):>9}{str(pos):>7}"
              f"{(f'{volts:.1f}' if volts else '--'):>8}"
              f"{(f'{temp}C' if temp is not None else '--'):>7}"
              f"   {'; '.join(notes)}")
        rows.append((sid, pos))
    return rows


def backdrive(bus, ids, seconds):
    print(f"\nSampling for {seconds:.0f} s. Move EVERY joint by hand, one at a "
          f"time, through a few degrees.")
    print("A joint you moved that shows ~0 counts of range is stuck.\n")
    seen = {sid: [] for sid in ids}
    end = time.time() + seconds
    while time.time() < end:
        for sid in ids:
            v = bus.read2(sid, REG_PRESENT_POS, tries=1)
            if v is not None:
                seen[sid].append(v)
        left = end - time.time()
        print(f"\r  {left:4.0f} s remaining ...", end="", flush=True)
    print("\r" + " " * 32 + "\r", end="")

    print(f"{'id':>4}{'samples':>9}{'min':>7}{'max':>7}{'range':>8}"
          f"{'degrees':>10}   verdict")
    for sid in ids:
        vals = seen[sid]
        if not vals:
            print(f"{sid:>4}{0:>9}{'--':>7}{'--':>7}{'--':>8}{'--':>10}   NO REPLY")
            continue
        span = max(vals) - min(vals)
        verdict = "moved" if span > 20 else ("barely moved" if span > 5
                                             else "DID NOT MOVE")
        print(f"{sid:>4}{len(vals):>9}{min(vals):>7}{max(vals):>7}{span:>8}"
              f"{span / DEG:>9.1f}   {verdict}")
    print("\nA joint you did not touch will also read DID NOT MOVE -- only "
          "trust rows\nfor joints you actually tried to move.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument("--ids", default=DEFAULT_IDS)
    parser.add_argument("--release", action="store_true",
                        help="write 0 to Torque_Enable before reading")
    parser.add_argument("--backdrive", type=float, metavar="SECONDS",
                        help="then sample while you move joints by hand")
    args = parser.parse_args()

    ids = [int(x) for x in args.ids.split(",")]
    bus = Bus(args.port, args.baud)
    try:
        rows = survey(bus, ids, args.release)
        alive = [sid for sid, pos in rows if pos is not None]
        print(f"\n{len(alive)}/{len(ids)} servos answered")
        if args.backdrive:
            backdrive(bus, alive, args.backdrive)
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
