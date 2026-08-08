#!/usr/bin/env python3
"""
analyze_stand.py -- summarise a recorded standing run.

    ros2 run ... # (not a ROS node -- just run it with python3)
    python3 tools/analyze_stand.py stand_012701

Answers the three questions a stand attempt can have:

  holds     tilt stays put, commands are quiet          -> the policy balances
  drifts    tilt trends one way                         -> a steady bias
  oscillates commands reverse sign many times a second  -> too slow for its gains

and reports the things that decide whether it was safe: how long it ran, what
faulted it if anything, how well each servo tracked its command, and the peak
temperature.

Needs the ROS 2 environment sourced (it uses rosbag2_py), but is not a node.
"""

import argparse
import re
import sys
from collections import defaultdict


def read_bag(path):
    """Yield (topic, message, t_seconds) for every message in the bag."""
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the rosbag2 python API ({exc}).\n"
            "Source your ROS 2 environment first:  source install/setup.bash"
        ) from exc

    reader = SequentialReader()
    try:
        reader.open(StorageOptions(uri=path, storage_id="sqlite3"),
                    ConverterOptions("", ""))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot open bag {path!r}: {exc}") from exc

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    t0 = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in types:
            continue
        seconds = stamp * 1e-9
        t0 = seconds if t0 is None else t0
        try:
            message = deserialize_message(data, get_message(types[topic]))
        except Exception:  # noqa: BLE001 - a truncated tail must not kill the run
            continue
        yield topic, message, seconds - t0


_TILT = re.compile(r"tilt=([-\d.]+)deg")
_STATE = re.compile(r"state=(\w+)")
_TEMP = re.compile(r"maxtemp=([-\d.]+)C")
_FAULT = re.compile(r"fault=(.+?)(?:\s\s|$)")


def summarise(path, joint_report):
    status, tilts, temps, faults = [], [], [], []
    measured, commanded = {}, {}
    first_running = last_time = None

    for topic, message, t in read_bag(path):
        if topic.endswith("/status"):
            text = message.data
            status.append((t, text))
            state = _STATE.search(text)
            if state and state.group(1) == "running" and first_running is None:
                first_running = t
            tilt = _TILT.search(text)
            if tilt:
                tilts.append((t, float(tilt.group(1))))
            temp = _TEMP.search(text)
            if temp:
                temps.append(float(temp.group(1)))
            fault = _FAULT.search(text)
            if fault:
                faults.append(fault.group(1).strip())
        elif topic == "/humanoid/joint_states":
            for name, value in zip(message.name, message.position):
                measured.setdefault(name, []).append((t, value))
        elif topic.endswith("/command"):
            for name, value in zip(message.name, message.position):
                commanded.setdefault(name, []).append((t, value))
        last_time = t

    if last_time is None:
        raise SystemExit(f"{path}: no messages read")

    print(f"\n=== {path} ===")
    print(f"recording  {last_time:.1f} s")
    if first_running is not None:
        print(f"RUNNING    from {first_running:.1f} s "
              f"({last_time - first_running:.1f} s under policy control)")
    else:
        print("RUNNING    never reached -- it did not finish ramping")
    if faults:
        print(f"FAULT      {faults[0]}")
    else:
        print("FAULT      none")
    if temps:
        print(f"maxtemp    {min(temps):.0f} -> {max(temps):.0f} C")

    # --- the verdict, from tilt after the ramp completes ---
    start = first_running if first_running is not None else 0.0
    window = [(t, v) for t, v in tilts if t >= start]
    if len(window) < 5:
        print("\nnot enough tilt samples after the ramp to judge.")
    else:
        values = [v for _, v in window]
        span = window[-1][0] - window[0][0]
        half = len(window) // 2
        early = sum(v for _, v in window[:half]) / max(1, half)
        late = sum(v for _, v in window[half:]) / max(1, len(window) - half)
        drift = late - early
        print(f"\ntilt       mean {sum(values)/len(values):5.1f} deg   "
              f"min {min(values):5.1f}   max {max(values):5.1f}")
        print(f"           first half {early:.1f} -> second half {late:.1f} "
              f"({drift:+.1f} deg over {span:.0f} s)")

        if max(values) > 25.0:
            verdict = "FELL or nearly did"
        elif abs(drift) > 4.0:
            verdict = f"DRIFTING {'+' if drift > 0 else '-'} -- a steady bias"
        elif max(values) - min(values) > 12.0:
            verdict = "UNSTABLE -- large tilt excursions"
        else:
            verdict = "HOLDING -- tilt is bounded and not trending"
        print(f"           verdict: {verdict}")

    # --- per joint: tracking error and how busy the command was ---
    names = [n for n in commanded if n in measured]
    if names and joint_report:
        print(f"\n{'joint':<26}{'cmd mean':>10}{'cmd swing':>11}"
              f"{'reversals/s':>13}{'track err':>11}")
        for name in names:
            cmd = [v for t, v in commanded[name] if t >= start]
            meas = dict(measured[name])
            if len(cmd) < 5:
                continue
            mean = sum(cmd) / len(cmd)
            swing = max(cmd) - min(cmd)
            # Sign changes in the command's increments: a balancing policy
            # makes a few per second, a policy fighting its own lag makes many.
            deltas = [b - a for a, b in zip(cmd, cmd[1:]) if abs(b - a) > 1e-4]
            reversals = sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0)
            duration = max(1e-6, last_time - start)
            # Tracking: compare each command against the nearest measurement.
            errors = []
            for t, value in commanded[name]:
                if t < start:
                    continue
                nearest = min(meas, key=lambda k: abs(k - t)) if meas else None
                if nearest is not None and abs(nearest - t) < 0.1:
                    errors.append(abs(meas[nearest] - value))
            track = sum(errors) / len(errors) if errors else float("nan")
            print(f"{name:<26}{_deg(mean):>9}{_deg(swing):>10}"
                  f"{reversals / duration:>12.1f}{_deg(track):>10}")
        print("\ncmd swing = peak-to-peak commanded angle; a quiet stand is "
              "under ~3 deg.\nreversals/s above ~8 means the command is "
              "chattering, not balancing.")
    return 0


def _deg(radians):
    from math import degrees, isnan
    return "  n/a" if isnan(radians) else f"{degrees(radians):+.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="bag directory, e.g. stand_012701")
    parser.add_argument("--no-joints", action="store_true",
                        help="tilt summary only")
    args = parser.parse_args()
    return summarise(args.bag, joint_report=not args.no_joints)


if __name__ == "__main__":
    sys.exit(main())
