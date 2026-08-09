#!/usr/bin/env python3
"""
replay_motion.py -- can this robot actually execute the reference gait?

Open-loop replay of the walking animation, at the deployment control rate,
with the robot SUSPENDED. No policy, no balance, no feedback: it just streams
`ref_q(phase)` to the servos exactly the way the walking node would, and
records what the joints actually did.

That is the question a step-response test cannot answer. A step measures how a
servo accelerates from rest to a new target; walking asks it to follow a moving
trajectory it is already chasing. Those can differ a lot, and the difference
decides whether a gait is deployable or has to be slowed down.

    # feet OFF THE GROUND, robot hanging
    python3 tools/replay_motion.py --speed 0.25 --cycles 2
    python3 tools/replay_motion.py --speed 0.6  --cycles 2     # as trained

THE ROBOT MUST BE SUSPENDED. Every leg joint moves through its full gait range
and the legs will swing. Nothing balances it.

It ramps into the start pose over `--ramp` seconds, replays `--cycles` gait
cycles, then returns to the standing pose and releases torque.

Read the RMS and peak tracking error. Under ~5 deg means the hardware follows
the gait and the trained policy has a chance. Tens of degrees means the servos
cannot do it and the animation has to be slowed (error scales roughly with
speed) before any amount of training will help.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, os.path.join(_SRC, "humanoid_deploy"))
sys.path.insert(0, os.path.join(_SRC, "humanoid_calibration"))

from humanoid_calibration.calibration_store import CalibrationStore  # noqa: E402
from humanoid_deploy.servo_bus import ServoBus  # noqa: E402


def load_reference(bundle_path, motion_path):
    """Reference joint angles in ACTUATOR order, clipped to the model limits.

    Mirrors mjx_walk_env exactly: the npz stores joints in its own order, the
    env reindexes to actuator order and clips to jnt_range so the target is
    physically reachable. Any difference here would replay a trajectory the
    policy was never trained against.

    The joint order and the limits come from the exported policy bundle rather
    than from robot.xml, so this runs on the robot with nothing but numpy --
    mujoco is a training-machine dependency and has no business on a Jetson.
    The bundle carries both because they were read out of the same XML at
    export time.
    """
    from humanoid_deploy.policy import Policy
    policy = Policy(bundle_path)
    names = list(policy.joint_names)
    lo, hi = policy.xml_lower, policy.xml_upper

    motion = np.load(motion_path, allow_pickle=True)
    npz_names = list(motion["dof_names"])
    missing = [n for n in names if n not in npz_names]
    if missing:
        raise SystemExit(f"the motion file has no column for {missing}")
    col = [npz_names.index(n) for n in names]
    ref = np.clip(motion["dof_positions"][:, col], lo, hi)
    return names, np.asarray(ref, dtype=np.float64), float(motion["fps"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument("--bundle", default=os.path.join(
        _SRC, "humanoid_deploy", "models", "policy_bundle.npz"),
        help="any exported bundle -- only its joint order and limits are used")
    parser.add_argument("--motion", default=os.path.join(
        _SRC, "humanoid_deploy", "models", "biped_walk_amp.npz"))
    parser.add_argument("--speed", type=float, default=0.6,
                        help="motion_speed, as passed to the trainer")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--ramp", type=float, default=3.0,
                        help="seconds to ease into the first reference pose")
    parser.add_argument("--goal-speed", type=int, default=3000)
    parser.add_argument("--acc", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and report, write nothing to the servos")
    args = parser.parse_args()

    names, ref, fps = load_reference(args.bundle, args.motion)
    n_cycle = ref.shape[0] - 1
    period = n_cycle / (fps * args.speed)
    dt = 1.0 / args.rate
    dphase = dt / period
    print(f"reference {ref.shape[0]} frames, {fps:.0f} fps, speed {args.speed}")
    print(f"gait period {period:.3f} s, {period/dt:.1f} control steps per cycle")
    print(f"peak reference speed "
          f"{np.degrees(np.abs(np.diff(ref, axis=0)).max() * fps * args.speed):.0f} deg/s")

    store = CalibrationStore.load()
    if not store.joints:
        raise SystemExit("no calibration found; run the calibration first")
    missing = [n for n in names if n.replace("_joint", "") not in store.joints]
    if missing:
        raise SystemExit(f"calibration has no entry for {missing}")
    cals = [store.joints[n.replace("_joint", "")] for n in names]
    ids = [c.servo_id for c in cals]

    def to_counts(q_rad):
        out = []
        for c, q in zip(cals, q_rad):
            raw = c.zero_raw + c.direction * np.degrees(q) * c.counts_per_rev / 360.0
            out.append(int(round(raw)) % c.counts_per_rev)
        return out

    def to_rad(counts):
        return np.array([np.radians(c.raw_to_deg(v)) for c, v in zip(cals, counts)])

    # Refuse to replay anything outside the measured travel -- this tool has no
    # safety supervisor behind it.
    for i, c in enumerate(cals):
        lo_d, hi_d = min(c.min_deg, c.max_deg), max(c.min_deg, c.max_deg)
        r_lo, r_hi = np.degrees(ref[:, i].min()), np.degrees(ref[:, i].max())
        if r_lo < lo_d - 1e-6 or r_hi > hi_d + 1e-6:
            raise SystemExit(
                f"{names[i]}: reference spans [{r_lo:+.1f},{r_hi:+.1f}] deg but "
                f"the joint is only calibrated for [{lo_d:+.1f},{hi_d:+.1f}]")

    if args.dry_run:
        print("\n--dry-run: nothing written. Reference fits the calibrated travel.")
        return 0

    print("\n*** THE ROBOT MUST BE SUSPENDED -- feet off the ground ***")
    input("press ENTER when it is hanging clear, or Ctrl-C to abort... ")

    bus = ServoBus(args.port, args.baud, timeout_ms=30)
    log = []
    try:
        silent = [i for i in ids if not bus.ping(i)]
        if silent:
            raise SystemExit(f"these servos did not answer: {silent}")
        bus.set_speed_and_acc(ids, args.goal_speed, args.acc)

        start = to_rad([bus.read_positions(ids)[i] for i in ids])
        target0 = ref[0]
        steps = max(1, int(args.ramp / dt))
        print(f"easing into the start pose over {args.ramp:.1f} s ...")
        bus.set_torque(ids, True)
        for k in range(steps):
            a = (k + 1) / steps
            bus.write_goal_positions(ids, to_counts((1 - a) * start + a * target0))
            time.sleep(dt)

        total = int(args.cycles * period / dt)
        print(f"replaying {args.cycles:.1f} cycles = {total} steps "
              f"({total * dt:.1f} s) ...")
        phase = 0.0
        t0 = time.perf_counter()
        for k in range(total):
            # Same midpoint sampling the env uses: a zero-order hold on the
            # interval's leading edge lags the animation by half a cycle.
            f = np.mod(phase + 0.5 * dphase, 1.0) * n_cycle
            i0 = int(np.floor(f))
            w = f - i0
            target = (1 - w) * ref[i0] + w * ref[i0 + 1]

            bus.write_goal_positions(ids, to_counts(target))
            raw = bus.read_positions(ids)
            if len(raw) == len(ids):
                log.append((time.perf_counter() - t0, target.copy(),
                            to_rad([raw[i] for i in ids])))
            phase = np.mod(phase + dphase, 1.0)
            time.sleep(max(0.0, dt - (time.perf_counter() - t0) % dt))

        print("returning to the standing pose ...")
        last = log[-1][1] if log else target0
        for k in range(steps):
            a = (k + 1) / steps
            bus.write_goal_positions(ids, to_counts((1 - a) * last))
            time.sleep(dt)
    finally:
        try:
            bus.set_torque(ids, False)
        finally:
            bus.close()

    if len(log) < 10:
        print("not enough samples logged")
        return 1

    cmd = np.degrees(np.array([c for _, c, _ in log]))
    meas = np.degrees(np.array([m for _, _, m in log]))
    err = cmd - meas
    print(f"\n{len(log)} samples\n")
    print(f"{'joint':<26}{'cmd range':>16}{'RMS err':>10}{'peak err':>10}{'lag':>8}")
    for i, n in enumerate(names):
        rms = float(np.sqrt(np.mean(err[:, i] ** 2)))
        peak = float(np.abs(err[:, i]).max())
        span = cmd[:, i].max() - cmd[:, i].min()
        # Cross-correlation lag: how many control cycles the measurement trails.
        lag = ""
        if span > 2.0:
            best, best_r = 0, -2.0
            for s in range(0, 13):
                a = cmd[:len(cmd) - s, i]
                b = meas[s:, i]
                if a.std() > 1e-6 and b.std() > 1e-6:
                    r = float(np.corrcoef(a, b)[0, 1])
                    if r > best_r:
                        best_r, best = r, s
            lag = f"{best} cyc"
        print(f"{n:<26}{cmd[:,i].min():>7.1f}..{cmd[:,i].max():<7.1f}"
              f"{rms:>9.2f}{peak:>10.2f}{lag:>8}")

    moving = [i for i in range(len(names)) if cmd[:, i].max() - cmd[:, i].min() > 2.0]
    worst = max(float(np.sqrt(np.mean(err[:, i] ** 2))) for i in moving)
    print(f"\nworst RMS tracking error on a moving joint: {worst:.1f} deg")
    if worst < 5.0:
        print("  -> the hardware follows this gait. The policy has a chance.")
    elif worst < 15.0:
        print("  -> marginal. Expect the real gait to differ visibly from sim; "
              "consider a lower --speed.")
    else:
        print("  -> the servos cannot execute this animation. Retrain with a "
              "lower motion_speed; tracking error scales roughly with speed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
