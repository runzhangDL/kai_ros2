#!/usr/bin/env python3
"""sim_handover.py -- does the stand->walk->stand handover survive, in sim?

Runs the deployment on a simulated robot. Not a re-implementation of it: this
imports the same ``GaitSequencer`` the ROS node uses to sequence the modes and
the same numpy ``Policy`` it uses for inference, and drives them with the same
observation layout at the same 25 Hz through the same measured actuator model.
What is stubbed out is the robot, not the software.

That matters because the handover is the part no amount of training validates.
The trainer measured the walking policy cold-started from its nominal pose and
run to the end of the episode. Deployment does something it has never seen: it
walks the robot into that pose under a *different* policy, hands over, and then
takes the robot back mid-stride. Those two seconds either side of the walk are
where this falls over, and they are cheaper to get wrong here.

    # the deployment sequence, once
    /home/run/env_mujoco/bin/python tools/sim_handover.py

    # is there a good phase of the gait to stop on?
    /home/run/env_mujoco/bin/python tools/sim_handover.py --sweep stop-phase

    # how slow does the handback have to be?
    /home/run/env_mujoco/bin/python tools/sim_handover.py --sweep recover

TRAINING MACHINE ONLY. It needs mujoco and the training env for the measured
actuator constants; the robot needs neither.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "src", "humanoid_deploy")
sys.path.insert(0, _PKG)

from humanoid_deploy.gait import GaitConfig, GaitMode, GaitSequencer  # noqa: E402
from humanoid_deploy.policy import Policy  # noqa: E402

MODELS = os.path.join(_PKG, "models")
#: Base height below this, or tilt beyond this, and it is on the floor.
FALL_HEIGHT_M = 0.32
FALL_TILT_DEG = 50.0


class SimRobot:
    """MuJoCo plus the measured servo model, driven by joint position targets.

    The servo model is the rate/acceleration-limited tracker with transport
    delay that the walking policy was trained against, reproduced here from the
    env's own constants rather than re-fitted. It is the difference between a
    policy that looks fine in sim and one that works: without it the simulated
    servos are perfect and the gait it learns is not executable.
    """

    def __init__(self, xml, match_solver=True):
        import mujoco
        from mjx_walk_free_env import MuJoCoBipedWalkFreeMJX

        self.mujoco = mujoco
        env = MuJoCoBipedWalkFreeMJX(xml_path=xml, control_rate_hz=25.0,
                                     actuator_model=True, obs_noise_scale=0.0)
        self.nominal = np.asarray(env._nominal_np, dtype=np.float64)
        #: Base height with the legs crouched (the walking policy's pose) and
        #: with them straight (the standing policy's). These differ by 28 mm,
        #: and starting the straight-legged robot at the crouched height buries
        #: its feet in the floor -- it then gets ejected upward and spends the
        #: first second recovering from a fault of the harness's own making.
        self.crouch_z = float(env._stand_z)
        self.stand_z = float(env._standing_height(
            self.__class__._raw_model(mujoco, xml), np.zeros(env.sys.nu)))
        self.v_max = np.asarray(env._v_max, dtype=np.float64)
        self.a_max = np.asarray(env._a_max, dtype=np.float64)
        self.delay = np.asarray(env._delay)
        self.qlen = int(env._queue_len)
        self.quant = float(env._joint_vel_quant)
        self.frame_skip = int(env._n_frames)

        self.m = mujoco.MjModel.from_xml_path(xml)
        if match_solver:
            self.m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            self.m.opt.iterations, self.m.opt.ls_iterations = 1, 4
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep * self.frame_skip
        self.imu = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        jids = self.m.actuator_trnid[:, 0]
        self.lo = self.m.jnt_range[jids, 0].copy()
        self.hi = self.m.jnt_range[jids, 1].copy()
        self.nu = self.m.nu
        self.rows = self.qlen - self.delay
        self.idx = np.arange(self.nu)

    @staticmethod
    def _raw_model(mujoco, xml):
        return mujoco.MjModel.from_xml_path(xml)

    def reset_standing(self, tilt_deg=0.0, joint_deg=0.0, rng=None):
        """The pose the robot is in when the operator arms it: legs straight.

        ``tilt_deg`` and ``joint_deg`` are standard deviations, not bounds, and
        they are the whole point of the robustness sweep: the operator cannot
        set the robot down to better than a degree or two of upright, and the
        calibration locates each joint to about a degree. Starting every trial
        from a perfect pose measures a condition the hardware never produces.
        """
        self.mujoco.mj_resetData(self.m, self.d)
        rng = rng or np.random.default_rng(0)
        joints = np.zeros(self.nu)
        if joint_deg > 0.0:
            joints = np.clip(np.radians(rng.normal(0.0, joint_deg, self.nu)),
                             self.lo, self.hi)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if tilt_deg > 0.0:
            roll, pitch = np.radians(rng.normal(0.0, tilt_deg, 2))
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            quat = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
        self.d.qpos[0:2] = 0.0
        self.d.qpos[2] = self.stand_z + 0.004
        self.d.qpos[3:7] = quat
        self.d.qpos[7:] = joints
        self.d.qvel[:] = 0.0
        self.d.xfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.m, self.d)
        self.cq = np.tile(joints, (self.qlen, 1))
        self.sp = joints.copy()
        self.sv = np.zeros(self.nu)

    def apply(self, target_rad):
        """One control cycle: target -> servo model -> physics."""
        ctrl = np.clip(np.asarray(target_rad, dtype=np.float64), self.lo, self.hi)
        delayed = self.cq[self.rows, self.idx]
        self.cq = np.concatenate([self.cq[1:], ctrl[None]], axis=0)
        want = np.clip((delayed - self.sp) / self.dt, -self.v_max, self.v_max)
        acc = np.clip((want - self.sv) / self.dt, -self.a_max, self.a_max)
        self.sv = self.sv + acc * self.dt
        self.sp = np.clip(self.sp + self.sv * self.dt, self.lo, self.hi)
        self.d.ctrl[:] = self.sp
        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.m, self.d)

    # -- what the node would read ------------------------------------------

    def sensors(self):
        imu_mat = self.d.site_xmat[self.imu].reshape(3, 3)
        gravity = imu_mat.T @ np.array([0.0, 0.0, -1.0])
        qvel = self.d.qvel[6:].copy()
        if self.quant > 0:
            qvel = np.round(qvel / self.quant) * self.quant
        return (self.d.sensordata[0:3].copy(), self.d.sensordata[3:6].copy(),
                gravity, self.d.qpos[7:].copy(), qvel)

    @property
    def height(self):
        return float(self.d.qpos[2])

    @property
    def tilt_deg(self):
        imu_mat = self.d.site_xmat[self.imu].reshape(3, 3)
        gravity = imu_mat.T @ np.array([0.0, 0.0, -1.0])
        return float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))

    @property
    def travel(self):
        return float(self.d.qpos[0]), float(self.d.qpos[1])


def run_once(robot, stand, walk, cfg, hold_s=4.0, trace=False,
             tilt_deg=0.0, joint_deg=0.0, rng=None):
    """One full stand -> walk -> stand cycle. Returns a result dict."""
    robot.reset_standing(tilt_deg, joint_deg, rng)
    seq = GaitSequencer(cfg, walk.default_pose)

    stand_action = np.zeros(stand.nu)
    walk_action = np.zeros(walk.nu)
    obs = np.zeros(walk.obs_size)

    def observe():
        accel, gyro, gravity, qpos, qvel = robot.sensors()
        obs[0:3], obs[3:6], obs[6:9] = accel, gyro, gravity
        obs[9:9 + robot.nu] = qpos
        obs[9 + robot.nu:9 + 2 * robot.nu] = qvel
        return obs

    # Let the standing policy take the robot before anything else happens --
    # the same thing arming does on hardware.
    peak_tilt, min_height, rows = 0.0, 1.0, []
    fell_in = None
    stages = [("pre", 1.5)]
    total = (1.5 + cfg.crouch_s + cfg.settle_s + cfg.walk_duration_s
             + cfg.recover_s + hold_s + 2.0)
    requested = False
    t = 0.0
    while t < total:
        observe()
        obs[9 + 2 * robot.nu:9 + 3 * robot.nu] = stand_action
        stand_action = stand.act(obs[:stand.obs_size])
        stand_target = stand.action_to_targets(stand_action)

        walk_target = None
        if seq.walking:
            obs[9 + 2 * robot.nu:9 + 3 * robot.nu] = walk_action
            obs[walk.layout.tail] = np.concatenate(
                [seq.phase_features(), seq.command()])
            walk_action = walk.act(obs)
            walk_target = walk.action_to_targets(walk_action)

        target, status = seq.step(stand_target, walk_target)
        if status.walk_started:
            # Cold start means last_action = 0, the way reset() defines it.
            walk_action = np.zeros(walk.nu)
        robot.apply(target)
        t += cfg.control_dt

        if not requested and t >= 1.5 and seq.mode is GaitMode.STAND:
            seq.request_walk()
            requested = True

        peak_tilt = max(peak_tilt, robot.tilt_deg)
        min_height = min(min_height, robot.height)
        if trace:
            rows.append((t, status.mode.value, status.phase, robot.tilt_deg,
                         robot.height, status.walk_authority))
        if fell_in is None and (robot.height < FALL_HEIGHT_M
                                or robot.tilt_deg > FALL_TILT_DEG):
            # "stand" spans two very different situations -- the settled pose
            # the robot arms in, and the pose the handback leaves it in -- and
            # confusing them would blame the standing policy for a bad
            # handover, or the reverse.
            fell_in = status.mode.value
            if fell_in == "stand":
                fell_in = "stand(after)" if requested else "stand(before)"
            break
        if requested and seq.mode is GaitMode.STAND and status.walk_finished:
            total = t + hold_s
    del stages

    x, y = robot.travel
    return {
        "fell": fell_in, "peak_tilt": peak_tilt, "min_height": min_height,
        "end_tilt": robot.tilt_deg, "end_height": robot.height,
        "x": x, "y": y, "exit": seq.last_exit, "trace": rows,
    }


def summarise(label, result):
    verdict = f"FELL during {result['fell']}" if result["fell"] else "stayed up"
    print(f"{label:<22}{verdict:<22}peak tilt {result['peak_tilt']:5.1f} deg  "
          f"min h {result['min_height']:.3f}  end tilt {result['end_tilt']:5.1f} "
          f"deg  end h {result['end_height']:.3f}  x{result['x']:+.3f} "
          f"y{result['y']:+.3f}")
    return result["fell"] is None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", default="/home/run/Project/robot_mujoco/robot/robot.xml")
    parser.add_argument("--stand-bundle", default=os.path.join(MODELS, "policy_bundle.npz"))
    parser.add_argument("--walk-bundle", default=os.path.join(MODELS, "walk_bundle.npz"))
    parser.add_argument("--crouch-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=0.75)
    parser.add_argument("--walk-s", type=float, default=5.0)
    parser.add_argument("--recover-s", type=float, default=2.0)
    parser.add_argument("--cmd-vx", type=float, default=0.085)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--stop-phase", type=float, default=None)
    parser.add_argument("--crouch-style", default="ramp",
                        choices=("ramp", "blend_stand", "blend_walk"))
    parser.add_argument("--recover-style", default="blend",
                        choices=("blend", "freeze", "via_crouch"))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", choices=("none", "stop-phase", "recover",
                                            "crouch", "walk-len", "robustness",
                                            "crouch-style", "recover-style"),
                        default="none")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(args.xml)))
    stand = Policy(args.stand_bundle)
    walk = Policy(args.walk_bundle)
    print(f"stand {stand.describe().splitlines()[0]}")
    print(f"walk  {walk.describe().splitlines()[0]}")

    robot = SimRobot(args.xml)
    gait = walk.gait
    if np.max(np.abs(walk.default_pose - robot.nominal)) > 1e-6:
        raise SystemExit("the bundle's nominal pose disagrees with the env's")

    def config(**over):
        cfg = GaitConfig(
            control_dt=walk.control_dt,
            crouch_s=args.crouch_s, settle_s=args.settle_s,
            walk_duration_s=args.walk_s, recover_s=args.recover_s,
            phase_increment=float(gait["phase_increment_per_cycle"]),
            cmd_vx=args.cmd_vx, cmd_wz=args.cmd_wz,
            cmd_vx_range=tuple(gait["cmd_vx_range"]),
            cmd_wz_range=tuple(gait["cmd_wz_range"]),
            stop_phase=args.stop_phase,
            crouch_style=args.crouch_style,
            recover_style=args.recover_style,
        )
        for key, value in over.items():
            setattr(cfg, key, value)
        cfg.validate()
        return cfg

    if args.sweep == "none":
        print()
        result = run_once(robot, stand, walk, config(), trace=True)
        ok = summarise("full sequence", result)
        print("\n  mode timeline (tilt deg, height m):")
        last = None
        for t, mode, phase, tilt, height, authority in result["trace"]:
            if mode != last:
                print(f"    {t:6.2f}s  -> {mode:<10} tilt {tilt:5.1f}  "
                      f"h {height:.3f}  walk authority {authority:.2f}")
                last = mode
        return 0 if ok else 1

    print()
    if args.sweep == "stop-phase":
        # Vary how long the walk runs, in fractions of a gait cycle. Stopping
        # is triggered by the duration timer, so this samples the phase the
        # handback begins at without needing the sequencer to wait for one.
        period = walk.control_dt / float(gait["phase_increment_per_cycle"])
        print(f"stop phase sweep -- gait period {period:.2f}s, "
              f"walk length varied over one cycle")
        survived = 0
        for k in range(12):
            extra = k * period / 12.0
            length = args.walk_s + extra
            phase = (length / period) % 1.0
            result = run_once(robot, stand, walk, config(walk_duration_s=length))
            survived += summarise(f"stop at phase {phase:.2f}", result)
        print(f"\n{survived}/12 survived")
    elif args.sweep == "recover":
        survived = 0
        values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        for value in values:
            result = run_once(robot, stand, walk, config(recover_s=value))
            survived += summarise(f"recover_s {value:.1f}", result)
        print(f"\n{survived}/{len(values)} survived")
    elif args.sweep == "crouch":
        survived = 0
        values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        for value in values:
            result = run_once(robot, stand, walk, config(crouch_s=value))
            survived += summarise(f"crouch_s {value:.1f}", result)
        print(f"\n{survived}/{len(values)} survived")
    elif args.sweep == "robustness":
        # The question hardware actually asks. Everything else here varies a
        # timing the node controls; this varies the one thing it does not --
        # how accurately the robot was stood up and calibrated before arming.
        print(f"survival of the full sequence over {args.trials} trials, by how "
              f"well the start pose is known\n")
        for tilt, joint in ((0.0, 0.0), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0)):
            rng = np.random.default_rng(args.seed)
            up, where = 0, {}
            for _ in range(args.trials):
                result = run_once(robot, stand, walk, config(),
                                  tilt_deg=tilt, joint_deg=joint, rng=rng)
                if result["fell"] is None:
                    up += 1
                else:
                    where[result["fell"]] = where.get(result["fell"], 0) + 1
            detail = ", ".join(f"{n} in {m}" for m, n in sorted(where.items()))
            print(f"  tilt sd {tilt:.1f} deg, joints sd {joint:.2f} deg   "
                  f"{up:3d}/{args.trials} survived   {detail}")
    elif args.sweep in ("crouch-style", "recover-style"):
        key = args.sweep.replace("-", "_")
        values = (("ramp", "blend_stand", "blend_walk") if key == "crouch_style"
                  else ("blend", "freeze", "via_crouch"))
        survived = 0
        for value in values:
            result = run_once(robot, stand, walk, config(**{key: value}))
            survived += summarise(value, result)
        print(f"\n{survived}/{len(values)} survived")
    elif args.sweep == "walk-len":
        survived = 0
        values = [2.0, 5.0, 8.0, 10.0, 15.0, 20.0]
        for value in values:
            result = run_once(robot, stand, walk, config(walk_duration_s=value))
            survived += summarise(f"walk {value:.0f}s", result)
        print(f"\n{survived}/{len(values)} survived")
    return 0


if __name__ == "__main__":
    sys.exit(main())
