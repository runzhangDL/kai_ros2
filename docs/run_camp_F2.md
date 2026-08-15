# Running `camp_F2.pkl.best` on the robot

**Read this first.** This checkpoint does **not** walk. Everything below is a
full-pipeline rehearsal — arm, crouch, hand the robot to the walking policy,
take it back, stand up — with a policy that is stable but stationary. It is
worth running because it exercises every part of the deployment that a real
walking policy will also need, on hardware, at the real bus rate, with the real
handback. It is not worth running if what you want to see is a step.

## What the checkpoint actually does

Measured in `tools/sim_handover.py` at the robot's own stiffness (kp_scale 0.90
= kp 44.9 N·m/rad, from `tools/measure_kp.py`):

| | |
|---|---|
| full sequence, known start | **30/30** survived |
| start perturbed 0.5° tilt / 0.25° joints | 28/30 |
| start perturbed 1.0° tilt / 0.50° joints | 25/30 |
| start perturbed 2.0° tilt / 1.00° joints | 27/30 |
| peak tilt, nominal run | 14.7° (abort is 22°, fall cut is 40°) |
| end state | upright, 2.6° tilt, height 0.520 m |
| **net travel over a 20 s walk phase** | **+0.046 m, all of it in the first 1.5 s** |

That last row is the one that matters. Traced second by second, x sits at
+0.058 m and oscillates ±0.002 m from t=6 s to t=24 s. It lurches once and then
rocks in place. Airborne fraction is 0.006–0.035 against the 0.15 that the
training script itself sets as the deploy threshold.

Note the failures at perturbed starts are all *inside the walk phase*, and they
appear only at the real stiffness — the earlier 30/30/30/29 result was taken at
kp_scale 0.20, a stiffness this robot turned out not to have.

## Before you start

Five files changed, not just the bundle. From the dev machine:

```bash
cd /home/run/Project/ROS2_Humble
ROBOT=<user>@<robot-host>          # e.g. kai@192.168.1.50

rsync -av \
  src/humanoid_deploy/models/walk_bundle_F2b.npz \
  $ROBOT:~/kai_ros2-main/src/humanoid_deploy/models/

rsync -av \
  src/humanoid_deploy/humanoid_deploy/gait.py \
  src/humanoid_deploy/humanoid_deploy/walk_node.py \
  $ROBOT:~/kai_ros2-main/src/humanoid_deploy/humanoid_deploy/

rsync -av src/humanoid_deploy/config/walk.yaml \
  $ROBOT:~/kai_ros2-main/src/humanoid_deploy/config/

rsync -av src/humanoid_deploy/launch/walk.launch.py \
  $ROBOT:~/kai_ros2-main/src/humanoid_deploy/launch/
```

`gait.py` and `walk_node.py` carry the `via_crouch` handback, `walk.yaml` the
new defaults, and `walk.launch.py` the `crouch_only` argument used in run 2.
Skipping any of them leaves the robot running the `blend` handback, which is
the one that fails.

Then on the robot, rebuild so the new config defaults and the bundle are
installed:

```bash
cd ~/kai_ros2-main
colcon build --packages-select humanoid_deploy --symlink-install
source install/setup.bash
```

The config now ships the configuration that was validated:
`crouch_style: ramp`, `recover_style: via_crouch`, `recover_s: 2.0`,
`crouch_only: true`. You do not need to pass any of them.

## Terminal 1 — the IMU, on its own

```bash
ros2 launch humanoid_deploy imu.launch.py
```

Keep this in its own terminal. A Ctrl-C in terminal 2 goes to the whole process
group, and a handback flying on a frozen gravity vector is not a handback.

## Terminal 2 — dry run first

Nothing is written to any servo. Confirm both policies load and the rate is
25 Hz.

```bash
ros2 launch humanoid_deploy walk.launch.py \
    bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/policy_bundle.npz \
    walk_bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/walk_bundle_F2b.npz
```

Expect in the log:

```
standing obs=48 nu=13 action_scale=0.4 trained at 25.0 Hz
walking  obs=53 nu=13 action_scale=0.4 trained at 25.0 Hz
```

Ctrl-C twice to exit the dry run.

## Run 1 — crouch only (the one to do first)

`crouch_only` is already true in the config, so this is the same command with
`dry_run:=false`. The walking policy is evaluated and published every cycle but
given **zero authority**: the robot squats, holds, and stands back up.

```bash
ros2 launch humanoid_deploy walk.launch.py \
    bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/policy_bundle.npz \
    walk_bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/walk_bundle_F2b.npz \
    dry_run:=false
```

**How to hold the robot.** Stand behind it. Support it at the torso, under the
arms, with enough grip to take its full weight but loose enough that it can lean
a few degrees on its own — you are a backstop, not a fixture. The last crouch
experiment leaned backward and nearly went over on the way *up*, so keep your
hands where they can catch a backward fall for the whole run, especially the
last 2 seconds.

Terminal 3, once you are holding it:

```bash
ros2 service call /humanoid_servo/arm  std_srvs/srv/Trigger    # it stands

# CONFIRM it armed before going on -- expect state=running
ros2 topic echo /humanoid_servo/status --once

ros2 service call /humanoid_policy/walk std_srvs/srv/Trigger   # sequence starts
```

The confirmation is not optional. On 2026-08-15 a run went through the whole
sequence twice -- crouch, settle, walk at authority 1.00, handback -- with the
servos disarmed the entire time, and every log line looked correct. `~/walk`
now refuses when the state is not `running`, but check anyway.

If `arm` is refused with a `qpos_N = ... sigma` message, the robot is resting
in a pose the standing policy never saw in training. Stand it upright with the
legs straight and call it again -- this happened on the first attempt with the
right knee at 15.6 deg, and cleared on the second.

Timeline, about 10 seconds end to end:

| t | what |
|---|---|
| 0.0 s | crouch begins, open loop over 2 s |
| 2.0 s | settle, 0.75 s at rest |
| 2.75 s | walking policy takes over, zero authority — nothing visible changes |
| 7.75 s | `via_crouch` handback: settles to the static crouch, then stands |
| 9.75 s | standing policy has it back |

Ctrl-C once at any point does the same thing early and the robot stays up.
Ctrl-C **twice** exits and drops torque — hold the robot before the second one.

**What to watch for.** It should stay within a few degrees of vertical the whole
way. If it leans backward more than about 10°, or if the stand-up at the end is
the part that struggles, that is the same CoM error the ankle trim was fixing —
say so and we adjust `ankle_pitch_trim_deg` before going further.

## Run 2 — walk phase live

Only after run 1 is clean, and only with the robot on the gantry or with a
second person's hands on it.

```bash
ros2 launch humanoid_deploy walk.launch.py \
    bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/policy_bundle.npz \
    walk_bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/walk_bundle_F2b.npz \
    dry_run:=false \
    crouch_only:=false \
    walk_duration_s:=3.0
```

3 seconds, not 5, for the first one: the sim says distance stops accumulating
after 1.5 s anyway, so a longer run buys no information and only adds exposure.

**Expect the robot to sway in place and not travel.** That is the correct
outcome for this checkpoint, not a failure of the deployment. What you are
testing is that the handover happens cleanly, the servos track a live policy at
25 Hz, and the handback puts it back on its feet.

## Please record it

```bash
ros2 bag record -o walk_F2_run1 /humanoid/joint_states \
    /humanoid_policy/joint_command /humanoid_policy/gait \
    /humanoid_servo/status /humanoid_servo/command /imu/data
```

`/humanoid_policy/gait` is the one to read afterwards: it carries the mode and
the tilt every cycle, so it says exactly where in the sequence anything went
wrong.

Tar it and drop it in the project directory as before — `tools/read_bag.py`
reads it without a ROS install.

## Abort conditions

| you see | what happens | what you do |
|---|---|---|
| tilt past 22° | node abandons the walk, standing policy takes over | let it try; hands ready |
| tilt past 40° | `servo_node` cuts torque — it goes limp | catch it |
| a joint buzzing or stalled | stall guard trips at 25° error for 1.5 s | Ctrl-C once |
| anything you do not like | — | Ctrl-C once, then hold and Ctrl-C again |

## After this

This rehearsal does not produce a walking robot; it clears the deployment path
so that the next checkpoint that *does* walk can be dropped in as a single
`walk_bundle:=` argument. The open training question is unchanged and is
written up in `docs/slow_gait_experiment.md` — `left_airborne` peaks near 0.3
and decays to ~0.05 in every campaign, and the split-ablation of
`UNMEASURED_LIMITS` (four roll joints vs all seven) has still not been run.
