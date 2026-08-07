# humanoid_deploy

Runs the trained MJX/brax standing policy on the real robot. Two nodes:
`policy_node` computes, `servo_node` owns the motors and every safety decision.

**Nothing here runs on an uncalibrated robot** — `humanoid_calibration`'s guard
fails closed, because encoder counts only mean radians via the calibration.

---

## What I verified, and what I could not

Verified on this machine, against the real `stand_params.pkl`:

| Check | Result |
|---|---|
| numpy inference vs brax's own `make_inference_fn` | max abs diff **4.1e-06** over 1024 random observations |
| Network semantics | swish, no layer norm, `normalize=(obs-mean)/std` (no clipping), action = **`tanh(logits[:13])`** |
| IMU chain at rest | `accel=(0,0,9.807) m/s²`, `gravity=(0,0,-1)`, `tilt=0.000°` |
| Omitting the IMU rotation | caught at **49σ** by the arm gate |
| Closed loop, 200 cycles | every command inside the envelope, every count in 0..4095 |
| Joint-order extraction from your real `robot.xml` | left leg is **reversed** vs the calibration order — `action[0]` drives servo **6**, not 1 |
| Unit tests | 68 passing |

**Not verified — needs hardware:** actual bus timing, real IMU behaviour, and
whether the policy balances. Run `bus_benchmark` and a `dry_run` session first.

## Export the policy (on the training machine)

The robot loads a `.npz`, not the `.pkl` — no jax on the Jetson.

```bash
python3 tools/export_policy.py \
    --params models/stand_params.pkl \
    --xml robot/robot.xml \
    --out models/policy_bundle.npz
```

**`robot.xml` is required.** The bundle carries the weights *and* the MuJoCo
**joint order**, the XML joint limits, the action scale and the control period —
all read from the same model that produced the policy. Re-typing the joint order
into a YAML file would be the single most dangerous thing in this system: get it
wrong and every command goes to the wrong motor. The exporter also refuses to
write if the numpy forward pass disagrees with brax.

## The six conversions

| # | Concern | What happens |
|---|---|---|
| 1 | IMU axes | `R = [[0,0,-1],[-1,0,0],[0,1,0]]` — your matrix, verified: det=+1, and it maps a resting IMU to `gravity=(0,0,-1)`. A determinant of −1 is rejected, because a reflection would silently mirror left and right. |
| 2 | Gyro units | **Already rad/s.** `wit_ros2_imu` calls `np.radians` on it. Configurable anyway. |
| 2b | **Accel units** | **Your driver outputs g, not m/s².** It scales by `16/32768`. MuJoCo's `accel_z` has training mean **+7.02**, so m/s² is required. `accel_units: auto` measures a resting sample and decides. Feeding g shifts commands by up to **10.5°** — measured. |
| 3 | Joint units | Counts → radians via calibration, shortest-way-round so a zero near the seam still reads continuously. |
| 4 | Normalization | `mean`/`std` come straight out of the checkpoint (EMA mode) into the bundle. |
| 5 | Action | Raw output is 26 numbers = `(loc, scale)`. Deterministic action is `tanh(loc[:13])`, then `clip(0 + a*0.4, limits)`. Using the raw output directly would be wrong. |
| 6 | Control rate | See below. |

## Control rate: 125 Hz is not reachable at 250000 baud

A cycle costs one sync read of 13 positions (~6–8 ms of airtime) plus one sync
write (~2 ms). 125 Hz needs the whole cycle in 8 ms. Measure yours:

```bash
ros2 run humanoid_deploy bus_benchmark
```

It reports p50/p99 cycle time and a recommended rate (~70% of the ceiling). The
default is **50 Hz**. Running slower than training degrades push recovery — the
policy's notion of a timestep no longer matches reality, and `servo_node` warns
about it at startup. Your options, best first:

1. **Retrain** with `frame_skip` set so the simulated rate matches what the bus
   achieves (`frame_skip=10` → 50 Hz).
2. Raise the baud rate and re-qualify with `sts_tool.py bustest` — but you
   chose 250000 for reliability, and a dropped read is worse than a slow loop.
3. Accept 50 Hz. Likely fine for standing (the servo's own loop is much faster
   and the policy commands positions, not torques); worse under shove.

## Safety

Layered, each independent. In order of how early they catch a mistake:

1. **Calibration gate** — launch aborts on an uncalibrated robot.
2. **Encoder-seam check** — refuses to start if any joint's reachable command
   window crosses 0/4095. This is the failure you asked about: commanding
   across the wrap makes the servo take the long way round, nearly a full
   revolution at speed. There is no safe way to issue that command, so it is a
   hard stop rather than a clamp.
3. **Arm preflight** — must be held upright, all servos reading, every joint
   inside its envelope, and *the observation must look statistically like
   training data*. That last one is the cheap detector for a swapped axis, an
   accelerometer in g, or a gyro in deg/s: `grav_z` has a training σ of 0.02,
   so pointing it wrong lands ~50σ out. Verified — omitting the rotation is
   caught at 49σ.
4. **Ramp** — blends from the pose the robot is *actually* in toward the policy
   target over 3 s, so arming never snaps a joint.
5. **Envelope clamp** — intersection of calibrated travel and XML limits, minus
   a margin. Applied to every command, twice.
6. **Rate limit** — caps per-step change, turning a bad action into a slow move.
7. **Fall detection** — sustained tilt or free-fall → **FAULT: torque off,
   latched**. Terminal by design; automatic recovery is how a robot destroys
   itself in a stand-up/fall-over loop. `servo_node` watches the IMU itself, so
   a hung policy node cannot suppress it.
8. **Watchdog / bus / thermal** — stale command, lost servos, over-temperature
   or under-voltage all latch FAULT.
9. **EEPROM lock** — every write is checked against the RAM block (40..54). The
   deployment path physically cannot re-id a servo, change its baud, or touch
   its angle limits.

`humanoid_calibration.feetech_bus` stays read-only; this package imports only
its pure packet helpers, so that audit property is preserved.

## Bring-up, in order

```bash
colcon build && source install/setup.bash

# 1. measure the bus (torque off; the write is a no-op)
ros2 run humanoid_deploy bus_benchmark

# 2. DRY RUN -- full loop, zero bytes written to any servo
ros2 launch humanoid_deploy stand.launch.py \
    bundle:=$(ros2 pkg prefix humanoid_deploy)/share/humanoid_deploy/models/policy_bundle.npz

#    check: joint map table, tilt ~0 when upright, no OOD warnings
ros2 topic echo /humanoid_servo/status
ros2 topic echo /humanoid_policy/observation --once

# 3. arm in dry run -- proves the preflight passes without moving anything
ros2 service call /humanoid_servo/arm std_srvs/srv/Trigger

# 4. LIVE, robot on a gantry or held, hand on the power switch
ros2 launch humanoid_deploy stand.launch.py bundle:=... dry_run:=false
ros2 service call /humanoid_servo/arm std_srvs/srv/Trigger

# stop
ros2 service call /humanoid_servo/estop std_srvs/srv/Trigger
```

Release torque before handling the robot:
`python3 sts_tool.py --baud 250000 torque <id> 0`

## Topics

| | |
|---|---|
| `/humanoid/joint_states` | measured, rad, named (from `servo_node`) |
| `/humanoid_policy/joint_command` | targets, rad, named — a *request* |
| `/humanoid_policy/observation` | the exact 48-vector fed to the net |
| `/humanoid_policy/action` | raw tanh action, [-1, 1] |
| `/humanoid_servo/status` | state, fault, tilt, max temperature |
| `/humanoid_servo/{arm,disarm,estop}` | `std_srvs/Trigger` |

## Things to know

**Two nodes cost one control period of latency** (sensor → policy → executor).
`policy_node` is driven by `/humanoid/joint_states` rather than its own timer to
keep it to one rather than two. At 50 Hz that is 20 ms. If that proves too slow,
the two can be merged into one process — the logic is already separated into
plain classes with no ROS in them.

**Four of your joints stand on a hard limit.** `left_knee_pitch [-90, 0]`,
`left_hip_roll [-90, 0]`, `right_hip_roll [0, 90]`, `right_knee_pitch [0, 90]`
all put the standing pose exactly at an endpoint — correct for a knee that
cannot hyper-extend. The safety margin is suppressed there rather than
excluding the nominal pose (which would make arming impossible), and
`servo_node` warns that those joints have no cushion on that side.

**The policy's standing pose is not zero.** At a perfect standing observation
its fixed point commands up to ~9.7° off zero. That is the posture it learned;
it is not a bug.

**`xml_lower/upper` come from `robot.xml`, not from `joint_limits.yaml`.** I
checked: on this robot they agree exactly. If they ever disagree the tighter
wins, and `JointMap` refuses to build if the nominal pose falls outside the
intersection.

**Joint names differ between the model and the calibration.** The XML suffixes
everything with `_joint`; `joint_limits.yaml` does not. `resolve_joint_names`
matches exactly first, then on the suffix-stripped name, verifies the result is
a bijection, and logs the resolved mapping at startup. Nothing is positional.
Override with `joint_name_aliases:=["model_name:calibration_name"]` if needed.

**Export verification and TF32.** `export_policy.py` forces
`jax_default_matmul_precision="highest"` and evaluates the brax reference on the
CPU. Without that, an Ampere-or-newer GPU computes float32 matmuls in TF32
(~10 mantissa bits) and the comparison drifts to ~5e-3 — which tests the GPU's
numerics, not this port. On CPU at full precision it is 5.0e-06.
