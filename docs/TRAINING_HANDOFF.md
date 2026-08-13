# Walking policy — what deployment measured, and what it needs next

**For:** whoever runs the next training campaign in `~/Project/robot_mujoco`.
**From:** the ROS 2 deployment side (`~/Project/ROS2_Humble`).
**Date:** 2026-08-10. Covers every checkpoint through campaign 6.

The deployment node is finished and its parameters are exhausted. What blocks a
hardware walk is now entirely on the training side, and this is the evidence.

---

## 1. The one result that matters

Six checkpoints, each exported to a deployment bundle and verified against brax
to <4e-06, then run through **the actual deployment sequence** (stand → crouch
→ hand over → walk 5 s → hand back → stand) in MuJoCo with the measured
actuator model:

| checkpoint | cold-start speed | step advance | full sequence, exact start | at tilt σ1.0° / joints σ0.5° | fails during |
|---|---|---|---|---|---|
| `camp_D3` | −0.002 m/s | −3.7 mm | — | — | never advances |
| `camp_E1` | **0.035 m/s** | **+31.4 mm** | 0/20 | 0/20 | `walk` |
| `camp_F1` | 0.029 m/s | +25.2 mm | 0/20 | 0/20 | `walk` |
| `camp_F2` | 0.022 m/s | +18.7 mm | 0/20 | 4/20 | `to_stand` |
| `camp_G1` | 0.007 m/s | +4.6 mm | **20/20** | **12/20** | `to_stand` |
| `camp_G2` | topples | — | 0/20 | 0/20 | `walk` |

**Step length and handover survival trade against each other, monotonically,
across the whole set.** Every checkpoint that walks properly cannot be handed
the robot from a standstill; the one that can be handed the robot barely
advances. No existing checkpoint gives both, and that is the gap.

`camp_G1` is the only deployable-shaped result and its gait is not worth
deploying: 4.6 mm per footfall, and it drifts 0.45 m sideways per 0.21 m
forward.

---

## 2. Why `camp_E1` cannot be handed over — the specific mechanism

Deployment cannot drop the robot into a pose. It walks it from the standing
pose into the crouched nominal pose under the standing policy, lets it settle,
then hands over. That is the only handover that exists.

Survival of `camp_E1` against **initial pitch**, cold-started at the nominal
pose, 0.25° steps:

```
   -4.0 deg                                +4.0 deg
   XX....................XXX.......X
   FALL [-4.00,-3.75]
   UP   [-3.50,+1.25]     <- a 4.75 deg wide safe band
   FALL [+1.50,+2.00]     <- a 0.5 deg wide hole inside it
   UP   [+2.25,+3.75]
   FALL [+4.00,+4.00]
```

**The crouch settles at +1.53°** — deterministically, at 1 s, 2 s and 4 s ramp
speeds alike — which lands in the hole. A half-degree failure pocket interior to
an otherwise-working band is a memorised trajectory, not a basin of attraction.

Campaign 6's `--init-static-frac 0.4` removed that pocket, which was the right
diagnosis and the right fix. It also cost 85% of the step length.

---

## 3. The process finding

**Training `ep_len` is not measuring the deployment condition.** `camp_G1`'s
log reports `forward_vel=0.0454`; its cold-start forward speed is 0.007 m/s — a
factor of six apart. Randomised resets let a policy coast on initial velocity it
will never have on the robot, because deployment always hands over from rest.

This has now hidden a gait regression twice, independently:

| | training `ep_len` | cold-start reality |
|---|---|---|
| `camp_E2` | 67/350 (visibly bad) | 0% survival |
| `camp_G2` | 134.8/350 (looks fine, near G1's 139.4) | **0% survival, swing air time 0.000 s** |

`camp_G2` looked like a mild regression in the log and is in fact completely
broken — it has stopped lifting its feet. Both hardening attempts (E2 and G2)
regressed the gait while the training metric moved only modestly.

**Ask:** run `vis_walk_free.py --cold-start` inside the training loop, or at
minimum on every checkpoint before calling it an improvement. Everything else
below is worth less than this.

---

## 4. What to change, in priority order

### 4.1 `--zero-cmd-prob 0.15`, warm-started from `camp_G1`

This is the remaining deployment-side gap, and the env already anticipates it.
The comment on `_sample_command` reads:

> *"A fraction of commands are exactly zero. Without them the policy never
> learns to stop, and 'stand still' is the state the deployment hands over to
> the standing policy from."*

It was never enabled — `zero_cmd_prob` defaults to 0 and no campaign raised it.

Every `camp_G1` failure is at the **walk → stand seam**, where the standing
policy is handed a robot in a staggered stance with forward momentum, which it
never saw in training. `stand(before)` is empty across 80 trials, so the
standing policy itself is sound; it is the state it inherits that kills it.

If the walking policy can be commanded to a standstill, the handback becomes:
ramp `cmd_vx` to zero → let it settle with its feet together → *then* blend to
the standing policy. That removes the seam instead of tuning across it. The
deployment node will gain a `SLOWING` mode to match; that is my side and it is
a small change to `gait.py`.

### 4.2 Recover the step length `camp_G1` lost

Warm-start from G1 and push step advance back up, at whatever survival cost.
Deployment can trade — what it cannot do is choose between two checkpoints that
each fail a different half of the test.

Target: **+20 mm per footfall or better while keeping the full-sequence gate
in §5**. `camp_F1` at +25.2 mm shows the gait is reachable; it just has not been
reached from a static start.

### 4.3 Fix `cmd_vx_range` in the sidecar — already done, noted for completeness

`train_walk_free.py` wrote `[0.0, args.cmd_vx_max]`, dropping `cmd_vx_min`, so
every pre-campaign-6 sidecar claims a zero forward command was trained when it
was not. Campaign 6's sidecars carry `zero_cmd_prob` and are correct. The
exporter keys off that field and patches only the older ones.

---

## 5. The acceptance gate

Two commands. A checkpoint that passes both is deployable; one that fails
either is not, regardless of what training reported.

```bash
# 1. does it still walk, from the condition deployment actually produces?
cd ~/Project/robot_mujoco
/home/run/env_mujoco/bin/python vis_walk_free.py -c camp_XX.pkl \
    --headless 750 --episodes 8 --cmd-vx 0.085 --cmd-wz 0 \
    --push-force 0 --match-solver --cold-start
#    require: 100% survival, step advance >= +20 mm, |lateral| < forward

# 2. does it survive the handover, both ways?
cd ~/Project/ROS2_Humble
/home/run/env_mujoco/bin/python src/humanoid_deploy/tools/export_policy.py \
    --params ~/Project/robot_mujoco/camp_XX.pkl \
    --xml ~/Project/robot_mujoco/robot/robot.xml \
    --sidecar ~/Project/robot_mujoco/camp_XX.pkl.json \
    --out src/humanoid_deploy/models/walk_bundle_XX.npz

PYTHONPATH=~/Project/robot_mujoco /home/run/env_mujoco/bin/python \
    tools/sim_handover.py --sweep robustness --trials 30 \
    --walk-bundle src/humanoid_deploy/models/walk_bundle_XX.npz
#    require: >= 27/30 at tilt sd 1.0 / joints sd 0.5
#             (the sd 2.0 row is the honest hardware estimate; report it)
```

`sim_handover.py` is not a model of the deployment — it imports the same
`GaitSequencer` the ROS node sequences with and the same numpy `Policy` it runs
inference with, at 25 Hz through the measured actuator model. What is stubbed
is the robot, not the software.

The thresholds are why they are: 30 trials at σ1.0° because the calibration
locates each joint to about 1° and one encoder count is 0.088°; +20 mm because
below that the robot is marching in place and there is nothing to deploy.

---

## 5a. Hardware, re-measured 2026-08-13 (bus, actuators, new knee)

The bus was re-bauded to 500000, the right knee servo replaced and recalibrated,
and the actuator model measured properly for the first time. Three results
change what training should assume.

### The control loop is 4.7x faster

`tcdrain()` costs a flat ~11.95 ms per transmit on this Tegra UART -- measured
identical at every packet size from 8 to 47 bytes, against at most 0.94 ms of
airtime. Paid twice a cycle, it *was* the 24 ms control cycle. It is not needed:
the bus is half-duplex and already waits for its own transmit to echo back,
which is a stronger barrier. Removed.

| | cycle p50 | p99 | read success | ceiling |
|---|---|---|---|---|
| before | 24.00 ms | 31.71 ms | 100% | 31.5 Hz |
| after | **5.07 ms** | **6.58 ms** | 100% | **152 Hz** |

Baud had nothing to do with it -- 250000 and 500000 measured identically at
24 ms. **50 Hz is now comfortable** (3x headroom on p99). 100 Hz is reachable
but buys little: dead time is unchanged at ~80 ms, so total loop lag only goes
120 ms -> 100 ms -> 90 ms, while training cost scales with `frame_skip`.

### Dead time: ~80 ms, unchanged, and lower under load

Seven joints, `servo_trace.py` at acc=100: **72-92 ms free, 72 ms loaded**.
Identical to the 84 ms measured at 250000 baud, confirming transport delay is
the servo's internal loop and not the bus. At 25 Hz that is 2 cycles; at 50 Hz
it is 4. `_LIMITS_MEASURED_AT_HZ` already converts correctly.

### The roll joints are far weaker under load than the guess assumed

This is the important one. Measured loaded, 12 deg steps, robot standing:

| joint | loaded a_max |
|---|---|
| right_knee_pitch (11) | 6.7 rad/s^2 |
| left_knee_pitch (3) | 4.0 rad/s^2 |
| **right_hip_roll (9)** | **1.8 rad/s^2** |
| **left_hip_roll (5)** | **stalled at 86%, residual 1.67 deg** |

**Hip roll under load is 3-4x weaker than knee under load, symmetrically on
both sides.** `UNMEASURED_LIMITS` models the rolls at `a_max` 11.0 against the
knees' 17.0 -- a ratio of 1:1.5, where the measurement says 1:3.7.

Use the *ratio*, not the absolute number: step response and the trajectory fit
that produced the existing table disagree by a consistent factor, but that bias
cancels between joints measured the same way. Scaling off the knee's 17.0:

```python
UNMEASURED_LIMITS = (1.35, 4.6, 2)      # was (1.35, 11.0, 2)
```

This is the single most likely explanation for the lateral drift that never
yielded: the joints that reject a lateral disturbance were modelled 2.4x
stronger than they are, so every policy trained so far learned to lean on
authority the robot does not have.

### Free-hanging numbers, all seven leg joints, acc=100

Small steps bias `a_max` low -- `t90` includes the ~80 ms dead time, which is a
large fraction of a short move. The 60 deg steps are the trustworthy ones:

| joint | a_max @20 deg | a_max @60 deg | peak @60 deg |
|---|---|---|---|
| left_hip_pitch (6) | 5.7 | **7.3** | 154 deg/s |
| right_knee_pitch (11) | 6.8 | **6.7** | 146 deg/s |
| right_hip_roll (9) | 2.9 | **6.9** | 156 deg/s |
| left_hip_roll (5) | 6.3 | — | — |
| left_knee_pitch (3) | 6.1 | — | — |
| left_ankle_roll (2) | 6.1 | — | — |
| right_ankle_roll (12) | 4.7 | — | — |

Free `a_max` is ~7 rad/s^2 and remarkably uniform across joint types.

### v_max has never actually been observed

Every step, including 60 deg, was still accelerating on arrival -- measured peak
/ acceleration-implied peak was 0.96-1.01, never below 0.75. So **v_max >= 2.7
rad/s** on hip pitch, knee pitch and hip roll, and the table's 1.35-2.45 values
are all *below* a speed the joints were observed passing through without
plateauing. That column is too low and is a fit artifact, not a measurement.

Note `servo_goal_speed: 2000` (176 deg/s) is close to the measured 156 deg/s
free peak; `walk.yaml` already raises it to 3000.

---

## 6. Deployment-side facts worth having

- **Control law**, unchanged and correct in the bundle:
  `ctrl = clip(nominal_pose_rad + action * 0.4, joint_limits)`, residual about
  the **crouched** pose. (The env docstring still says "the 0-rad standing
  pose"; the code and the bundle both use `NOMINAL_POSE_DEG`.)
- **Observation** 53 = the standing policy's 48 + `phase_sin, phase_cos,
  cmd_vx, cmd_vy, cmd_wz`. One builder feeds both policies; the node evaluates
  both every cycle so whichever is idle stays warm.
- **Handover timings**, measured, not guessed: crouch ramp 2.0 s, settle
  0.75 s, handback `blend` over 2.0 s. Alternatives were tested and are worse —
  keeping the standing policy live during the descent falls over (it saturates
  fighting a descent it cannot see the reason for), and settling the stride into
  the crouch before standing up (`via_crouch`) scores 3/20 against `blend`'s
  12/20.
- **`stop_phase` buys nothing**, retested after G1 made it testable: 11/16 with
  no phase preference against 6–11/16 for every specific phase. Consistent with
  56% schedule agreement against a 50% chance baseline — the clock barely
  predicts which foot is down.
- **What the robot can run today:** `crouch_only` mode, which runs the whole
  sequence with the walking policy given zero authority (20/20, 20/20, 19/20;
  peak tilt 6–7°). It validates the bus, sequencing, handback and Ctrl-C on
  hardware without trusting any gait.
- **Hardware measurements still owed to you:** loaded hip-roll and ankle-roll
  actuator limits still carry the guessed `(1.35 rad/s, 11.0 rad/s²)`, and this
  gait's worst defect is lateral. Also the robot's settled tilt in the crouch —
  hardware's version of the simulated +1.53°.
