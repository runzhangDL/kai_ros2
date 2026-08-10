# Walking deployment — what was built, what it measures, and what is blocked

**Checkpoints tested**, all exported and brax-verified to <4e-06. The full
head-to-head and the recommendation to the training side are in
[`TRAINING_HANDOFF.md`](TRAINING_HANDOFF.md); this document is the deployment
record.

| checkpoint | bundle | cold-start step | full sequence @σ1.0° |
|---|---|---|---|
| `camp_D3` | `walk_bundle_D3.npz` | −3.7 mm | never advances |
| `camp_E1` = `walkfree_deploy.pkl` | `walk_bundle.npz` | **+31.4 mm** | 0/20 |
| `camp_F1` (duty 0.60) | `walk_bundle_F1.npz` | +25.2 mm | 0/20 |
| `camp_F2` (duty 0.70) | `walk_bundle_F2.npz` | +18.7 mm | 4/20 |
| `camp_G1` (static resets) | `walk_bundle_G1.npz` | +4.6 mm | **12/20** |
| `camp_G2` (G1 + hardening) | `walk_bundle_G2.npz` | topples | 0/20 |

Step length and handover survival trade against each other monotonically across
the whole set. No checkpoint gives both.

**Bottom line:** the deployment is built; the sequencing, the handback and
Ctrl-C all work. No checkpoint yet walks well enough to deploy. `camp_E1`
cannot be handed over at all (§4). `camp_G1` hands over cleanly but advances
4.6 mm per step instead of 31.4 (§4a). `camp_G2` is broken. Run the robot in
`crouch_only` mode — the config default — until a checkpoint clears both
`tools/sim_handover.py --sweep robustness` and a `--cold-start` speed check.

---

## 1. What was verified before anything else

The checkpoint reproduces the training author's numbers exactly:

```
8 episodes, cap 750 steps (30.0s), cold start
  survived full episode : 100%       forward speed : +0.035 m/s
  518 footfalls         : +31.4 mm/footfall
  schedule agreement    : 56.5%      [50% = chance]
```

The numpy inference port used on the robot agrees with brax to **3.576e-06**
over 1024 random observations, so nothing below is an export artifact.

Two documentation defects were found and corrected in the bundle:

- **`cmd_vx_range` in the sidecar is wrong.** `train_walk_free.py` writes
  `[0.0, args.cmd_vx_max]`, dropping `args.cmd_vx_min`. Every sidecar therefore
  claims a zero forward command was trained. It was not: `zero_cmd_prob`
  defaults to 0 and campaign 4 did not raise it, so the lowest command the
  policy has ever seen is 0.05 m/s. `DEPLOY_WALKFREE.md` §5 is right and the
  sidecar is wrong. **Commanding `vx = 0` is not a way to stop**, and the node
  refuses it.
- The env docstring still says the residual is about "the 0-rad standing pose".
  The code uses the crouched `NOMINAL_POSE_DEG`, and so does the bundle.

---

## 2. The architecture, and why Ctrl-C forced it

The requirement was: interrupting the walk hands the robot back to the standing
policy rather than dropping it. That single requirement determines the topology.

**One process owns the bus, both policies, and the interrupt.** A robot whose
control process has exited is a robot on the floor, so `walk_node.main()` spins
`ServoNode` and `WalkPolicyNode` in one single-threaded executor and installs
its own SIGINT handler. The first Ctrl-C asks the sequencer to stop and the
process keeps running until the robot is standing again; the second exits. A
bounded grace period covers the case where the bus dies and the sequencer stops
advancing.

**The IMU is deliberately not in `walk.launch.py` by default.** Ctrl-C goes to
every process in a launch group. The walking node survives it on purpose; the
IMU driver does not, and a handback flying on a frozen gravity vector is not a
handback. Run `imu.launch.py` in its own terminal.

**`servo_node` is unchanged and unaware there are two policies.** The walking
node publishes on the topics it already subscribes to, so every safety rule —
envelope clamp, rate limit, seam check, fall detection, stall guard, arming
preflight — applies to walking exactly as it did to standing.

The observation makes this cheap: `mjx_walk_free_env` appends 5 entries to the
standing policy's 48, so one builder feeds both. Both policies are evaluated
every cycle, each shown its own action history, so whichever is not driving is
warm for the moment it takes over.

### The sequence

```
STAND --/walk--> TO_CROUCH --> SETTLE --> WALK --> TO_STAND --+
  ^                                                           |
  +-----------------------------------------------------------+
```

Every exit — the duration timer, Ctrl-C, `~/stop`, or tilt over
`walk_abort_tilt_deg` — runs the same `TO_STAND` path. There is one way this
ends and it is the one that gets tested.

---

## 3. What the simulation decided, rather than what looked sensible

`tools/sim_handover.py` drives the *real* `GaitSequencer` and the *real* numpy
`Policy` against MuJoCo with the measured actuator model. Three design choices
came out opposite to intuition.

**The crouch entry must be open loop.** Keeping the standing policy live all
the way down (`crouch_style: blend_stand`) reads as the safest option and it
falls over. That policy saturates fighting a descent it has no way to
understand, and averaging a saturated controller with a pose target tracks
neither. The crouch is a statically stable symmetric squat — it does not need a
balance controller to reach, it needs to be reached slowly. Ramping open loop
settles at 1.5° of tilt from 1 s, 2 s or 4 s alike.

**The stop phase buys nothing.** Feet agree with the gait clock 56.5% of the
time against a 50% chance baseline, so the phase barely predicts which foot is
down. `stop_phase` is implemented and disabled.

**The handback keeps both policies live** so the walking policy finishes its
stride instead of parking a swing foot in the air. Once `camp_G1` made the walk
survive long enough to test this properly (it could not be tested on E1 — the
robot was down before the handback was reached), all three styles were measured
at tilt σ1.0°/joints σ0.5°:

| `recover_style` | `recover_s` 2.0 | 3.0 | 4.0 |
|---|---|---|---|
| `blend` | 12/20 | 13/20 | — |
| `freeze` | 12/20 | 11/20 | — |
| `via_crouch` | 3/20 | 2/20 | 1/20 |

`via_crouch` — settle the stride into the static crouch, then stand up out of
it, mirroring the entry — was the intuitive favourite and is by far the worst.
Forcing a walking robot into a static pose mid-stride is a disturbance, not a
stabiliser. `blend` at 2.0 s stays, and it is also the only setting that
survives the *unperturbed* case 20/20 (`recover_s` 3.0 fails it 30/30, which is
the same knife-edge chaos §4 describes, now on the exit instead of the entry).

`stop_phase` was retested here too, since the earlier verdict was reached
before any gait survived to the handback. It still buys nothing: 11/16 with no
preference against 6–11/16 for every specific phase.

---

## 4. Why the gait is blocked, exactly

Survival of the walking policy against **initial pitch**, cold-started at the
nominal pose, in 0.25° steps:

```
        -4.0 deg                                        +4.0 deg
        XX....................XXX.......X
        bands:  FALL [-4.00,-3.75]
                UP   [-3.50,+1.25]      <- a 4.75 deg wide safe band
                FALL [+1.50,+2.00]      <- a 0.5 deg wide hole inside it
                UP   [+2.25,+3.75]
                FALL [+4.00,+4.00]
```

**The crouch settles at +1.53°, in the hole.** Deterministically, at every ramp
speed. That is the whole failure: not a bad handover, not a bad sequencer — the
handover lands the robot on the one initial condition in the middle of the safe
band that does not work.

Perturbing the start pose gives the dose-response:

| start pose known to | survives 10 s |
|---|---|
| exactly | 100% (12/12 across all initial phases) |
| tilt σ 0.5°, joints σ 0.25° | 39/40 |
| tilt σ 1.0°, joints σ 0.50° | 33/40 |
| tilt σ 2.0°, joints σ 1.00° | 17/40 |

A failure pocket half a degree wide, sitting inside an otherwise-safe band, is
the signature of a memorised trajectory rather than a controller with a basin
of attraction. On hardware the initial lean cannot be known to better than a
degree or two — the calibration locates each joint to about 1°, one encoder
count is 0.088°, and the robot's actual settled lean in the crouch is a
property of the floor and the mass distribution, not of anything the node
controls. **The bottom row of that table is the honest estimate of hardware.**

### The fix belongs in training, not in the node

Trimming the crouch to land at −1° instead of +1.53° would pass this test and
would be tuning the deployment to hit a knife edge measured in simulation. The
lean on hardware will be a different number, and the pocket may be somewhere
else entirely.

What is needed is initial-state randomisation at deployment magnitude. It is
cheap — warm-start from `camp_E1`, change one thing:

```bash
train_walk_free.py --restore camp_E1.pkl --actuator-delay-scale 1.0 \
    --timesteps 60000000 --out camp_F1.pkl        # + reset tilt/joint noise
```

Note this is **not** what `camp_E2` did. E2 changed four things at once
(`--domain-rand --actuator-rand 0.25 --action-delay-steps 1 --push-force 6
--obs-noise-scale 1.0`) and collapsed to `ep_len 67/350` against E1's 173.
`--action-delay-steps 1` alone pushes total delay to 3 cycles, which the
earlier dose-response says never walks. Add reset noise **only**, and keep the
delay at 2.

Acceptance test, before touching the robot:

```bash
/home/run/env_mujoco/bin/python tools/sim_handover.py --sweep robustness
```

The `tilt sd 2.0 / joints sd 1.00` row should read comfortably above 0/30.

---

## 4a. Campaign 6 — the cold start is fixed, the gait is not

Campaign 6 ran the recommended experiment: G1 = `camp_E1` + `--init-static-frac
0.4` and nothing else; G2 = G1 + pushes and observation noise, added alone.
Both export and verify against brax (3.3e-06, 2.6e-06).

Full deployment sequence, `tools/sim_handover.py --sweep robustness`,
`recover_s 2.0`:

| | E1 | G1 | G2 |
|---|---|---|---|
| exact start | 0/20 | **20/20** | 0/20 |
| tilt σ 1.0°, joints σ 0.50° | 0/20 | 12/20 | 0/20 |
| where it fails | all in `walk` | all in `to_stand`/`stand` | all in `walk` |

**G1 fixed the thing it was meant to fix.** It never falls during the walk at
any perturbation level, and the `stand(before)` column is empty in 80 trials,
so the standing policy is not the problem. The cold-start failure pocket is
gone.

**But the gait it fixed is no longer a gait.** On the author's own harness,
cold-started, 8 × 30 s:

| | E1 | G1 | G2 |
|---|---|---|---|
| survived 30 s | 100% | 100% | **0%** |
| forward speed | 0.035 m/s | **0.007 m/s** | −0.367 m/s (topples) |
| step advance | +31.4 mm | **+4.6 mm** | — |
| forward / lateral | 1.07 / 0.53 m | **0.21 / 0.45 m** | — |
| footfalls | 518 | 535 | — |

G1 still steps — 535 footfalls — but advances 4.6 mm per step instead of 31.4,
and drifts sideways more than twice as far as it goes forward. It is marching
in place with a veer. G2 is broken outright: swing air time 0.035 s / 0.000 s
means it has stopped picking its feet up at all.

**The pattern to take seriously:** hardening has now regressed the gait twice,
independently — E2 (four changes at once) and G2 (two). Both times the reported
training `ep_len` moved only modestly while the cold-start behaviour collapsed.
That is the actionable process finding: **training `ep_len` is not measuring
the deployment condition.** G1's training log reports `forward_vel=0.0454`; its
cold-start forward speed is 0.007 m/s, a factor of six apart. Randomised starts
let a policy coast on initial velocity it will never have on the robot. Until a
`--cold-start` evaluation runs inside the training loop, campaign metrics will
keep disagreeing with deployment.

### What to ask for next

The two objectives are in tension and each campaign has bought one by selling
the other. A run has to be scored on both:

1. **Cold-start eval in the loop**, or at minimum `vis_walk_free.py
   --cold-start` on every checkpoint before it is called an improvement.
2. **`--zero-cmd-prob 0.15`, warm-started from G1.** This is the remaining
   deployment-side gap and the env author already anticipated it — the comment
   on `_sample_command` reads *"Without them the policy never learns to stop,
   and 'stand still' is the state the deployment hands over to the standing
   policy from."* It was never enabled. Every G1 failure is at the walk→stand
   seam, where the standing policy is handed a robot in a staggered stance with
   forward momentum, which it never saw in training. If the walking policy can
   be commanded to a standstill, the handback becomes: ramp `cmd_vx` to zero,
   let it settle with its feet together, *then* blend. That removes the seam
   rather than tuning across it, and `gait.py` would gain a `SLOWING` mode.
3. Recover the step length G1 lost, at whatever survival rate that costs, and
   let the deployment decide the trade.

---

## 5. Running it on the robot now — `crouch_only`

`crouch_only: true` (the config default) runs the entire deployment — crouch,
hold, handback, Ctrl-C, tilt abort — with the walking policy evaluated and
published but given **zero authority**. The robot squats and stands up again.
This exercises everything except the gait, which is the only part that is not
trustworthy. In simulation:

| | |
|---|---|
| nominal start | survives, peak tilt 6.25° |
| Ctrl-C after 2 s | survives, peak tilt 5.54° |
| tilt σ 0.5°, joints σ 0.25° | 20/20 |
| tilt σ 1.0°, joints σ 0.50° | 20/20 |
| tilt σ 2.0°, joints σ 1.00° | 19/20 |

### Procedure

Build and export first:

```bash
# training machine
cd ~/Project/robot_mujoco
/home/run/env_mujoco/bin/python \
    ~/Project/ROS2_Humble/src/humanoid_deploy/tools/export_policy.py \
    --params walkfree_deploy.pkl --xml robot/robot.xml \
    --sidecar walkfree_deploy.pkl.json \
    --out ~/Project/ROS2_Humble/src/humanoid_deploy/models/walk_bundle.npz

# robot
cd ~/Project/ROS2_Humble && colcon build --packages-select humanoid_deploy
source install/setup.bash
```

Then, **with the robot on a gantry or held**:

```bash
# terminal 1 -- leave running
ros2 launch humanoid_deploy imu.launch.py

# terminal 2 -- DRY RUN first, nothing is written to a servo
ros2 launch humanoid_deploy walk.launch.py \
    bundle:=install/humanoid_deploy/share/humanoid_deploy/models/policy_bundle.npz \
    walk_bundle:=install/humanoid_deploy/share/humanoid_deploy/models/walk_bundle.npz

# terminal 3
ros2 topic echo /humanoid_policy/gait --once
ros2 service call /humanoid_servo/arm   std_srvs/srv/Trigger
ros2 service call /humanoid_policy/walk std_srvs/srv/Trigger
```

Check in the dry run that the joint map, the nominal pose and the mode
timeline all look right, then repeat with `dry_run:=false`. Record it:

```bash
ros2 bag record -o crouch_test /humanoid/joint_states \
    /humanoid_policy/joint_command /humanoid_policy/gait \
    /humanoid_servo/status /humanoid_servo/command /imu/data
```

**What to check, and it is the number that matters most for the retrain:** the
tilt reported in `/humanoid_policy/gait` while the mode is `settle`. That is
the hardware's equivalent of the simulated +1.53°, and it tells the training
side how wide the safe band actually has to be.

Ctrl-C in terminal 2 during the crouch or the hold should bring the robot back
to standing and keep it there; a second Ctrl-C exits and drops torque, so hold
the robot before pressing it.

---

## 6. Changes to existing code

- **`joint_map.py`** — `limit_margin_deg` now shrinks only the *calibrated*
  limit, not the model's. The margin exists to keep a servo off a physical hard
  stop and only the calibration knows where one is; the XML limit is a software
  bound already honoured exactly. This mattered here: the walking pose puts
  `right_ankle_pitch` exactly on its model limit, and the old behaviour clipped
  away 2° the policy uses on every step. It also fixes the 4 ankle joints that
  were being clipped during standing.
- **`joint_map.py`** — `command_window()` is centred on the policy's nominal
  pose instead of on zero. Centring the walking window on zero would understate
  a knee's reach by 40°, which is exactly the error the seam check exists to
  catch. With the crouch, worst-case seam slack is still large; no joint comes
  near the wrap.
- **`policy.py`** — `ObservationLayout` takes an `extra` tail so one layout
  describes both the 48- and 53-entry observations, and `Policy` carries the
  gait metadata block.
- **`export_policy.py`** — `--sidecar` exports the walking policy, reading the
  nominal pose, action scale, control rate, gait clock and command ranges from
  the trainer's JSON and checking its actuator order against the XML first.

## 7. Still open

- The **loaded hip-roll and ankle-roll** actuator limits still carry the
  guessed `(1.35 rad/s, 11.0 rad/s²)`. This gait's lateral balance depends on
  them and lateral drift is its worst defect (0.43 m sideways per 1.09 m
  forward).
- **End-to-end loop delay** measured in the deployed stack rather than summed
  from components.
- Fall detection and estop have still never fired on hardware.
