# The slow-walk experiment — design notes before running it

**Goal, in the operator's words:** *"I don't mind the robot moving very slowly,
as long as it walks the way a human does."* So: real steps, one foot at a time,
each foot visibly leaving the ground. Speed is not a requirement. Shuffling with
both feet down is not a pass.

**Status:** not yet run. Nothing is training.

---

## Why the earlier slow attempt did not test this

Campaign 25 set `--gait-period 6.0` and nothing else, on the belief that the
only thing a long period changes is how much time the hip rolls get. That is
wrong. Four separate quantities scale with the gait period or silently assume a
fast one, and at 6 s all four are broken:

| quantity | value | in gait cycles at 6 s | what it does |
|---|---|---|---|
| `air_min` (credit threshold) | 0.60 s | — | a step earns **nothing** unless the foot stays up 0.6 s |
| `progress_tau_s` | 2.0 s | **0.33** | velocity EMA dips between steps |
| `discount_horizon_s` | 6.0 s | **1.0** | policy cannot see one stride ahead |
| `cmd_resample_s` | 5.0 s | **0.83** | the command changes mid-stride |
| `episode_seconds` | 14 s | **2.3** | about two steps per episode |

The first is the one that makes campaign 25 uninterpretable: `air_time_target`
defaults to the schedule's own swing, so lengthening the period to give slow
joints more time **simultaneously demanded proportionally longer steps**. It
did not test a slow gait; it tested whether the robot could take huge slow ones.

The second may be worse. `term_no_progress` ends the episode when the
forward-velocity EMA falls below 25% of the command. At one step every 3 s with
a 2 s EMA, forward velocity is deeply intermittent — the EMA dips between steps
and kills a robot that is walking correctly. That mechanism alone could produce
the "learns to step, then unlearns it" curve that four campaigns have shown, and
it would have nothing to do with the actuators.

`--air-time-target`, `--progress-tau-s` and `--cmd-resample-s` did not exist as
flags. They do now.

---

## What is actually established

Measured on the robot, and not in doubt:

| | |
|---|---|
| servo stiffness | kp 44.9 N·m/rad (`measure_kp.py`, residual 0.65°) |
| transport delay | 72–92 ms, ~2 cycles at 25 Hz |
| loaded hip roll | ~1.8 rad/s², vs ~6.7 for the loaded knee |
| nominal pose | ankle pitch +4° from the model's, hardware-validated to 0.5° flat |
| bus | 5.07 ms cycle, 100% reads, 152 Hz ceiling |

Established in simulation, single-variable:

- Roll `a_max` 11.0 sustains stepping (`left_airborne` ~0.40); 4.6 collapses it
  (~0.05). Same everything else. **But that parameter covers seven joints** —
  both hip rolls, both hip yaws, both ankle rolls and the waist — so it shows
  the roll/yaw group matters, not the hip rolls specifically.
- Eliminated as causes: kp span (three values), gait period alone (1.2 vs 2.0),
  `zero_cmd_prob` (0.35 vs 0.15), `target_height` desync.

**The open question is whether the collapse is the actuators or the reward
plumbing above.** Four broken period-dependent parameters is enough to explain
it without invoking hardware at all, and they have never been corrected
together.

---

## The run to do

One run, everything scaled coherently to the period. `--gait-period 3.0` rather
than 6.0: swing is then 0.60 s against the 0.46 s the weight shift needs, which
is enough margin without pushing every other timescale as far out of range.

```bash
cd /home/run/Project/robot_mujoco
/home/run/env_mujoco/bin/python train_walk_free.py \
    --gait-period 3.0 \
    --air-time-target 0.25 \
    --progress-tau-s 9.0 \
    --cmd-resample-s 9.0 \
    --discount-horizon-s 12.0 \
    --episode-seconds 30 \
    --zero-cmd-prob 0.15 --domain-rand --init-static-frac 0.4 \
    --actuator-delay-scale 1.0 --push-force 0 --obs-noise-scale 0 \
    --timesteps 300000000 --out camp_C2.pkl --logdir ./logs/campaign26/tb_C2
```

Why each:

- **`--gait-period 3.0`** — swing 0.60 s, 1.3× the measured weight-shift time.
- **`--air-time-target 0.25`** — a 0.125 s lift now earns partial credit and
  0.25 s earns full. Decoupled from the period, so a slow cadence no longer
  demands huge steps. This is the single most important change.
- **`--progress-tau-s 9.0`** — three gait periods, so the EMA spans several
  steps and cannot dip between them.
- **`--cmd-resample-s 9.0`** — three periods, so the command holds across a
  stride.
- **`--discount-horizon-s 12.0`** — four periods, so a full stride is inside
  the horizon.
- **`--episode-seconds 30`** — ten gait cycles per episode.

## What to watch, in priority order

1. **`left_airborne`.** Every failed campaign shows the same signature: it peaks
   near 0.3 by the second eval and decays to ~0.05. If it **holds** above 0.25,
   the reward plumbing was the problem and the robot can walk slowly. If it
   peaks and decays again, that is four coupled fixes ruled out and the
   actuator reading stands.
2. **`mean_air_time`** — should sit near 0.25 s, not 0.005.
3. **`ep_len`** — if this stays near 70/750 while airborne is healthy, the
   no-progress termination is still firing and `progress_tau_s` needs to go
   higher still.
4. **`forward_vel`** — expected around 0.01–0.02 m/s. Slow is fine; the
   operator has said so explicitly.

## If it works

The bundle carries `phase_increment_per_cycle`, so the deployment picks up the
3 s period automatically — no node change. Then the gate in
`TRAINING_HANDOFF.md` §5, then `crouch_only` on hardware, then a short walk
with the standing policy ready to take over.

## If it does not

Then the roll authority reading is the honest answer, and the options are the
ones in the campaign-24 write-up: widen the feet inward (cheapest, printable,
~25% less lateral CoM travel needed), stronger roll actuators, or accept a
robot that stands, crouches, balances and recovers — all of which work on
hardware today.

Before concluding that, run the ablation properly: `UNMEASURED_LIMITS` covers
seven joints, and only the four roll joints are implicated by the lateral-balance
argument. Splitting them would say whether it is the hips, the ankles, or both.
