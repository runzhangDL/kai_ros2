# Hardware actuator model — what `mjx_walk_env.py` must change and why

**Audience:** whoever is next working on `~/Project/robot_mujoco` (the training
side). This is a spec derived from measurements on the physical robot. Every
number here was measured, not assumed; the tools that produced them are in
`tools/` in the ROS repo and can be re-run.

**Summary in one line:** MuJoCo's position actuator tracks its command almost
perfectly, the real STS3215 lags it by ~120 ms and saturates at ~2 rad/s, and
`walk25_slow.pkl` is undeployable because of it. Model the lag, pre-compensate
it in the control law, and a `motion_speed` of 0.30 becomes achievable — which
is *better* tracking than 0.12 without the fix.

---

## 1. The measurement

`tools/replay_motion.py` streams the reference animation to the servos
open-loop, at the deployment rate (25 Hz), with the robot suspended. No policy,
no balance — just "here is the trajectory, what does the hardware do".

Worst-joint RMS tracking error, measured, two runs per speed (reproducible to
±0.1°):

| motion_speed | gait period | worst RMS | worst peak | lag |
|---|---|---|---|---|
| 0.60 | 1.94 s | **23.9°** | 60.9° | 8–10 cycles |
| 0.40 | 2.92 s | 11.9° | 30.3° | 5 cycles |
| 0.25 | 4.67 s | 6.2° | 16.5° | 3–4 cycles |

The worst joints are always the knees, then the ankle pitches — the ones the
animation moves fastest. `left_knee_pitch` peaks at 292 °/s in the reference at
speed 0.6; the hardware manages about 117 °/s.

This is why `walk25_slow.pkl` (trained at `motion_speed 0.6`) cannot be
deployed. A knee that arrives 320 ms late and misses its target by 60° is not
executing the gait the policy learned.

## 2. The fitted model

Fitted by grid search against the measured RMS at four speeds. **Per-joint**,
because the joints differ substantially and MuJoCo takes vectors here as easily
as scalars:

| joint | V_MAX rad/s | A_MAX rad/s² | delay cycles | fit residual |
|---|---|---|---|---|
| left_hip_pitch | 1.35 | 14.5 | 3 | 0.25° |
| right_hip_pitch | 2.15 | 13.0 | 3 | 0.30° |
| left_knee_pitch | 2.45 | 17.5 | **4** | 0.54° |
| right_knee_pitch | 2.20 | 17.0 | **4** | 0.46° |
| left_ankle_pitch | 1.90 | 13.0 | 3 | 0.12° |
| right_ankle_pitch | 1.60 | 11.0 | 3 | 0.36° |
| *everything else* | 1.35 | 11.0 | 4 | *not measured* |

The seven remaining joints (rolls, yaws, waist) barely move in this animation,
so there is no data on them — use the conservative row. If a future gait uses
them, measure them the same way.

Two things worth noting. **The hips are slower than the knees**, which is
physically sensible: a hip swings the whole leg, a knee only the shank. And
**both knees show a 4-cycle delay against 3 elsewhere** — they are also the
largest-amplitude joints, so their parameters are the best-determined.

The `left_hip_pitch` V_MAX of 1.35 against `right_hip_pitch` at 2.15 is a large
left/right asymmetry. It may be real — those two links were reprinted and
recalibrated recently — or an artifact of fitting to summary statistics. Worth
confirming with `tools/servo_step.py --id 6` versus `--id 8` before relying on
it.

### Model structure

A pure transport delay feeding a rate- and acceleration-limited tracker:

```python
delayed = ctrl[k - DELAY]
want    = clip((delayed - pos) / dt, -V_MAX, V_MAX)
acc     = clip((want - vel) / dt,   -A_MAX, A_MAX)
vel    += acc * dt
pos    += vel * dt
```

### Which term dominates, and when

This is the part that changes the strategy. Worst-joint RMS with each effect in
isolation:

| motion_speed | delay only | rate/accel only | both |
|---|---|---|---|
| 0.60 | 12.1° | 16.9° | 23.7° |
| 0.40 | 8.3° | 6.6° | 12.8° |
| 0.25 | 5.3° | **0.35°** | 5.4° |
| 0.15 | 3.2° | **0.04°** | 3.2° |

**Below `motion_speed` ~0.3 the rate and acceleration limits stop mattering
entirely and the error is pure delay.** Slowing the animation further therefore
buys almost nothing: it reduces error only because a slower trajectory covers
less ground in a fixed 120 ms. The naive "just slow it down" plan bottoms out
around 2.5° at `motion_speed 0.12` — a 9.7-second gait cycle, which is not a
walk.

### Validation

The shared-parameter version of this model was fitted on speeds 0.60/0.40/0.25
and used to predict 0.30 **before it was measured**. Predicted 6.8°, measured
**7.7°** — 13% under, correct in direction and magnitude. The per-joint fit
above includes that fourth point.

### A caveat on the delay

Of the 3–4 cycles, roughly 1 is measurement structure: `replay_motion.py` writes
the command then reads position in the same cycle, so the reading reflects the
state before the write took effect. The deployment loop (`servo_node`) has the
*same* read/write ordering, so the figure transfers directly. It would be one
cycle less measured with an external encoder.

## 3. The fix: pre-compensate the delay

The delay is fixed and known, so it does not have to be learned. Command the
reference *ahead* of where you want the joint to be, and the servo arrives on
time. **The optimal lead turns out to equal each joint's own delay**, which is
both the obvious answer and a useful check that the model is sane:

```python
LEAD = DELAY          # per joint: 3 or 4 cycles
phase_cmd = phase + (0.5 + LEAD) * dphase
```

Predicted worst-joint RMS, per-joint model:

| motion_speed | period | no lead | lead 3 | **lead = delay** |
|---|---|---|---|---|
| 0.40 | 2.92 s | 11.8° | 5.7° | 4.2° |
| 0.35 | 3.33 s | 10.1° | 4.7° | 3.8° |
| **0.30** | 3.89 s | 8.4° | 2.7° | **1.75°** |
| 0.25 | 4.67 s | 7.0° | 2.0° | 1.3° |

Leading fixes *phase* error, not *amplitude* error — above `motion_speed` ~0.35
the rate limit clips the peaks and lead alone cannot recover them. Below that it
is close to a complete fix.

**Recommended target: `motion_speed 0.30`, lead = each joint's delay.** A 3.89 s
gait cycle at ~1.75° tracking error. Compare with the no-lead plan, which needed
`motion_speed 0.12` (a 9.7 s cycle) to reach 2.5°: this is three times the gait
speed at better tracking.

Put the lead in the control law of **both** the env and the deployment node. If
only the env has it, sim and real diverge; if only deployment has it, the policy
never learned to expect it. It belongs in the feedforward path so the residual
stays free for balance instead of funding lag compensation.

## 4. Exactly what to change in `mjx_walk_env.py`

### 4a. Add the actuator model to `step()`

Currently `ctrl` goes straight to `pipeline_step`, and MuJoCo's `kp=50` servo
puts the joint essentially where it is told. Insert the limiter between them:

```python
# --- measured on the robot; see the table in section 2 -----------------
# Order is ACTUATOR order, i.e. the order mj_id2name gives for actuators.
_UNMEASURED = (1.35, 11.0, 4)          # conservative, for joints with no data
ACTUATOR_LIMITS = {
    "left_hip_pitch_joint":   (1.35, 14.5, 3),
    "left_knee_pitch_joint":  (2.45, 17.5, 4),
    "left_ankle_pitch_joint": (1.90, 13.0, 3),
    "right_hip_pitch_joint":  (2.15, 13.0, 3),
    "right_knee_pitch_joint": (2.20, 17.0, 4),
    "right_ankle_pitch_joint":(1.60, 11.0, 3),
}
# Build V_MAX / A_MAX as (nu,) jnp arrays and DELAY as a per-joint int array,
# falling back to _UNMEASURED. The delay buffer must then be sized to
# max(DELAY) and indexed per joint.
```

In `__init__`, size the delay buffer (it must live in `info`, not on `self`,
because brax vmaps over environments):

```python
# (nu,) arrays in actuator order, built from ACTUATOR_LIMITS above
self._v_max = jnp.array(v_max_per_joint)          # rad/s
self._a_max = jnp.array(a_max_per_joint)          # rad/s^2
self._delay = np.array(delay_per_joint, dtype=int)   # 3 or 4, per joint
self._max_delay = int(self._delay.max())
# Per-joint lead, in phase units. Equals each joint's delay -- see section 3.
self._lead = jnp.array(self._delay, dtype=jnp.float32)
```

In `reset()`, seed the state:

```python
info["ctrl_queue"] = jnp.tile(ref_q, (self._max_delay, 1))
info["servo_pos"] = ref_q          # where the "servo" currently is
info["servo_vel"] = jnp.zeros(self.sys.nu)
```

In `step()`, after computing `ctrl` and before `pipeline_step`:

```python
# The real STS3215 does not arrive where it is told. It lags ~120 ms and
# saturates at ~2 rad/s; MuJoCo's position servo does neither, which is why
# a policy trained without this produces footfalls the robot cannot make.
# Per-joint delay: pick row (max_delay - delay_j) out of the queue for each
# joint, so a 3-cycle joint reads one row newer than a 4-cycle one.
queue = info["ctrl_queue"]                       # (max_delay, nu)
rows = self._max_delay - self._delay             # (nu,)
delayed = queue[rows, jnp.arange(self.sys.nu)]
info["ctrl_queue"] = jnp.concatenate([queue[1:], ctrl[None]], axis=0)

want = jnp.clip((delayed - info["servo_pos"]) / self.dt, -self._v_max, self._v_max)
acc = jnp.clip((want - info["servo_vel"]) / self.dt, -self._a_max, self._a_max)
servo_vel = info["servo_vel"] + acc * self.dt
servo_pos = info["servo_pos"] + servo_vel * self.dt
info["servo_pos"], info["servo_vel"] = servo_pos, servo_vel

data = self.pipeline_step(state.pipeline_state.replace(xfrc_applied=xfrc), servo_pos)
```

Note this **replaces** `ctrl` as the argument to `pipeline_step` — the physics
now sees the servo's achievable trajectory, not the raw command.

### 4b. Add the lead to the reference sample

```python
phase0 = info["phase"]
# 0.5 is the existing midpoint sampling. The extra LEAD cycles pre-compensate
# the measured 120 ms actuator lag, so the joint ARRIVES at ref(phase).
# Per-joint lead, so each joint is commanded ahead by its own delay. This
# needs a per-joint _ref_pose lookup rather than the scalar one:
#   f_j = mod(phase + (0.5 + lead_j) * dphase, 1) * n_cycle_frames
# then gather ref[floor(f_j), j] and interpolate, per joint.
phase_cmd = jnp.mod(phase0 + (0.5 + self._lead) * self._dphase, 1.0)  # (nu,)
ctrl_ref = self._ref_pose_per_joint(phase_cmd)
```

Keep the reward comparing against the scalar `ref(phase)` (interval end) as it
does now — the lead is a control-law detail, not a change to what "tracking"
means. Only the *command* is led; the *target* is unchanged.

If a per-joint reference lookup is more surgery than you want for a first pass,
a single scalar `LEAD = 4` costs only ~0.9° at `motion_speed 0.30` (2.67° vs
1.75°) and is a one-line change. Start there.

### 4c. Keep `--action-delay-steps 1`

That models bus/compute latency: the policy decides at cycle *k*, the servos
receive it at *k+1*. It is a **different** effect from the actuator lag and both
are real. Total decision-to-motion is then ~4 cycles, which matches the
deployment pipeline.

### 4d. Reconsider these, in light of the model

- **`action_scale 0.35`** is probably still right. At `motion_speed 0.30` the
  peak reference velocity is 2.5 rad/s; with the lead in the feedforward path
  the residual no longer needs to fund lag compensation, so 0.35 rad stays
  available for balance.
- **`OBS_NOISE["joint_vel"] = 0.15`** is now understated in an interesting way.
  On hardware, `qvel` is finite-differenced at 25 Hz from a 4096-count encoder,
  so it is *quantised* to 0.038 rad/s steps and reads exactly 0.0 about 85% of
  the time when the robot is nearly still (measured in a real standing run).
  Gaussian noise is the wrong shape. Consider quantising instead:
  `qvel = round(qvel / 0.038) * 0.038`.
- **`term_pose_error 1.5`** was calibrated against a policy that tracked well.
  With the actuator model in place, early training will track worse; watch that
  this does not terminate every episode before learning starts.

## 5. How to validate before spending a training run

The model is falsifiable and cheap to check:

```bash
# on the robot, suspended
python3 tools/replay_motion.py --speed 0.30 --cycles 2 --save replay_030.npz
```

Predicted worst-joint RMS **without** lead at 0.30 is **6.8°**. If the
measurement comes back near that, the model holds. If it does not, re-fit before
trusting anything above.

Then, once the env change is in, compare a sim rollout of the same open-loop
reference against `replay_030.npz`. Sim and hardware trajectories should now
overlay. That is the whole point of the exercise: a training environment whose
actuator tells the truth.

## 6. Known limits of this model

- **Fitted to summary statistics, not raw traces.** 18 joint/speed points, 0.54°
  residual. The raw traces (`--save`) would tighten it, particularly the split
  between delay and rate limiting at intermediate speeds.
- **Measured unloaded**, robot suspended. That is the correct condition for the
  *swing* leg, which is where the speed demand is. Stance joints bearing the
  robot's weight will be slower, and hip roll / ankle roll under load were never
  measured — they barely move in this animation, so it did not matter here, but
  it would for a gait that uses them.
- **`V_MAX` and `A_MAX` are shared across all 13 joints.** Per-joint values are
  extractable from the same data if the gait turns out to be sensitive to it.
- **`servo_acc` on the robot is now 100** (10000 counts/s²), not the old 30. The
  measurements above were taken at 100. If that config value changes, `A_MAX`
  changes with it.

## 7. Other constraints the deployment already imposes

For completeness, verified against this robot's calibration:

- **Control rate is 25 Hz** and not negotiable — the STS bus benchmarks at a
  24 ms read+write cycle.
- **No encoder-seam violations** anywhere in the gait at any speed, after the
  servo re-centring. The full ±0.35 rad command window around the reference
  stays clear of the 0/4095 wrap on all 13 joints.
- **The trajectory fits the calibrated travel**, with the four ankle joints
  needing the full ±20° XML range.
- **The observation layout must not change.** `obs[0:48]` is byte-identical to
  the standing policy and the deployment reuses that builder verbatim;
  `obs[48:50]` is `phase_sin, phase_cos`. Anything added must go after index 49.
- **A walk→stand handover cannot be a direct switch.** A mid-stride pose is
  9–36σ outside the standing policy's training distribution, and this gait has
  exactly one frame of double support. Deployment handles it with a scripted
  return; the training side does not need to care, but do not assume the
  standing policy can catch a walk that stops abruptly.
