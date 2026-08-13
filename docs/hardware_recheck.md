# Hardware re-characterisation after the 500000-baud change

Run in order. Each phase gates the next — if a phase fails, stop and send the
output rather than continuing, because everything after it assumes the bus is
sound.

Everything here is run on the **robot** (Jetson), from `~/Project/ROS2_Humble`.

```bash
cd ~/Project/ROS2_Humble
colcon build --packages-select humanoid_calibration humanoid_deploy
source install/setup.bash
mkdir -p ~/hwcheck
```

**Throughout: the robot hangs from the gantry, feet clear of the ground**,
except where a phase says otherwise.

---

## Phase 0 — who is on the bus (read-only, nothing moves)

The new right-knee servo is the unknown. A factory STS3215 ships as **id 1 at
1000000 baud**, and id 1 is already `left_ankle_pitch` on this robot. Two servos
sharing an id answer simultaneously and corrupt each other, which reads as "id 1
is flaky" rather than as an obvious error — so this runs before anything drives
a motor.

```bash
python3 tools/sts_tool.py scan --bauds 500000,1000000,250000,128000,115200 --full \
  2>&1 | tee ~/hwcheck/00_scan.txt
```

**Expect:** 13 servos, ids 1–13, all at 500000, nothing at any other rate.

| what you see | what it means |
|---|---|
| 13 ids at 500000 | good, go to Phase 2 |
| id 11 missing, something extra at 1000000 | the new servo was never re-bauded → Phase 1 |
| id 11 missing, nothing else anywhere | wiring or a dead servo — stop |
| id 1 flaky/intermittent | **likely two servos both on id 1** → Phase 1 |

---

## Phase 1 — only if Phase 0 found the new servo misconfigured

`sts_tool.py` is the only thing in this repo that writes EEPROM. Do this with
**the new servo as the only one powered**, or at minimum with the id it
currently holds not duplicated.

```bash
# talk to it at whatever rate/id Phase 0 found. Example: factory id 1 @ 1000000
python3 tools/sts_tool.py --baud 1000000 ping 1
python3 tools/sts_tool.py --baud 1000000 set-id 1 11      # -> becomes id 11
python3 tools/sts_tool.py --baud 1000000 set-baud 11 500000
python3 tools/sts_tool.py --baud 500000  ping 11          # confirm
```

Then re-run Phase 0 and confirm a clean 13.

---

## Phase 2 — does 500000 actually hold on this cabling

```bash
python3 tools/sts_tool.py --baud 500000 bustest 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  2>&1 | tee ~/hwcheck/02_bustest.txt
```

**Requires 100%.** Calibration's `min_read_success` is 1.0 and hard-stops below
it. Anything less and the answer is to step back down to 250000 — the robot
completed a full standing run at that rate, so it is a known-good fallback, not
a defeat.

```bash
python3 tools/motor_check.py 2>&1 | tee ~/hwcheck/02_motor_check.txt
```

**Expect:** 13/13 answer, all `torque off`, 37–42 °C, ~12 V. The new servo's
model number should match the others.

Then confirm the new knee is mechanically free — the old one was seized, which
is how this started:

```bash
python3 tools/motor_check.py --backdrive 20
# move the RIGHT KNEE by hand through a few degrees during the 20 s
```

**Expect:** id 11 shows hundreds of counts of range, verdict `moved`.

---

## Phase 3 — the new knee: recentre, then calibrate

A new servo's electrical zero is wherever the factory left it, so its travel may
straddle the 0/4095 encoder seam. Commanding across that seam makes the servo
take the long way round — nearly a full turn, at speed, through the leg. The
deployment refuses to start in that state, and this is the fix.

```bash
# read-only first: where does id 11 currently sit?
python3 tools/sts_recenter.py probe --id 11 2>&1 | tee ~/hwcheck/03_probe.txt
```

Then, **with the right knee posed at its standing position** (leg straight, the
pose the calibration calls 0°) and the robot supported:

```bash
python3 tools/sts_recenter.py one --id 11            # dry run, shows the plan
python3 tools/sts_recenter.py one --id 11 --apply
```

This only touches id 11. It backs up the old correction value first and updates
the stored calibration by exactly how far the servo moved, so the other twelve
joints are untouched and do **not** need recalibrating — the calibration
fingerprint deliberately excludes bus settings, so the baud change alone
invalidates nothing.

Now calibrate the one replaced joint:

```bash
ros2 run humanoid_calibration calibrate --only right_knee_pitch \
  2>&1 | tee ~/hwcheck/03_calibrate.txt

ros2 run humanoid_calibration verify_calibration 2>&1 | tee ~/hwcheck/03_verify.txt
```

---

## Phase 4 — the new bus timing

This is the number that decides the control rate.

```bash
ros2 run humanoid_deploy bus_benchmark --baud 500000 \
  2>&1 | tee ~/hwcheck/04_bench_500k.txt

# and the old rate, for a controlled comparison on the same cabling
ros2 run humanoid_deploy bus_benchmark --baud 250000 \
  2>&1 | tee ~/hwcheck/04_bench_250k.txt
```

**Prediction, so the result can be checked rather than just accepted:** one
control cycle puts 224 bytes on the wire (sync-read request + 13 reply frames +
sync-write). That is 8.96 ms of airtime at 250000 and 4.48 ms at 500000. The
measured cycle at 250000 was 24.0 ms, so **15.0 ms of it is rate-invariant** —
per-servo turnaround, syscalls, the servos' own response latency. Halving the
baud rate therefore buys about 4.5 ms, not half:

| | predicted |
|---|---|
| cycle at 500000 | **~19.5 ms** (ceiling ~51 Hz) |
| headroom at 33 Hz (30 ms) | +10.5 ms — comfortable |
| headroom at 40 Hz (25 ms) | +5.5 ms — workable |
| headroom at 50 Hz (20 ms) | +0.5 ms — no |

If the measurement lands near 19–20 ms the model is right and **33 Hz is the
sensible new target**. If it lands near 12 ms, something else was dominating
and 40–50 Hz opens up.

---

## Phase 5 — what the motors can do, at the new rate

Two different questions, two tools. `servo_step.py` gives speed / acceleration /
rise time; `servo_trace.py` gives **dead time**, which is the parameter the
walking policy is most sensitive to and the one a step fit determines worst.

### 5a. Suspended (feet clear)

```bash
for ID in 3 6 11; do
  python3 tools/servo_step.py  --id $ID --amplitude 20 2>&1 | tee ~/hwcheck/05a_step_$ID.txt
  python3 tools/servo_trace.py --id $ID --amplitude 20 --save ~/hwcheck/05a_trace_$ID.npz \
    2>&1 | tee ~/hwcheck/05a_trace_$ID.txt
done

# the roll joints, which have NEVER been measured and still carry a guessed
# (1.35 rad/s, 11.0 rad/s^2). Smaller amplitude: the ankle rolls only span +-20 deg.
for ID in 5 9; do
  python3 tools/servo_step.py  --id $ID --amplitude 15 2>&1 | tee ~/hwcheck/05a_step_$ID.txt
  python3 tools/servo_trace.py --id $ID --amplitude 15 --save ~/hwcheck/05a_trace_$ID.npz \
    2>&1 | tee ~/hwcheck/05a_trace_$ID.txt
done
for ID in 2 12; do
  python3 tools/servo_step.py  --id $ID --amplitude 10 2>&1 | tee ~/hwcheck/05a_step_$ID.txt
  python3 tools/servo_trace.py --id $ID --amplitude 10 --save ~/hwcheck/05a_trace_$ID.npz \
    2>&1 | tee ~/hwcheck/05a_trace_$ID.txt
done
```

### 5b. Bearing weight — the numbers the stance leg actually needs

Everything in `docs/hardware_actuator_model.md` was measured suspended. This is
the gap that has been open since the beginning.

**Stand the robot on the ground with the gantry preventing a fall but not
carrying its weight.** Then:

```bash
python3 tools/motor_check.py --hold        # every joint stiffens where it is
```

`--hold` reads each servo's present position and writes it back as the goal
*before* enabling torque, so nothing snaps to a stale target. Check the robot is
standing and stable, then measure the load-bearing joints one at a time:

```bash
for ID in 3 5 9 11; do
  python3 tools/servo_step.py  --id $ID --amplitude 12 2>&1 | tee ~/hwcheck/05b_step_$ID.txt
  python3 tools/servo_trace.py --id $ID --amplitude 12 --save ~/hwcheck/05b_trace_$ID.npz \
    2>&1 | tee ~/hwcheck/05b_trace_$ID.txt
  python3 tools/motor_check.py --hold      # re-stiffen; the step tool releases its joint
done

python3 tools/motor_check.py --release
```

Smaller amplitude here on purpose — the robot is standing on these joints.

### What to send back

`~/hwcheck/*.txt` (the `.npz` traces only if something looks odd and I need the
raw waveform). Tar it:

```bash
tar czf ~/hwcheck.tar.gz -C ~ hwcheck
```

---

## What these results will and will not change

**Will change:** the control rate, and with it a retrain. If Phase 4 supports
33 Hz, `frame_skip` goes 20 → 15 and both `mjx_stand_env` and
`mjx_walk_free_env` need rerunning at the new rate, because one policy step must
equal one physical timestep or every joint velocity in the observation is scaled
wrong.

**Will change:** the roll joints' `ACTUATOR_LIMITS` row, currently a guess. This
gait's worst defect is lateral drift and lateral balance runs through exactly
those four joints.

**Will not change: the dead time, much.** Transport delay is dominated by the
servo's internal loop, not the bus. Doubling the baud removes maybe 4–5 ms from
an ~80 ms round trip. Raising the control rate gives fresher observations and
finer commands — worth having — but it does not fix the delay, and delay is what
blocked walking. Note also that the same physical delay becomes *more cycles* at
a higher rate (2 cycles at 25 Hz ≈ 3 at 33 Hz); the env converts correctly via
`_LIMITS_MEASURED_AT_HZ`, so this is bookkeeping, not a regression, but the
"3 cycles never walks" result from the 25 Hz dose-response must **not** be
carried over to 33 Hz without re-deriving it in milliseconds.

**Will not bring back imitation learning.** Baud changes communication, not
mechanics. The reference gait needs 292 °/s peak; the servos deliver roughly
120 °/s loaded, and the same motors are still fitted. Unless Phase 5 shows joint
speeds far above what was measured before — and it should not, these are the
same STS3215s — the animation still has to be slowed to `motion_speed` ≈ 0.30,
which is where the reference-free approach came from in the first place. If the
loaded numbers do come back surprisingly high, that conclusion is worth
revisiting and I will say so.
