# Bench tools

Standalone scripts. Not ROS packages, not built by colcon — run them with
`python3` directly. They talk to the servo bus outside the ROS stack, which is
exactly what you want for bring-up and for diagnosing a robot that will not
start.

| script | what it is for |
|---|---|
| `sts_tool.py` | Feetech STS3215 bring-up: scan, ping, set id, set baud, move, bus qualification. The reference for this robot's bus behaviour — the ROS packages' packet handling was written against it. |
| `motor_check.py` | Read-only survey: who answers, model, torque, position, volts, temperature. `--backdrive` finds a joint that will not move by hand. |
| `sts_recenter.py` | Sets a servo's electrical midpoint to count 2048, so its travel sits far from the 0/4095 encoder seam. Needed for any **newly installed** servo before calibrating it. |
| `servo_step.py` | Step response for one joint: peak speed, 90% rise time, acceleration, steady-state error. Run free and loaded. |
| `servo_trace.py` | Raw `(t, position)` step trace for one joint, for reading **dead time** straight off. This is the parameter the walking policy is most sensitive to, and the one a step response fits worst. |
| `replay_motion.py` | Open-loop replay of a reference gait at the deployment rate, robot suspended. Answers whether the hardware can execute an animation at all. |
| `sim_handover.py` | Training-machine only. Runs the real `GaitSequencer` and the real numpy `Policy` against MuJoCo to test the stand→walk→stand handover before hardware. |

All of these default to `--baud 500000`. Pass `--baud` explicitly if the bus has
been stepped back down.

`sts_tool.py` is the **only** thing in this repository that writes servo EEPROM
(`set-id`, `set-baud`). The ROS packages cannot: `humanoid_calibration` never
writes at all, and `humanoid_deploy.servo_bus` rejects any write outside the RAM
block 40..54. Keeping the destructive operations in one hand-run script is
deliberate.

## Qualifying a baud rate

```bash
python3 sts_tool.py --baud 500000 bustest 1,2,3,4,5,6,7,8,9,10,11,12,13
```

Anything below 100% at a given rate means step down. The robot ran a full
standing deployment at 250000; 500000 is the current setting and must be
re-qualified whenever a servo is replaced or the loom is disturbed.

## Finding a servo that has gone missing

```bash
python3 sts_tool.py scan --bauds 1000000,500000,250000,128000,115200 --full
```
