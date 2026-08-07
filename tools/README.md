# Bench tools

Standalone scripts. Not ROS packages, not built by colcon — run them with
`python3` directly. They talk to the servo bus outside the ROS stack, which is
exactly what you want for bring-up and for diagnosing a robot that will not
start.

| script | what it is for |
|---|---|
| `sts_tool.py` | Feetech STS3215 bring-up: scan, ping, set id, set baud, move, bus qualification. The reference for this robot's bus behaviour — the ROS packages' packet handling was written against it. |

`sts_tool.py` is the **only** thing in this repository that writes servo EEPROM
(`set-id`, `set-baud`). The ROS packages cannot: `humanoid_calibration` never
writes at all, and `humanoid_deploy.servo_bus` rejects any write outside the RAM
block 40..54. Keeping the destructive operations in one hand-run script is
deliberate.

## Qualifying a baud rate

```bash
python3 sts_tool.py --baud 250000 bustest 1,2,3,4,5,6,7,8,9,10,11,12,13
```

Anything below 100% at a given rate means step down. 250000 is what this robot
was qualified at.

## Finding a servo that has gone missing

```bash
python3 sts_tool.py scan --bauds 1000000,500000,250000,128000,115200 --full
```
