# humanoid_calibration

One-time zero and travel-limit calibration for the 13-DoF humanoid's Feetech
STS3215 servos, plus the guard that stops motion nodes from running on an
uncalibrated robot.

**The calibration tool never writes to a servo.** It only issues `READ` (and
`PING`) instructions — see the safety note in `feetech_bus.py`.

---

## This robot's bus

| | |
|---|---|
| Port | `/dev/ttyTHS1` — Jetson Orin Nano 40-pin UART, pins 8/10 |
| Baud | **250000** (all 13 servos were moved down from the STS factory 1000000) |
| Wiring | Single-wire half-duplex: 74LVC1G125 open-drain TX + pull-up, RXD tapped off the same wire |

Three consequences of that wiring are handled in `feetech_bus.py`, all of them
established with `sts_tool.py` on the real robot:

1. **Every transmit is echoed back on RXD.** Not an edge case — it happens on
   every transaction. This matters more than it sounds: a `READ` of N bytes and
   its status reply are structurally *identical* (same id, same `LEN` of N+2,
   both with valid checksums), so an unhandled echo makes the driver return the
   echoed register address `0x38,0x02` as a position of **568 counts** — a
   plausible-looking, completely wrong number. The driver cuts the exact bytes
   it sent, and `test_read_raw_never_returns_the_echoed_register_address` pins
   the behaviour.
2. **Tegra HS-UARTs often ignore the baud divisor on the first `open()`.** The
   port opens at some default rate and every byte is malformed. The driver
   opens, closes and reopens.
3. **The first transaction after opening is routinely lost** while the line
   settles. The driver burns a throwaway probe (addressed to unused id `0xFD`)
   and flushes before any real read.

If you re-baud the bus, re-qualify it against the real cable lengths first —
`min_read_success` in the config defaults to 100%, and calibration hard-stops
below it:

```bash
python3 sts_tool.py --baud 250000 bustest 1,2,3,4,5,6,7,8,9,10,11,12,13
```

## Before you start

Fill in the real mechanical limits in `config/joint_limits.yaml`. Every
`min_deg` / `max_deg` in there is a placeholder.

**Servo torque.** This tool only reads, so torque is left exactly as it was at
power-on and you will not be able to back-drive the joints. Release it first
with the bring-up tool:

```bash
for i in {1,2,3,4,5,6,7,8,9,10,11,12,13}; do
  python3 sts_tool.py --port /dev/ttyTHS1 --baud 250000 torque $i 0
done
```

The calibration tool prints this exact command at start-up.

## Build and run

```bash
cd ~/Project/ROS2_Humble
colcon build --packages-select humanoid_calibration
source install/setup.bash

ros2 run humanoid_calibration calibrate
```

Off-robot, you can walk the whole flow with a simulated bus:

```bash
ros2 run humanoid_calibration calibrate --mock   # arrow keys / j,k move the fake encoder
```

## What calibration does

The tool runs a `bustest`-style pre-flight first — 20 reads per servo — and
refuses to continue unless every one of them succeeds. A single ping proves the
wiring but says nothing about whether the baud rate holds up over this robot's
cabling; a bus dropping one read in twenty still lets calibration *finish*, it
just records limits from whichever sample happened to land. That is the silent
corruption this tool exists to prevent, so it is a hard stop.

Then, for each joint **in the order given in `joint_limits.yaml`** — left leg
bottom-up, waist, right leg top-down:

| Step | What you do | What is recorded |
|---|---|---|
| **1 / 3 — Zero** | Live encoder counts are shown. Hand-position the link so it is vertical to the ground, press <kbd>ENTER</kbd>. | That reading becomes 0°. |
| **2 / 3 — Minimum** | The target from the config is shown in degrees and radians, alongside the live angle *relative to the zero you just set* and a position bar. Move the joint toward the minimum. | Recorded **automatically** once you hold at or past the target for 0.2 s. |
| **3 / 3 — Maximum** | Same, the other way. | Recorded automatically. |

Keys: <kbd>ENTER</kbd> confirm zero · <kbd>r</kbd> restart this joint ·
<kbd>s</kbd> skip this joint · <kbd>q</kbd> quit.

Each joint is saved the moment it finishes, so quitting or losing power never
costs you completed work. Re-running picks up exactly where you left off;
already-calibrated joints are skipped unless you pass `--force`.

If a limit is `0.0°` (a knee that does not hyper-extend, say) the zero pose
already satisfies it, so auto-detection would fire instantly — the tool asks
for an explicit <kbd>ENTER</kbd> instead.

### If the angle moves the wrong way

If the live angle goes negative when you expect positive, the servo is mounted
against the joint's positive direction. Set `direction: -1` for that joint in
`joint_limits.yaml` and re-run it:

```bash
ros2 run humanoid_calibration calibrate --only left_hip_roll
```

## Where the calibration lives

```
$HUMANOID_CALIBRATION_FILE                        # explicit override
$XDG_CONFIG_HOME/humanoid_robot/calibration.yaml  # if XDG_CONFIG_HOME is set
~/.config/humanoid_robot/calibration.yaml         # default
```

It is deliberately **outside the ROS install space**: calibration belongs to
this physical robot and must survive `colcon build`, workspace wipes and
package reinstalls. Writes are atomic, so a power cut mid-save leaves either
the old file or the new one, never a corrupt one.

Each joint stores the raw counts at zero, min and max, plus the config it was
captured against. A fingerprint of `joint_limits.yaml` is stored too — edit a
limit later and the affected joints are flagged **stale** and must be redone.
Editing the port or baud rate does *not* invalidate anything.

Delete the file to force a full recalibration.

## Stopping uncalibrated nodes from starting

ROS 2 has no built-in "is this robot calibrated" concept, so this package
defines one. There are three enforcement points; use the first two together.

### 1. Launch time — nothing starts at all

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from humanoid_calibration.launch_guard import calibration_gate

def generate_launch_description():
    return LaunchDescription([
        calibration_gate(),          # aborts the launch before anything spawns
        Node(package='humanoid_walk', executable='walking', output='screen'),
    ])
```

See `launch/guarded_bringup.launch.py` for a working template.

### 2. Node start-up — a hand-started node still refuses

```python
from humanoid_calibration.guard import require_calibration

class WalkingNode(Node):
    def __init__(self):
        super().__init__('walking')
        self.calib = require_calibration(self)   # raises NotCalibrated
```

`require_calibration` *returns* the per-joint calibration, so the call that
enforces the check is also the call that hands you the numbers — there is no
way to skip the check and still get the data.

```python
joint = self.calib['left_knee_pitch']
angle = joint.raw_to_deg(raw_counts)      # counts -> degrees about the zero
counts = joint.deg_to_raw(target_deg)     # degrees -> counts
safe   = joint.clamp_deg(target_deg)      # clamp to calibrated travel
```

### 3. Runtime — a latched topic to observe

```bash
ros2 run humanoid_calibration calibration_status
```

| | |
|---|---|
| `/calibration_status/ok` | `std_msgs/Bool`, latched |
| `/calibration_status/status` | `std_msgs/String`, latched detail |
| `/calibration_status/check` | `std_srvs/Trigger`, re-validate now |

Latched (`TRANSIENT_LOCAL`), so a node that starts late still gets the last
value immediately. This is for observability — enforcement should use (1) and
(2), which fail closed rather than waiting on a topic that may never arrive.

### Bench escape hatch

`HUMANOID_ALLOW_UNCALIBRATED=1` downgrades the node-level failure to a warning.
`calibration_gate(strict=True)` — the default — ignores it, so it cannot be
used to launch something that moves the legs.

## Command reference

```bash
ros2 run humanoid_calibration calibrate                       # calibrate what is not done
ros2 run humanoid_calibration calibrate --force               # redo everything
ros2 run humanoid_calibration calibrate --only left_hip_roll  # redo specific joints
ros2 run humanoid_calibration calibrate --start-at waist_yaw  # resume from a joint
ros2 run humanoid_calibration calibrate --status              # per-joint status
ros2 run humanoid_calibration calibrate --port /dev/ttyTHS1   # override the bus
ros2 run humanoid_calibration calibrate --mock                # no hardware

ros2 run humanoid_calibration check_calibration               # exit 0 = ready, 1 = blocked
ros2 run humanoid_calibration check_calibration -q            # silent, for scripts
```

`check_calibration` is a plain exit code, so it drops into systemd units,
`ExecStartPre=`, or shell scripts.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot open /dev/ttyTHS1` — permission denied | `sudo usermod -aG dialout $USER`, then log out and back in |
| `NO REPLY` from some servos | `python3 sts_tool.py scan --bauds 1000000,500000,250000,115200 --full` — a servo that was missed during re-bauding will still be on its old rate |
| `FLAKY 18/20` in pre-flight | The bus is dropping reads. Confirm with `sts_tool.py bustest`, then step the whole bus down to a slower rate |
| Servo won't turn by hand | Torque is still enabled — `sts_tool.py torque <id> 0` (see above) |
| Every byte malformed right after boot | The Tegra first-open baud bug; the driver already opens/closes/reopens, but check nothing else holds the port |
| Position jumps by ~4096 counts | Normal encoder wrap; it is unwrapped internally and does not affect the recorded values |
| Angle sign is backwards | Set `direction: -1` for that joint and re-run it |

If you move to a USB-TTL adapter, prefer a stable device path over
`/dev/ttyUSB0`, which moves between boots:

```bash
ls -l /dev/serial/by-id/    # then put that path in joint_limits.yaml
```

## Tests

```bash
colcon test --packages-select humanoid_calibration
colcon test-result --verbose
```

47 unit tests cover config validation, the store round-trip and its atomicity,
staleness detection, angle conversion across the encoder seam, the pre-flight
reliability gate, and the Feetech wire protocol — packet layout, checksums,
resynchronisation, and this bus's echo behaviour including a partial echo left
by a dropped turnaround byte.

## Layout

```
config/joint_limits.yaml       limits, servo ids, bus settings   <- FILL THIS IN
humanoid_calibration/
  calibrate_cli.py             the interactive tool
  guard.py                     require_calibration(), check_calibration CLI
  launch_guard.py              calibration_gate() for launch files
  calibration_status_node.py   latched status topic + service
  calibration_store.py         the persistent calibration file
  feetech_bus.py               read-only STS3215 protocol
  joint_config.py              config loading and validation
  term_ui.py                   raw-mode keys, in-place redraw
launch/
  guarded_bringup.launch.py    template for your own launch files
  calibration_status.launch.py
```
