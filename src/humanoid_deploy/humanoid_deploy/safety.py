"""The safety supervisor that sits between the policy and the motors.

Deliberately free of ROS and of the serial bus so that every rule here is
directly unit-testable; the nodes own I/O, this owns the decisions.

State machine::

    DISARMED --arm() passes preflight--> RAMPING --ramp completes--> RUNNING
        ^                                   |                          |
        +--------------- fault() -----------+--------------------------+
                                            v
                                          FAULT  (terminal, torque off)

Layers, outermost first. Each is independent, so a bug in one does not disarm
the others:

1. **Preflight.** Refuses to energise unless the calibration is valid, no joint
   command window crosses the encoder seam, every servo reads, the robot is
   upright, measured joints are inside the safe envelope, and the assembled
   observation is statistically consistent with training.
2. **Ramp.** Blends from the pose the robot is actually in toward the policy's
   target over a few seconds. Without this, arming snaps every joint to the
   policy's command at whatever speed the servos manage.
3. **Envelope clamp.** Every target is clipped to the intersection of the
   calibrated travel and the model's limits.
4. **Rate limit.** Caps per-step change, so a bad action becomes a slow move
   rather than a hammer blow.
5. **Fall / fault detection.** Sustained tilt, free-fall, stale commands, lost
   servos, overtemperature, undervoltage, and a joint that never reaches what
   it was told (a stall). Any of these latches FAULT.

FAULT is terminal on purpose. An automatic retry after a fall is how a robot
gets destroyed by a loop of standing up and falling over; recovery requires a
human deciding the machine is fit to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SafetyState(Enum):
    DISARMED = "disarmed"
    RAMPING = "ramping"
    RUNNING = "running"
    FAULT = "fault"


@dataclass(frozen=True)
class Fault:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.detail}"


@dataclass
class SafetyConfig:
    # ramp
    ramp_s: float = 3.0
    # envelope
    max_joint_rate: float = 4.0          # rad/s on the commanded target
    ramp_joint_rate: float = 1.0         # rad/s while ramping in
    # fall detection
    fall_tilt_deg: float = 40.0          # sustained tilt from upright
    fall_persist_cycles: int = 3
    freefall_accel: float = 4.0          # m/s^2; |a| below this is not standing
    freefall_persist_cycles: int = 5
    # liveness
    command_timeout_s: float = 0.20
    max_consecutive_read_failures: int = 5
    # stall / tracking
    stall_error_rad: float = 0.2618      # 15 deg
    stall_persist_s: float = 2.0
    # servo health
    max_temperature_c: int = 65
    min_voltage_v: float = 9.5
    # preflight
    arm_tilt_deg: float = 12.0           # must be near upright to arm
    ood_sigma: float = 6.0               # per-element z-score bound at arm time
    ood_max_elements: int = 0            # how many may exceed it
    # what to do on fault
    fault_action: str = "torque_off"     # or "hold"


@dataclass
class SafetyStatus:
    state: SafetyState
    fault: Fault | None = None
    ramp_alpha: float = 0.0
    clamped_joints: list[str] = field(default_factory=list)
    rate_limited_joints: list[str] = field(default_factory=list)
    tilt_deg: float = 0.0


class SafetySupervisor:
    def __init__(self, joint_map, config: SafetyConfig) -> None:
        self.map = joint_map
        self.cfg = config
        self.state = SafetyState.DISARMED
        self.fault: Fault | None = None

        self._last_target: np.ndarray | None = None
        self._ramp_start: np.ndarray | None = None
        self._ramp_elapsed = 0.0
        self._tilt_strikes = 0
        self._freefall_strikes = 0
        self._read_failures = 0
        self._stall_seconds = np.zeros(len(joint_map.names))

    # -- faults ------------------------------------------------------------

    def trip(self, code: str, detail: str) -> Fault:
        """Latch a fault. Idempotent -- the first cause is the one kept."""
        if self.state is not SafetyState.FAULT:
            self.state = SafetyState.FAULT
            self.fault = Fault(code, detail)
        return self.fault

    @property
    def faulted(self) -> bool:
        return self.state is SafetyState.FAULT

    # -- preflight ---------------------------------------------------------

    def preflight(self, measured_rad, imu_sample, observation, policy) -> list[str]:
        """Everything that must hold before torque is allowed. Returns problems."""
        problems: list[str] = []

        seam = self.map.seam_violations()
        if seam:
            problems += [
                "encoder seam: " + violation.describe() for violation in seam
            ] + [
                "  -> re-zero that joint away from the wrap, or remount the horn; "
                "commanding across the wrap would drive it a full turn the wrong way"
            ]

        if imu_sample.tilt_deg > self.cfg.arm_tilt_deg:
            problems.append(
                f"robot is {imu_sample.tilt_deg:.1f} deg from upright "
                f"(must be within {self.cfg.arm_tilt_deg:.0f} deg to arm); "
                "hold it vertical"
            )

        measured = np.asarray(measured_rad, dtype=np.float64)
        outside = np.nonzero(
            (measured < self.map.safe_lower - 1e-6)
            | (measured > self.map.safe_upper + 1e-6)
        )[0]
        for index in outside:
            problems.append(
                f"{self.map.names[index]} is at {np.degrees(measured[index]):+.1f} deg, "
                f"outside its safe range "
                f"[{np.degrees(self.map.safe_lower[index]):+.1f}, "
                f"{np.degrees(self.map.safe_upper[index]):+.1f}]"
            )

        offenders = policy.out_of_distribution(observation, self.cfg.ood_sigma)
        if len(offenders) > self.cfg.ood_max_elements:
            problems.append(
                f"{len(offenders)} observation element(s) are more than "
                f"{self.cfg.ood_sigma:g} sigma from anything seen in training:"
            )
            for _, label, value, z in offenders[:8]:
                problems.append(f"    {label:<10} = {value:+9.3f}  ({z:+.1f} sigma)")
            problems.append(
                "  -> this is what a wrong IMU axis, an accelerometer still in g, "
                "a gyro still in deg/s, or a joint sign flip looks like. Fix the "
                "conditioning before energising."
            )
        return problems

    def arm(self, measured_rad, imu_sample, observation, policy) -> list[str]:
        """Attempt to arm. Empty list means armed; otherwise nothing changed."""
        if self.faulted:
            return [f"cannot arm from FAULT: {self.fault}"]
        problems = self.preflight(measured_rad, imu_sample, observation, policy)
        if problems:
            return problems
        self._ramp_start = np.asarray(measured_rad, dtype=np.float64).copy()
        self._last_target = self._ramp_start.copy()
        self._ramp_elapsed = 0.0
        self._tilt_strikes = 0
        self._freefall_strikes = 0
        self._read_failures = 0
        self._stall_seconds[:] = 0.0
        self.state = SafetyState.RAMPING
        return []

    def disarm(self) -> None:
        if not self.faulted:
            self.state = SafetyState.DISARMED
        self._last_target = None
        self._ramp_start = None

    # -- per-cycle monitors ------------------------------------------------

    def check_health(self, imu_sample, command_age_s, read_ok,
                     temperatures=None, voltages=None) -> Fault | None:
        """Run the continuous monitors. Returns a Fault if one tripped."""
        if self.faulted:
            return self.fault

        if imu_sample.tilt_deg > self.cfg.fall_tilt_deg:
            self._tilt_strikes += 1
            if self._tilt_strikes >= self.cfg.fall_persist_cycles:
                return self.trip(
                    "FALL",
                    f"tilt {imu_sample.tilt_deg:.1f} deg exceeded "
                    f"{self.cfg.fall_tilt_deg:.0f} deg for "
                    f"{self._tilt_strikes} cycles",
                )
        else:
            self._tilt_strikes = 0

        accel_magnitude = float(np.linalg.norm(imu_sample.accel))
        if accel_magnitude < self.cfg.freefall_accel:
            self._freefall_strikes += 1
            if self._freefall_strikes >= self.cfg.freefall_persist_cycles:
                return self.trip(
                    "FREEFALL",
                    f"|accel| {accel_magnitude:.2f} m/s^2 below "
                    f"{self.cfg.freefall_accel:.1f} for "
                    f"{self._freefall_strikes} cycles",
                )
        else:
            self._freefall_strikes = 0

        if read_ok:
            self._read_failures = 0
        else:
            self._read_failures += 1
            if self._read_failures >= self.cfg.max_consecutive_read_failures:
                return self.trip(
                    "BUS_LOST",
                    f"{self._read_failures} consecutive failed servo reads",
                )

        if self.state is SafetyState.RUNNING and command_age_s > self.cfg.command_timeout_s:
            return self.trip(
                "COMMAND_STALE",
                f"no policy command for {command_age_s * 1000:.0f} ms "
                f"(limit {self.cfg.command_timeout_s * 1000:.0f} ms)",
            )

        if temperatures:
            for name, celsius in temperatures.items():
                if celsius is not None and celsius > self.cfg.max_temperature_c:
                    return self.trip(
                        "OVERTEMP", f"{name} at {celsius} C (limit "
                                    f"{self.cfg.max_temperature_c} C)")
        if voltages:
            for name, volts in voltages.items():
                if volts is not None and volts < self.cfg.min_voltage_v:
                    return self.trip(
                        "UNDERVOLT", f"{name} at {volts:.1f} V (limit "
                                     f"{self.cfg.min_voltage_v:.1f} V)")
        return None

    def check_tracking(self, measured_rad, dt: float) -> Fault | None:
        """Fault a joint that is commanded somewhere it never arrives.

        This policy leans on the model's joint limits: measured in MuJoCo, it
        commands the knees and the right hip roll to their endpoint on 74-97%
        of cycles. In simulation that costs nothing -- the limit constraint
        supplies the holding force. On hardware it costs nothing only if a
        mechanical stop really sits at the calibrated endpoint. If it does not,
        the servo supplies that force itself, indefinitely, and the only
        outward sign is heat.

        A persistent gap between the commanded and the measured angle is the
        signature of that, and of every other way a joint gets stuck: a jammed
        linkage, a stripped horn, a servo that quietly browned out. In the same
        MuJoCo rollout the steady-state tracking error stays under 0.4 deg and
        peaks at 2.5 deg, so the 15 deg default has a wide margin over anything
        normal operation produces.

        Only armed once RUNNING: during RAMPING the command is deliberately
        being dragged away from where the robot is.
        """
        if self.faulted or self.state is not SafetyState.RUNNING:
            self._stall_seconds[:] = 0.0
            return None
        if self._last_target is None:
            return None

        error = np.abs(np.asarray(measured_rad, dtype=np.float64) - self._last_target)
        stuck = error > self.cfg.stall_error_rad
        self._stall_seconds = np.where(stuck, self._stall_seconds + max(dt, 0.0), 0.0)

        over = np.nonzero(self._stall_seconds >= self.cfg.stall_persist_s)[0]
        if over.size:
            worst = int(over[np.argmax(error[over])])
            return self.trip(
                "STALL",
                f"{self.map.names[worst]} has sat "
                f"{np.degrees(error[worst]):.1f} deg away from its commanded "
                f"angle for {self._stall_seconds[worst]:.1f} s "
                f"(limit {np.degrees(self.cfg.stall_error_rad):.0f} deg for "
                f"{self.cfg.stall_persist_s:.1f} s). The joint is jammed, "
                "mis-calibrated, or the servo has lost power.",
            )
        return None

    # -- command shaping ---------------------------------------------------

    def shape(self, policy_target, dt: float) -> tuple[np.ndarray | None, SafetyStatus]:
        """Clamp, ramp and rate-limit a policy target into a safe command.

        Returns ``(command_rad, status)``; a None command means *do not write*.
        """
        status = SafetyStatus(state=self.state, fault=self.fault)
        if self.faulted or self.state is SafetyState.DISARMED:
            return None, status
        if self._last_target is None:
            return None, status

        target = np.asarray(policy_target, dtype=np.float64)

        clamped = self.map.clamp(target)
        status.clamped_joints = [
            self.map.names[i]
            for i in np.nonzero(np.abs(clamped - target) > 1e-9)[0]
        ]
        target = clamped

        if self.state is SafetyState.RAMPING:
            self._ramp_elapsed += dt
            alpha = min(1.0, self._ramp_elapsed / max(self.cfg.ramp_s, 1e-3))
            target = (1.0 - alpha) * self._ramp_start + alpha * target
            status.ramp_alpha = alpha
            rate = self.cfg.ramp_joint_rate
            if alpha >= 1.0:
                self.state = SafetyState.RUNNING
        else:
            status.ramp_alpha = 1.0
            rate = self.cfg.max_joint_rate

        step = max(rate, 0.0) * max(dt, 0.0)
        delta = np.clip(target - self._last_target, -step, step)
        status.rate_limited_joints = [
            self.map.names[i]
            for i in np.nonzero(np.abs(target - self._last_target) > step + 1e-12)[0]
        ]
        command = self._last_target + delta

        # Clamp again: ramping and rate limiting both interpolate from a
        # previous value, and that value is only inside the envelope because we
        # put it there. Re-clamping makes the guarantee independent of history.
        command = self.map.clamp(command)
        self._last_target = command

        status.state = self.state
        return command, status

    @property
    def last_target(self) -> np.ndarray | None:
        return None if self._last_target is None else self._last_target.copy()
