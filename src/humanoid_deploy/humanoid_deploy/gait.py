"""The stand -> walk -> stand sequencer.

Two trained policies share one robot, and neither knows the other exists. This
decides which one is driving, blends between them, and owns the gait clock.

Deliberately free of ROS and of the serial bus, for the same reason
``safety.py`` is: every rule here is directly testable, and -- more usefully --
the identical object can be stepped inside MuJoCo. ``tools/sim_handover.py``
drives *this* class against the simulator, so the handover is validated as
deployed code rather than as a description of it.

The sequence
------------
::

    STAND --request_walk()--> TO_CROUCH --> SETTLE --> WALK --> TO_STAND --+
      ^                                                                    |
      +--------------------------------------------------------------------+

``STAND`` is the resting state and the state the robot returns to, always. The
walking policy is a guest: it is handed the robot, given a fixed number of
seconds, and taken off again. Every exit -- the timer expiring, Ctrl-C, an
operator service call, tilting too far -- runs the same ``TO_STAND`` path, so
there is exactly one way this ends and it is the one that gets tested.

Why the crouch is a separate state
----------------------------------
The walking policy's action is a residual about a crouched pose (knees ~41 deg,
hips ~23 deg), not about the standing pose. Handing it a robot with straight
legs starts it 40 deg away from anything it has seen. The training author
measured this: cold-started from the nominal pose it survived 30 s in 8 of 8
episodes; dropped into a randomised state it survived ~42%. So the robot is
walked into the crouch first, under the standing policy's fading authority, and
the walking policy takes over from the pose it was validated from -- including
``last_action = 0`` and ``phase = 0``, which is what the trainer's ``reset()``
hands the policy and therefore what "cold start" actually means.

Why nobody is ever left uncommanded
-----------------------------------
Both transitions are blends of two *live* controllers, not open-loop pose
ramps. During ``TO_CROUCH`` the standing policy is still balancing, with its
authority fading as the crouch deepens; during ``TO_STAND`` the standing policy
is taking authority back while the walking policy finishes its stride. At no
point does the robot follow a scripted trajectory with nothing watching the
IMU, which is the failure mode that would put it on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GaitError(Exception):
    """Raised when the sequencer is configured with something unusable."""


class GaitMode(Enum):
    STAND = "stand"
    TO_CROUCH = "to_crouch"
    SETTLE = "settle"
    WALK = "walk"
    TO_STAND = "to_stand"


@dataclass
class GaitConfig:
    """Timings and the velocity command. Seconds unless stated."""

    control_dt: float = 0.04

    #: Standing pose -> crouch, with the standing policy's authority fading out.
    crouch_s: float = 2.0
    #: Hold the crouch before the walking policy takes over. This is what makes
    #: the handover a *cold* start: the joints stop moving, so the velocities
    #: the policy reads on its first cycle are ~0, as they were at reset.
    settle_s: float = 0.75
    #: How long to walk before returning to standing, unprompted.
    walk_duration_s: float = 5.0
    #: Walking -> standing, with the standing policy's authority fading in.
    recover_s: float = 2.0

    #: Gait clock, from the bundle. Advanced once per control cycle.
    phase_increment: float = 0.0333333

    #: Peak velocity command for obs[50:53]. Reached by ramping, not stepping.
    cmd_vx: float = 0.085
    #: Seconds to ramp the velocity command between 0 and ``cmd_vx``.
    #:
    #: This is what makes the whole sequence work on a compliant robot. The
    #: walking policy is trained with ``zero_cmd_prob`` > 0, so a zero command
    #: is a behaviour it has: stand still, in the crouch, feet together. That
    #: turns both handovers into something it already knows -- it takes the
    #: robot at zero command, is asked to accelerate, and is brought back to
    #: zero before the standing policy takes over. Nothing has to be blended
    #: across a stride.
    cmd_ramp_s: float = 1.0
    #: Ramp the velocity command up *during* the descent into the crouch,
    #: rather than holding zero until the crouch is reached.
    #:
    #: Off by default. It was added because no checkpoint could hold a zero
    #: command at what was then believed to be the robot's stiffness (kp 10) --
    #: but that stiffness figure turned out to be an artifact of a clamped
    #: ankle, and at the real kp the crouch is stable and the zero-command hold
    #: is the better design: the robot comes to a stop before either policy
    #: changes hands. Turn it on only if a checkpoint again cannot hold still.
    cmd_during_crouch: bool = False
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0

    #: Legal command range, from the bundle. Commands outside it are refused --
    #: vx in particular, because 0.0 was never trained and is the value someone
    #: will reach for when they want it to stop.
    cmd_vx_range: tuple[float, float] = (0.05, 0.12)
    cmd_wz_range: tuple[float, float] = (-0.3, 0.3)

    #: Stop only when the gait clock is near this phase, or ``None`` to stop on
    #: the cycle the request arrives. Capped by ``max_stop_wait_s`` regardless.
    stop_phase: float | None = None
    max_stop_wait_s: float = 1.5

    #: How to get from the standing pose into the crouch:
    #:
    #: ``ramp``         interpolate from the pose held at entry to the nominal
    #:                  pose, open loop.
    #: ``blend_stand``  interpolate from the standing policy's live target to
    #:                  the nominal pose.
    #: ``blend_walk``   interpolate from the standing policy's live target to
    #:                  the walking policy's live target, clock running.
    #:
    #: ``blend_stand`` reads as the safest of the three -- a balance controller
    #: stays live all the way down -- and in simulation it is the one that
    #: falls over. The standing policy saturates fighting the descent it cannot
    #: see the reason for, and averaging a saturated controller with a pose
    #: target tracks neither. See tools/sim_handover.py --sweep crouch-style.
    crouch_style: str = "blend_walk"

    #: How to get back:
    #:
    #: ``blend``       fade the standing policy in while the walking policy
    #:                 finishes its stride.
    #: ``freeze``      hold the last walking target instead, which parks a
    #:                 swing foot in the air for the duration.
    #: ``via_crouch``  reverse the entry: settle the walk into the static
    #:                 crouch over the first half, then stand up out of it over
    #:                 the second. The crouch is passively stable, so this puts
    #:                 the robot somewhere it can simply be left before asking
    #:                 the standing policy -- which has never seen a walking
    #:                 robot -- to take it over.
    recover_style: str = "blend"

    def validate(self) -> None:
        lo, hi = self.cmd_vx_range
        if not lo - 1e-9 <= self.cmd_vx <= hi + 1e-9:
            raise GaitError(
                f"cmd_vx {self.cmd_vx} is outside the trained range "
                f"[{lo}, {hi}] m/s. In particular 0.0 is not a stop command: "
                "this policy was never trained with a zero velocity command "
                "and has no standstill behaviour. Stopping is what TO_STAND "
                "is for."
            )
        lo, hi = self.cmd_wz_range
        if not lo - 1e-9 <= self.cmd_wz <= hi + 1e-9:
            raise GaitError(
                f"cmd_wz {self.cmd_wz} is outside the trained range [{lo}, {hi}]"
            )
        if abs(self.cmd_vy) > 1e-9:
            raise GaitError(
                "cmd_vy was never trained non-zero; leave it at 0.0")
        if self.control_dt <= 0.0:
            raise GaitError("control_dt must be positive")
        for name in ("crouch_s", "recover_s"):
            if getattr(self, name) <= 0.0:
                raise GaitError(f"{name} must be positive -- a zero-length "
                                "blend is a step change in every leg joint")
        if self.walk_duration_s <= 0.0:
            raise GaitError("walk_duration_s must be positive")
        if self.cmd_ramp_s <= 0.0:
            raise GaitError("cmd_ramp_s must be positive -- stepping the "
                            "velocity command is a disturbance the policy has "
                            "not seen")
        if self.crouch_style not in ("ramp", "blend_stand", "blend_walk"):
            raise GaitError(f"unknown crouch_style {self.crouch_style!r}")
        if self.recover_style not in ("blend", "freeze", "via_crouch"):
            raise GaitError(f"unknown recover_style {self.recover_style!r}")


@dataclass
class GaitStatus:
    mode: GaitMode
    phase: float
    #: Authority of the *walking* policy, 0..1. Also the blend weight.
    walk_authority: float
    #: Seconds left in the current mode, where that is meaningful.
    remaining_s: float
    #: Velocity command actually applied this cycle, m/s.
    cmd_vx: float = 0.0
    #: Set once a stop has been asked for but not yet begun.
    stop_reason: str | None = None
    #: True on the cycle the walking policy is handed the robot.
    walk_started: bool = False
    #: True on the cycle the robot is back under the standing policy alone.
    walk_finished: bool = False


class GaitSequencer:
    """Owns the mode, the gait clock, and the blend between the two policies.

    Call ``step()`` once per control cycle with each policy's target, in
    radians, already put through its own control law. It returns the target to
    command. The caller still clamps that to the safety envelope -- this class
    decides *what* to ask for, never whether it is allowed.
    """

    def __init__(self, config: GaitConfig, nominal_pose_rad) -> None:
        config.validate()
        self.cfg = config
        self.nominal = np.asarray(nominal_pose_rad, dtype=np.float64)
        self.mode = GaitMode.STAND
        self.phase = 0.0
        self._elapsed = 0.0
        self._stop_reason: str | None = None
        self._stop_waited = 0.0
        self._walk_started = False
        #: Pose held at the moment a blend began, for the open-loop styles.
        self._blend_from: np.ndarray | None = None
        #: Velocity command as a fraction of ``cmd_vx``, slewed not stepped.
        self._cmd_scale = 0.0
        #: Reason the last walk ended, for the log.
        self.last_exit: str | None = None

    # -- requests ----------------------------------------------------------

    def request_walk(self) -> tuple[bool, str]:
        if self.mode is not GaitMode.STAND:
            return False, f"already {self.mode.value}; wait for it to finish"
        self.mode = GaitMode.TO_CROUCH
        self._elapsed = 0.0
        self._stop_reason = None
        self._stop_waited = 0.0
        self._walk_started = False
        self._blend_from = None
        self._cmd_scale = 0.0
        self.phase = 0.0
        return True, (f"crouching over {self.cfg.crouch_s:.1f}s, then walking "
                      f"{self.cfg.walk_duration_s:.1f}s at "
                      f"{self.cfg.cmd_vx:.3f} m/s")

    def request_stop(self, reason: str) -> tuple[bool, str]:
        """Ask to return to standing. Idempotent; the first reason is kept.

        Safe to call from a signal handler: it only sets a flag, and the work
        happens on the next control cycle. Doing anything more from a handler
        would mean touching the serial bus from a context that can interrupt a
        write halfway through a packet.
        """
        if self.mode is GaitMode.STAND:
            return False, "already standing"
        if self._stop_reason is not None:
            return True, f"stop already pending ({self._stop_reason})"
        self._stop_reason = reason
        return True, f"stopping: {reason}"

    @property
    def stop_pending(self) -> bool:
        return self._stop_reason is not None

    @property
    def walking(self) -> bool:
        """Does the caller need to evaluate the walking policy this cycle?

        True whenever its output is used -- which includes the descent under
        ``crouch_style='blend_walk'``, where it is already part of the blend.
        """
        if self.mode in (GaitMode.TO_CROUCH, GaitMode.SETTLE):
            return self.cfg.crouch_style == "blend_walk"
        return self.mode in (GaitMode.WALK, GaitMode.TO_STAND)

    # -- the clock ---------------------------------------------------------

    def command(self) -> np.ndarray:
        """The velocity command for obs[50:53], as ramped this cycle.

        Zero everywhere except during the walk, and it slews between the two.
        A step change in this input is a disturbance the policy only ever saw
        at ``cmd_resample_s`` boundaries in training, and asking a compliant
        robot to go from standing still to commanded speed in one cycle is the
        kind of thing that shows up as a stumble on hardware and nowhere else.
        """
        return np.array([self._cmd_scale * self.cfg.cmd_vx,
                         self.cfg.cmd_vy,
                         self._cmd_scale * self.cfg.cmd_wz])

    @property
    def stopped(self) -> bool:
        """True once the velocity command has been ramped all the way down."""
        return self._cmd_scale <= 1e-6

    def _slew_command(self, want: float, dt: float) -> None:
        step = dt / max(self.cfg.cmd_ramp_s, 1e-6)
        if want > self._cmd_scale:
            self._cmd_scale = min(want, self._cmd_scale + step)
        else:
            self._cmd_scale = max(want, self._cmd_scale - step)

    def phase_features(self) -> np.ndarray:
        """``[sin(2*pi*phase), cos(2*pi*phase)]`` for obs[48:50]."""
        angle = 2.0 * np.pi * self.phase
        return np.array([np.sin(angle), np.cos(angle)])

    def _near_stop_phase(self) -> bool:
        """Is the clock within half a cycle's travel of the chosen phase?

        Only a scheduling preference, never a requirement: measured on this
        checkpoint the feet agree with the clock 56.5% of the time against a
        50% chance baseline, so the phase says much less about which foot is
        down than the schedule implies. ``max_stop_wait_s`` is what actually
        bounds the delay.
        """
        if self.cfg.stop_phase is None:
            return True
        gap = abs((self.phase - self.cfg.stop_phase + 0.5) % 1.0 - 0.5)
        return gap <= 0.5 * self.cfg.phase_increment + 1e-9

    # -- the cycle ---------------------------------------------------------

    def step(self, stand_target, walk_target) -> tuple[np.ndarray, GaitStatus]:
        """Advance one control cycle and return the target to command.

        :param stand_target: the standing policy's target, radians. Required in
            every mode -- it is the fallback the robot is always blended back
            toward, so it must be live even while the walking policy drives.
        :param walk_target: the walking policy's target, radians. May be
            ``None`` in modes where the walking policy is not running.
        """
        dt = self.cfg.control_dt
        stand_target = np.asarray(stand_target, dtype=np.float64)
        started = finished = False

        # The velocity command is full only while walking, and only while no
        # stop is pending and there is more than one ramp left on the clock.
        # Everywhere else it slews to zero, which for this policy means "stand
        # still in the crouch" -- a behaviour it has, because the run that
        # produced it used zero_cmd_prob 0.15.
        want = 0.0
        if self.mode is GaitMode.WALK and self._stop_reason is None:
            left = self.cfg.walk_duration_s - self._elapsed
            want = 0.0 if left <= self.cfg.cmd_ramp_s else 1.0
        elif (self.cfg.cmd_during_crouch and self._stop_reason is None
                and self.cfg.crouch_style == "blend_walk"
                and self.mode in (GaitMode.TO_CROUCH, GaitMode.SETTLE)):
            want = 1.0
        self._slew_command(want, dt)

        if self.mode is GaitMode.STAND:
            target, authority, remaining = stand_target, 0.0, 0.0

        elif self.mode is GaitMode.TO_CROUCH:
            if self._blend_from is None:
                # The standing policy's target, not the measured pose: the ramp
                # then starts from the angle already being commanded, so the
                # first cycle of the descent is not itself a step change.
                self._blend_from = stand_target.copy()
            self._elapsed += dt
            alpha = min(1.0, self._elapsed / self.cfg.crouch_s)
            authority = 0.0
            if self.cfg.crouch_style == "ramp":
                target = (1.0 - alpha) * self._blend_from + alpha * self.nominal
            elif self.cfg.crouch_style == "blend_stand":
                target = (1.0 - alpha) * stand_target + alpha * self.nominal
            else:                                   # blend_walk
                if walk_target is None:
                    raise GaitError("crouch_style 'blend_walk' needs a walk target")
                target = (1.0 - alpha) * stand_target + alpha * np.asarray(
                    walk_target, dtype=np.float64)
                authority = alpha
                self.phase = (self.phase + self.cfg.phase_increment) % 1.0
            remaining = max(0.0, self.cfg.crouch_s - self._elapsed)
            if self._stop_reason is not None:
                # Abandoning the descent: there is no walking policy to hand
                # back from, so go straight to the standing policy and let the
                # recover blend lift it out of a partial crouch.
                self._enter_to_stand()
            elif alpha >= 1.0:
                self.mode, self._elapsed = GaitMode.SETTLE, 0.0

        elif self.mode is GaitMode.SETTLE:
            self._elapsed += dt
            if self.cfg.crouch_style == "blend_walk":
                # The walking policy owns the crouch, at zero command. It is
                # the only controller trained around this pose, and at the
                # robot's measured stiffness the crouch is NOT passively
                # stable -- open loop it sags from 6.8 deg of lean to 23 deg
                # in under a second. Something has to be holding it.
                if walk_target is None:
                    raise GaitError("crouch_style 'blend_walk' needs a walk "
                                    "target in SETTLE")
                target = np.asarray(walk_target, dtype=np.float64)
                authority = 1.0
                self.phase = (self.phase + self.cfg.phase_increment) % 1.0
            else:
                target, authority = self.nominal, 0.0
            remaining = max(0.0, self.cfg.settle_s - self._elapsed)
            if self._stop_reason is not None:
                self._enter_to_stand()
            elif self._elapsed >= self.cfg.settle_s:
                # Cold start, exactly as the trainer's reset() defines it: at
                # the nominal pose, joints at rest, clock at zero. The caller
                # zeroes the walking policy's last_action on seeing
                # walk_started, which is the remaining half of that definition.
                self.mode, self._elapsed = GaitMode.WALK, 0.0
                if self.cfg.crouch_style != "blend_walk":
                    self.phase = 0.0
                    self._walk_started = started = True

        elif self.mode is GaitMode.WALK:
            if walk_target is None:
                raise GaitError("no walking target while in WALK")
            self._elapsed += dt
            self.phase = (self.phase + self.cfg.phase_increment) % 1.0
            target, authority = np.asarray(walk_target, dtype=np.float64), 1.0
            remaining = max(0.0, self.cfg.walk_duration_s - self._elapsed)
            if self._elapsed >= self.cfg.walk_duration_s:
                self.request_stop("walk duration reached")
            # Begin slowing one ramp before the end, so the timer expiring and
            # a Ctrl-C take the same path: command to zero, come to a stop,
            # and only then hand the robot over.
            if self._stop_reason is not None:
                self._stop_waited += dt
                if (self.stopped
                        and (self._near_stop_phase()
                             or self._stop_waited >= self.cfg.max_stop_wait_s)):
                    self._enter_to_stand()

        elif self.mode is GaitMode.TO_STAND:
            self._elapsed += dt
            alpha = min(1.0, self._elapsed / self.cfg.recover_s)
            authority = 1.0 - alpha
            if walk_target is None:
                target = stand_target
            elif self.cfg.recover_style == "via_crouch":
                # Mirror of the entry, in reverse. First half: bring the stride
                # to rest at the nominal pose, with the walking policy's share
                # fading so it is still the thing reacting while it slows.
                # Second half: leave the crouch under the standing policy,
                # which is the same open-loop-ish move the entry made and which
                # ends exactly where that policy expects the robot to be.
                walk_target = np.asarray(walk_target, dtype=np.float64)
                if alpha < 0.5:
                    beta = alpha * 2.0
                    target = (1.0 - beta) * walk_target + beta * self.nominal
                    authority = 1.0 - beta
                    self.phase = (self.phase + self.cfg.phase_increment) % 1.0
                else:
                    beta = (alpha - 0.5) * 2.0
                    target = (1.0 - beta) * self.nominal + beta * stand_target
                    authority = 0.0
            elif self.cfg.recover_style == "freeze":
                # Hold whatever the walking policy last asked for and fade the
                # standing policy in over the top of it.
                if self._blend_from is None:
                    self._blend_from = np.asarray(walk_target, dtype=np.float64).copy()
                target = authority * self._blend_from + alpha * stand_target
            else:
                # Both controllers stay live. The walking policy finishes
                # whatever stride it is in rather than parking a swing foot in
                # the air, and the standing policy takes the robot over as its
                # share rises.
                walk_target = np.asarray(walk_target, dtype=np.float64)
                target = authority * walk_target + alpha * stand_target
                self.phase = (self.phase + self.cfg.phase_increment) % 1.0
            remaining = max(0.0, self.cfg.recover_s - self._elapsed)
            if alpha >= 1.0:
                self.mode, self._elapsed = GaitMode.STAND, 0.0
                self.last_exit = self._stop_reason
                self._stop_reason = None
                self._stop_waited = 0.0
                self._walk_started = False
                self._blend_from = None
                finished = True

        else:  # pragma: no cover - the enum is closed
            raise GaitError(f"unreachable mode {self.mode}")

        return target, GaitStatus(
            mode=self.mode,
            phase=self.phase,
            walk_authority=float(authority),
            remaining_s=float(remaining),
            cmd_vx=float(self._cmd_scale * self.cfg.cmd_vx),
            stop_reason=self._stop_reason,
            walk_started=started,
            walk_finished=finished,
        )

    def _enter_to_stand(self) -> None:
        self.mode, self._elapsed = GaitMode.TO_STAND, 0.0
        self._blend_from = None

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        return (
            f"gait: crouch {self.cfg.crouch_s:.1f}s -> settle "
            f"{self.cfg.settle_s:.2f}s -> walk {self.cfg.walk_duration_s:.1f}s "
            f"at vx={self.cfg.cmd_vx:.3f} wz={self.cfg.cmd_wz:+.2f} -> recover "
            f"{self.cfg.recover_s:.1f}s\n"
            f"  clock {self.cfg.phase_increment:.6f}/cycle "
            f"({self.cfg.control_dt * 1e3:.0f} ms) = "
            f"{self.cfg.control_dt / self.cfg.phase_increment:.2f}s period; "
            f"stop_phase={self.cfg.stop_phase}"
        )
