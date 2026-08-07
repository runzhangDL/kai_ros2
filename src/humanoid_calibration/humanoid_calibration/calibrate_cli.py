"""Interactive, read-only motor calibration for the 13-DoF humanoid.

Run once per robot (or after any mechanical work on a joint)::

    ros2 run humanoid_calibration calibrate

For each joint, in the order given by ``joint_limits.yaml``:

  1. ZERO -- the live encoder value is shown; you hand-position the link so it
     is vertical to the ground and press ENTER. That reading becomes 0 deg.
  2. MINIMUM -- the target from the config is shown together with the live
     angle relative to the zero you just set. Move the joint towards the
     minimum; once you hold it at or past the target it is recorded
     automatically.
  3. MAXIMUM -- same, in the other direction.

The joint is written to the persistent calibration file as soon as it is
finished, so an interrupted session never loses completed work.

Nothing is ever written to a servo. See ``feetech_bus`` for why that matters
and what it implies about torque.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from . import term_ui as ui
from .calibration_store import CalibrationStore, JointCalibration, utc_now
from .feetech_bus import (
    AngleTracker,
    BusError,
    FeetechReader,
    MockReader,
    counts_to_deg,
)
from .joint_config import ConfigError, JointConfig, RobotConfig, load_config

# Outcomes of a per-joint interaction.
_DONE = "done"
_SKIP = "skip"
_QUIT = "quit"
_REDO = "redo"


class _Aborted(Exception):
    """Operator asked to quit."""


def _c(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{ui.RESET}" if enabled else text


class Calibrator:
    def __init__(
        self,
        config: RobotConfig,
        store: CalibrationStore,
        reader,
        terminal: ui.RawTerminal,
        mock: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.reader = reader
        self.term = terminal
        self.mock = mock
        self.color = ui.supports_color()
        self.cpr = config.servo.counts_per_rev
        self.tuning = config.calibration
        self._period = 1.0 / max(1.0, config.calibration.refresh_hz)

    # -- helpers -----------------------------------------------------------

    def _paint(self, text: str, color: str) -> str:
        return _c(text, color, self.color)

    def _health_line(self, joint: JointConfig, cache: dict) -> str:
        """Voltage/temperature, refreshed at ~2 Hz so it never stalls the loop."""
        now = time.monotonic()
        if now - cache.get("at", -99.0) > 0.5:
            cache["at"] = now
            cache["value"] = self.reader.read_health(joint.id)
        volts, celsius = cache.get("value", (None, None))
        parts = []
        parts.append(f"{volts:.1f} V" if volts is not None else "-- V")
        parts.append(f"{celsius} C" if celsius is not None else "-- C")
        return "  ".join(parts)

    def _handle_mock_key(self, key: str | None, joint: JointConfig) -> bool:
        """Let the mock bus be driven from the keyboard. Returns True if consumed."""
        if not self.mock or key is None:
            return False
        step = {
            ui.KEY_LEFT: -6,
            ui.KEY_RIGHT: 6,
            "j": -6,
            "k": 6,
            "J": -60,
            "K": 60,
        }.get(key)
        if step is None:
            return False
        self.reader.nudge(joint.id, step)
        return True

    def _header(self, joint: JointConfig, index: int, total: int) -> list[str]:
        title = f" JOINT {index}/{total}   {joint.name}"
        meta = f"servo id {joint.id}   range {joint.min_deg:+.1f} .. {joint.max_deg:+.1f} deg "
        pad = max(1, ui.term_width() - ui.visible_len(title) - len(meta))
        return [
            ui.rule("=", ui.CYAN if self.color else ""),
            self._paint(title, ui.BOLD) + " " * pad + self._paint(meta, ui.DIM),
            ui.rule("=", ui.CYAN if self.color else ""),
        ]

    # -- stage 1: zero -----------------------------------------------------

    def _capture_zero(
        self, joint: JointConfig, index: int, total: int, tracker: AngleTracker
    ) -> tuple[str, float | None]:
        health: dict = {}
        block = ui.LiveBlock()
        misses = 0

        while True:
            raw = self.reader.read_raw(joint.id)
            if raw is None:
                misses += 1
            else:
                misses = 0
                tracker.update(raw)

            unwrapped = tracker.unwrapped
            lines = self._header(joint, index, total)
            lines += [
                "",
                self._paint(" STEP 1/3   SET ZERO", ui.BOLD),
                "",
                "   Hand-position the link so it is "
                + self._paint("vertical to the ground", ui.BOLD)
                + ", then press ENTER.",
                "",
            ]
            if raw is None:
                lines.append(
                    "   " + self._paint(f"no reply from servo {joint.id} "
                                        f"({misses} reads missed)", ui.RED)
                )
                lines.append("")
                lines.append("")
            else:
                turn_deg = counts_to_deg(unwrapped or 0.0, self.cpr)
                lines.append(f"   raw counts   {self._paint(f'{raw:5d}', ui.BOLD)}"
                             f"   of {self.cpr}")
                lines.append(f"   shaft angle  {turn_deg:+8.2f} deg  (absolute, "
                             f"not yet referenced)")
                lines.append(f"   servo        {self._health_line(joint, health)}")
            lines += [
                "",
                "   " + self._paint("[ENTER]", ui.GREEN) + " set this as 0 deg    "
                + self._paint("[s]", ui.YELLOW) + " skip joint    "
                + self._paint("[q]", ui.RED) + " quit",
            ]
            block.draw(lines)

            key = self.term.read_key(self._period)
            if self._handle_mock_key(key, joint):
                continue
            if key == ui.KEY_ENTER:
                if raw is None or unwrapped is None:
                    continue  # refuse to zero on a dead read
                block.finish()
                return _DONE, unwrapped
            if key in ("s", "S"):
                block.finish()
                return _SKIP, None
            if key in ("q", "Q"):
                block.finish()
                return _QUIT, None

    # -- stages 2 & 3: limits ---------------------------------------------

    def _capture_limit(
        self,
        joint: JointConfig,
        index: int,
        total: int,
        tracker: AngleTracker,
        zero_unwrapped: float,
        which: str,
    ) -> tuple[str, float | None]:
        is_min = which == "min"
        target_deg = joint.min_deg if is_min else joint.max_deg
        step_no = 2 if is_min else 3
        label = "MINIMUM" if is_min else "MAXIMUM"
        tol = self.tuning.reach_tolerance_deg

        def angle_of(unwrapped: float) -> float:
            return joint.direction * counts_to_deg(unwrapped - zero_unwrapped, self.cpr)

        def reached(angle: float) -> bool:
            return angle <= target_deg + tol if is_min else angle >= target_deg - tol

        # A limit of exactly 0 deg (e.g. a knee that does not hyper-extend) is
        # already satisfied at the zero pose, so auto-detection would fire
        # instantly. Fall back to an explicit confirmation in that case.
        manual = reached(0.0)

        # Bar track spans the full range with a margin so overshoot stays visible.
        margin = max(5.0, 0.15 * (joint.max_deg - joint.min_deg))
        track_low, track_high = joint.min_deg - margin, joint.max_deg + margin

        block = ui.LiveBlock()
        reached_since: float | None = None
        health: dict = {}

        while True:
            raw = self.reader.read_raw(joint.id)
            if raw is not None:
                tracker.update(raw)
            unwrapped = tracker.unwrapped
            angle = angle_of(unwrapped) if unwrapped is not None else None

            now = time.monotonic()
            if angle is not None and not manual and reached(angle):
                reached_since = reached_since if reached_since is not None else now
                if now - reached_since >= self.tuning.dwell_s:
                    block.finish()
                    return _DONE, unwrapped
            else:
                reached_since = None

            lines = self._header(joint, index, total)
            lines += [
                "",
                self._paint(f" STEP {step_no}/3   {label} LIMIT", ui.BOLD)
                + self._paint(
                    f"      target {target_deg:+.2f} deg "
                    f"({target_deg * 3.141592653589793 / 180.0:+.4f} rad)",
                    ui.CYAN,
                ),
                "",
            ]

            if angle is None:
                lines += [
                    "   " + self._paint(f"no reply from servo {joint.id}", ui.RED),
                    "",
                    "",
                    "",
                ]
            else:
                remaining = (angle - target_deg) if is_min else (target_deg - angle)
                direction_word = "negative" if is_min else "positive"
                if reached(angle):
                    status = self._paint("AT TARGET -- hold still", ui.GREEN)
                else:
                    status = f"{remaining:6.2f} deg further in the {direction_word} direction"

                marks = {
                    0.0: "0",
                    joint.min_deg: "<",
                    joint.max_deg: ">",
                }
                bar = ui.span_bar(angle, track_low, track_high, marks)
                lines += [
                    f"   angle        {self._paint(f'{angle:+8.2f} deg', ui.BOLD)}"
                    f"   ({angle * 3.141592653589793 / 180.0:+.4f} rad)",
                    f"   remaining    {status}",
                    f"   {bar}",
                    f"   {'':13}{self._paint('< = min   0 = zero   > = max   # = you', ui.DIM)}",
                ]

            lines.append("")
            if manual:
                lines.append(
                    "   " + self._paint("[ENTER]", ui.GREEN)
                    + f" confirm {label.lower()} here    "
                    + self._paint("[r]", ui.YELLOW) + " redo joint    "
                    + self._paint("[s]", ui.YELLOW) + " skip    "
                    + self._paint("[q]", ui.RED) + " quit"
                )
                lines.append(
                    "   " + self._paint(
                        f"({label.lower()} is {target_deg:+.1f} deg, i.e. the zero pose "
                        "itself -- confirm manually)", ui.DIM)
                )
            else:
                lines.append(
                    "   " + self._paint(
                        f"records automatically after {self.tuning.dwell_s:.2f} s "
                        f"at or past the target (+/-{tol:.1f} deg)", ui.DIM)
                )
                lines.append(
                    "   " + self._paint("[r]", ui.YELLOW) + " redo joint    "
                    + self._paint("[s]", ui.YELLOW) + " skip    "
                    + self._paint("[q]", ui.RED) + " quit"
                )
            block.draw(lines)

            key = self.term.read_key(self._period)
            if self._handle_mock_key(key, joint):
                continue
            if manual and key == ui.KEY_ENTER and unwrapped is not None:
                block.finish()
                return _DONE, unwrapped
            if key in ("r", "R"):
                block.finish()
                return _REDO, None
            if key in ("s", "S"):
                block.finish()
                return _SKIP, None
            if key in ("q", "Q"):
                block.finish()
                return _QUIT, None

    # -- one joint ---------------------------------------------------------

    def run_joint(self, joint: JointConfig, index: int, total: int) -> str:
        while True:
            tracker = AngleTracker(self.cpr)
            self.term.drain()

            outcome, zero_unwrapped = self._capture_zero(joint, index, total, tracker)
            if outcome != _DONE or zero_unwrapped is None:
                return outcome
            print(f"   {self._paint('zero set', ui.GREEN)} at raw "
                  f"{int(round(zero_unwrapped)) % self.cpr}\n")

            def angle_at(unwrapped: float) -> float:
                offset = unwrapped - zero_unwrapped
                return joint.direction * counts_to_deg(offset, self.cpr)

            limits: dict[str, float] = {}
            restart = False
            for which, word in (("min", "minimum"), ("max", "maximum")):
                outcome, unwrapped = self._capture_limit(
                    joint, index, total, tracker, zero_unwrapped, which
                )
                if outcome == _REDO:
                    print(self._paint("   restarting this joint\n", ui.YELLOW))
                    restart = True
                    break
                if outcome != _DONE or unwrapped is None:
                    return outcome
                limits[which] = unwrapped
                print(f"   {self._paint(word + ' recorded', ui.GREEN)} at "
                      f"{angle_at(unwrapped):+.2f} deg\n")
            if restart:
                continue

            calibration = JointCalibration(
                name=joint.name,
                servo_id=joint.id,
                zero_raw=int(round(zero_unwrapped)) % self.cpr,
                min_raw=int(round(limits["min"])) % self.cpr,
                max_raw=int(round(limits["max"])) % self.cpr,
                min_deg=joint.min_deg,
                max_deg=joint.max_deg,
                direction=joint.direction,
                counts_per_rev=self.cpr,
                calibrated_at=utc_now(),
            )
            self.store.record(calibration, self.config.fingerprint, self.config.source_path)
            self.store.save()
            print(self._paint(f"   saved {joint.name} -> {self.store.path}", ui.GREEN))
            print()
            return _DONE


# -- preflight ------------------------------------------------------------


def preflight(config: RobotConfig, reader, color: bool) -> list[JointConfig]:
    """Check every configured servo answers, and answers *reliably*.

    Modelled on ``sts_tool.py bustest``: a single successful ping proves the
    wiring but says nothing about whether the baud rate holds up over this
    robot's cable lengths and motor count. A bus that drops one read in twenty
    still lets calibration finish -- it just records limits from whatever
    sample happened to land, which is exactly the silent corruption this tool
    exists to avoid. So we sample repeatedly and treat anything short of the
    configured success rate as a failure.

    Returns the joints that did not meet the bar.
    """
    reads = max(1, config.calibration.preflight_reads)
    threshold = config.calibration.min_read_success
    print(_c(f"Checking the bus (read-only, {reads} reads per servo)...", ui.BOLD, color))

    bad: list[JointConfig] = []
    for joint in config.joints:
        samples = [reader.read_raw(joint.id) for _ in range(reads)]
        good = [s for s in samples if s is not None]
        rate = len(good) / reads

        if not good:
            bad.append(joint)
            print(f"  id {joint.id:>3}  {joint.name:<20} "
                  + _c("NO REPLY", ui.RED, color))
            continue

        shaft = counts_to_deg(good[-1], config.servo.counts_per_rev)
        detail = f"  raw {good[-1]:5d}  ({shaft:7.2f} deg)"
        if rate < threshold:
            bad.append(joint)
            status = _c(f"FLAKY {len(good)}/{reads}", ui.RED, color)
        else:
            status = _c(f"ok {len(good)}/{reads}", ui.GREEN, color)
        print(f"  id {joint.id:>3}  {joint.name:<20} {status}{detail}")
    print()
    return bad


def print_status(config: RobotConfig, store: CalibrationStore, color: bool) -> None:
    report = store.validate(config)
    print(_c("Calibration status", ui.BOLD, color))
    print(f"  file          {store.path}")
    print(f"  config        {config.source_path}")
    print(f"  updated       {store.updated_at or '(never)'}")
    print()
    for joint in config.joints:
        stored = store.joints.get(joint.name)
        if stored is None:
            mark = _c("NOT CALIBRATED", ui.RED, color)
            detail = ""
        elif joint.name in report.stale:
            mark = _c("STALE", ui.YELLOW, color)
            detail = "  (config changed since calibration)"
        else:
            mark = _c("ok", ui.GREEN, color)
            detail = (
                f"  zero {stored.zero_raw:5d}  min {stored.min_raw:5d}  "
                f"max {stored.max_raw:5d}  {stored.calibrated_at}"
            )
        print(f"  {joint.name:<20} {mark}{detail}")
    print()
    verdict = _c("READY", ui.GREEN, color) if report.ok else _c("BLOCKED", ui.RED, color)
    print(f"  {verdict}: {report.reason()}")


# -- entry point ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Interactive read-only zero/limit calibration for the humanoid's servos.",
    )
    parser.add_argument("--config", help="path to joint_limits.yaml")
    parser.add_argument("--store", help="path to the calibration file to write")
    parser.add_argument("--port", help="override the serial port from the config")
    parser.add_argument("--baud", type=int, help="override the baud rate from the config")
    parser.add_argument(
        "--only",
        metavar="JOINT",
        nargs="+",
        help="calibrate only these joints (implies --force for them)",
    )
    parser.add_argument(
        "--start-at", metavar="JOINT", help="begin the sequence at this joint"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recalibrate joints that are already recorded",
    )
    parser.add_argument(
        "--status", action="store_true", help="print calibration status and exit"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="simulate the bus; drive the fake encoder with arrow keys or j/k",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    color = ui.supports_color()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(_c(f"config error: {exc}", ui.RED, color), file=sys.stderr)
        return 2

    # Bus overrides do not affect the fingerprint, so a calibration stays valid
    # across a swapped USB adapter.
    overrides = {}
    if args.port:
        overrides["port"] = args.port
    if args.baud:
        overrides["baudrate"] = args.baud
    if overrides:
        config = replace(config, bus=replace(config.bus, **overrides))

    store = CalibrationStore.load(args.store)

    if args.status:
        print_status(config, store, color)
        return 0 if store.validate(config).ok else 1

    # Work out the joint sequence.
    joints = list(config.joints)
    if args.start_at:
        names = [j.name for j in joints]
        if args.start_at not in names:
            print(_c(f"unknown joint '{args.start_at}'", ui.RED, color), file=sys.stderr)
            return 2
        joints = joints[names.index(args.start_at):]
    if args.only:
        unknown = [n for n in args.only if n not in config.joint_names]
        if unknown:
            print(_c(f"unknown joints: {', '.join(unknown)}", ui.RED, color), file=sys.stderr)
            return 2
        joints = [j for j in joints if j.name in args.only]

    report = store.validate(config)
    if not (args.force or args.only):
        already = {j.name for j in joints if j.name in store.joints and j.name not in report.stale}
        if already:
            joints = [j for j in joints if j.name not in already]
            print(
                _c(f"skipping {len(already)} already-calibrated joint(s); "
                   "use --force to redo them", ui.DIM, color)
            )

    if not joints:
        print(_c("nothing to calibrate -- every joint is already done.", ui.GREEN, color))
        print_status(config, store, color)
        return 0

    # Open the bus.
    try:
        reader = MockReader(config.servo) if args.mock else FeetechReader(config.bus, config.servo)
    except BusError as exc:
        print(_c(f"bus error: {exc}", ui.RED, color), file=sys.stderr)
        print(
            "\nCheck that the adapter is plugged in, that the port in "
            f"{config.source_path} is right, and that your user is in the "
            "'dialout' group (`sudo usermod -aG dialout $USER`, then log out "
            "and back in).",
            file=sys.stderr,
        )
        return 3

    print()
    print(ui.rule("=", ui.CYAN if color else ""))
    print(_c("  HUMANOID MOTOR CALIBRATION", ui.BOLD, color)
          + _c("   (read-only: no servo is ever written to)", ui.DIM, color))
    print(ui.rule("=", ui.CYAN if color else ""))
    print(f"  config       {config.source_path}")
    print(f"  calibration  {store.path}")
    bus_label = "MOCK" if args.mock else f"{config.bus.port} @ {config.bus.baudrate}"
    print(f"  bus          {bus_label}")
    print(f"  joints       {len(joints)} to do, in order: "
          f"{', '.join(j.name for j in joints)}")
    print()
    ids = ",".join(str(j.id) for j in config.joints)
    print(_c(
        "  This tool only reads, so servo torque is left exactly as you powered up.\n"
        "  If the joints do not back-drive freely by hand, release torque first:\n"
        f"    for i in {{{ids}}}; do python3 sts_tool.py --port {config.bus.port} "
        f"--baud {config.bus.baudrate} torque $i 0; done",
        ui.YELLOW, color))
    print()

    exit_code = 0
    try:
        bad = preflight(config, reader, color)
        if bad:
            names = ", ".join(f"{j.name}(id {j.id})" for j in bad)
            print(_c(f"Bus check failed for: {names}", ui.RED, color))
            print(
                "Calibrating over an unreliable bus records limits from whichever\n"
                "sample happened to land, so this is a hard stop. To diagnose:\n"
                f"  python3 sts_tool.py --port {config.bus.port} "
                f"--baud {config.bus.baudrate} scan\n"
                f"  python3 sts_tool.py --port {config.bus.port} "
                f"--baud {config.bus.baudrate} bustest "
                + ",".join(str(j.id) for j in config.joints) + "\n"
                "If reads are dropping, step the whole bus down to a slower rate."
            )
            return 4

        input("Press ENTER to begin, or Ctrl-C to abort... ")
        print()

        total = len(joints)
        completed, skipped = 0, []
        with ui.RawTerminal() as terminal:
            calibrator = Calibrator(config, store, reader, terminal, mock=args.mock)
            for index, joint in enumerate(joints, start=1):
                outcome = calibrator.run_joint(joint, index, total)
                if outcome == _QUIT:
                    raise _Aborted
                if outcome == _SKIP:
                    skipped.append(joint.name)
                    print(_c(f"   skipped {joint.name}", ui.YELLOW, color) + "\n")
                else:
                    completed += 1

        print(ui.rule("=", ui.CYAN if color else ""))
        print(_c(f"  {completed} joint(s) calibrated"
                 + (f", {len(skipped)} skipped: {', '.join(skipped)}" if skipped else ""),
                 ui.BOLD, color))
        print(ui.rule("=", ui.CYAN if color else ""))
        print()
        print_status(config, store, color)
        exit_code = 0 if store.validate(config).ok else 1

    except (_Aborted, KeyboardInterrupt):
        print()
        print(_c("aborted -- joints finished so far are already saved.", ui.YELLOW, color))
        print(f"resume with:  ros2 run humanoid_calibration calibrate")
        exit_code = 130
    finally:
        reader.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
