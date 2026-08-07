"""Post-calibration verification: are the recorded numbers actually sane?

``calibrate --status`` answers "was this joint recorded, and is it stale?".
That is presence, not correctness -- a joint calibrated by mashing ENTER three
times without moving anything shows up as a perfectly healthy ``ok``.

This tool checks the recorded numbers against the geometry they claim to
describe:

* the zero maps back to exactly 0 deg;
* each recorded limit, run back through the same conversion a motion node will
  use, lands at the limit the config asked for;
* the joint genuinely travelled -- a limit that is not 0 deg cannot share its
  encoder count with the zero;
* the total travel in counts matches the travel in degrees;
* every raw value is inside one encoder turn.

With ``--live`` it also reads the bus and reports what each joint is at *right
now*. Stand the robot in its calibration pose and every joint should read close
to 0 deg -- that is the real end-to-end proof that the zeros mean what you
think they mean.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace

from . import term_ui as ui
from .calibration_store import CalibrationStore, CalibrationStoreError, JointCalibration
from .feetech_bus import BusError, FeetechReader, MockReader
from .joint_config import ConfigError, JointConfig, RobotConfig, load_config


@dataclass
class JointReport:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rec_min: float | None = None
    rec_max: float | None = None
    live_deg: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_joint(
    cfg: JointConfig,
    cal: JointCalibration,
    cpr: int,
    tol: float,
    overshoot_warn: float,
) -> JointReport:
    """Run every offline consistency check for one joint."""
    report = JointReport(name=cfg.name)
    err = report.errors.append
    warn = report.warnings.append

    # -- raw values must lie inside one encoder turn ----------------------
    for label, raw in (("zero", cal.zero_raw), ("min", cal.min_raw), ("max", cal.max_raw)):
        if not 0 <= raw < cpr:
            err(f"{label}_raw={raw} is outside 0..{cpr - 1}")
    if cal.counts_per_rev != cpr:
        err(f"counts_per_rev={cal.counts_per_rev} but config says {cpr}")
    if report.errors:
        return report

    # -- the conversion a motion node will use ----------------------------
    rec_zero = cal.raw_to_deg(cal.zero_raw)
    report.rec_min = cal.raw_to_deg(cal.min_raw)
    report.rec_max = cal.raw_to_deg(cal.max_raw)

    if abs(rec_zero) > 1e-9:
        err(f"zero_raw maps to {rec_zero:+.4f} deg, not 0")

    # A limit is captured the moment the operator crosses it, so undershoot is
    # a real failure but overshoot only means they swung further than asked.
    if report.rec_min > cfg.min_deg + tol:
        err(
            f"min never reached: min_raw maps to {report.rec_min:+.2f} deg, "
            f"short of the configured {cfg.min_deg:+.2f} deg"
        )
    elif report.rec_min < cfg.min_deg - overshoot_warn:
        warn(
            f"min overshot by {abs(report.rec_min - cfg.min_deg):.1f} deg "
            f"({report.rec_min:+.2f} vs {cfg.min_deg:+.2f})"
        )

    if report.rec_max < cfg.max_deg - tol:
        err(
            f"max never reached: max_raw maps to {report.rec_max:+.2f} deg, "
            f"short of the configured {cfg.max_deg:+.2f} deg"
        )
    elif report.rec_max > cfg.max_deg + overshoot_warn:
        warn(
            f"max overshot by {abs(report.rec_max - cfg.max_deg):.1f} deg "
            f"({report.rec_max:+.2f} vs {cfg.max_deg:+.2f})"
        )

    # -- the joint must actually have moved -------------------------------
    # A limit of 0 deg legitimately shares its count with the zero (the tool
    # asks for a manual confirm there); any other limit must not.
    if abs(cfg.min_deg) > tol and cal.min_raw == cal.zero_raw:
        err("min_raw equals zero_raw -- the joint was never moved to its minimum")
    if abs(cfg.max_deg) > tol and cal.max_raw == cal.zero_raw:
        err("max_raw equals zero_raw -- the joint was never moved to its maximum")
    if cal.min_raw == cal.max_raw:
        err("min_raw equals max_raw -- both limits captured at the same position")

    # -- travel in counts must agree with travel in degrees ---------------
    if not report.errors:
        recovered_span = report.rec_max - report.rec_min
        if recovered_span <= 0:
            err(
                f"recovered range is inverted: {report.rec_min:+.2f} .. "
                f"{report.rec_max:+.2f} deg (check `direction`)"
            )
        else:
            expected_span = cfg.max_deg - cfg.min_deg
            if recovered_span + 1e-6 < expected_span - 2 * tol:
                err(
                    f"travel is {recovered_span:.2f} deg but the config asks for "
                    f"{expected_span:.2f} deg"
                )
    return report


def verify(
    config: RobotConfig,
    store: CalibrationStore,
    overshoot_warn: float,
    reader=None,
    live_tolerance: float = 10.0,
) -> list[JointReport]:
    """Verify every configured joint, optionally cross-checking against the bus."""
    cpr = config.servo.counts_per_rev
    tol = config.calibration.reach_tolerance_deg
    reports: list[JointReport] = []

    for cfg in config.joints:
        cal = store.joints.get(cfg.name)
        if cal is None:
            report = JointReport(name=cfg.name)
            report.errors.append("not calibrated")
            reports.append(report)
            continue

        report = verify_joint(cfg, cal, cpr, tol, overshoot_warn)

        if reader is not None:
            raw = reader.read_raw(cfg.id)
            if raw is None:
                report.errors.append(f"no reply from servo {cfg.id}")
            else:
                report.live_deg = cal.raw_to_deg(raw)
                if abs(report.live_deg) > live_tolerance:
                    report.warnings.append(
                        f"currently at {report.live_deg:+.1f} deg -- expected near 0 "
                        "if the robot is standing in its calibration pose"
                    )
        reports.append(report)
    return reports


def print_reports(
    config: RobotConfig,
    store: CalibrationStore,
    reports: list[JointReport],
    color: bool,
    live: bool,
) -> None:
    def paint(text: str, colour: str) -> str:
        return f"{colour}{text}{ui.RESET}" if color else text

    header = f"  {'joint':<20}{'zero':>6}{'min':>6}{'max':>6}   {'recovered range':<22}"
    if live:
        header += f"{'now':>9}   "
    print(paint(header + "verdict", ui.BOLD))

    for report in reports:
        cal = store.joints.get(report.name)
        if cal is None:
            print(f"  {report.name:<20}{'--':>6}{'--':>6}{'--':>6}   "
                  + f"{'':<22}" + (f"{'':>9}   " if live else "")
                  + paint("NOT CALIBRATED", ui.RED))
            continue

        span = ""
        if report.rec_min is not None and report.rec_max is not None:
            span = f"{report.rec_min:+7.2f} .. {report.rec_max:+7.2f}"
        row = (
            f"  {report.name:<20}{cal.zero_raw:>6}{cal.min_raw:>6}{cal.max_raw:>6}   "
            f"{span:<22}"
        )
        if live:
            row += (f"{report.live_deg:+8.2f}   " if report.live_deg is not None
                    else f"{'--':>8}   ")
        if report.errors:
            verdict = paint("FAIL", ui.RED)
        elif report.warnings:
            verdict = paint("warn", ui.YELLOW)
        else:
            verdict = paint("ok", ui.GREEN)
        print(row + verdict)

    print()
    for report in reports:
        for message in report.errors:
            print("  " + paint(f"FAIL  {report.name}: {message}", ui.RED))
        for message in report.warnings:
            print("  " + paint(f"warn  {report.name}: {message}", ui.YELLOW))
    if any(r.errors or r.warnings for r in reports):
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_calibration",
        description="Check that a completed calibration is internally consistent.",
    )
    parser.add_argument("--config", help="path to joint_limits.yaml")
    parser.add_argument("--store", help="path to the calibration file")
    parser.add_argument(
        "--live",
        action="store_true",
        help="also read the bus; stand the robot in its calibration pose first",
    )
    parser.add_argument("--port", help="override the serial port")
    parser.add_argument("--baud", type=int, help="override the baud rate")
    parser.add_argument("--mock", action="store_true", help="simulate the bus")
    parser.add_argument(
        "--overshoot-warn",
        type=float,
        default=5.0,
        metavar="DEG",
        help="warn when a limit was captured this far past its target (default: 5)",
    )
    parser.add_argument(
        "--live-tolerance",
        type=float,
        default=10.0,
        metavar="DEG",
        help="with --live, warn when a joint is this far from 0 (default: 10)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="exit code only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    color = ui.supports_color() and not args.quiet

    def emit(*parts: str) -> None:
        if not args.quiet:
            print(*parts)

    try:
        config = load_config(args.config)
        store = CalibrationStore.load(args.store)
    except (ConfigError, CalibrationStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    overrides = {}
    if args.port:
        overrides["port"] = args.port
    if args.baud:
        overrides["baudrate"] = args.baud
    if overrides:
        config = replace(config, bus=replace(config.bus, **overrides))

    emit(ui.rule("=", ui.CYAN if color else ""))
    emit("  CALIBRATION VERIFICATION")
    emit(ui.rule("=", ui.CYAN if color else ""))
    emit(f"  calibration  {store.path}")
    emit(f"  config       {config.source_path}")
    emit(f"  recorded     {store.updated_at or '(never)'}")
    emit()

    # Presence and staleness first: a numeric check against limits the file was
    # not captured against would be meaningless.
    presence = store.validate(config)
    if not presence.exists:
        if not args.quiet:
            print(f"  {presence.reason()}")
        return 1
    if presence.stale and not args.quiet:
        print("  " + f"{ui.YELLOW}stale: {presence.reason()}{ui.RESET}"
              if color else f"  stale: {presence.reason()}")
        print()

    reader = None
    if args.live:
        try:
            reader = MockReader(config.servo) if args.mock else FeetechReader(
                config.bus, config.servo
            )
        except BusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        emit(f"  reading {'MOCK' if args.mock else config.bus.port} "
             f"@ {config.bus.baudrate}...")
        emit()

    try:
        reports = verify(config, store, args.overshoot_warn, reader, args.live_tolerance)
    finally:
        if reader is not None:
            reader.close()

    if not args.quiet:
        print_reports(config, store, reports, color, args.live)

    failed = [r for r in reports if r.errors]
    warned = [r for r in reports if r.warnings and not r.errors]
    ok = len(reports) - len(failed)

    if args.quiet:
        return 0 if not failed and presence.ok else 1

    summary = f"  {ok}/{len(reports)} joints consistent"
    if warned:
        summary += f", {len(warned)} with warnings"
    print(summary)
    if failed or not presence.ok:
        print("  " + (f"{ui.RED}FAIL{ui.RESET}" if color else "FAIL")
              + " -- re-run the affected joints:")
        names = " ".join(r.name for r in failed) or " ".join(presence.missing)
        print(f"      ros2 run humanoid_calibration calibrate --only {names}")
        return 1
    print("  " + (f"{ui.GREEN}PASS{ui.RESET}" if color else "PASS")
          + " -- calibration is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
