"""Small terminal helpers: raw key input and flicker-free in-place redraw.

Deliberately dependency-free (no curses, no rich): the calibration tool is run
over SSH on a Jetson, often through a plain ``ssh`` session or a serial
console, and it needs to behave on a dumb TTY.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import tty
from types import TracebackType

# -- ANSI ------------------------------------------------------------------

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
WHITE = "\x1b[37m"

_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"

# Named keys returned by RawTerminal.read_key()
KEY_ENTER = "ENTER"
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_LEFT = "LEFT"
KEY_RIGHT = "RIGHT"
KEY_ESC = "ESC"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")


def term_width(default: int = 80) -> int:
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns)
    except OSError:
        return default


def visible_len(text: str) -> int:
    """Length of ``text`` ignoring ANSI escape sequences."""
    length = 0
    index = 0
    while index < len(text):
        if text[index] == "\x1b":
            end = text.find("m", index)
            index = len(text) if end == -1 else end + 1
            continue
        length += 1
        index += 1
    return length


class RawTerminal:
    """Context manager putting stdin in cbreak mode for single-key reads."""

    def __init__(self, hide_cursor: bool = True) -> None:
        self._fd = sys.stdin.fileno()
        self._saved: list | None = None
        self._hide_cursor = hide_cursor
        self.interactive = sys.stdin.isatty()

    def __enter__(self) -> "RawTerminal":
        if self.interactive:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            if self._hide_cursor:
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.restore()

    def restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        if self.interactive and self._hide_cursor:
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

    def read_key(self, timeout: float = 0.0) -> str | None:
        """Return one key press, or None if ``timeout`` elapses.

        Returns a single character, or one of the ``KEY_*`` constants.
        Ctrl-C is re-raised as KeyboardInterrupt so the caller's normal
        cleanup path runs (cbreak mode suppresses the usual SIGINT).
        """
        if not self.interactive:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        char = sys.stdin.read(1)
        if char == "\x03":
            raise KeyboardInterrupt
        if char in ("\r", "\n"):
            return KEY_ENTER
        if char == "\x1b":
            # Possible CSI escape sequence; peek without blocking.
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return KEY_ESC
            sequence = sys.stdin.read(1)
            if sequence != "[":
                return KEY_ESC
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return KEY_ESC
            final = sys.stdin.read(1)
            return {
                "A": KEY_UP,
                "B": KEY_DOWN,
                "C": KEY_RIGHT,
                "D": KEY_LEFT,
            }.get(final, KEY_ESC)
        return char

    def drain(self) -> None:
        """Discard any buffered input (e.g. keys mashed during a long read)."""
        if not self.interactive:
            return
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return
            sys.stdin.read(1)


class LiveBlock:
    """Repaints a fixed block of lines in place, without scrolling the screen."""

    def __init__(self) -> None:
        self._lines_drawn = 0

    def draw(self, lines: list[str]) -> None:
        out = []
        if self._lines_drawn:
            out.append(f"\x1b[{self._lines_drawn}A")
        width = term_width()
        for line in lines:
            if visible_len(line) > width:
                line = _truncate(line, width)
            out.append("\x1b[2K" + line + "\n")
        # Clear any leftovers if this frame is shorter than the previous one.
        for _ in range(max(0, self._lines_drawn - len(lines))):
            out.append("\x1b[2K\n")
        trailing = max(0, self._lines_drawn - len(lines))
        if trailing:
            out.append(f"\x1b[{trailing}A")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._lines_drawn = len(lines)

    def finish(self) -> None:
        """Leave the last frame on screen and stop tracking it."""
        self._lines_drawn = 0


def _truncate(text: str, width: int) -> str:
    result = []
    length = 0
    index = 0
    while index < len(text) and length < width:
        if text[index] == "\x1b":
            end = text.find("m", index)
            if end == -1:
                break
            result.append(text[index : end + 1])
            index = end + 1
            continue
        result.append(text[index])
        length += 1
        index += 1
    result.append(RESET)
    return "".join(result)


def rule(char: str = "=", color: str = "") -> str:
    line = char * term_width()
    return f"{color}{line}{RESET}" if color else line


def span_bar(
    value: float,
    low: float,
    high: float,
    marks: dict[float, str],
    width: int = 46,
) -> str:
    """Render ``value`` on a ``low``..``high`` track with labelled tick marks.

    ``marks`` maps a position on the track to the single character drawn there
    (e.g. ``{0.0: '0', -40.0: '<'}``). The current value is drawn last so it
    always wins a collision.
    """
    if high <= low:
        high = low + 1.0
    track = ["-"] * width

    def slot(position: float) -> int:
        fraction = (position - low) / (high - low)
        return max(0, min(width - 1, int(round(fraction * (width - 1)))))

    for position, char in marks.items():
        track[slot(position)] = char

    cursor = slot(max(low, min(high, value)))
    track[cursor] = "#"
    return "[" + "".join(track) + "]"
