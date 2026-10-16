"""The progress line of a long job — a small tqdm, written here.

There is no tqdm in the requirements and there is no need for one: a bar is a
few lines of arithmetic and a carriage return, and keeping it in the project
means one thing less to bundle into the exe.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, TextIO

from ssc.format import fmt_elapsed
from ssc.status import flash

# How often the line may be redrawn on a terminal. Faster than this and the
# characters only blur; slower and a fast job looks stuck.
REFRESH = 1.0 / 12.0

# How often a plain line is printed when the output is not a terminal — the
# window of the gui, or a run piped into a file. On a terminal this does not
# apply at all: there the bar is redrawn in place (see REFRESH) and nothing is
# left behind either way. Two seconds is short enough that a long job reads as
# working rather than hung, and it is one line either way.
INTERVAL = 2.0


def _is_a_terminal(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class Progress:
    """
    One line saying how far a job is, redrawn in place the way tqdm does it.

    On a terminal the line is overwritten after every step, so the screen keeps
    one moving bar. When the output is a pipe — the window of the gui streams
    it, or the run is written to the log file — there is nothing to overwrite,
    so a plain line is printed every few seconds instead, which lands in the
    window and in the log as an ordinary progress report. Both say the same
    thing, and neither writes anything when the job is over.

    step() may be called from several threads: the index is built by a pool of
    them.
    """

    def __init__(self, total: int, text: str = "", width: int = 26,
                 stream: Optional[TextIO] = None,
                 interval: float = INTERVAL, done: int = 0,
                 unit: str = "file"):
        self.total = max(0, int(total))
        self.text = text
        self.width = max(4, width)
        self.out = sys.stdout if stream is None else stream
        self.interval = interval
        self.done = max(0, int(done))
        # what is being counted, so that the rate says it: the walk counts files
        # of the collection, the precise pass counts the windows it reads out of
        # them, and "12.4 file/s" over a row of windows would be a lie about
        # what the bar is doing
        self.unit = unit
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        self._drawn_at = 0.0
        self._drawn_len = 0
        self._last_line = ""
        self._live = _is_a_terminal(self.out)

    # ---------- the line itself ----------

    def _line(self) -> str:
        done, total = self.done, self.total
        elapsed = time.perf_counter() - self._start
        if total > 0:
            filled = int(round(self.width * min(done, total) / total))
            bar = f"[{'=' * filled}{'-' * (self.width - filled)}]"
            count = f"{done}/{total}  {done * 100 // total:3d}%"
        else:
            bar, count = "", f"{done} {self.unit}(s)"
        # the same guess tqdm makes: what is done so far took this long, so
        # what is left of it will take about as much again. With nothing done
        # yet there is nothing to guess from, and the line says no ETA at all.
        left = (f"  ETA {fmt_elapsed(elapsed / done * (total - done))}"
                if done and total and done < total else "")
        rate = (f"  {done / elapsed:.1f} {self.unit}/s"
                if done and elapsed > 0.3 else "")
        parts = [p for p in (self.text, bar, count,
                             fmt_elapsed(elapsed) + rate + left) if p]
        return " ".join(parts)

    # ---------- drawing ----------

    def _write(self, text: str) -> None:
        # a bar is never worth failing a run over: the file may be closed, the
        # console may be a strange one, and the work itself is unaffected
        try:
            self.out.write(text)
            self.out.flush()
        except Exception:
            pass

    def _draw(self, force: bool = False) -> None:
        line = self._line()
        now = time.perf_counter()
        if not self._live:
            # nothing to overwrite: one line every few seconds is the progress
            # report a pipe or a log file can carry. It is a status line, so a
            # window reading the stream shows it under the output instead of
            # filling the report with a line per interval.
            if not force and now - self._drawn_at < self.interval:
                return
            self._drawn_at = now
            self._last_line = line
            flash(line)
            return
        if not force and now - self._drawn_at < REFRESH:
            return
        self._drawn_at = now
        self._last_line = line
        # what is left of a longer line has to be rubbed out, or the bar grows
        # to the right with every step
        pad = max(0, self._drawn_len - len(line))
        self._drawn_len = len(line)
        self._write("\r" + line + " " * pad)

    # ---------- use ----------

    def start(self) -> "Progress":
        """Puts the first line on the screen."""
        self._draw(force=True)
        return self

    def step(self, n: int = 1) -> None:
        """One unit of the work is done."""
        with self._lock:
            self.done += n
        self._draw()

    def close(self) -> None:
        """
        The last line, and the end of the line on screen.

        A line that says exactly what the last one said is not written again —
        a job that had nothing to do must not print its bar three times — and
        the same holds for a job that died half way: what it managed is what the
        last line says.
        """
        if self._line() == self._last_line:
            if self._live and self._drawn_len:
                self._drawn_len = 0
                self._write("\n")
            return
        self._draw(force=True)
        if self._live:
            self._drawn_len = 0
            self._write("\n")