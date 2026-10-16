"""
A print for the lines that matter while they are the newest thing on the screen
and stop mattering once the next one comes.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

# In front of such a line when the output is not a terminal. The window of the
# gui reads it, shows the text in the status line under the output and keeps it
# out of the report; the log of the run is written without it, so nothing is
# lost, only moved. On a terminal there is nothing to tell anyone and the line
# is an ordinary one.
STATUS_MARK = "\x1f"

# how much of the last replaced line is still on screen, so the next one can
# rub it out instead of leaving the tail of it behind
_pad = 0


def is_status(line: str) -> bool:
    """Whether the line came from flash() and is meant for the status line."""
    return line.startswith(STATUS_MARK)


def plain(line: str) -> str:
    """The line as it is written down: without the mark that routed it."""
    return line[1:] if line.startswith(STATUS_MARK) else line


def _is_a_terminal(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _write(text: str) -> None:
    # a status line is never worth failing a run over: the file may be closed,
    # the console may be a strange one, and the work itself is unaffected
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass


def flash(text: str, replace: bool = False,
          stream: Optional[TextIO] = None) -> None:
    """
    Says what is happening NOW, rather than adding to what has happened.

    Started from a terminal it is an ordinary print - or, with `replace`, a line
    the next flash() rubs out the way a progress bar does, so that a run which
    produces thousands of them leaves one behind instead of thousands. Started
    from the window of the gui it goes to the status line under the output and
    not into it: a message about a moment that has already gone past is not
    something to scroll back to.

    Any print() in the program becomes one of these by changing nothing but the
    name, which is what it is for.
    """
    global _pad
    out = sys.stdout if stream is None else stream
    try:
        if not _is_a_terminal(out):
            out.write(STATUS_MARK + text + "\n")
        elif replace:
            pad = max(0, _pad - len(text))
            _pad = len(text)
            out.write("\r" + text + " " * pad)
        else:
            _pad = 0
            out.write(text + "\n")
        out.flush()
    except Exception:
        pass


def end_flash() -> None:
    """
    Closes a line that was being replaced, so that whatever is printed next
    starts on a line of its own instead of on top of it.
    """
    global _pad
    if _pad:
        _pad = 0
        _write("\n")