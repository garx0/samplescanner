"""The name and the number of the program — THE ONE PLACE THEY ARE WRITTEN.

Bump the version here, before a build, and nowhere else.

Everything else that shows it is made from this file, not typed out again: the
title of the window, `SampleScanner.exe --version`, the first line of every run
(and so of every log file), and the VERSION.txt that the build drops next to the
exe. That is the whole point of putting it here — a number that has to be
remembered in four places is a number that is wrong in at least one of them, and
the person who finds out is the one who copied an old build into the community.

It lives in code rather than in a VERSION file on purpose: a plain file would
have to be bundled by PyInstaller and read back at run time, and then a folder
that was unpacked and left alone could disagree with the program inside it.
Here the number is fixed when the exe is built, which is exactly when a version
is decided.

Pure python — no imports, so the window can read it before anything else loads.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# CHANGE THIS LINE — and only this line — when the program gets a new version
# --------------------------------------------------------------------------- #
VERSION = "0.2.7"

# The name it is distributed under. Not part of the number, but it belongs to the
# same question, so it is answered here too.
APP_NAME = "SampleScanner"


def version_line() -> str:
    """'SampleScanner v<version>' — one line, the way a person reads a version."""
    return f"{APP_NAME} v{VERSION}"
