"""Where the program really is — independent of what packed it into one exe.

Two different packers are used on this program, and they disagree about what
"the folder the program is in" means:

* **PyInstaller** sets `sys.frozen` and points `sys.executable` at the real exe.
* **Nuitka** sets neither. It points `sys.executable` at its OWN internal python,
  inside a folder made for the run — which is not where the user put the program,
  is a different name every time, and is gone when the program ends. The real
  launcher is the thing the program was started as, and that is `sys.argv[0]`.

Getting this wrong is not cosmetic. The results and the index are written into
that folder, so on the wrong answer every search starts from nothing, the window
cannot start a search at all, and ffmpeg placed beside the program goes unseen.
Hence one place that answers it, used by all three.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Nuitka injects this name into the modules it compiles. Reading it by name is
# the documented way; there is nothing to import and nothing to install.
try:
    _compiled = __compiled__                      # noqa: F821
except NameError:                                 # not a Nuitka build
    _compiled = None


def is_packed() -> bool:
    """Whether this program was packed into one launcher, rather than run from
    sources. True under either packer."""
    return getattr(sys, "frozen", False) or _compiled is not None


def launcher_dir() -> Optional[str]:
    """
    The folder the launcher itself is in, or None when running from sources.

    That is the folder the results and the index belong to, the folder a search
    started from the window must be run in, and the folder to look for ffmpeg in.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    if _compiled is not None:
        # the launcher is the program as it was started, not the python inside it
        arg0 = getattr(_compiled, "original_argv0", None) or sys.argv[0]
        return os.path.dirname(os.path.abspath(arg0))
    return None


def launcher_exe() -> Optional[str]:
    """
    The launcher itself as a file - the thing an update has to replace.

    None when the program is run from sources: there is no exe there, and a
    copy of the program that replaces nothing is not an update.
    """
    if getattr(sys, "frozen", False):
        return os.path.abspath(sys.executable)
    if _compiled is not None:
        arg0 = getattr(_compiled, "original_argv0", None) or sys.argv[0]
        return os.path.abspath(arg0)
    return None


def launcher_argv() -> list:
    """
    The command that runs this same program again — how the window starts a
    search in a separate process.

    Under either packer that is the launcher itself with new arguments, the same
    way `SampleScanner.exe -q ... -s ...` runs the program instead of the window.
    From sources it is the interpreter with the script, which is what has always
    worked.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable]
    if _compiled is not None:
        return [sys.argv[0]]
    # this module sits in ssc/, the entry point beside that folder
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [sys.executable, "-u",
            os.path.join(project, "samplescanner_gui.py")]


# The argument that makes the frozen launcher be the converter instead of the
# window or the search. One exe is three programs, told apart by what they are
# asked to do — the same arrangement the search itself already uses.
CONVERT_FLAG = "--convert-caches"

# The argument that makes the frozen launcher be the second copy of the program
# that puts a downloaded version in place. It is the same arrangement again: one
# exe, several things it can be, told apart by the first argument.
UPDATE_FLAG = "--apply-update"


def converter_argv() -> list:
    """
    The command that converts the old indexes.

    A program of its own, not a mode of the search: it is a one-off maintenance
    job that has to keep working long after the format it reads is gone. Packed
    into one launcher it is the launcher with the flag that says so; from sources
    it is the interpreter with convert_caches_1to2.py, which is that program's own
    entry point and calls the very same main().
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, CONVERT_FLAG]
    if _compiled is not None:
        return [sys.argv[0], CONVERT_FLAG]
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [sys.executable, "-u",
            os.path.join(project, "convert_caches_1to2.py")]