"""Where the program writes: its own folder, and the right to do so."""

from __future__ import annotations

import os
from ssc.config import DEFAULT_RESULTS_PREFIX
from ssc.launcher import launcher_dir

def long_form(path: str) -> str:
    """
    The path in the form Windows accepts past MAX_PATH: made absolute and marked
    with the `\\\\?\\` prefix, which is the documented way past the limit — the
    setting LongPathsEnabled is not consulted for such a path, and it works on a
    path longer than 260 characters.

    Returned unchanged where none of that is needed: away from Windows the prefix
    means nothing, and a short path needs nothing. It is a path FOR THE FILESYSTEM
    — what os.walk() and ffmpeg are handed — and not something to print or to
    store in the index: a name carrying the prefix would no longer match anything
    the user can see.
    """
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    full = os.path.abspath(path)
    if len(full) < 250:
        return path
    return ("\\\\?\\UNC\\" + full[2:] if full.startswith("\\\\")
            else "\\\\?\\" + full)


def plain(path: str) -> str:
    """
    The same path WITHOUT the `\\\\?\\` prefix.

    long_form() exists to hand the filesystem something it will open; this exists
    to make sure that spelling goes no further than that call. Everything the user
    reads and everything the index stores — the collection folder, the file names,
    the paths in the manifest — has to be the ordinary one, or the next run would
    look for a folder that is spelled in a way nobody ever typed, and a report
    would show `\\\\?\\C:\\...` where the user means `C:\\...`.
    """
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def is_dir(path: str) -> bool:
    """os.path.isdir, but one that also sees a path Windows calls too long."""
    if os.path.isdir(path):
        return True
    prefixed = long_form(path)
    if prefixed == path:
        return False
    try:
        return os.path.isdir(prefixed)
    except (OSError, ValueError):
        return False


def is_file(path: str) -> bool:
    """os.path.isfile, but one that also sees a path Windows calls too long."""
    if os.path.isfile(path):
        return True
    prefixed = long_form(path)
    if prefixed == path:
        return False
    try:
        return os.path.isfile(prefixed)
    except (OSError, ValueError):
        return False


def exists(path: str) -> bool:
    """Whether anything at all stands at `path`, long or not."""
    return is_dir(path) or is_file(path)


def program_dir() -> str:
    """
    The folder the program writes into. Packed into one launcher it is the folder
    that launcher sits in — a portable program unpacked anywhere keeps its cache
    and its results next to itself instead of scattering them over the computer;
    run from a source checkout it is the working directory, as it always was.
    """
    return launcher_dir() or os.getcwd()


def results_root(prefix: str = DEFAULT_RESULTS_PREFIX) -> str:
    """Where the run folders go: ./results, next to the program."""
    if os.path.isabs(prefix):
        return prefix
    return os.path.join(program_dir(), prefix)


def ensure_writable_dir(path: str, what: str) -> str:
    """
    Makes sure the program may write into `path`, and creates it. A portable
    program unpacked somewhere it has no rights for (Program Files, a read-only
    share) has to say so BEFORE a long search is started, in words a user can
    act on, rather than fail halfway through one.
    """
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".samplescanner-write-test")
        with open(probe, "w") as fh:
            fh.write("")
        os.remove(probe)
    except OSError as exc:
        raise PermissionError(
            f"Cannot write into {path} ({exc.strerror or exc}).\n"
            f"{what} go there, so the program cannot work from where it is "
            f"now. Move the whole program folder somewhere you own (your "
            f"Documents, for example), or point it at a folder you can write "
            f"into, and start it again."
        ) from exc
    return path

