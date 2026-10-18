#!/usr/bin/env python3
"""
Prepare the source-code copy that is shared separately from the exe.

The exe is built by SampleScanner.spec; the sources that travel with it are
laid out here, so that what is shared is what is being worked on, without the
things that are only of use in the project folder:

    python share_source.py
    python share_source.py -o D:\somewhere\else

src_toshare2/SampleScanner/ is emptied except for cache\\ and results\\, and the
program is put into it again:

    samplescanner.py, samplescanner_gui.py     searching, and the window that
    convert_caches_1to2.py                      starts it, and the converter of
                                                the indexes of an older format
    ssc\\*.py                                   the package, as a package
    README.md, requirements.txt, .gitignore    what is needed to run it

Note which entry point goes where: `samplescanner_gui.py` hands over to the other
two. With no arguments it opens the window, with arguments it is the search
itself, and with the one argument that says so it is the converter — so a copy
that left the converter out would still start and search, and would only fail
when someone pressed "Convert".

Two things are deliberate:

* **ssc\\ keeps its folder.** The program does `from ssc.config import ...`, so
  the modules copied out flat would not be importable. Copying the .py files into
  a `ssc` folder inside the copy is what makes the result runnable as it is.
* **cache\\ and results\\ are left alone.** They are the two folders the program
  writes into, and they may already hold something. Everything else there is
  whatever the last run left, and is what this script is here to clear.

Nothing outside the SampleScanner folder is touched, and the script says which
folder that is in a line before it starts: a script that empties a folder should
be boring to read.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAM_DIR = "SampleScanner"
DEFAULT_OUT = HERE / "src_toshare2"
KEEP = ("cache", "results")

# The three entry points, and the files a person needs beside them. The
# converter is an entry point like the other two: packed into one launcher it is
# reached by the --convert-caches argument, so without it in the copy the program
# in the copy cannot convert anything.
TOP_FILES = ("samplescanner.py", "samplescanner_gui.py",
             "convert_caches_1to2.py",
             "README.md", "requirements.txt", ".gitignore")
PACKAGE = "ssc"


def refuse_if_unexpected(dest: Path) -> None:
    """
    Check the one folder this script empties, before emptying it.

    With -o that folder is wherever the person running this said it should be,
    so there is less left to check than there was when the path was built from
    this file and could not wander. What is still worth refusing is a target
    that has the project folder inside it, because emptying that would take the
    working copy with it - a drive root is covered by the name check below, a
    target one level under one is an ordinary folder and is left alone.

    The name is fixed whatever is passed - the SampleScanner folder always goes
    inside the folder given - so that the blast radius is one folder deep no
    matter where it is aimed.
    """
    reasons = []
    if dest.name != PROGRAM_DIR:
        reasons.append(f"the folder has to be called {PROGRAM_DIR}")
    if HERE.is_relative_to(dest):
        reasons.append(f"the project folder {HERE.name} is inside it, and this "
                       f"script empties whatever it is given")
    if reasons:
        print(f"Refusing to empty {dest}:", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        raise SystemExit(1)


def clear(dest: Path) -> int:
    """Empties the folder except for cache\\ and results\\, creating those."""
    dest.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in dest.iterdir():
        if item.name in KEEP:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        removed += 1
    for name in KEEP:
        (dest / name).mkdir(exist_ok=True)
    return removed


def copy_sources(dest: Path) -> tuple:
    """Puts the program into the folder. Returns what was copied."""
    copied = []
    for name in TOP_FILES:
        source = HERE / name
        if not source.is_file():
            print(f"  MISSING: {name}", file=sys.stderr)
            continue
        shutil.copy2(source, dest / name)
        copied.append(name)

    # the package, folder and all: imports say `from ssc.config import ...`
    target = dest / PACKAGE
    target.mkdir(exist_ok=True)
    modules = 0
    for source in sorted((HERE / PACKAGE).glob("*.py")):
        shutil.copy2(source, target / source.name)
        modules += 1
    copied.append(f"{PACKAGE}/ ({modules} modules)")
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="share_source.py",
        description="Lay the sources out for sharing, into "
                    "<folder>/" + PROGRAM_DIR + "/.")
    ap.add_argument("-o", "--output", metavar="DIR", default=str(DEFAULT_OUT),
                    help="folder to put " + PROGRAM_DIR + "/ into; it and any "
                         "folders above it are created if they are not there "
                         "yet (default: the project's src_toshare2)")
    args = ap.parse_args()

    out = Path(args.output).expanduser().resolve()
    dest = out / PROGRAM_DIR

    refuse_if_unexpected(dest)
    print(f"  target : {dest}")
    removed = clear(dest)
    print(f"  removed: {removed} item(s), kept " + ", ".join(KEEP))
    for item in copy_sources(dest):
        print(f"  copied : {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
