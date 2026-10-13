#!/usr/bin/env python3
"""
Prepare the source-code copy that is shared separately from the exe.

The exe is built by SampleScanner.spec; the sources that travel with it are
laid out here, so that what is shared is what is being worked on, without the
things that are only of use in the project folder:

    python share_source.py

src_toshare/SampleScanner/ is emptied except for cache\\ and results\\, and the
program is put into it again:

    samplescanner.py, samplescanner_gui.py     the two entry points
    ssc\\*.py                                   the package, as a package
    README.md, requirements.txt, .gitignore    what is needed to run it

Two things are deliberate:

* **ssc\\ keeps its folder.** The program does `from ssc.config import ...`, so
  the modules copied out flat would not be importable. Copying the .py files into
  a `ssc` folder inside the copy is what makes the result runnable as it is.
* **cache\\ and results\\ are left alone.** They are the two folders the program
  writes into, and they may already hold something. Everything else there is
  whatever the last run left, and is what this script is here to clear.

Nothing outside src_toshare\\SampleScanner\\ is touched, and the script says so
in a line before it starts: a script that empties a folder should be boring to
read.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "src_toshare" / "SampleScanner"
KEEP = ("cache", "results")

# the program itself, and the three files a person needs beside it
TOP_FILES = ("samplescanner.py", "samplescanner_gui.py",
             "README.md", "requirements.txt", ".gitignore")
PACKAGE = "ssc"


def refuse_if_unexpected() -> None:
    """
    Check the one folder this script empties, before emptying it.

    The path is built from this file, so it cannot wander — but a script that
    deletes is worth one guard that stops it if this file is ever copied
    somewhere it should not have been run from.
    """
    if DEST.name != "SampleScanner" or DEST.parent.name != "src_toshare" \
            or not DEST.is_relative_to(HERE):
        print(f"Refusing to touch {DEST}", file=sys.stderr)
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
    refuse_if_unexpected()
    print(f"  target : {DEST}")
    removed = clear(DEST)
    print(f"  removed: {removed} item(s), kept " + ", ".join(KEEP))
    for item in copy_sources(DEST):
        print(f"  copied : {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())