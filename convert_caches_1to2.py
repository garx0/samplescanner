#!/usr/bin/env python3
"""
Converts the library indexes (the cache) of an older format into the current one.

    convert_caches_1to2.py                  everything in ./cache that is old
    convert_caches_1to2.py --dry-run        only say what it would do
    convert_caches_1to2.py --cache DIR      another cache folder
    convert_caches_1to2.py -v               say what each folder holds

What it does is a copy, not a rebuild: the features are already computed and sit
in the old folders as .npy pairs, and every old manifest already says which file
each pair belongs to. No audio file is opened, nothing is decoded, and the old
folders are left exactly as they are — so a run that was interrupted can simply
be run again, and a user who would rather go back to the previous version of the
program still has a cache that version can read.

It is a program of its own rather than a flag of samplescanner.py because it is
not part of searching: it is a one-off maintenance job the window offers once per
New search while an old cache is still lying around, and it has to keep working
after the old format stops being supported, which is exactly what putting it in
the search program would prevent.

The window runs this same main() in its own process (see
samplescanner_gui.open_converter), so the two are never out of step.
"""
import argparse
import os
import shutil
import sys
import time
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ssc import convert                                      # noqa: E402
from ssc.cache import cache_root                             # noqa: E402
from ssc.config import INDEX_VERSION                         # noqa: E402
from ssc.format import fmt_elapsed, fmt_size                 # noqa: E402
from ssc.paths import ensure_writable_dir                    # noqa: E402
from ssc.progress import Progress                            # noqa: E402
from ssc.version import version_line                         # noqa: E402


def free_bytes(path: str) -> int:
    """Room left on the drive `path` is on."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def say_old_folders(old, verbose: bool) -> None:
    """What is there to convert, before anything is promised about it."""
    print(f"OLD       : {len(old)} index folder(s) in the old format")
    for c in old:
        name = os.path.basename(c.path)
        why = convert.incompatible(c)
        if why is not None:
            print(f"           {name:<28} {len(c.entries):>5} file(s)  "
                  f"cannot be converted — {why}")
        else:
            print(f"           {name:<28} {len(c.entries):>5} file(s)  "
                  f"{fmt_size(c.size)}")
    if not verbose:
        return


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="convert_caches_1to2",
        description="Convert library indexes (cache) saved before "
                    f"SampleScanner v0.2.3 into the current format (index "
                    f"version {INDEX_VERSION}). Nothing is decoded and no audio "
                    "file is opened: the existing .npy are copied into the "
                    "current folder layout, and the old folders are left alone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The old folders are not deleted. Remove them with the "
               "Clear the cache button of the window once you are happy with "
               "the result.",
    )
    ap.add_argument("--cache", metavar="DIR", default=None,
                    help="the cache folder (default: ./cache next to the "
                         "program)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only report what would be done, change nothing")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="say more about what is there")
    args = ap.parse_args(list(argv) if argv is not None else sys.argv[1:])

    root = os.path.abspath(args.cache) if args.cache else cache_root()
    print(version_line())
    print(f"CACHE     : {root}")
    print(f"FORMAT    : index version {INDEX_VERSION}")

    if not os.path.isdir(root):
        print("NOTHING   : there is no cache folder here")
        return 0

    old = convert.find_old_caches(root)
    if not old:
        print("NOTHING   : no indexes in the old format — nothing to do")
        return 0

    say_old_folders(old, args.verbose)

    plan = convert.make_plan(old, root)
    print(f"PLAN      : {plan.files} file(s) into {len(plan.new_folders)} new "
          f"cache folder(s), {fmt_size(plan.bytes_needed)} to copy")
    if not plan.files:
        print("NOTHING   : there is nothing that can be copied")
        return 0

    # Room for the copy while the old folders are still lying there. The old ones
    # are moved rather than deleted, so they keep taking their space for good and
    # the copy really does need a second set of bytes - on a large library that is
    # gigabytes, and running out of room halfway through a copy of an hour's work
    # is the worst moment to find out.
    need = max(plan.bytes_needed, sum(c.size for c in old))
    free = free_bytes(root)
    print(f"SPACE     : {fmt_size(free)} free on this drive, "
          f"{fmt_size(need)} needed for the copy beside the old indexes")
    if free and need > free:
        print(f"ERROR     : not enough free space on this drive — "
              f"{fmt_size(need - free)} more is needed. Free up that much and "
              f"run this again; nothing has been written.")
        return 1

    if args.dry_run:
        print("DRY RUN   : nothing was written")
        return 0

    try:
        ensure_writable_dir(root, "the converted indexes")
    except PermissionError as exc:
        print(f"ERROR     : {exc}")
        return 1

    t0 = time.perf_counter()
    bar = Progress(plan.files, text="converting the indexes", unit="file")
    bar.start()
    try:
        done = convert.run(plan, root, bar)
    finally:
        bar.close()
    dt = time.perf_counter() - t0

    print(f"CONVERTED : {done.converted} file(s) in {len(plan.new_folders)} "
          f"cache folder(s), copied in {fmt_elapsed(dt)}")
    if done.skipped:
        print(f"           {done.skipped} file(s) skipped — their .npy pair is "
              f"not complete in the old folder; a search over them will build "
              f"them again")

    # Only now, with every pair written and every manifest in place, are the old
    # folders moved: a conversion that fell short leaves them exactly where they
    # were, which is the only thing a user needs after a failed run.
    # An old folder the run did not take anything from stays where it is: it is
    # still unreadable, and while it is there the window keeps saying so. Said
    # once, here, where it can be said with everything else that did not go to
    # plan - the listing at the top already named it and why.
    moved, failed = convert.stash(old, done.used, root)
    for name, why in failed:
        print(f"KEPT      : {name} could not be moved out of {root} — {why}. "
              f"It is still there and still not read; remove it by hand.")
    for name, why in plan.skipped:
        if name not in set(moved):
            print(f"KEPT      : {name} was not converted — {why}. It is still in "
                  f"{root} and still not read.")
    if moved:
        print()
        print(f"MOVED     : the {len(moved)} old index folder(s) were moved to "
              f"{convert.stash_dir(root)}")
        print("           They are not read any more and nothing was deleted.")
        print("           Delete them by hand whenever you like — or leave them "
              "there for now,")
        print("           in case the indexes in the new format turn out to "
              "misbehave: they have not")
        print("           yet been run against a real library for long enough to "
              "promise that they")
        print("           never will, and these are the only other copy of the "
              "features they hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())