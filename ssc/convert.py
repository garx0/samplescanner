"""
Turning index folders of an older format into the one this version reads.

The features themselves are not recomputed and the audio files are never opened.
Everything that is needed is already written down: the .npy pairs hold the
features, and the old manifest says which file each pair belongs to and when it
last changed. So the whole job is copying bytes from one folder tree to another
and writing down where they went.

What the older format looked like, and what differs:

    before   cache/<collection>/  *.npy + manifest.json      (one level)
    after    cache/<X>_/<Y>_/     *.npy + manifest.json      (two levels)

    * a name is one thing — the collection it was built for — while now the
      second level names the folder the audio file actually lies in, which is
      what a search looks a library folder up by;
    * "name" was the path of the file relative to the whole collection, and it is
      now relative to that folder, so for the same file it is usually just the
      file name, and the .npy name derived from it usually changes;
    * "folder" said what the collection was called and settled nothing, while now
      it is the working field: it says whose folder this is, and a folder that
      moved is recognised by it.

The old folders are copied, never moved or renamed. Nothing of theirs is touched,
so an interrupted run leaves the old cache exactly as it was and simply does less
work next time, and a user who would rather go back to the previous version of the
program still has a cache it can read.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ssc.cache import (DISK_ROOT, CacheFolder, energy_bin, index_bin_name,
                       is_root, level_name, nfc, safe_name, save_folder, scan)
from ssc.config import COARSE_BANDS, COARSE_HOP, COARSE_NFFT

# The version the folders being converted were written by. Read out of their
# manifests rather than assumed, so that a cache that is already in the current
# format is never mistaken for one that is not.
OLD_VERSION = 1


# --------------------------------------------------------------------------- #
# Finding what there is to convert
# --------------------------------------------------------------------------- #

@dataclass
class OldCache:
    """One index folder of the older format, read but not changed."""
    path: str
    source: str                 # what its manifest calls the collection
    entries: List[dict]
    size: int = 0               # bytes of .npy in it


def find_old_caches(root: str) -> List[OldCache]:
    """
    Every index folder under `root` that is in the older format.

    A folder is only in it when its own manifest says so: a current-format
    folder has no manifest of its own at that level, and a folder holding
    anything else is not an index at all. Ordered by name, so that the same tree
    is always converted the same way round.
    """
    out: List[OldCache] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        mpath = os.path.join(path, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if data.get("version") != OLD_VERSION:
            continue
        entries = data.get("entries")
        if not isinstance(entries, list):
            continue
        out.append(OldCache(path=os.path.abspath(path),
                            source=nfc(data.get("folder", "")),
                            entries=entries,
                            size=_folder_bytes(path)))
    return out


def _folder_bytes(folder: str) -> int:
    total = 0
    try:
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file():
                    total += e.stat().st_size
    except OSError:
        pass
    return total


def incompatible(c: OldCache) -> Optional[str]:
    """
    Why this folder cannot be converted, or None when it can.

    A manifest that does not match the feature parameters the program reads now
    describes numbers that cannot be compared with ours whatever the files look
    like now. The .npy are copied byte for byte and would be meaningless, so the
    folder is left alone and the user is told which one it was and why.
    """
    mpath = os.path.join(c.path, "manifest.json")
    try:
        with open(mpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return "the manifest cannot be read"
    if (data.get("coarse_nfft"), data.get("coarse_hop"), data.get("coarse_bands")) \
            != (COARSE_NFFT, COARSE_HOP, COARSE_BANDS):
        return ("its features were built with other settings "
                f"(nfft {data.get('coarse_nfft')}, hop {data.get('coarse_hop')}, "
                f"bands {data.get('coarse_bands')}; this version reads "
                f"{COARSE_NFFT}, {COARSE_HOP}, {COARSE_BANDS})")
    return None


# --------------------------------------------------------------------------- #
# The plan: which folder holds which file
# --------------------------------------------------------------------------- #

@dataclass
class Moved:
    """One file on its way from an old folder to a new one."""
    src_folder: str
    src_bin: str                # the pair as it is named where it stands now
    dest: str                   # the level-2 cache folder it goes to
    source: str                 # the library folder the new manifest records
    name: str                   # its name inside that folder
    bin: str                    # the pair under its new name
    n_coarse: int
    duration: float
    size: int
    mtime: float


@dataclass
class Plan:
    """Everything the conversion will do, worked out before it does any of it."""
    moves: List[Moved] = field(default_factory=list)
    new_folders: List[str] = field(default_factory=list)  # level-2, in order
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # why, what
    bytes_needed: int = 0

    @property
    def files(self) -> int:
        return len(self.moves)


def absolute_of(folder: str, name: str) -> str:
    """
    Where the file really is, from what an old manifest recorded.

    The two agree on what a name is meant to be but not on what they did: most
    old manifests hold a path relative to the collection, but a collection that
    sat on more than one drive had no folder to be relative to, and its names
    were written out in full with the collection left as whatever directory the
    program happened to start in. An absolute name is therefore taken as it
    stands, and only a relative one is joined to the collection.
    """
    name = nfc(name)
    if os.path.isabs(name):
        return os.path.abspath(name)
    return os.path.abspath(os.path.join(folder, name))


def make_plan(old: List[OldCache], root: str) -> Plan:
    """
    Where every file of every old folder goes.

    Nothing is written and nothing is read but the old manifests. Names are
    handed out in one pass over folders sorted by path, so that the same cache
    always converts to the same layout: the number on a name may differ from what
    another history of the cache would have produced, but it never changes from
    one run to the next, which is what matters for a thing that is done twice.
    """
    plan = Plan()
    tree = scan(root)
    taken1 = set(tree.level1)

    for cache_old in old:
        why = incompatible(cache_old)
        if why is not None:
            plan.skipped.append((os.path.basename(cache_old.path), why))
            continue

        # The library folders this old folder covers, grouped by what the first
        # level is named after and by the folder the file really lies in.
        by_x: Dict[str, Dict[str, List[Moved]]] = {}
        for entry in cache_old.entries:
            try:
                full = absolute_of(cache_old.source, str(entry["name"]))
            except (KeyError, TypeError):
                plan.skipped.append((os.path.basename(cache_old.path),
                                     "an entry without a name"))
                continue
            y = os.path.dirname(full)
            # A file lying straight in a drive root has no folder of its own to be
            # named after, so it stands under the pseudo-folder both levels share
            # - and then the collection is not what the first level is named after
            # either, for the same reason. Which is also why one old folder can end
            # up feeding two first-level folders of its own.
            x = DISK_ROOT if is_root(y) else \
                safe_name(os.path.basename(cache_old.path))
            name = os.path.basename(full)
            by_x.setdefault(x, {}).setdefault(y, []).append(
                Moved(src_folder=cache_old.path,
                      src_bin=str(entry.get("bin", "")), dest="", source=y,
                      name=name, bin=index_bin_name(name),
                      n_coarse=int(entry.get("n_coarse", 0)),
                      duration=float(entry.get("duration", 0.0)),
                      size=int(entry.get("size", -1)),
                      mtime=float(entry.get("mtime", -1.0))))

        # Every number goes on the first level, and the second level is always the
        # plain name of the folder the file is in.
        #
        # Two reasons, and the second is the one that decides it. First: nothing
        # here is ever written into a folder that already exists, so an index that
        # was in the cache before the conversion, and an index the conversion
        # itself wrote a moment ago, can never be half-overwritten by another. The
        # number is what says so - a fresh first level every time, with the second
        # level empty inside it because it was created a second ago.
        #
        # Second: a number on the second level would be indistinguishable from a
        # library folder that really is called `Y_2`, and the second level is the
        # one a search looks a folder up by. The drive roots make it plain: the
        # same collection with files in the roots of two drives must end up as
        # cache/_disk_root_/_disk_root_ and cache/_disk_root_2/_disk_root_ -
        # numbered once, on the level nothing reads.
        for x in sorted(by_x):
            inside: Dict[str, Set[str]] = {}
            made: List[str] = []          # first-level folders of this X, in order
            for y in sorted(by_x[x]):
                short = (DISK_ROOT if is_root(y)
                         else safe_name(os.path.basename(y))) + "_"
                l1_name = next((m for m in made
                                if short not in inside[m]), None)
                if l1_name is None:
                    # the name is already used under every first level of ours,
                    # so a new one is made for it rather than the second level
                    # carrying a number: an index folder and a library folder
                    # never end up with the same name and mean different things
                    l1_name = level_name(x, taken1)
                    taken1.add(l1_name)
                    inside[l1_name] = set()
                    made.append(l1_name)
                inside[l1_name].add(short)
                dest = os.path.join(tree.root, l1_name, short)
                plan.new_folders.append(dest)
                for mv in by_x[x][y]:
                    mv.dest = dest
                    plan.moves.append(mv)
                    plan.bytes_needed += _pair_bytes(cache_old.path, mv.src_bin)

    plan.moves.sort(key=lambda m: (m.dest, m.name))
    return plan


def _pair_bytes(src_folder: str, bin_name: str) -> int:
    """How much room one file's pair of .npy takes, or 0 when it is not there."""
    total = 0
    for name in (bin_name, energy_bin(bin_name)):
        try:
            total += os.path.getsize(os.path.join(src_folder, name))
        except OSError:
            pass
    return total


# --------------------------------------------------------------------------- #
# Doing it
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    """What a conversion actually did, which is not always what it planned."""
    converted: int = 0
    skipped: int = 0
    # the old folders at least one file was taken from - the ones that are no
    # longer read by anything and may be put out of the way afterwards
    used: Set[str] = field(default_factory=set)


def run(plan: Plan, root: str, bar=None) -> Result:
    """
    Copies every planned pair and writes the manifests that describe them.

    A file whose pair is not complete in the old folder cannot be converted - a
    half-written pair is worth nothing to anybody - and is counted as skipped
    rather than copied as what it is.

    Each pair lands in the new folder as a pair or not at all: both halves are
    copied to temporary names first and both are renamed afterwards. Two files
    cannot be renamed atomically as one, so a crash between the two renames can
    leave one half of a pair behind, and that is what the index cleans up the next
    time it is read.
    """
    out = Result()
    manifests: Dict[str, CacheFolder] = {}
    bar = bar or _NoBar()

    for mv in plan.moves:
        halves = [mv.bin, energy_bin(mv.bin)]
        srcs = [os.path.join(mv.src_folder, h)
                for h in (mv.src_bin, energy_bin(mv.src_bin))]
        if not mv.src_bin or not all(os.path.isfile(p) for p in srcs):
            out.skipped += 1
            continue
        try:
            os.makedirs(mv.dest, exist_ok=True)
            _copy_pair(srcs, mv.dest, halves)
        except OSError:
            out.skipped += 1
            continue
        folder = manifests.setdefault(
            mv.dest, CacheFolder(path=mv.dest, source=mv.source))
        folder.entries[mv.name] = {
            "bin": mv.bin, "n_coarse": mv.n_coarse, "duration": mv.duration,
            "size": mv.size, "mtime": mv.mtime}
        out.converted += 1
        out.used.add(mv.src_folder)
        bar.step()

    # the manifests last: one that lists a file whose features are not on the
    # disk is a lie the next run would believe
    for dest in sorted(manifests):
        save_folder(manifests[dest])
    return out


def _copy_pair(srcs: List[str], dest: str, halves: List[str]) -> None:
    """
    Copies both halves of one pair into `dest`, each under its final name only
    once its whole content is there and its size is known to be the size of what
    was copied.
    """
    staged: List[Tuple[str, str]] = []
    for src, half in zip(srcs, halves):
        tmp = os.path.join(dest, half + ".tmp")
        shutil.copyfile(src, tmp)
        if os.path.getsize(tmp) != os.path.getsize(src):
            os.remove(tmp)
            raise OSError(f"{half}: the copy is not the size of the original")
        staged.append((tmp, os.path.join(dest, half)))
    for tmp, final in staged:
        os.replace(tmp, final)


# --------------------------------------------------------------------------- #
# Putting the old folders out of the way
# --------------------------------------------------------------------------- #

STASH_NAME = "cache_todelete"


def stash_dir(root: str) -> str:
    """
    Where the converted-from index folders go instead of being deleted.

    A sibling of the cache folder, so that it is next to the program wherever the
    program is, and never inside the tree the program walks - which is what lets
    the old indexes sit there without anybody mistaking them for something the
    search still reads.
    """
    return os.path.join(os.path.dirname(os.path.abspath(root)), STASH_NAME)


def stash(old: List[OldCache], used: Set[str], root: str
          ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Moves the old folders whose files were converted out of the cache and into the
    folder beside it. Nothing is deleted.

    Nothing is deleted because the old index is the only other copy of the
    features it holds, and the format they were just converted into has not been
    run against a real library for long enough to promise it will never want them
    back. So the folder is moved, not emptied, and the user is told where it went
    and that it is theirs to remove.

    A folder whose files could not be converted is left exactly where it is: it is
    still unreadable, and while it is there the window keeps saying so, which is
    the truth about it.

    Returns (moved, failed) - the two are reported separately because a folder
    that would not move is one the user has to hear about rather than assume done.
    """
    target = stash_dir(root)
    moved: List[str] = []
    failed: List[Tuple[str, str]] = []
    for c in old:
        if c.path not in used:
            continue
        name = os.path.basename(c.path)
        try:
            os.makedirs(target, exist_ok=True)
            shutil.move(c.path, _free_path(target, name))
        except (OSError, shutil.Error) as exc:
            failed.append((name, " ".join(str(exc).split())[:120]))
        else:
            moved.append(name)
    return moved, failed


def _free_path(folder: str, name: str) -> str:
    """A path in `folder` that nothing is at yet - the same rule as a clash digit."""
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return path
    base, n = name, 2
    while os.path.exists(os.path.join(folder, f"{base}_{n}")):
        n += 1
    return os.path.join(folder, f"{base}_{n}")


class _NoBar:
    """
    The progress bar that is not there, so that the job can be run without one -
    a test, or a script that only wants the return code. It takes the same calls
    as the real one and drops them.
    """

    def step(self, n: int = 1) -> None:
        pass


__all__ = ["OLD_VERSION", "OldCache", "Plan", "Moved", "Result", "absolute_of",
           "find_old_caches", "incompatible", "make_plan", "run", "stash",
           "stash_dir"]