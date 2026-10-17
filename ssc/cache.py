"""The shared on-disk index (version 2).

The cache is a two-level tree under the program folder:

    cache/<X>_/<Y>_/*.npy  +  manifest.json

<X> is the folder the user picked, or for a drive the top folder on it; <Y> is
the folder that actually holds the audio file, that is the innermost one. The
level-1 name decides nothing about the search - it exists so that one enormous
folder does not end up holding every <Y> of the whole cache - but the level-2
name does: it is how a library folder is found among thousands of cache folders
without opening a single manifest, and it is also where the digit that resolves
a name clash goes.

Every level-2 folder records the source folder it stands for, so a folder that
was moved is still recognised, and one cache serves every search that touches
the same files.

The naming grammar puts the clash digit BEFORE the trailing underscore, never
after it. So `drums_` is the first `drums`, `drums_2` the second, and `drums_2_`
a real library folder named `drums_2`. Reading a name back means dropping the
trailing underscore if there is one, then dropping a trailing digit group after
the last underscore - which is why the underscore has to be the last thing in
the name: with it there, every folder of a given library name is found under
that one name, which is the whole of what the lookup does. The grammar is
therefore unambiguous in both directions, and resolving a clash can never turn
into a cascade.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ssc.config import COARSE_BANDS, COARSE_HOP, COARSE_NFFT, INDEX_VERSION
from ssc.format import _ILLEGAL, safe_stem
from ssc.paths import program_dir

# The pseudo-folder for "the files that sit in the root of a drive". Named so
# that moving those files to the root of another drive needs no renaming.
DISK_ROOT = "_disk_root"

# How long a folder name taken from the library may get. Windows caps a whole
# path rather than a name, and cache/<X>_/<Y>_/ is already using a good part of
# it, so a long library folder name has to give way.
NAME_MAX = 60


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

def safe_name(name: str) -> str:
    """A folder name taken from a library folder, made safe for any OS."""
    stem = _ILLEGAL.sub("_", name).strip(" .")
    stem = re.sub(r"\s+", " ", stem)
    if len(stem) > NAME_MAX:
        stem = stem[:NAME_MAX].rstrip(" .")
    return stem


def level_name(base: str, taken: Set[str]) -> str:
    """
    The folder name for `base`, with a clash digit if that name is taken.

    The digit goes BEFORE the trailing underscore, never after it: `data_` is
    the first `data`, `data_2` the second. That is what makes the name readable
    back - taking everything up to the last underscore has to give `data` again,
    or the second copy of a folder could not be found by its library folder name
    at all. A real library folder called `data_2` therefore becomes `data_2_`,
    which reads back as `data_2`, and never collides with the second `data`.

    Only the first free digit is looked for, so what a folder ends up called
    depends on what the cache already holds and not on the order the library was
    walked in - the same library laid down twice must not land in two places.
    Existing folders are never renamed to tidy the numbering: that would move
    data around on every run.
    """
    plain = f"{base}_"
    if plain not in taken:
        return plain
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def parse_level_name(name: str) -> str:
    """
    The library folder name a cache folder stands for, read back from it.

    The trailing underscore is the separator: what is left after removing it is
    the name itself, and a digit left after the last underscore of the rest is
    the clash number rather than part of the name. So `data_` and `data_2` both
    read as `data`, and `data_2_` reads as `data_2` - a folder really called that.
    A name with no underscore at all is taken as it stands, which is what a
    folder put into the cache by hand should be read as.
    """
    if name.endswith("_"):
        return name[:-1]
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


# --------------------------------------------------------------------------- #
# Names of the files inside a cache folder
# --------------------------------------------------------------------------- #

def index_bin_name(name: str) -> str:
    """
    The file one track's features are stored under — short and safe.

    The digest is not there to tell files with the same name apart, which one
    folder cannot hold anyway, but to keep apart two names that differ only past
    the 60th character: a shortened stem alone would flatten those into one file,
    and the manifest would then hold two entries pointing at the same features, one
    of which would be rebuilt on every run, forever.
    """
    stem = safe_stem(name, 60)
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{digest}_{stem}.npy"


def energy_bin(bin_name: str) -> str:
    """The frame-energy file that belongs to a feature file."""
    stem = bin_name[:-4] if bin_name.endswith(".npy") else bin_name
    return stem + "_e.npy"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def cache_root() -> str:
    """
    Where the whole cache lives: in the program's own folder, so that a portable
    program carries its index with it and a source checkout keeps it where the
    program is run from. One tree for everything - that is the point of v2.
    """
    return os.path.join(program_dir(), "cache")


def nfc(path: str) -> str:
    """NFC form, which is the form a path is written and compared in."""
    return unicodedata.normalize("NFC", path)


def key(path: str) -> str:
    """The form a path is compared in: absolute, NFC, case-folded by the OS."""
    return os.path.normcase(nfc(os.path.abspath(path)))


def is_root(path: str) -> bool:
    """True for a drive root such as `G:\\`, which has no parent to speak of."""
    return os.path.dirname(os.path.abspath(path)) == os.path.abspath(path)


def tail(path: str) -> List[str]:
    """
    The address of a folder without its own name - the part two paths must share
    to be considered the same place. A drive root leaves nothing behind, which is
    why files moved between drive roots rank no candidate above any other.
    """
    parts = os.path.abspath(path).rstrip("\\/").split(os.sep)
    return parts[:-1]


def suffix_len(a: str, b: str) -> int:
    """
    How many address components two folders agree on from the right, that is
    how far their paths run together once the leaf is removed.

    `F:\\stuff\\pack\\part1\\drums` and `G:\\pack\\part1\\drums` agree on `pack`
    and `part1`, two components, so the second is the better candidate for the
    first's files than `G:\\pack\\part2\\drums`, which agrees on none. Counting
    components rather than characters is what makes this work at all: a drive
    letter is one component, not one character.
    """
    ta, tb = tail(a), tail(b)
    n = 0
    while n < min(len(ta), len(tb)) \
            and os.path.normcase(ta[-1 - n]) == os.path.normcase(tb[-1 - n]):
        n += 1
    return n


def lib_name(folder: str) -> str:
    """
    The base name a cache level takes from a library folder. The two levels use
    the same rule, so a folder called `drums` is found under either spelling and
    a drive root is always the same pseudo-folder wherever it is asked about.
    """
    folder = os.path.abspath(folder)
    return DISK_ROOT if is_root(folder) else safe_name(os.path.basename(folder))


def lib_key(folder: str) -> str:
    """
    The form a library folder's name is looked up in.

    Windows does not care how a name is spelled, so the cache must not either:
    `My Music` and `my music` are one folder, and a folder moved from one to the
    other would otherwise be looked for under a spelling it does not have.
    """
    return os.path.normcase(lib_name(folder))


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #

@dataclass
class CacheFolder:
    """One level-2 cache folder: the source folder it stands for, and its files."""
    path: str                                     # absolute path of the folder
    source: str                                   # the source folder itself
    entries: Dict[str, dict] = field(default_factory=dict)   # name -> record

    def has(self, name: str) -> bool:
        return name in self.entries

    def matches(self, name: str, size: int, mtime: float) -> bool:
        """Whether this folder already holds exactly this file, features and all."""
        rec = self.entries.get(name)
        return rec is not None \
            and int(rec.get("size", -1)) == size \
            and abs(float(rec.get("mtime", -1.0)) - mtime) < 1e-6


def load_folder(path: str) -> Optional[CacheFolder]:
    """
    Reads one manifest, or None if there is nothing usable here.

    A manifest from another index version, or built with other feature
    parameters, describes numbers that cannot be compared with ours whatever
    the files look like now, so it is ignored as if it were absent. The folder
    is left where it is: data we cannot read is not ours to delete.
    """
    mpath = os.path.join(path, "manifest.json")
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != INDEX_VERSION \
                or data.get("coarse_nfft") != COARSE_NFFT \
                or data.get("coarse_hop") != COARSE_HOP \
                or data.get("coarse_bands") != COARSE_BANDS:
            return None
        return CacheFolder(path=os.path.abspath(path),
                           source=nfc(data.get("folder", "")),
                           entries={e["name"]: e
                                    for e in data.get("entries", [])})
    except (OSError, ValueError, KeyError):
        return None


def save_folder(folder: CacheFolder) -> None:
    """Writes the manifest, through a temporary file so a crash cannot cut it."""
    os.makedirs(folder.path, exist_ok=True)
    path = os.path.join(folder.path, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": INDEX_VERSION,
                   "coarse_nfft": COARSE_NFFT, "coarse_hop": COARSE_HOP,
                   "coarse_bands": COARSE_BANDS,
                   "folder": nfc(folder.source),
                   "entries": [{"name": n, "bin": e["bin"],
                                "n_coarse": int(e["n_coarse"]),
                                "duration": float(e["duration"]),
                                "size": int(e["size"]),
                                "mtime": float(e["mtime"])}
                               for n, e in sorted(folder.entries.items())]},
                  fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# One pass over the tree
# --------------------------------------------------------------------------- #

@dataclass
class CacheScan:
    """What a single pass over the cache found."""
    root: str = ""
    level1: Set[str] = field(default_factory=set)         # level-1 names in use
    level2: Dict[str, Set[str]] = field(default_factory=dict)   # level-1 -> names
    by_lib: Dict[str, List[str]] = field(default_factory=dict)  # base name -> paths


def scan(root: str) -> CacheScan:
    """
    One pass over both levels: which level-1 names exist, which level-2 names
    sit under each, and which cache folders answer to which library folder name.

    This is the whole of what the lookup costs, because manifests are NOT opened
    here. They are read only for the folders a library folder actually competes
    for, which is what keeps a search over a cache of hundreds of thousands of
    files from turning into hundreds of thousands of JSON parses. On a cache of
    a few thousand folders the pass itself costs single-digit milliseconds.

    A version-1 folder holds .npy files and no subfolder, so it contributes
    nothing to any of the three tables and is simply not seen - which is how the
    old cache stays harmless, and stays the user's to delete.
    """
    out = CacheScan(root=os.path.abspath(root))
    if not os.path.isdir(out.root):
        return out
    with os.scandir(out.root) as it1:
        for e1 in it1:
            if not e1.is_dir():
                continue
            out.level1.add(e1.name)
            under = out.level2.setdefault(e1.name, set())
            with os.scandir(e1.path) as it2:
                for e2 in it2:
                    if not e2.is_dir():
                        continue
                    under.add(e2.name)
                    out.by_lib.setdefault(
                        os.path.normcase(parse_level_name(e2.name)),
                        []).append(e2.path)
    return out


# --------------------------------------------------------------------------- #
# The library side
# --------------------------------------------------------------------------- #

@dataclass
class LibFile:
    """One audio file of the current search."""
    path: str                    # absolute
    name: str                    # its name inside Y
    size: int
    mtime: float


@dataclass
class LibFolder:
    """
    One folder of the library that holds audio files of this search.

    It is at once the folder the search asks about and the Y the cache folder is
    named after, which is why all its files lie directly in it: a subfolder with
    audio in it is a LibFolder of its own and lands in a cache folder of its own.
    """
    folder: str
    files: List[LibFile]

    def wanted(self) -> Set[Tuple[str, int, float]]:
        return {(f.name, f.size, f.mtime) for f in self.files}

    def has_lib(self, name: str, size: int, mtime: float) -> bool:
        return (name, size, mtime) in self.wanted()


def group_library(files: Sequence[Tuple[str, int, float]]
                  ) -> Dict[str, LibFolder]:
    """Groups the library's files by the folder holding them."""
    out: Dict[str, List[LibFile]] = {}
    for path, size, mtime in files:
        parent = os.path.dirname(os.path.abspath(path))
        out.setdefault(parent, []).append(
            LibFile(path=os.path.abspath(path), name=os.path.basename(path),
                    size=size, mtime=mtime))
    return {folder: LibFolder(folder=folder, files=fs)
            for folder, fs in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Step 1 - which cache folder belongs to which library folder
# --------------------------------------------------------------------------- #

@dataclass
class Step1:
    """The outcome of matching library folders against the cache."""
    folders: Dict[str, CacheFolder] = field(default_factory=dict)
    # library folder -> the cache folder that was taken for it
    assigned: Dict[str, str] = field(default_factory=dict)
    moved: List[Tuple[str, str]] = field(default_factory=list)  # old, new source


def step1(scanned: CacheScan, lib: Dict[str, LibFolder]) -> Step1:
    """
    Decides, for every library folder, which cache folder is its own.

    A cache folder stands for one library folder and to no other, so a folder
    that has been taken is never offered again - otherwise two library folders
    would end up sharing one, their files would be written into the same
    manifest, and the `folder` in it would name whichever of the two happened
    to be processed last.

    The two ways of recognising a folder are therefore tried in two passes over
    all the library folders, not in one pass per folder. Exact recognition comes
    first, for every folder, before any of them is guessed at. A folder that
    records whose it is is that folder's own, and a guesser must not be able to
    take it away just because its name sorts earlier: that would leave the real
    owner with a fresh cache folder and hand it to a copy.

    Only when there is no exact match are the candidates found by name considered,
    and then the test is strict: the cache folder must hold exactly the files of
    the library folder, all of them and nothing else. Half a match is worse than
    no match, because a partial one would leave the folder half-built forever. The
    candidates are tried from the one whose path runs together with the library
    folder's for the longest first, so that two identically organised packs give
    up the right one.
    """
    out = Step1()
    claimed: Set[str] = set()

    def load(path: str) -> Optional[CacheFolder]:
        if path not in out.folders:
            out.folders[path] = load_folder(path)      # may be None
        return out.folders[path]

    def candidates_of(folder: str) -> List[str]:
        # sorted, so that two folders claiming the same source are taken in the
        # same order whatever order the directory walk happened to produce
        return sorted(scanned.by_lib.get(lib_key(folder), []))

    def take(folder: str, path: str, cf: CacheFolder) -> None:
        out.assigned[folder] = path
        claimed.add(path)
        if key(cf.source) != key(folder):
            out.moved.append((cf.source, folder))

    order = sorted(lib.items())

    # pass 1 - the folder that says whose it is
    for folder, _lf in order:
        for path in candidates_of(folder):
            if path in claimed:
                continue
            cf = load(path)
            if cf is not None and key(cf.source) == key(folder):
                take(folder, path, cf)
                break

    # pass 2 - the folder that holds exactly the right files and nothing else
    for folder, lf in order:
        if folder in out.assigned:
            continue
        want = lf.wanted()
        ranked = []
        for path in candidates_of(folder):
            if path in claimed:
                continue
            cf = load(path)
            if cf is None or not cf.source:
                continue
            ranked.append((suffix_len(folder, cf.source),
                           os.path.normcase(cf.source), path, cf))
        ranked.sort(key=lambda t: (-t[0], t[1], t[2]))
        for _score, _src, path, cf in ranked:
            have = {(n, int(e.get("size", -1)), float(e.get("mtime", -1.0)))
                    for n, e in cf.entries.items()}
            if have == want:
                take(folder, path, cf)
                break
    return out


# --------------------------------------------------------------------------- #
# Step 2 - where the files nothing claimed go
# --------------------------------------------------------------------------- #

@dataclass
class Step2:
    """The cache folder each library folder ends up in, and which are new."""
    target: Dict[str, str] = field(default_factory=dict)   # lib folder -> path
    created: Dict[str, str] = field(default_factory=dict)  # path -> source


def step2(scanned: CacheScan, lib: Dict[str, LibFolder], matched: Step1,
          top_of: Dict[str, str]) -> Step2:
    """
    Gives a cache folder to every library folder that step 1 did not recognise.

    `top_of` maps a file to the X it belongs to: the folder of the library
    record that brought it in, which the caller resolves, because knowing what
    the user asked for is the caller's business and not the cache's.

    Files are collected first and named only afterwards. A folder's digit must
    not depend on the order the walk met its files in, or the same library laid
    down twice would land in two different places.
    """
    out = Step2()
    out.target.update(matched.assigned)

    # what wants a level-1 folder, and which library folders go under each
    tops: Dict[str, List[str]] = {}
    for folder, lf in sorted(lib.items()):
        if folder in out.target:
            continue
        for f in lf.files:
            # looked up case-folded: the map is keyed that way, and a path
            # spelled with another case still names the same file
            x = top_of.get(key(f.path), folder)
            tops.setdefault(x, []).append(folder)
    tops = {x: sorted(set(fs)) for x, fs in tops.items()}

    taken1 = set(scanned.level1)
    level1_of: Dict[str, str] = {}
    for x in sorted(tops):
        name = level_name(lib_name(x), taken1)
        taken1.add(name)
        level1_of[x] = name

    for x in sorted(tops):
        l1 = level1_of[x]
        l1_path = os.path.join(scanned.root, l1)
        taken2 = set(scanned.level2.get(l1, set()))
        for folder in tops[x]:
            name = level_name(lib_name(folder), taken2)
            taken2.add(name)
            path = os.path.join(l1_path, name)
            out.target[folder] = path
            out.created[path] = folder
    return out


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #

@dataclass
class Assignment:
    """Which cache folder, and which file in it, every library file belongs to."""
    target: Dict[str, str] = field(default_factory=dict)   # lib folder -> path
    files: Dict[str, Dict[str, LibFile]] = field(default_factory=dict)
    # cache folder path -> the source folder it must end up recording
    source: Dict[str, str] = field(default_factory=dict)
    folders: Dict[str, CacheFolder] = field(default_factory=dict)  # opened ones
    moved: List[Tuple[str, str]] = field(default_factory=list)


def assign(scanned: CacheScan, lib: Dict[str, LibFolder],
           top_of: Dict[str, str]) -> Assignment:
    """Runs both steps and lays every library file out on the cache tree."""
    matched = step1(scanned, lib)
    placed = step2(scanned, lib, matched, top_of)

    out = Assignment(target=dict(placed.target), folders=matched.folders,
                     moved=list(matched.moved))
    for folder, cf in matched.folders.items():
        if cf is not None:
            out.source[folder] = cf.source
    for folder, lf in sorted(lib.items()):
        path = placed.target[folder]
        out.files.setdefault(path, {}).update(
            {f.name: f for f in lf.files})
        out.source[path] = folder
    return out
