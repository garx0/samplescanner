"""What -q and -s may name, and how it becomes a list of files."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple
from ssc.audio import _is_audio, list_audio_files
from ssc.config import AUDIO_EXTENSIONS
from ssc.format import safe_stem
from ssc.paths import (exists as _exists, is_dir as _is_dir, is_file as _is_file,
                       long_form as _long_form, plain)

def _has_glob_magic(spec: str) -> bool:
    return any(c in spec for c in "*?[")


def _is_pattern(path: str) -> bool:
    """
    Whether this entry has to be read as a glob pattern instead of as a name.

    A name holding '*', '?' or '[' is normally a pattern, but a file or folder
    really called that exists just as well — and read as a pattern it matches
    only what the pattern says, not what is on the disk. So the literal name
    wins: whatever really stands under it is used as it is, and a name is only
    read as a pattern when nothing of that name exists.

    The existence test goes through _exists() rather than os.path.isdir(), so
    that a long name is still recognised as a real name on a machine where
    Windows itself will not look at it.
    """
    if not _has_glob_magic(path):
        return False
    return not _exists(path)


def read_source_list(path: str) -> List[str]:
    """
    Reads a text file that lists what to search: folders, files and glob
    patterns, one per line. Blank lines and lines starting with '#' are ignored,
    and everything after a '#' on a line is treated as a comment — so a path
    that really contains '#' has to be quoted with double quotes.

    A relative entry counts from the folder of this file, not from wherever the
    program happens to be started — see _against.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read the list {path}: {exc}")
    out: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if len(line) > 1 and line[0] in "\"'" and line[-1] == line[0]:
            line = line[1:-1].strip()          # a quoted path may contain '#'
        elif "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _against(list_path: str, entries: List[str]) -> List[str]:
    """
    Works out where the entries of a text list really are.

    A relative entry counts from the folder the LIST is in, not from the working
    directory of whoever started the program. A list of a collection is written
    next to that collection, so "Artist *\\*.flac" in it means that folder — while
    against the working directory it means nothing at all, or, worse, some
    unrelated folder that happens to sit in the program's own directory (which is
    exactly what the window uses). An absolute path, a ~ and a %VAR% are left as
    they are.
    """
    root = os.path.dirname(os.path.abspath(list_path))
    out: List[str] = []
    for entry in entries:
        entry = os.path.expandvars(os.path.expanduser(entry))
        out.append(entry if os.path.isabs(entry)
                   else os.path.join(root, entry))
    return out


def _expand_source_entry(entry: str, recursive: bool
                         ) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """
    Turns one entry of -s into concrete audio files, plus the record each of
    them was picked through.

    A record is what one entry stands for: a folder, one named file, or the
    whole set of hits of one pattern. Its Z is the folder that record stands
    for - the folder itself, the folder holding the named file, or, for a
    pattern, the deepest folder holding every hit. The cache groups by that, so
    the shape of what the user typed decides the shape of the cache and not the
    other way round.
    """
    path = os.path.expanduser(entry)
    entries = (sorted(glob.glob(path, recursive=recursive))
               if _is_pattern(path) else [path])

    anchors: List[str] = []
    records: List[Tuple[str, List[str]]] = []
    files: List[str] = []
    for p in entries:
        p = os.path.expandvars(p)
        if _is_dir(p):
            here: List[str] = []
            # os.walk needs the prefixed form too, or it returns nothing at all
            # for a long path and the collection looks empty - and what it hands
            # back has the prefix stripped again, so no name that reaches the
            # index or the report is spelled in a way nobody typed
            for root, dirs, fs in os.walk(_long_form(p)):
                dirs.sort()
                for n in sorted(fs):
                    if _is_audio(n):
                        here.append(plain(os.path.join(root, n)))
            if here:
                here.sort()
                anchors.append(os.path.abspath(p))
                records.append((os.path.abspath(p), here))
                files.extend(here)
        elif _is_file(p) and _is_audio(p):
            full = os.path.abspath(p)
            z = os.path.dirname(full) or full
            anchors.append(z)
            records.append((z, [p]))
            files.append(p)

    if not records:
        return [], []
    if _is_pattern(path) and len(records) > 1:
        # one pattern is one record: it stands for the deepest folder holding
        # every hit, so that a pattern over a whole library does not scatter the
        # cache across whatever folders the wildcard happened to reach
        merged: List[str] = []
        for _z, here in records:
            merged.extend(here)
        merged.sort()
        records = [(_common_folder(anchors), merged)]
    return files, records


def _common_folder(folders: List[str]) -> str:
    """The deepest folder that contains all of them."""
    if len(folders) == 1:
        return folders[0]
    try:
        common = os.path.commonpath(folders)
    except ValueError:             # not one root: nothing deeper than the first
        return folders[0]
    if not _is_dir(common):        # a single file: commonpath returns the file
        common = os.path.dirname(common) or common
    return common


def _drive_top(file_path: str) -> str:
    """The folder in the root of the drive that holds this file."""
    drive, rest = os.path.splitdrive(file_path)
    parts = [q for q in rest.split(os.sep) if q]
    return os.path.join(drive + os.sep, parts[0]) if parts else file_path


def _is_root(path: str) -> bool:
    path = os.path.abspath(path)
    return os.path.dirname(path) == path


def _depth(path: str) -> int:
    """How deep a folder sits, counted as address components."""
    return len([q for q in os.path.abspath(path).rstrip("\\/").split(os.sep) if q])


def top_folders(records: List[Tuple[str, List[str]]]) -> Dict[str, str]:
    """
    Which library folder X each file belongs to.

    X is the folder of the record, which is what the user named and therefore
    what they will recognise later. The exception is a drive: picking one means
    every folder on it, and there is no single folder to name, so each file goes
    under the top folder of its own drive instead.

    A file that several records cover belongs to the deepest of them - the most
    specific choice - and, at equal depth, to the first by name. Settling it
    here rather than in the cache is what makes the layout independent of the
    order the entries were given in.

    The keys are case-folded, because that is the form the cache looks files up
    in, and a path spelled with a different case still has to find its folder.
    """
    best: Dict[str, Tuple[int, str, str]] = {}     # file -> (-depth, X, X)
    for z, files in records:
        z = os.path.abspath(z)
        for p in files:
            full = os.path.abspath(p)
            if not _is_root(z):
                x = z
            elif os.path.dirname(full) == z:
                x = z                    # the file sits in the drive root itself
            else:
                x = _drive_top(full)
            k = os.path.normcase(os.path.abspath(p))
            cand = (-_depth(x), x, x)
            if k not in best or cand < best[k]:
                best[k] = cand
    return {k: v[2] for k, v in best.items()}


def _as_specs(specs) -> List[str]:
    """
    -q and -s take one path or several of them. A single string stays a single
    string, so every old command line keeps working untouched.
    """
    if isinstance(specs, (str, bytes, os.PathLike)):
        return [os.fspath(specs)]
    return [os.fspath(s) for s in specs]


def _spec_label(spec: str) -> str:
    """
    The name one -q/-s argument goes under in the name of the results folder:

    * a folder goes under its own name;
    * a single file — a query, a source, or a text list of either — goes under
      its name without the extension, which says what was picked better than the
      folder it happens to sit in;
    * a pattern goes under the deepest folder that exists in it, with "_glob"
      added, because a '*' cannot go into a real folder name and is therefore
      spelled out.
    """
    path = os.path.expanduser(spec)
    if os.path.isdir(path):
        return os.path.basename(os.path.normpath(path)) or path
    if os.path.isfile(path):
        # one file was named — an audio file just as much as a list of them
        return os.path.splitext(os.path.basename(path))[0]
    head = os.path.dirname(spec) or spec
    probe = head
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    label = os.path.basename(os.path.normpath(probe)) if probe else "search"
    return safe_stem(label, 36) + "_glob"


def _specs_label(specs: List[str], fallback: str) -> str:
    """
    The name of a run when several paths were given: after the folder they all
    sit in, and if they have none, after the first two of them joined with a
    '+' — which says what was picked better than a hash would.
    """
    if len(specs) == 1:
        return _spec_label(specs[0])
    parents = {os.path.normcase(os.path.dirname(os.path.abspath(
        os.path.expanduser(s)))) for s in specs}
    if len(parents) == 1:
        common = next(iter(parents))
        if os.path.isdir(common):
            return os.path.basename(common) or fallback
    names = [_spec_label(s) for s in specs]
    label = " + ".join(names[:2])
    if len(names) > 2:
        label += f" + {len(names) - 2} more"
    return label or fallback


def resolve_sources(specs, recursive: bool = True
                    ) -> Tuple[str, List[str], str]:
    base, names, _records, label = resolve_source_records(specs, recursive)
    return base, names, label


def resolve_source_records(specs, recursive: bool = True
                           ) -> Tuple[str, List[str],
                                      List[Tuple[str, List[str]]], str]:
    """
    Turns -s into (base folder, file names relative to it, records, label).

    Every argument may be a folder (what it always was), a glob pattern of
    folders and/or files, or a text file of any extension that lists folders,
    files and glob patterns, one per line - and there may be several arguments
    at once, which is what a file browser hands over when several items are
    picked at the same time:

        -s "G:\\music\\Artist *\\*.flac"
        -s my_sources.txt          (one entry per line, relative entries count
                                    from the folder the list itself is in)
        -s library loop.flac "G:\\music\\reprises" D:\\more\\*.wav

    All the entries end up under ONE base folder: the deepest directory that
    contains every file found. Names stay relative to it, which is what the
    report works with. When the files sit on different drives there is no such
    folder and the names are the full paths instead - os.path.join("", name)
    still opens them.

    The records say which folder each file was picked through. The search itself
    has no use for them; the index needs them to lay the cache out, because
    which folder the user named is the only thing that can say which cache
    folder a file belongs to when two of them could take it.
    """
    specs = _as_specs(specs)
    entries: List[str] = []
    for spec in specs:
        path = os.path.expanduser(spec)
        if _is_pattern(path):
            entries.append(spec)
        elif _is_dir(path):
            entries.append(path)
        elif _is_file(path) and not _is_audio(path):
            more = read_source_list(path)    # a text file with a list
            if not more:
                raise FileNotFoundError(
                    f"{spec} is empty - nothing to search.")
            entries.extend(_against(path, more))
        else:
            entries.append(spec)

    abs_paths: List[str] = []
    records: List[Tuple[str, List[str]]] = []
    for e in entries:
        found, recs = _expand_source_entry(e, recursive)
        abs_paths.extend(found)
        records.extend(recs)
    # de-duplicate, keeping the order stable
    seen = set()
    uniq = []
    for p in abs_paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if not uniq:
        raise FileNotFoundError(
            f"No audio files matched {' '.join(repr(s) for s in specs)} - "
            f"check the paths, the glob or the list file.")

    full = [os.path.abspath(p) for p in uniq]
    try:
        base = os.path.commonpath(full)
        if not _is_dir(base):
            # one single file matched: commonpath returns the file itself
            base = os.path.dirname(base) or base
        names = [os.path.relpath(p, base) for p in full]
    except (ValueError, OSError):        # different drives, mixed roots
        base = ""
        names = list(full)
    if any(os.path.isabs(n) for n in names):
        base = ""
    pairs = sorted(zip(names, full), key=lambda t: t[0].lower())
    names = [n for n, _ in pairs]

    return base, names, records, _specs_label(specs, "search")


def resolve_queries(specs, recursive: bool = True,
                    only: Optional[str] = None
                    ) -> Tuple[List[str], str]:
    """
    Every -q argument may be a folder (walked recursively), a single audio
    file, a glob pattern of files, or a text file listing any of those — and
    there may be several arguments at once:

        -q ./query
        -q one_query.wav "G:\\music\\Artist\\*.wav"
        -q my_queries.txt         (a relative entry counts from the folder the
                                   list itself is in)

    All of it ends up in one flat list of query files. The second value is the
    name of the run: the folder name, the stem of a single file, and for
    several paths the folder they share or the names of the first two.
    """
    specs = _as_specs(specs)
    paths: List[str] = []
    silent: List[str] = []          # folders that hold no audio at all
    for spec in specs:
        path = os.path.expanduser(spec)
        if _is_pattern(path):
            hits = sorted(glob.glob(path, recursive=recursive))
            if not hits:
                raise FileNotFoundError(f"No files matched {spec!r}.")
            sub = hits
        elif os.path.isfile(path) and not _is_audio(path):
            sub = _against(path, read_source_list(path))
            if not sub:
                raise FileNotFoundError(f"{spec} is empty — no queries in it.")
        else:
            sub = [path]
        for e in sub:
            e = os.path.expandvars(os.path.expanduser(e))
            if _is_dir(e):
                try:
                    names = list_audio_files(e, recursive=recursive)
                except FileNotFoundError:
                    # a folder without audio is only a problem if there is
                    # nothing else to search — the other picks still count
                    silent.append(e)
                    continue
                if only:
                    names = [n for n in names if only.lower() in n.lower()]
                paths.extend(os.path.join(e, n) for n in names)
            elif _is_file(e):
                if os.path.splitext(e)[1].lower() not in AUDIO_EXTENSIONS:
                    raise FileNotFoundError(f"Not an audio file: {e}")
                # a file named explicitly is never filtered by --only: that
                # filter is meant for picking tracks out of a whole collection
                paths.append(e)
            else:
                raise FileNotFoundError(f"Folder or file not found: {e}")

    seen = set()
    uniq = []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if not uniq:
        shown = ", ".join(specs)
        if only:
            raise FileNotFoundError(
                f"No queries left in {shown} after the --only {only!r} "
                f"filter.")
        if silent:
            raise FileNotFoundError(
                f"No audio files found in {', '.join(silent)}.")
        raise FileNotFoundError(f"No audio files found in {shown}.")
    return uniq, _specs_label(specs, "query")