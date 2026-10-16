"""The on-disk index of a collection, and the coarse search on it."""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import fftconvolve

from ssc.audio import (audio_duration, list_audio_files, load_audio,
                       load_audio_many)
from ssc.config import (BANK, BATCH_WINDOWS, COARSE_BANDS, COARSE_DTYPE,
                        COARSE_HOP,
                        COARSE_MIN_FRAMES, COARSE_NFFT, COARSE_PER_MEMBER,
                        ENERGY_FLOOR, EPS, FINE_CANDIDATES, FINE_MAX_TOTAL,
                        FINE_MIN_SAMPLES, FINE_PAD, HOP,
                        INDEX_BYTES_PER_FILE,
                        INDEX_BYTES_PER_SECOND, INDEX_SIZE_WARN,
                        INDEX_VERSION, LOCAL_EXCLUSION, LOCAL_REFINE,
                        MIN_STD_FLOOR,
                        MIN_STD_RATIO, REF_TAIL_PAD, SAMPLE_RATE)
from ssc.features import (PreparedRef, build_views, coarse_logmel, frame_center,
                          frame_correlations, logmel_bank, prepare_ref,
                          similarity_at, time_delta)
from ssc.format import _ILLEGAL, fmt_duration, fmt_size, safe_stem
from ssc.paths import program_dir
from ssc.progress import Progress

# --------------------------------------------------------------------------- #
# Search index
# --------------------------------------------------------------------------- #


@dataclass
class SourceIndex:
    name: str
    feats: Dict[Tuple[int, int], np.ndarray]
    n_frames: int
    duration: float
    views: Optional[Dict[Tuple[int, int, bool], np.ndarray]] = None
    energy: Optional[np.ndarray] = None      # energy of each frame
    prepared: Optional[Dict[Tuple[int, int, bool], PreparedRef]] = None

    def prepared_view(self, key: Tuple[int, int, bool]) -> PreparedRef:
        """Prepares (and caches) the source for searching with feature key."""
        if self.prepared is None:
            self.prepared = {}
        need = self.views[key].shape[0]
        prep = self.prepared.get(key)
        if prep is None or prep.pad < need:
            prep = prepare_ref(self.views[key], max(REF_TAIL_PAD, need))
            self.prepared[key] = prep
        return prep


def _index_file(folder: str, name: str,
                max_seconds: float = 0.0) -> SourceIndex:
    # max_seconds>0 reads only the beginning of the file — fast check mode
    x = load_audio(os.path.join(folder, name),
                   duration=max_seconds if max_seconds > 0 else None)
    feats = logmel_bank(x)
    n = max(m.shape[0] for m in feats.values())
    # frame energy from the most detailed band grid
    energy = np.exp(feats[(2048, 48)]).sum(axis=1).astype(np.float64)
    return SourceIndex(name=name, feats=feats, n_frames=n,
                       duration=len(x) / SAMPLE_RATE, energy=energy)


@dataclass
class SearchIndex:
    """Collection index: the slim on-disk one (CoarseIndex) plus, optionally,
    features fully computed in memory (used by --selftest)."""
    folder: str
    sources: List[SourceIndex] = field(default_factory=list)
    coarse: Optional[CoarseIndex] = None

    def by_name(self, name: str):
        if self.coarse is not None:
            return self.coarse.by_name(name)
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(name)


@dataclass
class CoarseEntry:
    """One source file inside the on-disk index."""
    name: str                  # path relative to the search root
    path: str                  # full path to the audio file
    bin: str                   # name of the file holding the features
    n_coarse: int
    duration: float
    size: int
    mtime: float


@dataclass
class IndexPlan:
    """What indexing a collection would cost, decided before anything is
    written: which files are missing, how much audio they hold and how many
    bytes that turns into on the drive the cache sits on."""
    total: int                  # files of the collection
    cached: int                 # of them, already in the cache
    missing: List[str]          # the ones that would have to be computed
    seconds: float              # audio length of the missing ones, s
    unknown: List[str]          # files whose length could not be read
    bytes_estimate: int         # what they will take on disk
    index_dir: str
    free_bytes: int             # free space on the drive holding the cache

    @property
    def fits(self) -> bool:
        return self.bytes_estimate <= self.free_bytes

    @property
    def is_big(self) -> bool:
        return self.bytes_estimate > INDEX_SIZE_WARN


def free_space_at(path: str) -> int:
    """
    Free bytes on the drive that will hold `path`. The cache folder itself may
    not exist yet, so the nearest existing parent is asked instead.
    """
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def format_plan(plan: IndexPlan) -> List[str]:
    """The plan as the lines a user reads before the index is built."""
    if not plan.missing:
        return [f"all {plan.cached} file(s) of the collection are already in "
                f"the cache; nothing new has to be written"]
    what = f"{len(plan.missing)} of {plan.total} file(s) are not in the cache"
    if plan.cached:
        what += f", {plan.cached} are"
    lines = [what,
             f"they hold ≈{fmt_duration(plan.seconds)} of audio and will "
             f"take about {fmt_size(plan.bytes_estimate)} of index "
             f"({INDEX_BYTES_PER_SECOND // 1024} KB/s)"
             f" — in cache/{os.path.basename(plan.index_dir)}",
             f"free space on the drive: {fmt_size(plan.free_bytes)}"]
    if plan.unknown:
        lines.append(f"the length of {len(plan.unknown)} file(s) could not be "
                     f"read; they are not in the estimate")
    return lines


class CoarseIndex:
    """
    The collection index, kept on disk. Every file gets a pair of .npy files:
    log-mel (float16) and frame energy (float32). A manifest records them, so
    on repeated runs everything unchanged is reused (size and modification
    time are compared).

    Only one file is loaded into memory at a time, so the memory cost depends
    on the length of the longest single track, not on the collection size.
    """
    def __init__(self, folder: str, index_dir: str):
        self.folder = folder
        self.abs_folder = os.path.abspath(folder)
        self.dir = index_dir
        self.entries: List[CoarseEntry] = []
        self.bytes_on_disk = 0

    # ---------- planning ----------

    def plan(self, names: Sequence[str]) -> IndexPlan:
        """
        What the index of this collection WOULD cost, before anything is
        written: which of the files are not in the cache yet, how much audio
        that is, and how many bytes it takes on disk.

        Nothing is created and nothing is decoded — the length of a file is
        read from its header. The number is an estimate, and it is meant to run
        slightly high: measured against real indexes it is about 5 % over,
        because the estimate cannot see the .npy headers and the slack of the
        file system.
        """
        todo, cached = self._partition(names)
        missing = [n for n, prev in todo if prev is None]
        seconds = 0.0
        unknown: List[str] = []
        for n in missing:
            try:
                seconds += audio_duration(os.path.join(self.folder, n))
            except (RuntimeError, OSError, ValueError):
                unknown.append(n)
        need = int(seconds * INDEX_BYTES_PER_SECOND
                   + len(missing) * INDEX_BYTES_PER_FILE)
        return IndexPlan(total=len(names), cached=cached, missing=missing,
                         seconds=seconds, unknown=unknown,
                         bytes_estimate=need,
                         index_dir=self.dir,
                         free_bytes=free_space_at(self.dir))

    def _partition(self, names: Sequence[str]
                   ) -> Tuple[List[Tuple[str, Optional[dict]]], int]:
        """
        Splits the files into (name, what the manifest has for it) — a dict to
        reuse, or None to compute. Used by plan() and by build() so that "what
        is missing" is decided in exactly one place.
        """
        old = self._load_manifest()
        wanted = {n: self._stat(n) for n in names}
        todo: List[Tuple[str, Optional[dict]]] = []
        cached = 0
        for n in names:
            size, mtime = wanted[n]
            prev = old.get(n)
            if prev and prev.get("size") == size \
                    and abs(prev.get("mtime", -1) - mtime) < 1e-6 \
                    and os.path.exists(os.path.join(self.dir, prev["bin"])) \
                    and os.path.exists(os.path.join(self.dir,
                                                    _energy_bin(prev["bin"]))):
                todo.append((n, prev))     # reuse
                cached += 1
            else:
                todo.append((n, None))
        return todo, cached

    # ---------- building ----------

    def build(self, names: Sequence[str], jobs: int = 0,
              verbose: bool = False) -> Tuple[int, int]:
        """
        Completes the index. Returns (rebuilt, taken from cache).

        A progress line runs while it works: the files already in the cache are
        counted as done straight away — there is nothing to compute for them —
        and the rest are counted one by one as their features land on the disk.

        A file that cannot be DECODED is a warning and nothing more: one damaged
        file in a folder of two thousand must not cost the user the search of
        the other 1999. It is named on the console, left out of the manifest and
        of the index, and remembered in `self.skipped`. Anything else — a full
        disk, a cache that cannot be written — still stops the run: those are not
        the file's fault, and a half-written index would be worse than none.
        """
        os.makedirs(self.dir, exist_ok=True)
        todo, cached = self._partition(names)
        wanted = {n: self._stat(n) for n in names}
        fresh = [t for t in todo if t[1] is None]
        if verbose and cached:
            print(f"  {cached} file(s) taken from the index cache, "
                  f"{len(fresh)} recomputed", flush=True)

        # the files already in the cache need nothing computed, so the bar
        # starts where the work really starts
        bar = Progress(len(todo), text="building index", done=cached).start()
        self.skipped: List[str] = []
        try:
            # only the files that are not in the cache yet: the cached ones are
            # already counted, and `one` never sees them
            def one(name):
                size, mtime = wanted[name]
                bin_name = _index_bin_name(name)
                try:
                    audio = load_audio(os.path.join(self.folder, name))
                except (RuntimeError, OSError) as exc:
                    self.skipped.append(name)
                    why = (" ".join(str(exc).split())[:150] or exc.__class__.__name__)
                    why = why.rstrip(": ")
                    print(f"  WARNING: skipping {name} — the file cannot be "
                          f"read: {why}", flush=True)
                    return name, None, False
                data, energy = coarse_logmel(audio)
                np.save(os.path.join(self.dir, bin_name),
                        data.astype(COARSE_DTYPE))
                np.save(os.path.join(self.dir, _energy_bin(bin_name)), energy)
                rec = {"bin": bin_name, "n_coarse": int(data.shape[0]),
                       "duration": float(len(data) * COARSE_HOP / SAMPLE_RATE),
                       "size": size, "mtime": mtime}
                bar.step()
                return name, rec, False

            fresh_names = [n for n, _ in fresh]
            if fresh_names and jobs != 1 and len(fresh_names) > 1:
                with futures.ThreadPoolExecutor(
                        max_workers=jobs if jobs > 0 else
                        max(1, (os.cpu_count() or 2) - 1)) as pool:
                    done = list(pool.map(one, fresh_names))
            else:
                done = [one(n) for n in fresh_names]
        finally:
            bar.close()            # also when a file turned out to be unreadable
        recs: Dict[str, dict] = {n: rec for n, rec, _ in done if rec is not None}
        for n, prev_rec in todo:
            if prev_rec is not None:
                recs[n] = prev_rec

        self.entries = []
        for n in names:
            rec = recs.get(n)
            if rec is None:      # skipped above: it is not in the index
                continue
            self.entries.append(CoarseEntry(
                name=n, path=os.path.join(self.folder, n), bin=rec["bin"],
                n_coarse=int(rec["n_coarse"]), duration=float(rec["duration"]),
                size=int(rec["size"]), mtime=float(rec["mtime"])))
        self._save_manifest()
        self._measure_disk()
        return len(fresh), cached

    def _stat(self, name: str) -> Tuple[int, float]:
        st = os.stat(os.path.join(self.folder, name))
        return st.st_size, st.st_mtime

    def _load_manifest(self) -> Dict[str, dict]:
        path = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("version") != INDEX_VERSION \
                    or data.get("coarse_nfft") != COARSE_NFFT \
                    or data.get("coarse_hop") != COARSE_HOP \
                    or data.get("coarse_bands") != COARSE_BANDS:
                # the numbers in the index were computed with other parameters,
                # so none of it can be reused whatever the files look like now
                return {}
            # The folder this cache was built for is recorded, but no longer
            # decides anything. A collection spread over several drives has no
            # one folder to name — for that, and for a program that has been
            # moved to another disk to escape a full one, the address was the
            # only thing tying a perfectly good index to where it was built, and
            # moving the program threw away gigabytes of decoding for nothing.
            # What the cache actually holds is decided per file, by size and
            # modification time, in _split() — a stricter test than comparing
            # one string with another, and the reason two collections of the same
            # name still do not get mixed up.
            return {e["name"]: e for e in data.get("entries", [])}
        except (OSError, ValueError, KeyError):
            return {}

    def _save_manifest(self) -> None:
        path = os.path.join(self.dir, "manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": INDEX_VERSION,
                       "coarse_nfft": COARSE_NFFT, "coarse_hop": COARSE_HOP,
                       "coarse_bands": COARSE_BANDS,
                       "folder": self.abs_folder,
                       "entries": [{"name": e.name, "bin": e.bin,
                                    "n_coarse": e.n_coarse,
                                    "duration": e.duration, "size": e.size,
                                    "mtime": e.mtime} for e in self.entries]},
                      fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def _measure_disk(self) -> None:
        self.bytes_on_disk = 0
        for e in self.entries:
            for p in (os.path.join(self.dir, e.bin),
                      os.path.join(self.dir, _energy_bin(e.bin))):
                if os.path.exists(p):
                    self.bytes_on_disk += os.path.getsize(p)

    # ---------- reading ----------

    def by_name(self, name: str) -> CoarseEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def load_entry(self, e: CoarseEntry) -> Tuple[np.ndarray, np.ndarray]:
        """Reads the features and energy of one file (as float32 in RAM)."""
        feats = np.load(os.path.join(self.dir, e.bin),
                        mmap_mode=None).astype(np.float32)
        energy = np.load(os.path.join(self.dir, _energy_bin(e.bin))
                         ).astype(np.float32)
        return feats, energy


def _energy_bin(bin_name: str) -> str:
    """Name of the frame-energy file belonging to a feature file."""
    stem = bin_name[:-4] if bin_name.endswith(".npy") else bin_name
    return stem + "_e.npy"


def _index_bin_name(name: str) -> str:
    """Short, safe file name for the features (no paths, no collisions)."""
    stem = safe_stem(name, 60)
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{digest}_{stem}.npy"


def default_index_dir(folder: str) -> str:
    """
    Default index location: <program folder>/cache/<collection> — next to the
    exe, so that a portable program keeps its index with itself and a source
    checkout keeps it in the folder the program is run from.

    Two collections with the same name in different folders would land in one
    cache folder. The manifest remembers the full path of the collection it was
    built for and refuses a cache belonging to another one, so they do not mix
    up even if their file names, sizes and timestamps happen to agree.
    """
    name = _ILLEGAL.sub("_", os.path.basename(os.path.abspath(folder)))
    name = re.sub(r"\s+", " ", name).strip(" .")[:60].rstrip(" .")
    return os.path.join(program_dir(), "cache", name or "collection")


# --------------------------------------------------------------------------- #
# Coarse search + precise verification of candidates
# --------------------------------------------------------------------------- #


@dataclass
class CoarseCandidate:
    file_name: str
    time: float             # approximate offset, s
    coarse_z: float
    similarity: float = 0.0
    frame_support: float = 0.0
    refined_time: float = 0.0


def coarse_views(x: np.ndarray) -> Dict[bool, np.ndarray]:
    """
    Coarse features of the QUERY: computed separately, with the COARSE_HOP
    step. They must not be taken from logmel_bank — that uses HOP, so the
    query frames would not be aligned in time with the index frames.

    A query shorter than COARSE_MIN_FRAMES coarse frames (a percussive hit of
    ~60 ms is already there) is padded with silence at its END. Padding at the
    end keeps the alignment — the content still starts at frame 0, so the peak
    of the profile is still the right position — and only dilutes the score,
    which is all the coarse stage is asked for. The precise pass afterwards
    works with the real, unpadded query, so nothing is lost.
    """
    need = COARSE_NFFT + (COARSE_MIN_FRAMES - 1) * COARSE_HOP
    if len(x) < need:
        x = np.concatenate([x, np.zeros(need - len(x), dtype=x.dtype)])
    base = frame_center(coarse_logmel(x)[0])
    return {False: base, True: frame_center(time_delta(base))}


class CoarseSide:
    """
    The part of the coarse scan of one file that no query can change.

    Everything in here comes from the stored features of the file alone: the
    centred frames of each feature variant, the running sum of their squares
    and the spread per band. A batch builds it once per file and then asks that
    one file for a profile against every query and every pitch shift of the
    batch, which is what lets the collection be walked once instead of once per
    shift. What is NOT in here is the convolution - that one is a question of
    the query, and it is the part that has to be done again for each of them.

    The variants are built only when a query needs them: a file shorter than
    the query has no use for the one it would be compared with.
    """

    __slots__ = ("n", "energy", "_views", "_energy_sum")

    def __init__(self, n: int, energy: np.ndarray) -> None:
        self.n = n
        self.energy = energy
        self._views: Dict[bool, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._energy_sum: Optional[Tuple[float, np.ndarray]] = None

    def view(self, feats: np.ndarray, delta: bool
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(frames, running sum of their squares, spread per band), one variant.

        The running sum is band major - (bands, frames+1), not (frames+1, bands)
        - because everything that reads it takes one band at a time (see
        _corr_profile), and a column of an (n, bands) array is touched again for
        every band, which is the whole array each time over.
        """
        got = self._views.get(delta)
        if got is None:
            # same as in the main path: centre across frames (removes the level),
            # then across bands along time - otherwise a spurious mean_Q * sum(R)
            # term enters the covariance and pollutes the profile
            R = frame_center(time_delta(feats) if delta else feats)
            R = np.ascontiguousarray(R - R.mean(axis=0, keepdims=True),
                                     dtype=np.float32)
            cs = np.zeros((self.n + 1, R.shape[1]))
            np.cumsum(R.astype(np.float64) ** 2, axis=0, out=cs[1:])
            got = self._views[delta] = (R, np.ascontiguousarray(cs.T),
                                        R.std(axis=0) + EPS)
        return got

    def energy_sum(self) -> Tuple[float, np.ndarray]:
        """(median energy of the file, running sum of it) - read by every window."""
        if self._energy_sum is None:
            cs = np.concatenate(
                [[0.0], np.cumsum(self.energy.astype(np.float64))])
            self._energy_sum = (float(np.median(self.energy)) + EPS, cs)
        return self._energy_sum


def _scan_side(feats: np.ndarray, side: CoarseSide,
               qviews: Dict[bool, np.ndarray]) -> np.ndarray:
    """
    Coarse match profile of one file against one query, over all offsets.
    Returns (n,) - a score in z-like units; -inf where the window is
    too quiet.
    """
    n = side.n
    acc = np.zeros(n)
    used = 0
    window = 1
    for delta, Q in qviews.items():
        T = Q.shape[0]
        window = T
        if T > n:
            continue
        R, cs, std = side.view(feats, delta)
        prof = _corr_profile(Q, R, cs, std)
        if prof is None:
            continue
        sd = prof.std()
        if sd < 1e-9:
            continue
        acc += (prof - prof.mean()) / sd
        used += 1
    if used == 0:
        return np.full(n, -np.inf)
    acc /= used
    return _apply_energy_gate(acc, side, window)


def coarse_scan_file(entry: CoarseEntry, feats: np.ndarray, energy: np.ndarray,
                     qviews: Dict[bool, np.ndarray]) -> np.ndarray:
    """
    Coarse match profile over all offsets of one file.
    Returns (n_coarse,) — a score in z-like units; -inf where the window is
    too quiet.
    """
    n = min(entry.n_coarse, feats.shape[0])
    side = CoarseSide(n, energy[:n])
    return _scan_side(np.ascontiguousarray(feats[:n], dtype=np.float32),
                      side, qviews)


def _corr_profile(Q: np.ndarray, R: np.ndarray, cs: np.ndarray,
                  global_std: np.ndarray) -> Optional[np.ndarray]:
    """
    Cosine profile of Q over all offsets of R (see profile_from_prepared).

    Everything the score is made of - the RMS of the window, how far its bands
    are from constant, the weighted correlation - is a quantity per offset AND
    per band, and written out for all of it at once that is eight or nine
    arrays of (n, bands) float64, four megabytes each. For a four-minute file
    that is some sixty megabytes of traffic per query, and it was where the time
    of a pass went: the convolution itself is a third of it.

    So the bands are taken one at a time. The working set is then a handful of
    n-element rows that stay in cache, and `cs` is stored band major so that a
    band is a contiguous row rather than a column of the whole array.

    The two sums over bands are still taken over a whole (n, bands) buffer,
    because numpy adds along an axis pairwise and adding by hand would round
    differently. Each band is the same operation in the same order on the same
    numbers as before, so what comes out is the same number down to the last
    bit - which is the whole requirement here: this may be quicker, never
    different.
    """
    T, B = Q.shape
    n = R.shape[0]
    if T == 0 or B == 0 or n < T:
        return None
    q_std = Q.std(axis=0, dtype=np.float64)
    Qc = np.ascontiguousarray(Q - Q.mean(axis=0, dtype=np.float64),
                              dtype=np.float32)
    # the tail is padded with zeros (= the file mean): it contributes nothing
    # to the covariance and makes the windows at the very end complete
    Rp = R
    if T > 1:
        Rp = np.vstack([R, np.zeros((T - 1, B), dtype=R.dtype)])
    cov = fftconvolve(Rp, Qc[::-1], mode="valid", axes=0)[:n]
    covT = np.ascontiguousarray(cov.T)          # a band is then one row
    # the two buffers the sums over bands are taken in, band major so that the
    # rows are written in one go - filling a column of an (n, bands) array costs
    # a cache line per offset, and that is as bad as the array it replaces.
    # They are turned round once at the end, so that the sums are taken over
    # the same layout as before.
    prodT = np.empty((B, n), dtype=np.float64)
    weightT = np.zeros((B, n), dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    denom = np.empty(n, dtype=np.float64)
    corr = np.empty(n, dtype=np.float64)
    ok = np.empty(n, dtype=bool)
    head = n - T                              # offsets whose window fits wholly
    for j in range(B):
        qj = q_std[j]
        # cs holds the running sum over R alone. Past its end the padded frames
        # are zeros, and adding zeros never changes a sum - so reading cs there
        # as its last value gives the number the padded running sum would hold.
        run = cs[j]
        np.subtract(run[T:n], run[:head], out=w[:head])
        np.subtract(run[n], run[head:n], out=w[head:])
        w /= T
        np.maximum(w, 0.0, out=w)
        np.sqrt(w, out=w)
        # a band nearly constant on either side says nothing about the match
        np.minimum(w, qj, out=lo)
        np.maximum(w, qj, out=hi)
        hi += EPS
        np.divide(lo, hi, out=hi)
        np.greater(hi, MIN_STD_RATIO, out=ok)
        ok &= w > MIN_STD_FLOOR * global_std[j]
        np.multiply(w, T * qj, out=denom)
        corr.fill(0.0)
        np.divide(covT[j], denom, out=corr, where=ok & (denom > 1e-9))
        np.clip(corr, -1.0, 1.0, out=corr)
        corr *= ok
        np.copyto(weightT[j], lo, where=ok)
        # corr is already zero where the band said nothing, so this is the same
        # product as corr * weight without reading the weight back
        np.multiply(corr, lo, out=prodT[j])
    wsum = np.ascontiguousarray(weightT.T).sum(axis=1)
    return np.clip(np.divide(np.ascontiguousarray(prodT.T).sum(axis=1), wsum,
                             out=np.zeros(n), where=wsum > EPS), -1.0, 1.0)


def _apply_energy_gate(prof: np.ndarray, side: CoarseSide,
                       window: int) -> np.ndarray:
    med, cs = side.energy_sum()
    idx = np.arange(side.n)
    end = np.minimum(idx + window, side.n)
    avg = (cs[end] - cs[idx]) / np.maximum(end - idx, 1)
    return np.where(avg >= ENERGY_FLOOR * med, prof, -np.inf)


def _coarse_peaks(prof: np.ndarray, count: int) -> List[int]:
    finite = np.isfinite(prof)
    if not finite.any():
        return []
    masked = np.where(finite, prof, -np.inf)
    order = np.argsort(masked)[::-1]
    excl = max(1, int(round(LOCAL_EXCLUSION * SAMPLE_RATE / COARSE_HOP)))
    picked: List[int] = []
    for k in order:
        if not np.isfinite(masked[k]) or masked[k] <= 0:
            break
        if all(abs(int(k) - p) > excl for p in picked):
            picked.append(int(k))
        if len(picked) >= count:
            break
    return picked


def fine_window(entry: CoarseEntry, t_guess: float,
                duration: float) -> Tuple[float, str, float]:
    """
    What one candidate has to be read: from where, how much, and of which file.

    Split out of fine_verify because the reads of a whole batch now happen
    together (load_audio_many) — this says what to ask for, that says when.
    """
    start = max(0.0, t_guess - FINE_PAD)
    return start, entry.path, duration + 2 * FINE_PAD


def fine_verify(entry: CoarseEntry, t_guess: float, duration: float,
                qviews: Dict[Tuple[int, int, bool], np.ndarray]
                ) -> Tuple[float, float, float]:
    """
    Precise check of one candidate on its own: re-read a window of a few
    seconds around the guessed position from the original file and compute the
    full feature bank. That way the final numbers are identical to what a search
    over a fully computed collection would give.
    """
    start, path, want = fine_window(entry, t_guess, duration)
    return fine_check(load_audio(path, sr=SAMPLE_RATE, start=start,
                                 duration=want), start, t_guess, duration,
                      qviews)


def fine_check(x: np.ndarray, start: float, t_guess: float, duration: float,
               qviews: Dict[Tuple[int, int, bool], np.ndarray]
               ) -> Tuple[float, float, float]:
    """
    Everything a candidate's numbers need except the reading of its window, so
    that the same work is done whether the window came alone or with thirty
    others out of one ffmpeg.
    """
    # A candidate near the end of a file short enough to run out before the query
    # does has nothing to compare with: the window ffmpeg returned stops in the
    # middle of the query, similarity_at can only answer "not measurable", and
    # that reads in the report as r=-1 and, in the end, as NOT FOUND for a file
    # that really holds the sample. A loop sample of 0.3 s is short, not
    # unfindable, so the window is lengthened with silence instead.
    #
    # `duration` is the length of the query BEFORE it was trimmed and before any
    # pitch shift, so it is never shorter than what is actually compared — which
    # is what covers the sweep as well: a positive shift shortens the query by
    # 2^(n/12), and the window does not shrink with it.
    #
    # The silence sits behind the content, so it adds no weight to the comparison
    # and shifts no offset.
    need = max(FINE_MIN_SAMPLES,
               int(round((t_guess - start + duration) * SAMPLE_RATE)) + 2048)
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    rviews = build_views(logmel_bank(x))
    stats = {key: (R.mean(axis=0), R.std(axis=0) + EPS)
             for key, R in rviews.items()}
    k0 = int(round((t_guess - start) * SAMPLE_RATE / HOP))
    best = (-2.0, k0)
    for probe in range(max(0, k0 - LOCAL_REFINE), k0 + LOCAL_REFINE + 1):
        num = den = 0.0
        for n_fft, n_mels, delta, weight in BANK:
            key = (n_fft, n_mels, delta)
            r_bar, gstd = stats[key]
            s = similarity_at(qviews[key], rviews[key], probe, r_bar, gstd)
            if s > -1.0:
                num += weight * s
                den += weight
        if den > 0 and num / den > best[0]:
            best = (num / den, probe)
    similarity, best_k = best
    n_fft, n_mels, delta = BANK[0][0], BANK[0][1], BANK[0][2]
    key0 = (n_fft, n_mels, delta)
    Q = qviews[key0]
    win = rviews[key0][best_k: best_k + Q.shape[0]]
    support = 0.0
    if win.shape[0] == Q.shape[0]:
        _pf, support = frame_correlations(Q, win)
    return start + best_k * HOP / SAMPLE_RATE, similarity, support


def coarse_search(index: CoarseIndex, query_x: np.ndarray,
                  qfeats: Dict[Tuple[int, int], np.ndarray],
                  duration: float, verbose: bool = False
                  ) -> Tuple[List[CoarseCandidate], List[CoarseCandidate]]:
    """
    Two-pass search of one query: the coarse pass over the on-disk index
    shortlists candidates, the precise pass reads only those from the original
    files and computes the full feature bank.
    """
    return coarse_search_many(index, [(query_x, qfeats, duration)])[0]


def coarse_search_many(
        index: CoarseIndex,
        queries: Sequence[Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray],
                                float]],
        progress: Sequence = None,
        verify=None,
        walk_done=None) -> List[Tuple[List[CoarseCandidate],
                                     List[CoarseCandidate]]]:
    """
    The same two-pass search for several queries at once - which is what a pitch
    sweep is: every shift of a query is a query of its own - with ONE walk over
    the collection instead of one per query.

    Each query gets exactly the list of candidates, the shortlist and the
    precise pass it would have got on its own: the score of a file against a
    query at a shift depends on nothing outside those three, so how many of
    them are answered in one walk cannot change it. What the walk saves is the
    part of the work that belongs to the FILE - reading its features, centring
    them, the running sum of their squares - which is the same for every query
    of the batch and is done once per file.

    `progress` is stepped once per file, so a caller can show how far the walk
    has got; a walk over a large collection is minutes long and silence in the
    middle of it reads as a hang. `verify` is called with the number of windows
    the precise pass is going to read, and the bar it returns is stepped once
    per window: the walk finishing is not the answer arriving, and without a
    second bar the time in between looks like the program stopped. `walk_done`
    is called at that same moment, so that the bar of the walk can be finished
    there - left open it would measure its own line after the precise pass had
    added its time and its rate to it, and say it had read the files slower than
    it did.

    Returns one (precise, rough) pair per query, in the order they came in.
    """
    qviews_coarse = [coarse_views(x) for x, _f, _d in queries]
    qviews_full = [build_views(f) for _x, f, _d in queries]
    # a query longer than a file cannot be compared with it at all; knowing the
    # shortest query of the batch saves reading a file none of them fits in
    shortest = [min(Q.shape[0] for Q in qc.values()) for qc in qviews_coarse]
    rough: List[List[CoarseCandidate]] = [[] for _ in queries]
    for entry in index.entries:
        if progress is not None:
            progress.step()
        if not any(s <= entry.n_coarse for s in shortest):
            continue
        feats, energy = index.load_entry(entry)
        n = min(entry.n_coarse, feats.shape[0])
        side = CoarseSide(n, energy[:n])
        feats = np.ascontiguousarray(feats[:n], dtype=np.float32)
        for i, qc in enumerate(qviews_coarse):
            if shortest[i] > n:
                continue
            prof = _scan_side(feats, side, qc)
            for k in _coarse_peaks(prof, COARSE_PER_MEMBER):
                rough[i].append(CoarseCandidate(
                    file_name=entry.name, time=k * COARSE_HOP / SAMPLE_RATE,
                    coarse_z=float(prof[k])))
        del feats, energy, side
    if walk_done is not None:
        walk_done()
    # the shortlists are drawn for the whole batch before the first window is
    # read, so that the precise pass knows how many it is going to read: that
    # number is what its bar counts, and guessing it would leave a bar that
    # fills past the end
    picked = [_shortlist(r) for r in rough]
    bar = verify(sum(len(p) for p in picked)) if verify is not None else None
    try:
        fine: List[List[CoarseCandidate]] = [[] for _ in picked]
        # The windows of every shortlisted candidate of the whole batch, read
        # together: one ffmpeg for a group of them instead of one each, which is
        # where most of the time of the precise pass went (see BATCH_WINDOWS).
        # The candidates are walked in order and only one group's worth of audio
        # is held at a time - a batch of 36 shifts has 4608 of them, and holding
        # every window would cost a gigabyte.
        plan = []
        for i, (_x, _f, dur) in enumerate(queries):
            for c in picked[i]:
                entry = index.by_name(c.file_name)
                start, path, want = fine_window(entry, c.time, dur)
                # A window that runs past the end of its source is left out of
                # the groups below and read on its own. How much audio there
                # actually is decides what such a window looks like - the rest is
                # silence - and a group read pads each window out to a length it
                # can only guess, which would put a different amount of silence
                # into it than a read of its own does. There are a handful of
                # such candidates: they are the ones at the end of a file.
                alone = start + want > entry.duration - 4.0 / SAMPLE_RATE
                plan.append((i, c, dur, start, path, want, alone))

        audio: List[Optional[np.ndarray]] = [None] * len(plan)
        group: List[int] = []

        def read_group(keys: List[int]) -> None:
            got = load_audio_many([(plan[k][4], plan[k][3], plan[k][5])
                                   for k in keys])
            for k, x in zip(keys, got):
                audio[k] = x

        for at, item in enumerate(plan):
            _i, _c, _d, start, path, want, alone = item
            if alone:
                audio[at] = load_audio(path, sr=SAMPLE_RATE, start=start,
                                       duration=want)
                continue
            group.append(at)
            if len(group) >= BATCH_WINDOWS:
                read_group(group)
                group = []
        if group:
            read_group(group)

        for (i, c, dur, start, _path, _want, _alone), x in zip(plan, audio):
            t, sim, support = fine_check(x, start, c.time, dur, qviews_full[i])
            fine[i].append(CoarseCandidate(
                file_name=c.file_name, time=c.time, coarse_z=c.coarse_z,
                similarity=sim, frame_support=support, refined_time=t))
            if bar is not None:
                bar.step()
        for f in fine:
            f.sort(key=lambda c: -c.similarity)
        return list(zip(fine, rough))
    finally:
        if bar is not None:
            bar.close()


def _shortlist(rough: List[CoarseCandidate]) -> List[CoarseCandidate]:
    """What the coarse pass shortlists out of its own rough candidates."""
    rough.sort(key=lambda c: -c.coarse_z)

    # The coarse z of one file is NOT comparable with the z of another: it is a
    # z-score inside that file's own profile, so a flat, repetitive file hands
    # out a high z for a mediocre peak, while a busy file has to be much better
    # in absolute terms to reach the same number. Ranking purely by z therefore
    # lets a handful of files push the true one out of the budget. Measured on
    # a 20-query set: the file that really holds the sample was rank 4 of 71 by
    # z for one query and rank 56 for another — and in the precise pass that
    # same candidate was the best match of all (r=0.729 against 0.535).
    #
    # So every file offers its best peak first, and only then do the second,
    # third, ... peaks of any file compete. That is what the coarse stage is
    # asked for: recall, not precision — the precise pass decides.
    by_file: Dict[str, List[CoarseCandidate]] = {}
    for c in rough:
        lst = by_file.setdefault(c.file_name, [])
        if len(lst) < COARSE_PER_MEMBER:
            lst.append(c)
    budget = min(len(rough), max(FINE_CANDIDATES * 2,
                                 min(len(by_file), FINE_MAX_TOTAL)))
    picked: List[CoarseCandidate] = []
    for round_no in range(COARSE_PER_MEMBER):
        if len(picked) >= budget:
            break
        for name, lst in by_file.items():        # insertion order = z order
            if len(picked) >= budget:
                break
            if round_no < len(lst):
                picked.append(lst[round_no])
    return picked


def build_index(folder: str, jobs: int = 0, verbose: bool = False,
                recursive: bool = True, only: Optional[str] = None,
                max_seconds: float = 0.0,
                names: Optional[Sequence[str]] = None) -> SearchIndex:
    # `names` lets the caller hand over a set that did not come from a single
    # folder — see resolve_sources; they are still relative to `folder`.
    if names is None:
        names = list_audio_files(folder, recursive=recursive)
    if only:
        names = [n for n in names if only.lower() in n.lower()]
        if not names:
            raise FileNotFoundError(
                f"Nothing left in {folder} after the --only {only!r} filter.")
    workers = jobs if jobs > 0 else min(len(names), max(1, (os.cpu_count() or 2) - 1))
    tail = (f", first {max_seconds:g} s of each" if max_seconds > 0 else "")
    if verbose:
        print(f"Indexing {folder} ({len(names)} file(s){tail}) ...", flush=True)
    work = _index_file if max_seconds <= 0 else \
        (lambda f, n: _index_file(f, n, max_seconds))
    bar = Progress(len(names), text="building index").start()
    try:
        def work_one(name):
            src = work(folder, name)
            bar.step()
            return src

        if workers > 1 and len(names) > 1:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                built = list(pool.map(work_one, names))
        else:
            built = [work_one(n) for n in names]
    finally:
        bar.close()
    if verbose:
        for s in built:
            note = " (truncated)" if max_seconds > 0 else ""
            print(f"  {s.name}  —  {s.duration:7.1f} s, {s.n_frames} frames{note}",
                  flush=True)
    return SearchIndex(folder=folder, sources=built)

