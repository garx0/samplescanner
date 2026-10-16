"""Audio in, audio out: decoding through ffmpeg, listing, trimming."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import resample_poly
from ssc.config import (AUDIO_EXTENSIONS, BATCH_MAX_COMMAND, BATCH_WINDOWS,
                        EXTRACT_PAD, SAMPLE_RATE)
from ssc.format import exe_name
from ssc.launcher import is_packed, launcher_dir
from ssc.paths import is_dir, is_file, long_form
from ssc.pitch import semitone_ratio

# --------------------------------------------------------------------------- #
# Audio: decoding through ffmpeg
# --------------------------------------------------------------------------- #

_FFMPEG: Optional[str] = None


# A program frozen as a window has no console, so every ffmpeg started from it
# would open a console window of its own for a moment — one per decoded file.
# This flag is what keeps them from appearing.
SUBPROCESS_FLAGS = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                    if os.name == "nt" else {})


def run_hidden(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run without a console window flashing up on Windows."""
    return subprocess.run(cmd, **SUBPROCESS_FLAGS, **kwargs)


def ffmpeg_bin() -> str:
    """
    Locate ffmpeg: in PATH first, then next to the program, then next to the
    exe it was frozen into — a frozen program cannot use its own folder for
    this, because that one is a temporary unpack directory.

    The names beside the program are the ones THIS system gives them, so a build
    unpacked next to the exe works the same on Windows, on macOS and on Linux —
    on the two others the binary carries no .exe, and a rule that looked only for
    ffmpeg.exe would report a program standing right there as missing.
    """
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return found
    # This module's own folder first — a build dropped into the package — and
    # then the folder the package sits in, which is where the program itself is.
    # That second one is the documented place to put ffmpeg ("next to
    # samplescanner.py"), and it is where the window looks as well: when the two
    # disagreed, the window saw ffmpeg standing right there and let the search
    # start, and only the first decoded file then failed with "ffmpeg not
    # found". Frozen, that folder is the unpack directory and the one below is
    # the real one — the folder of the exe, which is next to it anyway.
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.dirname(here)]
    if is_packed():
        # the folder the launcher is in, which is not where this module lives:
        # a packer unpacks the program somewhere else and runs it from there
        roots.append(launcher_dir())
    name = exe_name("ffmpeg")
    for root in roots:
        for cand in (os.path.join(root, "ffmpeg", "bin", name),
                     os.path.join(root, name)):
            if os.path.isfile(cand):
                _FFMPEG = cand
                return cand
    # a habit kept only where it is one: most Windows guides unpack ffmpeg into
    # C:\ffmpeg\bin, and nothing else on any platform ever has that folder
    fallback = os.path.join("C:\\", "ffmpeg", "bin", exe_name("ffmpeg"))
    if os.name == "nt" and os.path.isfile(fallback):
        _FFMPEG = fallback
        return _FFMPEG
    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg AND ffprobe and add them to PATH "
        f"(or place {exe_name('ffmpeg')} and {exe_name('ffprobe')} next to "
        "samplescanner.py).")


def load_audio(path: str, sr: int = SAMPLE_RATE, channels: int = 1,
               start: Optional[float] = None,
               duration: Optional[float] = None) -> np.ndarray:
    """
    Decodes a file to float32. If start/duration are given, only that slice
    is read (fast seek). With channels>1 returns (samples, channels).
    """
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.6f}"]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.6f}"]
    cmd += ["-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(channels), "-ar", str(sr), "-"]
    proc = run_hidden(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not read {path}:\n"
                           + proc.stderr.decode("utf-8", "replace").strip())
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if channels > 1:
        x = x[: (len(x) // channels) * channels].reshape(-1, channels)
    return x


def load_audio_many(windows: Sequence[Tuple[str, float, float]],
                    sr: int = SAMPLE_RATE
                    ) -> List[np.ndarray]:
    """
    Several windows through ONE ffmpeg, each decoded exactly as load_audio
    would have decoded it on its own.

    One candidate of the precise pass is one ffmpeg, and starting one costs far
    more than decoding a few seconds of audio does: measured at 77 ms a
    candidate, of which 57 ms was the process and 17 ms the feature bank. A
    search reads up to 128 candidates per shift, so the starts alone cost more
    than everything else the precise pass does.

    Each window keeps its own input and its own -ss, so it is decoded from the
    same place by the same decoder state as a read of its own - which is the
    whole point. The other way round, one read covering a span and sliced in
    memory, does NOT reproduce separate reads: ffmpeg's seek lands on its own
    landing point and the resampler starts from there, and the two answers
    differ measurably (up to completely unrelated audio for mp3).

    Each branch is padded and then cut to an exact number of samples, and that
    is what makes the segments land where they should: concat lays the stream
    out by the length ffmpeg really produced, which is not always the length
    asked for - one sample more often than not - so cutting is what keeps two
    branches from being read one sample apart. Measured over the test
    collections, what this returns is bit-identical to the individual reads.

    `windows` is a list of (path, start, duration). Falls back to reading the
    windows one by one whenever anything about the batched form is refused,
    which is also what happens when a group holds a single window.
    """
    out: List[np.ndarray] = []
    for group in _window_groups(windows):
        out.extend(_read_group(group, sr) or
                   [load_audio(p, sr=sr, start=s, duration=d)
                    for p, s, d in group])
    return out


def _window_groups(windows: Sequence[Tuple[str, float, float]]
                   ) -> List[List[Tuple[str, float, float]]]:
    """
    The windows split into groups ffmpeg can be asked for in one go: at most
    BATCH_WINDOWS of them, and not so many that the command line comes near
    what Windows will accept (a long collection path is repeated once per
    input, and CreateProcess stops at 32767 characters).
    """
    groups: List[List[Tuple[str, float, float]]] = []
    group: List[Tuple[str, float, float]] = []
    size = 0
    for w in windows:
        need = len(w[0]) + 48
        if group and (len(group) >= BATCH_WINDOWS or
                      size + need > BATCH_MAX_COMMAND):
            groups.append(group)
            group, size = [], 0
        group.append(w)
        size += need
    if group:
        groups.append(group)
    return groups


def _read_group(group: Sequence[Tuple[str, float, float]],
                sr: int) -> Optional[List[np.ndarray]]:
    """
    One ffmpeg for a group of windows, or None if it would not do it - in which
    case the caller reads them one by one. Refusing is always possible and costs
    only the time that was already going to be spent.
    """
    if len(group) < 2:
        # a group of one is a read with a filter graph around it: all the ways
        # to fail, and nothing to gain
        return None
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error"]
    counts: List[int] = []
    for path, start, duration in group:
        cmd += ["-ss", f"{max(0.0, start):.6f}",
                "-t", f"{max(0.0, duration):.6f}", "-i", path]
        counts.append(int(round(max(0.0, duration) * sr)))
    graph = []
    for i, n in enumerate(counts):
        graph.append(f"[{i}:a]aresample={sr},aformat=channel_layouts=mono,"
                     f"apad,atrim=end_sample={n},"
                     f"asetpts=PTS-STARTPTS[b{i}]")
    graph.append("".join(f"[b{i}]" for i in range(len(group)))
                 + f"concat=n={len(group)}:v=0:a=1[out]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[out]",
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = run_hidden(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    flat = np.frombuffer(proc.stdout, dtype=np.float32)
    if len(flat) < sum(counts):
        return None
    out, at = [], 0
    for n in counts:
        out.append(flat[at:at + n].copy())
        at += n
    return out


def ffprobe_bin() -> str:
    """ffprobe, next to ffmpeg; falls back to a name swap for a build that has
    both in one folder (which is every build)."""
    cand = os.path.join(os.path.dirname(ffmpeg_bin()), exe_name("ffprobe"))
    if os.path.isfile(cand):
        return cand
    return ffmpeg_bin().replace("ffmpeg", "ffprobe")


def probe_audio(path: str) -> Tuple[int, int]:
    """Sample rate and channel count of the source file."""
    try:
        out = run_hidden(
            [ffprobe_bin(), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        ).stdout.decode().strip().splitlines()[-1].split(",")
        return int(out[0]), int(out[1])
    except Exception:
        return 44100, 2


def audio_duration(path: str) -> float:
    """
    How long an audio file is, in seconds — read from its header, nothing is
    decoded. That is what makes it cheap enough to ask about every file of a
    collection before deciding whether to index it.

    Raises RuntimeError when the length cannot be read at all; the caller
    decides what to do with a file it cannot measure.
    """
    try:
        out = run_hidden(
            [ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        ).stdout.decode().strip()
        seconds = float(out.splitlines()[-1])
        if seconds > 0:
            return seconds
    except Exception:
        pass
    # ffprobe said nothing useful; ffmpeg's own banner line carries the length
    out = run_hidden([ffmpeg_bin(), "-i", path], stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)",
                  out.stderr.decode("utf-8", "replace"))
    if not m:
        raise RuntimeError(f"cannot read the length of {path}")
    hours, minutes, seconds = m.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def list_audio_files(folder: str, recursive: bool = False) -> List[str]:
    """
    Lists audio files in a folder. With recursive=True all subfolders are
    walked and the result contains paths relative to folder — so a name can
    both open the file and show the user which subfolder it lives in.
    Everything that is not audio (artwork, .cue, .nfo, .m3u, ...) is dropped
    by extension.
    """
    if not is_dir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names: List[str] = []
    # os.walk and listdir are given the long-path form as well: on a machine where
    # Windows will not look past MAX_PATH, the ordinary form sees nothing at all
    # and a collection full of audio reports itself as empty. The relative name is
    # worked out against that same form, so the prefix cancels out and never
    # reaches the index.
    walk_root = long_form(folder)
    if recursive:
        for root, dirs, files in os.walk(walk_root):
            dirs.sort()                       # predictable traversal order
            # rel is worked out against the same prefixed form os.walk was given,
            # so the prefix cancels out here and never reaches the names below
            rel = os.path.relpath(root, walk_root)
            for n in files:
                if os.path.splitext(n)[1].lower() in AUDIO_EXTENSIONS:
                    names.append(n if rel == os.curdir
                                 else os.path.join(rel, n))
    else:
        for n in os.listdir(walk_root):
            path = os.path.join(folder, n)
            if is_file(path) and \
                    os.path.splitext(n)[1].lower() in AUDIO_EXTENSIONS:
                names.append(n)
    if not names:
        what = "in the nested folders" if recursive else ""
        raise FileNotFoundError(f"No audio files found in {folder} {what}.")
    return sorted(names, key=str.lower)


def _is_audio(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS



# Trimming the silence around a query. The noise around a cut-out sample is
# rarely clean silence: it sits on a noise floor whose level has nothing to do
# with how loud the file is, and there can be two of them (a very quiet digital
# floor and a louder one next to the sample). So nothing is compared to 0 dBFS —
# both limits are derived from the file itself, which is what makes the trim
# work the same on a normalised and on a quiet query:
#   * the noise floor is estimated as a low percentile of the frame levels, so a
#     handful of loud frames cannot lift it;
#   * the cut line is the LOOSER of "within TRIM_PEAK_REL_DB of the peak" and
#     "TRIM_FLOOR_MARGIN_DB above the noise floor", so neither a loud file nor a
#     compressed one with a high noise floor can eat into the signal.
TRIM_FRAME_MS = 2.0            # resolution of the level profile; this is what
                                # actually decides where the cut lands, so it
                                # is fine enough to put it on the boundary
TRIM_NOISE_PERCENTILE = 10.0   # low percentile that stands for the noise floor
TRIM_FLOOR_MARGIN_DB = 12.0    # how far above the floor still counts as noise
TRIM_PEAK_REL_DB = -38.0       # never trim anything closer than this to the peak
TRIM_MAX_DYN = 50.0            # cap: the floor can never be read as peak - more
TRIM_KEEP_MS = 0.0             # the frame already contains the onset, so the
                                # cut lands on the boundary without extra slack
TRIM_MIN_KEEP_MS = 30.0        # below this the file is left alone entirely


def trim_silence(x: np.ndarray, sr: int = SAMPLE_RATE,
                 rel_db: float = TRIM_PEAK_REL_DB,
                 frame_ms: float = TRIM_FRAME_MS,
                 margin_db: float = TRIM_FLOOR_MARGIN_DB,
                 keep_ms: float = TRIM_KEEP_MS) -> Tuple[np.ndarray, float]:
    """
    Removes the silence and the noise around a query, keeping the useful part.
    Returns (trimmed signal, seconds cut from the front).
    """
    if x.size == 0:
        return x, 0.0
    xf = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(xf)))
    if peak <= 1e-9:
        return x, 0.0
    n = len(xf)
    frame = max(1, int(frame_ms * sr / 1000.0))
    # one RMS per frame, frame i covering exactly x[i : i+frame]; the last
    # frame may be shorter, which is why the divisor is clipped as well
    cum = np.concatenate([[0.0], np.cumsum(np.abs(xf) ** 2)])
    starts = np.arange(0, n, frame)
    ends = np.minimum(starts + frame, n)
    env = np.sqrt((cum[ends] - cum[starts]) / np.maximum(ends - starts, 1))
    peak_db = 20.0 * np.log10(peak)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
    # the floor estimate is capped, so a file that is signal almost everywhere
    # cannot turn its own quietest tenth into a noise floor
    floor_db = min(float(np.percentile(env_db, TRIM_NOISE_PERCENTILE)),
                   peak_db - TRIM_MAX_DYN)
    thr_db = max(peak_db + rel_db, floor_db + margin_db)
    loud = np.flatnonzero(env_db > thr_db)
    if loud.size == 0:
        return x, 0.0
    keep = max(0, int(keep_ms * sr / 1000.0))
    start = max(0, int(starts[loud[0]]) - keep)
    end = min(n, int(starts[loud[-1]]) + frame + keep)
    if end - start < TRIM_MIN_KEEP_MS * sr / 1000.0:
        return x, 0.0
    return np.ascontiguousarray(xf[start:end]), start / sr


def source_start_of_query(y_start: float, trim_shift: float,
                          pitch: float) -> float:
    """
    Where the query's OWN first sample sits in the source timeline.

    The search runs on the trimmed query resampled by semitone_ratio(pitch),
    and that signal is compared with the source sample by sample — so the
    position the search reports is where the TRIMMED query starts. The file
    itself began trim_shift earlier, and one second of the query covers
    1/semitone_ratio(pitch) seconds of the source (a query that was resampled
    down plays slower and therefore reaches less far into the source), so the
    head that trimming removed is trim_shift/ratio seconds of source and the
    real beginning of the query is that much EARLIER, not later.

    Getting this backwards costs twice the trimmed head: the reported position
    moves late by trim_shift*(1 + 1/ratio), and since the export starts one
    pre-roll before the reported position, the exported fragment then begins
    2*trim_shift/ratio - preroll after the sample's true onset — on a query
    with 100 ms of quiet lead that is a fifth of a second of missing audio.
    """
    return y_start - trim_shift / semitone_ratio(pitch)


def source_span_of_query(raw_duration: float, pitch: float) -> float:
    """
    How long the WHOLE query occupies in the source timeline — what the
    exported fragment has to cover, regardless of how much of it was trimmed
    off before searching. Without this the fragment is cut to the length of
    the trimmed search signal and the quiet head and tail are lost.
    """
    return raw_duration / semitone_ratio(pitch)


def extract_fragment(path: str, offset: float, duration: float,
                     sr: int, channels: int) -> np.ndarray:
    """
    Cuts the piece [offset, offset+duration) out of a source at its native
    sample rate and channel count. Slightly more than needed is read and then
    trimmed exactly by samples — that compensates for the seek inaccuracy.
    """
    duration = max(duration, 1.0 / sr)
    start = max(0.0, offset - EXTRACT_PAD)
    want_end = offset + duration + EXTRACT_PAD
    block = load_audio(path, sr=sr, channels=channels, start=start,
                       duration=want_end - start)
    a = int(round((offset - start) * sr))
    b = a + int(round(duration * sr))
    seg = block[max(0, a): max(0, b)]
    if seg.size == 0:                       # file shorter than expected
        seg = block
    return seg


# The resample ratio as an integer fraction: 1000 is fine enough that the
# nearest 12th of an octave is within a couple of cents.
PITCH_PRECISION = 1000


def shift_pitch(x: np.ndarray, semitones: float) -> np.ndarray:
    """
    Shifts the pitch by `semitones`, changing the tempo along with it (a plain
    resample by 2^(n/12)). A positive value raises the pitch and shortens the
    signal. Zero is returned untouched, so it is safe to call with 0.
    """
    if x.size == 0 or abs(semitones) < 1e-9:
        return x
    ratio = semitone_ratio(semitones)
    # resample_poly(x, up, down) makes the content that filled len samples
    # occupy len*up/down samples — it plays back at up/down of the original
    # speed, so the pitch moves with it. Raising the pitch by `ratio` therefore
    # means the content must shrink by the same factor: up/down = 1/ratio.
    # One formula covers both directions.
    up = PITCH_PRECISION
    down = max(1, int(round(PITCH_PRECISION * ratio)))
    y = resample_poly(np.asarray(x, dtype=np.float64), up, down)
    return np.ascontiguousarray(y, dtype=np.float32)


def _decode_slice(path: str, start: float, duration: float) -> np.ndarray:
    """Decodes only the needed slice of a file (faster than reading it all)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", path,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-"]
    proc = run_hidden(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not read the slice of {path}: "
                           + proc.stderr.decode("utf-8", "replace").strip())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()

