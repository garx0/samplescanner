"""Cutting the found fragment out, and putting one run together."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
import numpy as np
from ssc.audio import (extract_fragment, ffmpeg_bin, probe_audio, run_hidden,
                       shift_pitch)
from ssc.config import (EXPORT_PREROLL, MAX_PATH_CHARS,
                        MIN_STEM, ORIG_PITCH_SUBDIR, PATH_TAIL_RESERVE,
                        QUERY_STEM_MAX, RESULTS_STAMP, SOURCE_STEM_MAX)
from ssc.format import fmt_stamp, safe_stem
from ssc.paths import results_root
from ssc.pitch import fmt_semitones
from ssc.report import Hit, Report

# --------------------------------------------------------------------------- #
# Export of the found fragment
# --------------------------------------------------------------------------- #


def results_folder(prefix: str, query_root: str, search_root: str,
                   when: Optional[datetime] = None) -> Tuple[str, str]:
    """
    The folder of one run: <results folder>/<query>__<search>__<date and time>,
    plus the suffix that the log file of that run shares.

    query_root is the name of the -q folder (or the stem of the -q file when a
    single file was given), search_root the name of the -s folder. The
    timestamp is the moment of the run, which keeps the names unique — two
    runs of the same pair of folders never share a folder.
    """
    stamp = (when or datetime.now()).strftime(RESULTS_STAMP)
    suffix = f"{safe_stem(query_root, 40)}__{safe_stem(search_root, 40)}__{stamp}"
    return os.path.join(results_root(prefix), suffix), suffix


def unique_path(directory: str, stem: str, ext: str) -> str:
    """A path in directory; if the file exists, _2, _3, ... is appended."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, stem + ext)
    if not os.path.exists(path):
        return path
    n = 2
    while os.path.exists(os.path.join(directory, f"{stem}_{n}{ext}")):
        n += 1
    return os.path.join(directory, f"{stem}_{n}{ext}")


def _shorten(name: str, limit: int) -> str:
    """
    Cuts a name down to `limit` characters, keeping its BEGINNING: that is where
    the catalogue number and the artist sit, and they identify a track far
    better than its tail does.
    """
    if limit >= len(name):
        return name
    return name[:limit].rstrip(" .") or name[:1]


def fragment_stem(query_name: str, source_rel: str, offset: float,
                  directory: str, max_path: int = MAX_PATH_CHARS,
                  query_max: int = QUERY_STEM_MAX,
                  source_max: int = SOURCE_STEM_MAX,
                  tag: str = "") -> str:
    """
    The name of an exported fragment: <query>__<source>__<timestamp><tag>, cut
    down so the WHOLE path stays inside max_path. `tag` marks the
    pitch-corrected copy (e.g. "+3st") and goes right after the timestamp.

    What limits Windows is the total path, not one name: the results folder
    already carries both folder names and a timestamp, and the query name and
    the track name are both long. The source name therefore gives way first —
    the query name says which query this fragment belongs to, and the
    timestamp says where inside the track it was taken, so both stay whole as
    long as possible. The source name is only shortened after that.
    """
    stamp = fmt_stamp(offset) + tag
    q = safe_stem(query_name, query_max)
    src = safe_stem(source_rel, source_max)
    # room left for the query and the source together: the full path minus the
    # folder, minus the extension and the _2 of a repeated name, minus the two
    # separators and the timestamp, which are never shortened
    room = (max_path - len(os.path.abspath(directory))
            - PATH_TAIL_RESERVE - 2 * len("__") - len(stamp))
    if room < len(q) + len(src):
        src = _shorten(src, max(MIN_STEM, min(source_max, room - len(q))))
        q = _shorten(q, max(MIN_STEM, min(query_max, room - len(src))))
    return f"{q}__{src}__{stamp}"


def _encode_ffmpeg(x: np.ndarray, sr: int, channels: int, path: str,
                   codec_args: Sequence[str]) -> bool:
    """Encodes an array to a file through ffmpeg (from raw float32)."""
    payload = np.ascontiguousarray(x, dtype=np.float32)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels),
           "-ar", str(sr), "-i", "-"] + list(codec_args) + [path]
    try:
        proc = run_hidden(cmd, input=payload.tobytes(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return False
    return proc.returncode == 0 and os.path.exists(path) \
        and os.path.getsize(path) > 0


def save_fragment(seg: np.ndarray, sr: int, channels: int, directory: str,
                  stem: str) -> str:
    """
    Saves the fragment as a 16-bit WAV: the format every player, editor and
    sampler takes without asking, and no container or codec in the way.

    The name is built AFTER the extension is known — unique_path only compares
    files that would really be overwritten if the name were the same.
    """
    wav = unique_path(directory, stem, ".wav")
    if _encode_ffmpeg(seg, sr, channels, wav, ["-c:a", "pcm_s16le"]):
        return wav
    raise RuntimeError(f"Could not save the fragment to {directory}")


def exported_length(offset: float, duration: float,
                    preroll: float = EXPORT_PREROLL) -> float:
    """
    How long the exported fragment is: `duration` — the length of the WHOLE
    query in the source timeline — plus the pre-roll at both ends, so that a
    position which is a little late cannot eat into the sample from either
    side (only the one before the start of the file is missing).
    """
    return duration + preroll + min(preroll, max(0.0, offset))


def export_match(query_name: str, source_rel: str, source_path: str,
                 offset: float, duration: float, results_dir: str,
                 preroll: float = EXPORT_PREROLL,
                 pitch: float = 0.0) -> List[str]:
    """
    Cuts the found fragment out of the source and puts it into results_dir.
    File name: <query>__<source without folder>__<timestamp>, shortened if
    needed so the whole path fits into MAX_PATH_CHARS (see fragment_stem).

    The cut starts `preroll` seconds before the found position and ends that much
    later as well, so a position that is slightly late still yields the complete
    sample from both ends (see EXPORT_PREROLL). The name keeps reporting the
    found position. `duration` is the length of the WHOLE query in the source
    timeline (see source_span_of_query), not the length of the trimmed signal
    that was actually searched: the fragment has to hold the query from end to
    end, whatever trimming and pitch shifting took off before the search.

    With a non-zero `pitch` two files are written. The one in the run folder is
    shifted back by -pitch, so it sounds like the query that was searched, and
    it is marked with the shift after the timestamp, e.g. "...__0m25s686_+3st".
    The fragment as it lies in the source — the source's OWN pitch, untouched —
    goes into the orig_pitch subfolder, so the two are never mixed up. When the
    result is not confident the run folder is already unsure/, and the fragment
    lands in unsure/orig_pitch/.
    Returns the paths of the saved files, the source-pitch one first.
    """
    start = max(0.0, offset - preroll)
    sr, channels = probe_audio(source_path)
    seg = extract_fragment(source_path, start,
                           exported_length(offset, duration, preroll),
                           sr, channels)
    shifted = abs(pitch) > 1e-9
    plain_dir = os.path.join(results_dir, ORIG_PITCH_SUBDIR) if shifted \
        else results_dir
    if shifted:
        os.makedirs(plain_dir, exist_ok=True)
    saved = [save_fragment(seg, sr, channels, plain_dir,
                           fragment_stem(query_name, source_rel, offset,
                                         plain_dir))]
    if shifted:
        back = shift_pitch(seg, -pitch)
        tag = f"_{fmt_semitones(-pitch)}st"
        saved.append(save_fragment(
            back, sr, channels, results_dir,
            fragment_stem(query_name, source_rel, offset, results_dir,
                          tag=tag)))
    return saved


def export_report(rep: Report, run_dir: str, unsure_dir: str,
                  export_enabled: bool, search_folder: str,
                  unsure_min_sim: float) -> float:
    """
    Writes the fragments of one report and prints what was written. Returns the
    time it took.

    A convincing match goes into the run folder directly, a not convincing
    enough one into its unsure/ subfolder — so every query ends up with a
    fragment to listen to. rep.hits holds every equivalent copy of the
    recording, all of them confirmed, so all of them are exported.
    """
    t_one = time.perf_counter()
    if not export_enabled:
        return 0.0
    if rep.exportable:
        dest_dir, note = run_dir, ""
    elif rep.best.similarity >= unsure_min_sim:
        dest_dir = unsure_dir
        note = f"  [uncertain — {rep.export_reason}]"
    else:
        print(f"  FRAGMENT: not exported — {rep.export_reason} "
              f"(below --unsure-min-sim {unsure_min_sim:.2f})")
        return time.perf_counter() - t_one

    hits = rep.hits or [Hit(rep.best.file_name, rep.offset,
                            rep.best.similarity)]
    done: List[Tuple[List[str], Hit]] = []
    for hit in hits:
        try:
            saved = export_match(
                query_name=rep.query_name,
                source_rel=hit.file_name,
                source_path=os.path.join(search_folder, hit.file_name),
                offset=hit.offset, duration=rep.export_duration,
                results_dir=dest_dir,
                pitch=rep.pitch)
            done.append((saved, hit))
        except (RuntimeError, OSError) as exc:
            print(f"  FRAGMENT: could not save {hit.file_name} ({exc})")
    if len(done) > 1:
        print(f"  FRAGMENT: {len(done)} equivalent sources exported:")
    plain_is_shifted = abs(rep.pitch) > 1e-9
    for saved, hit in done:
        for i, path in enumerate(saved):
            if i == 0 and plain_is_shifted:
                tag = f"  [the source's own pitch, in {ORIG_PITCH_SUBDIR}/]"
            elif i == 0:
                tag = ""
            else:
                tag = (f"  [corrected by {fmt_semitones(-rep.pitch)} st, "
                       f"sounds like the query]")
            print(f"            {_shown_path(path)}{tag}"
                  f"{note if hit is done[0][1] and i == 0 else ''}")
    return time.perf_counter() - t_one


def _shown_path(path: str) -> str:
    """
    The path as the report prints it: from the `results` folder down, not the
    whole way from the drive.

    The line ends up in the console and in the log, and a full path is mostly
    the user's own folders repeated — with a deep source path and a long track
    name the useful part is pushed off the right edge. The results folder is
    found by name, so the way back to it is the same wherever the run put it;
    a path without that folder in it (a results-prefix pointing somewhere else)
    is printed as it is, rather than as a chain of `..`.
    """
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts) - 1):
        if parts[i] == "results":
            return os.sep.join(parts[i:])
    return path

