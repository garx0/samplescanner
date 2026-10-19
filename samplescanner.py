"""SampleScanner — find the source of an audio sample.

Takes a short audio file (the query) from ./query and reports which of the long
audio files in ./search contains the matching fragment, and at what timestamp.

This file is the command line: it parses the arguments, runs the queries and
prints what came out. Everything it does lives in the ssc package, one concern
per module:

    ssc/config.py     every number the program is tuned by
    ssc/audio.py      decoding through ffmpeg, listing files, trimming a query
    ssc/inputs.py     what -q and -s may name, and how it becomes a file list
    ssc/features.py   log-mel spectrograms and the correlation profiles
    ssc/index.py      the on-disk index of a collection, and the coarse search
    ssc/search.py     the precise pass, the margins and the decision
    ssc/report.py     the result of one query, and the text a human reads
    ssc/export.py     cutting the fragment out and putting one run together
    ssc/pitch.py      the --pitch argument and the resampling for each shift
    ssc/selftest.py   synthetic queries with a known source
    ssc/format.py     how a number, a time or a name is written
    ssc/paths.py      where the program writes, and the right to do so

samplescanner_gui.py is the window; it hands its arguments to main() here.

Task
----
Given a short query x (a cut-out sample fragment, ~0.2-1 s) and a set of long
sources y1..yN, find y_k and the offset t such that x ~= y_k[t:]. The query
differs from the original: different EQ, reverb, noise, a different level,
possibly vinyl noise. Direct waveform comparison therefore does not work
(correlation ~0.02); speed/pitch are assumed to be unchanged.

Method
------
1. Downmix everything to 22050 Hz mono.
2. Compute log-mel spectrograms at several resolutions.
   The main comparison tool is a per-band cosine similarity measured along the
   TIME axis inside each mel band: it is invariant to any fixed EQ (scaling a
   band by a constant) and largely to an added noise floor, because in the log
   domain such a floor is just a constant shift. Before comparing, every frame
   is centred across bands, which removes the overall level.
3. The profile over ALL offsets of a file is computed at once with a single
   FFT convolution (profile length = number of source frames; the query is
   aligned to the start of frame k). Each profile is then turned into a
   z-score and summed over a "bank" of feature sets: {1024, 2048} x
   {48, 64} bands x {static, first time difference}. Different resolutions
   fail in different places, so the combination is markedly more reliable
   than any single profile.
4. Candidates: the best offsets of each file (with neighbouring offsets
   suppressed). For each candidate the offset is refined over +-2 frames
   and the final metrics are computed: similarity, frame support (the
   fraction of frames with r > 0.5) and the margin over the best false
   candidate.

Large collections
-----------------
The full feature bank costs ~700 KB per second of audio (for a 12 GB mp3
collection that is 165 GB), so the wide search runs off a compact on-disk
"coarse" index (one resolution, 23 ms hop, float16 — 4.2 KB/s of audio).
Only the shortlisted candidates are re-read from the original files and
scored with the full bank, so all reported numbers and thresholds stay
exactly the same.

Output
------
One results folder per run: results/<query folder>__<search folder>__
<date and time of the run>. It holds the fragments of every query of that run
— the confident ones directly, the uncertain ones in its unsure/ subfolder —
and log_<same suffix>.txt with everything the run printed.

Usage
-----
    python samplescanner.py                        # ./query against ./search
    python samplescanner.py -q QUERY -s SEARCH -v  # own folders + details
    python samplescanner.py -q one_query.wav       # a single query file
    python samplescanner.py --selftest             # synthetic self-check
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ssc.audio import load_audio, shift_pitch, trim_silence
from ssc.config import (COARSE_BANDS, COARSE_HOP, DEFAULT_BATCH,
                        DEFAULT_RESULTS_PREFIX,
                        DEFAULT_UNSURE_SUBDIR, EQUIV_GAP, EQUIV_MIN_SIM,
                        EQUIV_TOL, EXPORT_MIN_MARGIN, EXPORT_MIN_SIM, HOP,
                        INDEX_SIZE_WARN, INTERMEDIATE_SUBDIR, LOG_PREFIX,
                        ROUND_STOP_SHARE, SAMPLE_RATE, UNSURE_MIN_SIM)
from ssc import cache
from ssc.export import export_report, results_folder
from ssc.features import logmel_bank
from ssc.format import fmt_duration, fmt_elapsed, fmt_size, fmt_time
from ssc.index import (CoarseIndex, CoarseCandidate, IndexPlan, SearchIndex,
                       coarse_search_many, format_plan)
from ssc.inputs import (resolve_queries, resolve_source_records,
                        resolve_sources, top_folders)
from ssc.paths import ensure_writable_dir
from ssc.pitch import (fmt_semitones, group_end, normalise_pitch_args,
                       parse_pitch_args)
from ssc.progress import INTERVAL, Progress
from ssc.report import (Report, fit_name, name_column_width, print_report,
                        report)
from ssc.search import Candidate
from ssc.selftest import selftest
from ssc.status import flash, is_status
from ssc.version import version_line

# --------------------------------------------------------------------------- #
# CLI
# Anything one query can throw at us. A single bad query must never stop the
# others, so the loop in main() catches exactly these and carries on.
QUERY_ERRORS = (RuntimeError, OSError, ValueError, ArithmeticError,
                subprocess.SubprocessError)

# The first word of the line the program prints when it wants an answer. The
# window watches for it, asks the question in a dialog and sends the answer
# back on stdin; from a terminal nobody does, and the size is only announced.
CONFIRM_MARKER = "CONFIRM    :"

# The first words of the two lines of a pass over the collection, so the window
# of the gui can tell them apart: SCAN says what the pass is for and belongs in
# the report, SCANNING is the bar of the pass itself and only ever shows in the
# status line under the output - on a terminal it is redrawn in place instead,
# and both of them start with the same word there.
SCAN_MARK = "SCAN"
SCAN_BAR_MARK = "SCANNING"
VERIFY_MARK = "VERIFYING"
PREPARE_MARK = "PREPARING"
# how many shifts the text of a bar writes out before it starts counting them
BAR_PITCHES = 5


def verify_bar(total: int, whose: str, interval: float):
    """
    The bar of the precise pass: the windows of the shortlisted candidates,
    read from the originals one by one and checked properly.

    It is a bar of its own because it is a different kind of work from the walk
    that came before it, and often a longer one. The walk counts files, and at
    100% of it the program has read every file of the collection and computed
    nothing that could answer a query - only the coarse profile. Everything that
    decides anything is the reading of the shortlisted windows that follows, one
    ffmpeg per window, which is where a large collection spends most of the time
    of a pass. With the status line holding the finished bar of the walk, all of
    that is silence, and a bar at 100% reads as a program that has stopped.
    """
    if interval < 0:
        return None
    return Progress(total, text=f"{VERIFY_MARK} candidates of current {whose}",
                    interval=interval, unit="candidates").start()


def verdict_note(nbatch: int) -> str:
    """
    What the pass is on now that the walk over the collection is over.

    The last thing to happen before the first verdict: the candidates of every
    shift are turned into reports, and whatever the lazy rule has accepted is cut
    out with ffmpeg. Short next to the walk itself, but it is not nothing, and
    the status line has to leave the finished bar of the walk behind - a line
    that keeps saying 100% of something already finished says nothing at all.
    """
    return (f"{PREPARE_MARK} verdicts of current batch" if nbatch > 1
            else f"{PREPARE_MARK} verdict of current query")


class NoSpaceError(RuntimeError):
    """The index would not fit on the drive — the search cannot start."""

    def __init__(self, plan: IndexPlan):
        self.plan = plan
        super().__init__(
            f"the index of this collection would take about "
            f"{fmt_size(plan.bytes_estimate)}, but only "
            f"{fmt_size(plan.free_bytes)} is free on the drive that holds "
            f"{plan.index_dir}.\n"
            f"The search is NOT started: free up some space on that drive, "
            f"or move the program's folder to another one, and run it again.")


def confirm_large_index(args, plan: IndexPlan) -> bool:
    """
    Asks whether a very large index may be built, and returns the answer.

    With --confirm-large-index the question goes to stdin and the window that
    started the program is the one that answers. Anything that is not a yes
    counts as a no, and a closed stdin counts as a no too: nobody is there, so
    the safe answer is not to spend gigabytes on their behalf. Without the flag
    — from a terminal, where there is nobody to ask — the size is announced and
    the answer is yes.
    """
    size = fmt_size(plan.bytes_estimate)
    if not args.confirm_large_index:
        print(f"WARNING   : the index of this collection will take about "
              f"{size}, which is a lot of disk — building it anyway "
              f"(the window asks before it does)")
        return True
    print(f"{CONFIRM_MARKER} the index of this collection will take about "
          f"{size} — that is a lot of disk. Build it anyway? [y/N]")
    sys.stdout.flush()
    try:
        answer = sys.stdin.readline()
    except Exception:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


@dataclass
class Shift:
    """
    One query at one pitch shift: what is to be looked for, and what the search
    of it gave. Kept apart because the two halves of the work are done at
    different moments - the query side is ready before the collection is
    touched, the collection side only after it has been walked.
    """
    label: str
    pitch: float
    y: np.ndarray                       # the shifted signal that is searched
    qf: Dict[Tuple[int, int], np.ndarray]
    duration: float                     # the query before trimming and shifting
    trim_shift: float
    trim_tail: float
    kept_duration: float
    coarse: Optional[Tuple[List[CoarseCandidate], List[CoarseCandidate]]] = None
    rep: Optional[Report] = None
    why: Optional[str] = None           # why it could not be searched, if so
    t_pass: float = 0.0                 # the whole pass that answered it
    # its share of that pass: the pass answered one query at one pitch per pair
    # it was given, so dividing by that count is what makes a time that can be
    # compared between a batch of many and a pass over every pitch
    t_walk: float = 0.0
    t_report: float = 0.0

    @property
    def t_search(self) -> float:
        return self.t_walk + self.t_report


def prepare_shift(label: str, x: np.ndarray, pitch: float,
                  no_trim: bool = False) -> Shift:
    """
    The query side of one search: the signal to look for, its log-mel bank and
    the timings the report needs. None of it depends on the collection, which is
    why it can be done for a whole batch of shifts before the library is walked
    once for all of them.

    The silence is trimmed on the ORIGINAL query, before the pitch is touched -
    otherwise the shift would be weighed against the shifted noise. The reported
    offset stays in the source timeline either way.
    """
    duration = len(x) / SAMPLE_RATE
    trim_shift = 0.0
    trim_tail = 0.0
    if not no_trim:
        kept, trim_shift = trim_silence(x, SAMPLE_RATE)
        trim_tail = max(0.0, duration - trim_shift - len(kept) / SAMPLE_RATE)
        x = kept
    kept_duration = len(x) / SAMPLE_RATE     # the trimmed query, query seconds
    y = shift_pitch(x, pitch)
    return Shift(label, pitch, y, logmel_bank(y), duration, trim_shift,
                 trim_tail, kept_duration)


def print_query_length(x: np.ndarray, pitches: Sequence[float],
                       no_trim: bool) -> None:
    """
    What the query itself is, before any pitch is tried: the length does not
    depend on the shift (trimming happens before it, on the original), and the
    frame count is the one of the first shift the sweep tries. Both cost a few
    milliseconds to work out here, which is the price of saying it before the
    table instead of after it.
    """
    raw_s = len(x) / SAMPLE_RATE
    kept_x = trim_silence(x, SAMPLE_RATE)[0] if not no_trim else x
    kept_s = len(kept_x) / SAMPLE_RATE
    qf0 = logmel_bank(shift_pitch(kept_x, pitches[0]))
    n_frames0 = max(m.shape[0] for m in qf0.values())
    line = f"         duration {raw_s:.3f} s"
    if kept_s < raw_s - 1e-6:
        line += f" -> {kept_s:.3f} s after trimming"
    print(line + f", {n_frames0} analysis frames")


def pitch_label(shifts: List[float]) -> str:
    """
    The shifts a pass is answering, written out for the status line.

    A sweep of a few semitones has a short list and says all of it. A wide one
    would push the bar itself past the edge of the status line of the window -
    the list is in front of the bar, not behind it - so past this many the rest
    is counted rather than written. The full list is on the SCAN line above,
    which belongs to the report and can be as long as it has to be.
    """
    out = [fmt_semitones(p) for p in shifts[:BAR_PITCHES]]
    if len(shifts) > BAR_PITCHES:
        out.append(f"… {len(shifts) - BAR_PITCHES} more")
    return ", ".join(out)


def walk_shifts(shifts: List[Shift], index: SearchIndex, args,
                note: str = "", nbatch: int = 1) -> None:
    """
    ONE pass over the collection, answering every shift it is handed.

    The line printed before it says what the pass is for - how many queries and
    at which shifts - because a pass over a large collection is minutes long and
    nothing else is said in the meantime. The bar below it says how far it has
    got: on a terminal it is redrawn in place, and where the output is a pipe -
    the window of the gui, a run written to a file - it is a plain line every
    --progress-interval seconds, which is a couple of seconds by default.

    The time the pass took belongs to the WHOLE batch: every shift of every
    query in it was computed in that one pass, so the time is what is reported
    against all of them, and the share of it that falls to one shift is kept
    only to add up the run.
    """
    if not shifts or index.coarse is None:
        return
    if note:
        flash(note)
    # which shifts this pass is answering, so that the bar says what it is
    # working on: in a round it is the one every query still in play has reached,
    # in a single pass over a batch it is all the rest of them
    seen: List[float] = []
    for sh in shifts:
        if sh.pitch not in seen:
            seen.append(sh.pitch)
    pitch_text = pitch_label(seen)
    # a pass of one query is not a batch, and saying "batch" of one reads as a
    # word put in where it does not fit
    whose = "batch" if nbatch > 1 else "query"
    t_pass = time.perf_counter()
    bar = None
    if args.progress_interval >= 0:
        bar = Progress(len(index.coarse.entries), interval=args.progress_interval,
                       text=f"{SCAN_BAR_MARK} LIBRARY for current {whose} with "
                            f"pitch(es) {pitch_text} st").start()
    # the bar of the walk is closed by the walk itself, not here: what it says
    # about its own time and rate has to be the walking, and this function does
    # not return until the precise pass is over as well. It is dropped once
    # closed, so that the call in the finally below - for a walk that raised
    # before getting there - writes nothing.
    def close_bar() -> None:
        nonlocal bar
        if bar is not None:
            bar.close()
            bar = None
    try:
        done = coarse_search_many(index.coarse,
                                  [(sh.y, sh.qf, len(sh.y) / SAMPLE_RATE)
                                   for sh in shifts], progress=bar,
                                  verify=lambda total: verify_bar(
                                      total, whose, args.progress_interval),
                                  walk_done=close_bar)
    finally:
        # close_bar has already run at the end of the walk; this is only here
        # for the walk that raises before getting there, and closing twice
        # writes nothing
        close_bar()
    elapsed = time.perf_counter() - t_pass
    share = elapsed / len(shifts)
    for sh, coarse in zip(shifts, done):
        sh.coarse = coarse
        sh.t_pass = elapsed
        sh.t_walk = share


def scan_note(count: int, pitches: List[float]) -> str:
    """
    What a pass over the collection is for, in one line. The shifts are the
    distinct ones the pass answers: in a round every query has its own, in one
    pass over a whole batch there are as many as the batch has.
    """
    uniq: List[float] = []
    for p in pitches:
        if p not in uniq:
            uniq.append(p)
    return (f"{SCAN_MARK:9s}: {count} quer(y/ies) at pitch(es) "
            + ", ".join(fmt_semitones(p) for p in uniq))


def report_intermediate(query: BatchQuery, index: SearchIndex, args,
                        run_dir: str, unsure_dir: str, s_base: str,
                        noted: Dict[str, bool], quiet: bool = False) -> None:
    """
    A confident match, the moment it is found rather than at the end of the
    sweep: said in one line, and cut out straight away. A search of a large
    collection over many shifts runs for hours, and a result that is only
    written when it is all over is a result that a stop halfway through throws
    away.

    WHERE it goes depends on the order. In lazy mode the query stops at this very
    shift, so this is the match it will be reported with: it goes straight into
    the run folder, where it stays, and the export at the end of the query's
    sweep is skipped - writing it again would put a second copy of the same
    file next to the first. In non-lazy mode the sweep goes on and a later shift
    may win, so what is found now is a provisional result and goes into
    `intermediate/` instead, leaving the run folder for the winner.

    The shift that is saved is the one that was found, not the best of the whole
    sweep - that one is still to come, and may well be elsewhere.

    `quiet` cuts the files out without a word on the screen. A caller showing
    what it found as it goes says so on the row of the shift instead, and the
    names are kept on the report so the verdict still says where they went -
    printed here as well, they would put every path on the screen twice and push
    the table of shifts out of sight.
    """
    rep = query.best
    flash(f"FOUND    : {os.path.basename(query.label)} -> "
          f"{rep.best.file_name} @ {fmt_time(rep.offset)}, "
          f"{fmt_semitones(rep.pitch)} st, r={rep.best.similarity:.3f} "
          f"[confident — saved as it was found, the sweep goes on]")
    if args.no_export:
        return
    if args.nonlazy:
        dest = os.path.join(run_dir, INTERMEDIATE_SUBDIR)
        header = ("confident results exported before the rest of pitches "
                  "are checked:")
    else:
        dest = run_dir
        header = ("confident results exported in advance (the rest will be "
                  "exported after the whole batch is processed):")
    # said once for the whole run, not over every result: it introduces the
    # block of file names below it, and a line repeating itself in front of
    # each of them reads as three separate things having happened
    if not quiet and not noted.get("header"):
        noted["header"] = True
        print(f"  {header}")
    # the cut itself is ffmpeg work, one call per exported source, and it is the
    # last thing before the next query is judged - saying so keeps the status
    # line from sitting on a finished bar while it happens
    flash(f"cutting the fragments of {os.path.basename(query.label)} "
          f"(in advance)")
    export_report(rep, run_dir, unsure_dir, True, s_base, args.unsure_min_sim,
                  dest_override=dest, quiet=quiet)
    query.exported_report = rep
    query.exported_final = dest == run_dir


def finish_shift(sh: Shift, index: SearchIndex, args) -> None:
    """
    Turns the candidates of one shift into its report, or into the reason it
    has none. Nothing is printed and nothing is written to disk, so a sweep over
    many shifts can search quietly and export only the one that wins.
    """
    t = time.perf_counter()
    try:
        sh.rep = report(sh.label, index, sh.qf,
                        max(m.shape[0] for m in sh.qf.values()),
                        sh.duration, sh.trim_shift, args.top, args.verbose,
                        export_min_sim=args.export_min_sim,
                        export_min_margin=args.export_min_margin,
                        query_duration=len(sh.y) / SAMPLE_RATE,
                        kept_duration=sh.kept_duration,
                        query_x=sh.y, query_trim_tail=sh.trim_tail,
                        equiv_tol=args.equiv_tol,
                        equiv_min_sim=args.equiv_min_sim,
                        equiv_gap=args.equiv_gap, pitch=sh.pitch, quiet=True,
                        coarse_result=sh.coarse)
    except QUERY_ERRORS as exc:
        sh.why = str(exc)
    sh.t_report = time.perf_counter() - t


class BatchQuery:
    """
    One query of a batch, and what has been worked out about it so far.

    Its shifts are made one at a time, when the walk that needs them comes
    round: a shift nobody is going to report is never even prepared.
    """
    __slots__ = ("label", "x", "shifts", "next", "best", "swept", "done",
                 "stopped_at", "found_confident", "told", "exported_report",
                 "exported_final", "name_w", "entries", "head", "rows",
                 "cut_row", "shown_live", "errs", "computed", "t_sum")

    def __init__(self, label: str, x: np.ndarray) -> None:
        self.label = label
        self.x = x
        self.shifts: List[Optional[Shift]] = []
        self.next = 0                  # the shift to try next
        self.best: Optional[Report] = None
        # every candidate of every shift, so the report can say which shift
        # each of them came from
        self.swept: Dict[Tuple[str, float], Tuple[Candidate, float]] = {}
        self.done = False              # nothing left to try
        self.stopped_at: Optional[int] = None   # the lazy rule cut it here
        self.found_confident = False       # a shift good enough to answer with
        self.told = False                  # and it has been written down already
        # what was written in advance, and whether it went where the winner of
        # the sweep belongs: the same report is cut twice otherwise, and the
        # second copy lands beside the first
        self.exported_report: Optional[Report] = None
        self.exported_final = False
        self.name_w = 0
        # (the shift came through, what it has to say) for every shift tried
        self.entries: List[Tuple[bool, List[str]]] = []
        # the heading of that table and one line per shift, kept out of
        # `entries` so that a search showing its shifts as it goes can print the
        # heading once and the newest line, without carrying the rest along
        self.head: List[str] = []
        self.rows: List[str] = []
        # which row is the one whose fragment has been cut out already, -1 while
        # none has: the caller that shows the rows as it goes says so on that
        # one, instead of printing a block of file names over the table
        self.cut_row = -1
        # whether this table has already been printed row by row as the shifts
        # were searched, so that the report does not print it a second time
        self.shown_live = False
        self.errs: List[Tuple[str, str]] = []
        self.computed = 0
        self.t_sum = 0.0

    def shift(self, pitches: List[float], args, at: int) -> Optional[Shift]:
        """Prepares the shift at position `at`, or None if there is none."""
        while len(self.shifts) < len(pitches):
            self.shifts.append(None)
        if at >= len(pitches):
            return None
        sh = self.shifts[at]
        if sh is None:
            sh = prepare_shift(self.label, self.x, pitches[at], args.no_trim)
            self.shifts[at] = sh
        return sh


def take(query: BatchQuery, sh: Shift, index: SearchIndex, args,
         pitches: List[float], lazy: bool, stop_searching: bool,
         nbatch: int) -> None:
    """
    Folds one searched shift into what is known about its query, and leaves
    behind the lines that shift has to say on the console.

    The lazy rule is here: a query stops at its first confident shift, so the
    shifts after it are never searched. `stop_searching` is False for the order
    that has already computed every shift of the batch in one pass - there the
    rule can only decide what is REPORTED, not what is searched.
    """
    finish_shift(sh, index, args)
    query.computed += 1
    query.t_sum += sh.t_search
    if sh.rep is None:
        # one query must never stop the others: whatever went wrong here is
        # reported and the loop moves on
        query.entries.append((False, [
            f"    {fmt_semitones(sh.pitch):>5} st: ERROR {sh.why} (this "
            f"attempt only, the rest continues)"]))
        query.errs.append((query.label, sh.why or ""))
        return
    rep = sh.rep
    for c in rep.final[:max(args.top, 1)]:
        key = (c.file_name, round(c.exact_frame * HOP / SAMPLE_RATE, 3))
        old = query.swept.get(key)
        if old is None or c.similarity > old[0].similarity:
            query.swept[key] = (c, sh.pitch)
    # Which shift wins is NOT the same question as which shift is confident:
    # the winner is the best by score (the similarity, cut down when the
    # candidate explains too little of the query), while confidence also asks
    # for the margin over the other files. So a later shift with a slightly
    # better r and a worse margin can outscore a confident one without being
    # confident itself.
    #
    # That is only allowed where the sweep really is still going. Once the lazy
    # rule has accepted a shift, the shifts after it are not part of the answer
    # — they are not even in the table — so the winner is frozen there and on.
    if query.stopped_at is None:
        if query.best is None or rep.score > query.best.score:
            query.best = rep
            mark = "  <- best so far" + (" [confident]" if rep.confident else "")
        else:
            mark = ""
    else:
        mark = ""
    lines = []
    # the name column is as wide as the one the candidate list of this query
    # uses, so the two read the same; it is measured on the first shift that
    # runs, and the header is printed with it
    if not query.name_w:
        query.name_w = name_column_width(index, rep.final)
        # the rows below are what each shift found in order, not the final
        # answer - the answer is in the report that comes after them
        lines.append("    Best matches for each pitch (intermediate results, "
                     "in order checked, see VERDICT below for final results "
                     "and closest candidates)")
        left = (f"    {'pitch':>6}  {'source':<{query.name_w}s} "
                f"{'position':>9}  ")
        lines.append(f"{left}{'time spent':>10}  {'r':>6}  {'frames':>6}")
        # the heading of the whole table, both of its lines, taken before the row
        # for this shift goes on the end of `lines` - a search that shows its
        # shifts as it goes prints this once, above the first of them
        query.head = list(lines)
    name = fit_name(rep.best.file_name, query.name_w)
    w = query.name_w
    # the pass that answered this shift answered every pair of query and pitch it
    # was given in one walk, and t_walk is this shift's share of it - the time for
    # ONE query at ONE pitch, which is the only form of the number that can be
    # read against another run: a batch of fifteen queries and a single query do
    # not take the same time for a shift, and saying so under the header instead
    # of dividing is what made the column unreadable
    row = (f"    {fmt_semitones(sh.pitch):>5} st  "
           f"{name:<{w}s} "
           f"{fmt_time(rep.offset):>9s}  "
           f"{fmt_elapsed(sh.t_walk):>10}  "
           f"{rep.score:6.3f}  "
           f"{rep.best.frame_support*100:3.0f}%{mark}")
    lines.append(row)
    query.rows.append(row)
    query.entries.append((True, lines))
    if (lazy and len(pitches) > 1 and query.best.confident
            and query.stopped_at is None):
        query.stopped_at = len(query.entries)
        if stop_searching:
            query.done = True
    if query.best.confident and not query.found_confident:
        # the first shift that is confident enough to be answered with, lazy or
        # not. In lazy mode the sweep is over here, so this is the result of the
        # query; in non-lazy mode it is only a provisional one - hence where it
        # is written (see report_intermediate). The row is remembered as the one
        # about to be cut out, so that a search printing its shifts as it goes
        # can say on the line itself that this one was written already.
        query.found_confident = True
        query.cut_row = len(query.rows) - 1


def report_query(query: BatchQuery, index: SearchIndex, args,
                 pitches: List[float], run_dir: str, unsure_dir: str,
                 s_base: str, batch_time: float, nbatch: int
                 ) -> Tuple[Optional[Report], int, float,
                            List[Tuple[str, str]]]:
    """
    Prints what was found for one query and exports the fragment of the shift
    that won. Everything about the query has been searched by the time this is
    called; what is left is to say it.

    `batch_time` is what the passes of the whole batch took, which is what the
    time of a shift is measured against; `nbatch` is how many queries shared
    them. What this query is charged for the run is its own share, not the
    batch - see the return value.
    """
    label = query.label
    shown = query.entries if query.stopped_at is None \
        else query.entries[:query.stopped_at]
    # the query is named BEFORE its shifts, not between them and the report:
    # after a batch there is one block per query, and a table of shifts with no
    # name above it belongs to nobody. The blank line before it keeps the blocks
    # apart, which is all that separates them once a batch is done.
    # A table that was printed row by row as the shifts came in is not printed
    # again here: that would put every row on the screen twice.
    if not query.shown_live:
        print()
        for _ok, lines in shown:
            for line in lines:
                print(line, flush=True)
    reported = sum(1 for ok, _lines in shown if ok)
    if query.stopped_at is not None and query.stopped_at < len(pitches):
        # one line about the shifts that were not tried at all: a line per
        # skipped shift buries the table under them on a wide sweep. It says
        # what is MISSING from the table, so it goes straight under it rather
        # than under the report that follows - otherwise the blank line and the
        # name of the query stand between the two and make them look unrelated.
        print(f"    Next pitches were skipped, "
              f"{fmt_semitones(query.best.pitch)} st was already confident "
              f"(use non-lazy mode to try all pitches anyway)")
    print()
    print(f"QUERY  : {label}", flush=True)
    best = query.best
    if best is None:
        return None, query.computed, query.t_sum, query.errs
    # hand the candidates of every shift to the report
    best.sweep_candidates = (
        sorted(query.swept.values(),
               key=lambda t: -t[0].similarity)[:max(args.top, 1)]
        if len(pitches) > 1 else [])
    print()
    print_report(best, index, args.top, args.verbose, with_query=False)
    if query.exported_final and query.exported_report is best:
        # this very report was cut out already, into the run folder, the moment
        # the sweep accepted it — cutting it again would only put a second copy
        # of the same file beside the first. Its verdict names the files the
        # same way a match cut out now would, and says nothing about having
        # been written down before: the reader of a verdict is looking at it
        # with the log behind him, not in front of him.
        # with --no-export nothing was written and nothing was named earlier, so
        # there is nothing to name now either
        if best.exported_lines:
            print("  FRAGMENT:")
            for line in best.exported_lines:
                print(line)
    else:
        flash(f"cutting the fragments of {os.path.basename(label)}")
        export_report(best, run_dir, unsure_dir, not args.no_export,
                      s_base, args.unsure_min_sim)
    # What a pass cost is what the whole batch cost: the shifts of every query
    # in it were computed in the same pass. The share of it that is one query's
    # is given in brackets, and the run is totalled from the shares, so that the
    # total stays the time that really passed.
    per_query = batch_time / max(1, nbatch)
    where = (f" for batch of {nbatch} queries "
             f"(≈{fmt_elapsed(per_query)} per query)" if nbatch > 1 else "")
    print(f"  TIME     : index search over {reported}/{query.computed} "
          f"pitch(es) took {fmt_elapsed(batch_time)}{where}. Use non-lazy mode "
          f"to check all pitches anyway")
    return best, query.computed, query.t_sum, query.errs

def shifts_this_round(pitches: List[float], at: int, one_pass: bool,
                      bundled: bool, order: str) -> range:
    """
    Which shifts one query is given in the round that starts at `at`, as a range
    of indexes into the ordered shifts.

    Three answers, and the difference between them is most of what a run costs:

    * a plain round hands over one shift, so a confident one ends the query at
      once and nothing past it is ever searched;
    * a bundled round hands over everything left in the group `at` sits in -
      one walk instead of one per shift, but still one group of fineness at a
      time;
    * a one-pass round hands over everything left whatever group it falls in,
      because that is what --pitchbatch=all and --nonlazy ask for.

    The group is where a bundled round stops: with the finer order the
    semitones are settled before the halves are looked at, which is the order
    the groups exist for. With a step of 1 every shift is a whole semitone, so
    there is a single group and a bundled round is the whole of the rest.
    """
    if one_pass:
        return range(at, len(pitches))
    if bundled:
        return range(at, group_end(pitches, at, order))
    return range(at, at + 1)


def resolve_pitchbatch(requested: str, nonlazy: bool) -> str:
    """
    Which order the collection is really walked in, out of what was asked for.

    `--nonlazy` says every shift has to be searched, whatever else becomes true
    of a query once one of them is confident. In the dynamic order that is also
    what empties the batch: nothing finishes early for the sake of a confident
    match, so the cap that stops splitting fires on the very first round and
    every shift that is left goes into one walk. That is the `all` order, arrived
    at by accident and by a rule nobody sees, so asking for `dynamic` together
    with `--nonlazy` is read as `all` instead - the same walk, arrived at on
    purpose.

    `single` is left as it is. One shift per walk is exactly what a person who
    wants to watch a query go through every shift asked for, and `--nonlazy`
    takes away the reason to stop early, not the reason to walk once per shift.
    """
    return "all" if nonlazy and requested == "dynamic" else requested


def show_shift(query: BatchQuery, headed: bool) -> bool:
    """
    Prints the line of the shift that has just been searched - one line, the one
    for this pitch, and not the ones above it that are already on the screen.

    The heading goes out with the first of them, so that a search showing its
    shifts as it goes reads as the table it is building rather than as a list
    of numbers with no columns named. It is the same heading the final report
    prints, and it fits the columns because the width of the source column is
    measured from the file names of this very query.

    Returns whether the heading has been printed by now, for the caller to pass
    on: this runs once per pitch and there is nothing in the query to remember
    that for.
    """
    if not headed and query.head:
        for line in query.head:
            print(line, flush=True)
        headed = True
    if query.rows:
        row = query.rows[-1]
        if query.cut_row == len(query.rows) - 1 and "[confident]" in row:
            # this shift was good enough to answer with, so its fragment has
            # been cut out already - said here, on the line about that shift,
            # rather than as a block of file names over the table
            row = row.replace("[confident]", "[confident, exported]")
        print(row, flush=True)
    return headed


def solve_batch(queries: List[BatchQuery], index: SearchIndex, args,
                pitches: List[float], lazy: bool, run_dir: str,
                unsure_dir: str, s_base: str, noted: Dict[str, bool]) -> None:
    """
    Answers a whole batch, one walk of the collection at a time, and leaves
    every query of it ready to be reported.

    A ROUND is one walk that gives every query still in play its next shift.
    Queries that come back confident leave the batch - and are written down
    there and then, not at the end - so a later round answers fewer of them.

    --pitchbatch says what a round is allowed to be, and the three answers are
    the three orders the collection is walked in:

    * **single** gives every round one shift and never bundles. It is the
      default, and with --batch 1 it prints the heading of the table of shifts
      once and then one line per shift as it is searched, so a long search is
      something to watch rather than to wait for.
    * **dynamic** stops splitting once a round has stopped too few queries to be
      worth its walk, and from there bundles the rest - one walk per group of
      pitches rather than one per shift. Bundling stops at the end of the group:
      with the finer order the whole semitones come first, so a round that found
      nothing at 0, ±1, ±2 says so after the semitones and leaves the halves for
      the next walk. A sample that is not at a whole semitone at all is the rarer
      case, and one that IS there but hard to match is better found by a walk
      over the halves than by a walk over everything at once, which only learns
      the same thing an hour later.
    * **all** has no rounds: one walk answers every remaining shift of every
      query of the batch. --nonlazy used to ask for it by itself, and no longer
      does - a walk that answers every shift at once shows nothing until it is
      over, which is no way to watch a search, and a person asking to see every
      pitch is asking to watch one.

    None of the three changes what is found: the shifts are tried in the same
    order whichever is asked for, so the verdict and the winner are the same and
    only the walks differ.
    """
    # One walk for the whole rest is asked for by name only. --nonlazy says
    # WHICH shifts are searched and --pitchbatch how they are walked; resolve_pitchbatch
    # has already folded the one combination that made those two disagree.
    one_pass = args.pitchbatch == "all"
    # the whole table after every shift, but only where there is one query to
    # show it for: with several in the batch the table of one of them would be
    # the only one on the screen, which says less than saying nothing until the
    # batch is done. That is what a batch is for.
    live = args.pitchbatch == "single" and len(queries) == 1
    bundled = False
    headed = False               # has the heading of the table gone out yet
    alive = list(range(len(queries)))
    while alive:
        todo: List[Tuple[BatchQuery, Shift]] = []
        for i in alive:
            q = queries[i]
            wanted = shifts_this_round(pitches, q.next, one_pass, bundled,
                                      args.pitchorder)
            try:
                for at in wanted:
                    sh = q.shift(pitches, args, at)
                    if sh is not None:
                        todo.append((q, sh))
            except QUERY_ERRORS as exc:
                # one query must never stop the others: a query whose signal
                # cannot be turned into features is out of this batch
                q.errs.append((q.label, str(exc)))
                q.entries.append((False, [f"    ERROR   : {exc}"]))
                q.done = True
                q.next = len(pitches)
        if not todo:
            break
        walk_shifts([sh for _q, sh in todo], index, args,
                    scan_note(len(todo), [sh.pitch for _q, sh in todo]),
                    len(queries))
        # the walk is over: what follows is the work that turns what it found
        # into answers, and the status line says so rather than leaving the
        # finished bar of the walk standing there
        flash(verdict_note(len(queries)))
        for q, sh in todo:
            take(q, sh, index, args, pitches, lazy, not one_pass,
                 len(queries))
            q.next += 1
            q.done = q.done or q.next >= len(pitches)
            if q.found_confident and not q.told:
                # the lazy rule has decided this query: its match is written
                # down now, because the rest of the sweep may take hours and
                # may never finish
                q.told = True
                report_intermediate(q, index, args, run_dir, unsure_dir,
                                   s_base, noted, quiet=live)
        if live:
            # after every shift, and only the query there is one of: the heading
            # once, and then the one line of this pitch under it, so the block
            # on the screen is the table the report would have printed, built a
            # row at a time as the shifts come in
            headed = show_shift(queries[0], headed)
            queries[0].shown_live = True
        if one_pass:
            break
        before = len(alive)
        alive = [i for i in alive if not queries[i].done]
        # A round that stopped almost nobody will not be followed by one that
        # stops a lot, and from there the rest is cheaper to compute in fewer
        # walks - one per group of pitches, so that a query still gets its answer
        # at the end of the semitones instead of at the end of everything. Only
        # the dynamic order does this: single asked for one shift per round and
        # stays one shift per round however badly the rounds go, and all is a
        # single walk from the first.
        if (alive and args.pitchbatch == "dynamic"
                and (before - len(alive)) * ROUND_STOP_SHARE < before):
            bundled = True


def sweep(queries: List[Tuple[str, Optional[np.ndarray]]],
          index: SearchIndex, args, pitches: List[float], lazy: bool,
          run_dir: str, unsure_dir: str, s_base: str
          ) -> Tuple[List[Tuple[str, Report]], List[Tuple[str, str]],
                     int, float]:
    """
    Searches every query at every shift, prints the table of shifts and the
    report of the shift that won, and exports that one.

    --pitchbatch says how a batch of queries is walked through the shifts
    (solve_batch); --nonlazy turns the lazy stopping off on top of any of them.

    There is one order this function chooses for itself: a collection held in
    memory has no index to walk once for a batch, so there is nothing to batch
    it over and every query is searched on its own (search_one).

    The order the shifts are tried in does not depend on any of this, so it
    makes no difference to what is found - only to how many walks it takes and
    how soon the first line of the answer appears.
    """
    results: List[Tuple[str, Report]] = []
    failed: List[Tuple[str, str]] = []
    attempts = 0
    t_search_all = 0.0
    ready = [(label, x) for label, x in queries if x is not None]
    # a collection held in memory has no index to walk once for a batch, so
    # there is nothing to batch it over
    batched = index.coarse is not None
    # said once for the whole run: it introduces the block of file names that
    # the matches exported in advance add up to
    noted: Dict[str, bool] = {}
    size = max(1, args.batch) if batched else 1
    for start in range(0, len(ready), size):
        batch: List[BatchQuery] = []
        # what this batch is, before the queries of it are listed: with a batch
        # of one there is no batch to speak of, and the line would only repeat
        # the query that comes next
        nbatch = len(ready[start:start + size])
        print(f"\nLoading batch of {nbatch} queries:" if nbatch > 1
              else "\nLoading query:")
        for label, x in ready[start:start + size]:
            flash(f"searching {os.path.basename(label)}")
            print(f"\nQUERY : {label}")
            try:
                print_query_length(x, pitches, args.no_trim)
                batch.append(BatchQuery(label, x))
            except QUERY_ERRORS as exc:
                # one query must never stop the others
                print(f"  ERROR   : cannot read {label}: {exc}")
                failed.append((label, str(exc)))
        if batched:
            solve_batch(batch, index, args, pitches, lazy, run_dir,
                        unsure_dir, s_base, noted)
        else:
            for query in batch:
                search_one(query, index, args, pitches, lazy, run_dir,
                            unsure_dir, s_base, noted)
        nbatch = len(batch)
        batch_time = sum(q.t_sum for q in batch)
        for query in batch:
            rep, computed, t_sum, errs = report_query(
                query, index, args, pitches, run_dir, unsure_dir, s_base,
                batch_time, nbatch)
            attempts += computed
            t_search_all += t_sum
            failed.extend(errs)
            if rep is not None:
                results.append((query.label, rep))
    return results, failed, attempts, t_search_all


def search_one(query: BatchQuery, index: SearchIndex, args,
               pitches: List[float], lazy: bool, run_dir: str,
               unsure_dir: str, s_base: str, noted: Dict[str, bool]) -> None:
    """
    One query, one shift, one walk of the collection: the order for a collection
    that has no on-disk index to share between the queries.

    Nothing is shared with another query, so a result is printed the moment its
    own shifts are done - at the price of walking the collection again for every
    single one of them.
    """
    while not query.done:
        try:
            sh = query.shift(pitches, args, query.next)
        except QUERY_ERRORS as exc:
            query.errs.append((query.label, str(exc)))
            query.entries.append((False, [f"    ERROR   : {exc}"]))
            break
        if sh is None:
            break
        walk_shifts([sh], index, args, scan_note(1, [sh.pitch]), 1)
        flash(verdict_note(1))
        query.next += 1
        query.done = query.next >= len(pitches)
        take(query, sh, index, args, pitches, lazy, True, 1)
        if query.found_confident and not query.told:
            query.told = True
            report_intermediate(query, index, args, run_dir, unsure_dir, s_base, noted)


# --------------------------------------------------------------------------- #


class Tee:
    """
    A stdout replacement that writes everything into a file as well, so the
    whole report of a run is saved next to the fragments it exported.
    """

    def __init__(self, stream, path: str):
        self.stream = stream
        self.path = path
        self.broken = ""      # why the log stopped being written, once known
        self.fh = open(path, "w", encoding="utf-8", errors="replace",
                        newline="\n")

    def write(self, data: str) -> int:
        self.stream.write(data)
        if self.broken:
            return len(data)    # already said once; do not try every line
        # A status line says what the run is doing RIGHT NOW - the bar of a
        # pass, the pass it is passing, a match found a moment ago. It belongs
        # on the screen and in the status line of the window; read an hour
        # later, out of a file, it is either stale or a wall of the same bar
        # repeated once every couple of seconds. The log keeps the report,
        # which is what it is for, and leaves those out.
        if is_status(data):
            return len(data)
        try:
            # A progress bar redraws its line with a carriage return, which a
            # file has no way to show: there is no line to return to. The log
            # gets one ordinary line per redraw instead of a stack of overwrites
            # nobody can read — the same information, in a form that opens.
            self.fh.write(data.replace("\r", "\n") if "\r" in data else data)
        except Exception as exc:
            # a full disk, a folder that went away, a file an antivirus holds:
            # the run itself is still perfectly fine, and a search that dies
            # because its LOG cannot be written is the worse of the two
            self.broken = str(exc)
            try:
                self.stream.write(
                    f"\n  NOTE: the log file {self.path} cannot be written "
                    f"({self.broken}). The search continues, but nothing more "
                    f"is saved into it.\n")
                self.stream.flush()
            except Exception:
                pass
        return len(data)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for s in (self.stream, self.fh):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        """
        What the console underneath is, not what the log file is. The progress
        bar asks before it starts overwriting a line: on a terminal it is redrawn
        in place, and into a pipe it prints whole lines instead.
        """
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass

    # a few attributes the standard stream has, so that anything expecting a
    # text stream keeps working
    @property
    def encoding(self) -> str:
        return "utf-8"


def report_error(text: str) -> None:
    """
    Says what went wrong without ever dying while saying it.

    A program frozen as a window has no console: stderr can be nothing at all,
    or a handle that has already gone away, and a plain print() to one of those
    raises a second, more confusing error on top of the first one. The run log
    is a safe place for it either way — it holds everything the run printed.

    Both streams get the line, and one that fails does not stop the other: the
    error is exactly what the log is needed for afterwards, so writing it to
    stderr and returning — which is what this used to do — left the log ending
    at the last thing that had gone right.
    """
    said = False
    seen: set = set()
    for stream in (sys.stderr, sys.stdout):
        if stream is None:
            continue
        # stderr is redirected into the log for the length of a run, so writing
        # to both would put the same line there twice
        if id(stream) in seen:
            continue
        seen.add(id(stream))
        try:
            stream.write(text + "\n")
            stream.flush()
            said = True
        except Exception:
            continue
    if not said:
        try:
            print(text, file=sys.stderr, errors="backslashreplace")
        except Exception:
            pass


def _index_files(base, names):
    """
    The library's files as (absolute path, size, modification time).

    Size and modification time are what the index decides a cached file by, so
    they have to be read before the plan is made. One stat per file, and the
    same stat the old index needed to make the same decision.
    """
    out = []
    for n in names:
        full = os.path.join(base, n) if base else n
        try:
            st = os.stat(full)
        except OSError:
            continue
        out.append((os.path.abspath(full), st.st_size, st.st_mtime))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            # A report that arrives in pieces is worse than useless when the
            # search takes minutes: a pipe (the window of the gui, a file) is
            # block buffered, and the verdict of one query would only appear
            # when the next query wrote its first line — hours later, and after
            # a line that says the search of that pitch took three minutes.
            # Line buffering is what a person reading a long run expects.
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    if sys.stdout is None:
        # a windowed program has no console to print into; the run log still
        # holds everything, so the output is sent somewhere harmless
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    ap = argparse.ArgumentParser(
        prog="SampleScanner" if getattr(sys, "frozen", False)
            else "samplescanner.py",
        description="SampleScanner: a fragment from ./query is "
                    "searched for in the files of ./search.",
        epilog="Examples:\n"
               "  python samplescanner.py\n"
               "  python samplescanner.py -v\n"
               "  python samplescanner.py -q ./query -s ./search\n"
               "  python samplescanner.py -q one_query.wav -s ./search\n"
               "  python samplescanner.py -q a.wav b.wav -s ./one ./two\n"
               "  python samplescanner.py --pitch=-2:3\n"
               "  python samplescanner.py --pitch=-2:3:0.5\n"
               "  python samplescanner.py --pitch=-2:3 --nonlazy\n"
               "  python samplescanner.py --pitch=-1:1:0.25 --pitchorder symm\n"
               "  python samplescanner.py --selftest --selftest-cases 24\n")
    # The question "is this the build I was sent?" has to be answerable without
    # guessing from a file date, so it is asked of the program itself.
    ap.add_argument("-V", "--version", action="version",
                    version=version_line(),
                    help="print the version and stop")
    ap.add_argument("-q", "--query", nargs="+", action="extend", default=[],
                    metavar="PATH",
                    help="the queries: one or several folders (walked "
                         "recursively), audio files, glob patterns or text "
                         "files listing them — the code tells them apart. "
                         "Several paths at once are what a file browser gives "
                         "when several items are picked, and the flag may be "
                         "repeated")
    ap.add_argument("-s", "--search", nargs="+", action="extend", default=[],
                    metavar="PATH",
                    help="the sources: one or several folders (walked "
                         "recursively: every audio file of every subfolder is "
                         "taken), glob patterns of folders and/or files "
                         "('music\\Artist *\\*.flac'), or text files of any "
                         "extension that list folders, files and glob "
                         "patterns, one per line ('#' starts a comment). May "
                         "be repeated")
    ap.add_argument("--no-trim", action="store_true",
                    help="do not trim the silence at the edges of the query")
    ap.add_argument("--no-recurse", action="store_true",
                    help="do not descend into the query/search subfolders "
                         "(recursive traversal by default)")
    ap.add_argument("--top", type=int, default=5,
                    help="how many candidates to print in -v")
    ap.add_argument("--only", default=None, metavar="SUBSTR",
                    help="index only the source files whose name contains "
                         "this substring, and only the queries of a -q folder "
                         "(a quick check on a single track)")
    ap.add_argument("--max-source-seconds", type=float, default=0.0,
                    metavar="SEC",
                    help="index only the first SEC seconds of every source "
                         "(a quick check)")
    ap.add_argument("--no-export", action="store_true",
                    help="do not cut any fragment to disk at all")
    ap.add_argument("--export-min-sim", type=float, default=EXPORT_MIN_SIM,
                    metavar="R",
                    help=f"minimum similarity for exporting the fragment "
                         f"(default {EXPORT_MIN_SIM})")
    ap.add_argument("--export-min-margin", type=float, default=EXPORT_MIN_MARGIN,
                    metavar="M",
                    help=f"minimum margin over the other candidates for "
                         f"exporting (default {EXPORT_MIN_MARGIN})")
    ap.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX,
                    metavar="NAME",
                    help="the folder the run folders go into, next to the "
                         "program: one folder per run inside it, named "
                         "<query>__<search>__<date and time> "
                         f"(default {DEFAULT_RESULTS_PREFIX})")
    ap.add_argument("--unsure-subdir", default=DEFAULT_UNSURE_SUBDIR,
                    metavar="NAME",
                    help="subfolder of the run folder for the matches "
                         "that are not convincing enough "
                         f"(default {DEFAULT_UNSURE_SUBDIR})")
    ap.add_argument("--unsure-min-sim", type=float, default=UNSURE_MIN_SIM,
                    metavar="R",
                    help=f"do not even export into the unsure subfolder below "
                         f"this similarity (default {UNSURE_MIN_SIM}; "
                         f"0 = always)")
    ap.add_argument("--equiv-tol", type=float, default=EQUIV_TOL, metavar="R",
                    help="a candidate within this similarity of the best may be "
                         "the same recording in another mix: all such files "
                         f"are reported and exported (default {EQUIV_TOL}; "
                         f"0 = off)")
    ap.add_argument("--equiv-min-sim", type=float, default=EQUIV_MIN_SIM,
                    metavar="R",
                    help="minimum similarity for the equivalence rule to "
                         f"apply (default {EQUIV_MIN_SIM})")
    ap.add_argument("--equiv-gap", type=float, default=EQUIV_GAP, metavar="R",
                    help="the equivalent group must be separated from the rest "
                         "by at least this much, otherwise the candidates are "
                         f"genuinely ambiguous (default {EQUIV_GAP})")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="threads used when indexing (0 = auto)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="verbose output")
    ap.add_argument("--confirm-large-index", action="store_true",
                    help="when the index of the collection would take more than "
                         f"{INDEX_SIZE_WARN // 1024 ** 3} GB, stop and wait for "
                         "a yes on stdin (the window adds this by itself and "
                         "puts the question in a dialog; from a terminal the "
                         "size is only announced)")
    ap.add_argument("-p", "--pitch", nargs="+", default=None, metavar="ST",
                    help="search the query at these pitch shifts, in semitones: "
                         "--pitch -1 0 1 2, or a range --pitch=-2:3:0.5 "
                         "(start:end:step, like python range). Each shift is a "
                         "plain resample by 2^(n/12), so pitch and tempo "
                         "change together — no time-stretching. Default: 0")
    ap.add_argument("--pitchorder", choices=("finer", "symm"), default="finer",
                    help="the order the pitch shifts are tried in: "
                         "'finer' (default) whole semitones first, then the "
                         "halves, then the quarters, each time by distance "
                         "from zero with the upward shift first; 'symm' simply "
                         "by distance from zero. With 'finer' the step of a "
                         "--pitch range must be a power of two")
    ap.add_argument("--nonlazy", action="store_true",
                    help="try every pitch shift even after one of them gave a "
                         "confident match: slower, and it finds a better match "
                         "at a later shift more often. It says which shifts are "
                         "searched, not how they are walked: with "
                         "--pitchbatch=single (the default) each of them still "
                         "gets its own pass over the collection and the table "
                         "still prints after it, so a long search can be watched "
                         "the whole way")
    ap.add_argument("--pitchbatch", choices=("single", "dynamic", "all"),
                    default="single",
                    help="how a batch of queries is walked through the shifts. "
                         "'single' (default): one shift per pass over the "
                         "collection, every time, so a search reports what it "
                         "found as it finds it - with --batch 1 the table of "
                         "shifts searched is printed after every one of them. "
                         "'dynamic': the same at first, but once a pass stops too "
                         "few queries to be worth its walk the rest is bundled, "
                         "one walk per group of pitches - quicker to finish, "
                         "quieter while it runs. 'all': no rounds, every shift of "
                         "every query of the batch in a single walk. All three "
                         "find the same matches in the same order")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    metavar="batch_size",
                    help="how many queries are checked at once while going "
                         "through the whole library once. A bigger batch means a "
                         "quicker search, but the report of an individual query "
                         "is updated less often - you see it once the whole "
                         f"batch is done (default {DEFAULT_BATCH}; 1 shows every "
                         "query as it goes)")
    ap.add_argument("--progress-interval", type=float,
                    default=INTERVAL, metavar="SECONDS",
                    help="how often the bar of a pass over the collection, and of the "
                         "index being built, is written out when the output "
                         "is not a terminal - the window of the gui shows it "
                         "in its status line, a run piped into a file keeps it "
                         "in the log. On a terminal the bar is redrawn in place "
                         "and this does not apply; "
                         f"-1 turns it off (default {INTERVAL:g})")
    ap.add_argument("--selftest", action="store_true",
                    help="check on synthetic distorted queries")
    ap.add_argument("--selftest-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severity", choices=("mild", "normal", "harsh"),
                    default="normal",
                    help="distortion strength in --selftest (normal by "
                         "default — that is a stress test: EQ + reverb + noise 6-22 dB)")
    args = ap.parse_args(normalise_pitch_args(
        list(sys.argv[1:] if argv is None else argv)))
    if not args.query:
        args.query = ["query"]
    if not args.search:
        args.search = ["search"]
    # settled once, here, so that everything below reads the order that will
    # really be walked rather than the one that was asked for
    args.pitchbatch = resolve_pitchbatch(args.pitchbatch, args.nonlazy)

    try:
        if args.selftest:
            st_base, st_names, _ = resolve_sources(
                args.search, recursive=not args.no_recurse)
            return selftest(st_base, n_cases=args.selftest_cases,
                            seed=args.seed, severity=args.severity,
                            verbose=args.verbose, names=st_names)

        # the shift list is read before any real work, so a typo in --pitch
        # costs nothing and is reported as a plain error, not a traceback
        pitches = parse_pitch_args(args.pitch, args.pitchorder)

        t_run = time.perf_counter()
        # the queries are resolved first (just a listing, no decoding) so that
        # the name of the run folder is known before anything is printed — the
        # log of the run then holds the whole report, index build included
        qpaths, q_root = resolve_queries(args.query,
                                         recursive=not args.no_recurse,
                                         only=args.only)
        if not qpaths:
            raise FileNotFoundError(
                f"No queries left in {', '.join(args.query)} after the --only "
                f"{args.only!r} filter.")
        # -s may be a folder, a glob pattern or a text file listing either, so
        # it is resolved into ONE base folder plus the names below it
        s_base, names, records, s_root = resolve_source_records(
            args.search, recursive=not args.no_recurse)
        top_of = top_folders(records)

        # ONE results folder per run inside results/, holding the fragments of
        # every query of this run: the confident ones directly, the weak ones in
        # unsure/, plus log_<same suffix>.txt with everything printed here. The
        # timestamp in the name makes each run its own folder, so nothing of
        # different runs ever mixes.
        run_dir, run_suffix = results_folder(args.results_prefix, q_root,
                                             s_root)
        # both folders are checked before anything expensive is started: a
        # portable program unpacked where it may not write has to say so now,
        # not after the index of a whole collection has been built
        ensure_writable_dir(os.path.dirname(run_dir) or ".",
                            "the results and the log of every run")
        unsure_dir = os.path.join(run_dir, args.unsure_subdir)
        # The log is written for EVERY run, --no-export included: it costs a few
        # kilobytes, and a run that found nothing but has no record of why is the
        # one a person comes back to. The flag is about the fragments.
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, f"{LOG_PREFIX}_{run_suffix}.txt")
        # stderr goes into the log as well: numpy and scipy write their warnings
        # to the real file descriptor 2, and a warning that only ever reaches
        # the console is a warning nobody has afterwards
        tee = Tee(sys.stdout, log_path)
        tee.stderr_backup = sys.stderr          # put back in the finally
        sys.stdout = sys.stderr = tee
        # first line of the run, and so the first line of the log: a log that
        # says which build wrote it answers "what was this?" on its own, months
        # later, without anybody having to remember which exe it came from
        print(version_line())
        print(f"RESULTS   : {run_dir}\\")
        print(f"LOG       : {log_path}")
        if args.no_export:
            print("FRAGMENTS : not cut (--no-export) — this log is written "
                  "all the same")

        if args.only:
            names = [n for n in names if args.only.lower() in n.lower()]
        if not names:
            raise FileNotFoundError(
                f"Nothing left in {', '.join(args.search)} after the --only {args.only!r} filter.")
        if args.verbose:
            print(f"Coarse index: {cache.cache_root()}", flush=True)

        # --- what the index of this collection will cost, before it exists ---
        # The plan reads the manifests and the headers of the missing files and
        # writes NOTHING: a search over a big collection should be able to say
        # "this needs 3.4 GB" while the user can still do something about it.
        # It is said out loud, because on a large collection the estimate is
        # itself a long silence — every file of it is asked which cache folder it
        # belongs in, whether the index already holds it, and how long it is.
        flash("Estimating size of index to build (or update)")
        coarse = CoarseIndex(s_base)
        plan = coarse.plan(_index_files(s_base, names), top_of, jobs=args.jobs)
        for i, line in enumerate(format_plan(plan)):
            print(f"{'PLAN      :' if i == 0 else '            '}{line}")
        if not plan.fits:
            raise NoSpaceError(plan)
        if plan.is_big:
            if not confirm_large_index(args, plan):
                print("STOPPED   : the index was not built — it was "
                      "confirmed away")
                return 3
        ensure_writable_dir(coarse.root, "the index (the cache folder)")

        flash("building the index")
        t0 = time.perf_counter()
        fresh, cached = coarse.build(jobs=args.jobs, verbose=args.verbose)
        coarse.finish()
        t_index = time.perf_counter() - t0
        t_index = time.perf_counter() - t0
        total_audio = sum(e.duration for e in coarse.entries)
        index = SearchIndex(folder=s_base, sources=[], coarse=coarse)
        # a file that could not be read was counted as work but left no index
        # entry: the line must not claim it was indexed
        skipped = getattr(coarse, "skipped", [])
        indexed = fresh - len(skipped)
        print(f"INDEX     : build in {t_index:.2f} s — {len(coarse.entries)} "
              f"file(s), {fmt_duration(total_audio)} of audio")
        print(f"            {coarse.root}, {coarse.bytes_on_disk/1024/1024:.1f} MB "
              f"({indexed} recomputed, {cached} from cache), "
              f"hop {COARSE_HOP/SAMPLE_RATE*1000:.0f} ms, "
              f"{COARSE_BANDS} bands")
        if skipped:
            print(f"            {len(skipped)} file(s) ignored, they could "
                  f"not be read: "
                  + ", ".join(os.path.basename(s) for s in skipped[:5])
                  + (" ..." if len(skipped) > 5 else ""))

        flash(f"reading {len(qpaths)} quer(y/ies)")
        t_q = time.perf_counter()
        queries: List[Tuple[str, Optional[np.ndarray]]] = []
        failed: List[Tuple[str, str]] = []
        for p in qpaths:
            try:
                queries.append((p, load_audio(p)))
            except QUERY_ERRORS as exc:
                print(f"  ERROR   : cannot read {p}: {exc}")
                failed.append((p, str(exc)))
        t_read = time.perf_counter() - t_q
        print(f"Queries: {len(qpaths)}"
              + (f", {len(failed)} could not be read" if failed else ""))

        lazy = not args.nonlazy
        print(f"Pitch (semitones): {', '.join(fmt_semitones(p) for p in pitches)}"
              + (f"  ({len(pitches)} shifts, order '{args.pitchorder}')"
                 if len(pitches) > 1 else "")
              + ("  [lazy: a query stops at its first confident pitch; "
                  "--nonlazy to try them all]"
                 if lazy and len(pitches) > 1 else ""), flush=True)

        # There are three orders to walk the collection in, and one switch on
        # top of them. What a file costs to be made ready for a comparison does
        # not depend on the query, so the orders differ only in how they share
        # that work - the shifts themselves are tried in the same order either
        # way, and the verdict is the same.
        if args.pitchbatch == "all":
            order = (f"the whole collection is scanned once per "
                     f"{args.batch} quer(y/ies), every shift answered from "
                     f"that pass")
        elif args.pitchbatch == "dynamic":
            order = (f"the whole collection is scanned in rounds, {args.batch} "
                     f"quer(y/ies) and one (or more) pitch at a time")
        elif args.nonlazy:
            order = (f"the whole collection is scanned in rounds, one pitch "
                     f"at a time, every pitch searched")
        elif args.batch > 1:
            order = (f"the whole collection is scanned in rounds, {args.batch} "
                     f"quer(y/ies) and one pitch at a time")
        else:
            order = (f"the whole collection is scanned in rounds, one pitch "
                     f"at a time")
        print(f"Order     : {order}", flush=True)

        results, sweep_failed, attempts, t_search_all = sweep(
            queries, index, args, pitches, lazy, run_dir, unsure_dir, s_base)
        failed.extend(sweep_failed)

        if len(results) > 1:
            print("\nSummary   (one line per query: where it was found, at "
                  "which pitch shift, and how similar that was)")
            w = max(len(os.path.basename(lbl)) for lbl, _ in results)
            # the header repeats the layout of the lines below it, arrow and
            # all, so every word stands exactly above its column
            print(f"  {'query':<{w}s} -> {'source':<46} {'position':>9}  "
                  f"{'shift':>5}   r   frames")
            for label, rep in results:
                extra = (f" (+{len(rep.hits)-1} equivalent)"
                         if len(rep.hits) > 1 else "")
                mark = "" if rep.exportable else "  [unsure]"
                pitch = f"{fmt_semitones(rep.pitch):>5} st"
                print(f"  {os.path.basename(label):<{w}s} -> "
                      f"{fit_name(rep.best.file_name, 46):<46} "
                      f"{fmt_time(rep.offset):>9s}  {pitch}  "
                      f"r={rep.best.similarity:.3f}  "
                      f"{rep.best.frame_support*100:3.0f}%{extra}{mark}")

        if failed:
            print(f"\n{len(failed)} query/queries could not be processed:")
            for label, why in failed:
                print(f"  {os.path.basename(label)}: {why}")

        # the timing summary: the index is built once, the queries run against it
        total = time.perf_counter() - t_run
        if attempts:
            sweeps = (f"x{len(pitches)} pitches" if len(pitches) > 1
                      else f"{len(queries)} quer(y/ies)")
            print(f"\nTOTAL     : index {fmt_elapsed(t_index)} (once) + "
                  f"query reading {fmt_elapsed(t_read)} + search "
                  f"{fmt_elapsed(t_search_all)} for {attempts} attempt(s) = "
                  f"{sweeps} "
                  f"(avg {fmt_elapsed(t_search_all/attempts)}) = "
                  f"{fmt_elapsed(total)}")
        else:
            print(f"\nTOTAL     : index {fmt_elapsed(t_index)} + query reading "
                  f"{fmt_elapsed(t_read)} = {fmt_elapsed(total)}")
        # a run where nothing at all worked is a failure, not a success
        return 1 if failed and not results else 0
    except (RuntimeError, FileNotFoundError, ValueError, OSError,
            subprocess.CalledProcessError) as exc:
        report_error(f"Error: {exc}")
        return 2
    except KeyboardInterrupt:
        report_error("\nInterrupted — the search was stopped here. Everything "
                     "printed up to this point is in the log file.")
        return 130
    except Exception as exc:
        # Something the program did not plan for. For those the place and the
        # line matter more than the message, so the WHOLE traceback goes into
        # both streams: without it the log ends at the last thing that worked
        # and the failure is a mystery afterwards. MemoryError on a very large
        # library lives here too — it is not an OSError, and it was the one
        # that used to vanish.
        report_error(f"Unexpected error: {exc.__class__.__name__}: {exc}\n"
                     + "".join(traceback.format_exc()))
        return 2
    finally:
        # close the log and give stdout and stderr back, even if the run died
        # halfway
        if isinstance(sys.stdout, Tee):
            tee, sys.stdout = sys.stdout, sys.stdout.stream
            sys.stderr = tee.stderr_backup
            tee.flush()
            tee.close()


if __name__ == "__main__":
    sys.exit(main())
