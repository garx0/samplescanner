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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ssc.audio import load_audio, shift_pitch, trim_silence
from ssc.config import (COARSE_BANDS, COARSE_HOP, DEFAULT_RESULTS_PREFIX,
                        DEFAULT_UNSURE_SUBDIR, EQUIV_GAP, EQUIV_MIN_SIM,
                        EQUIV_TOL, EXPORT_MIN_MARGIN, EXPORT_MIN_SIM, HOP,
                        INDEX_SIZE_WARN, LOG_PREFIX, SAMPLE_RATE,
                        UNSURE_MIN_SIM)
from ssc.export import export_report, results_folder
from ssc.features import logmel_bank
from ssc.format import fmt_duration, fmt_elapsed, fmt_size, fmt_time
from ssc.index import (CoarseIndex, IndexPlan, SearchIndex, default_index_dir,
                       format_plan)
from ssc.inputs import resolve_queries, resolve_sources
from ssc.paths import ensure_writable_dir
from ssc.pitch import (fmt_semitones, normalise_pitch_args, parse_pitch_args)
from ssc.report import (Report, fit_name, name_column_width, print_report,
                        report)
from ssc.search import Candidate
from ssc.selftest import selftest
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


class NoSpaceError(RuntimeError):
    """The index would not fit on the drive — the search cannot start."""

    def __init__(self, plan: IndexPlan):
        self.plan = plan
        super().__init__(
            f"the index of this collection would take about "
            f"{fmt_size(plan.bytes_estimate)}, but only "
            f"{fmt_size(plan.free_bytes)} is free on the drive that holds "
            f"{plan.index_dir}.\n"
            f"The search is NOT started: free up some space, or point the "
            f"cache elsewhere with --index-dir, and run it again.")


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


def search_query(label: str, x: np.ndarray, index: SearchIndex,
                 top_n: int, verbose: bool, pitch: float = 0.0,
                 export_min_sim: float = EXPORT_MIN_SIM,
                 export_min_margin: float = EXPORT_MIN_MARGIN,
                 equiv_tol: float = EQUIV_TOL,
                 equiv_min_sim: float = EQUIV_MIN_SIM,
                 equiv_gap: float = EQUIV_GAP,
                 no_trim: bool = False
                 ) -> Tuple[Report, float, float]:
    """
    Searches one query at one pitch shift and returns (report, seconds the
    search took). Nothing is printed and nothing is written to disk, so a sweep over
    many shifts can search quietly and export only the one that wins.

    The silence is trimmed on the ORIGINAL query, before the pitch is touched —
    otherwise the shift would be weighed against the shifted noise. The reported
    offset stays in the source timeline either way.
    """
    duration = len(x) / SAMPLE_RATE
    trim_shift = 0.0
    trim_tail = 0.0
    t_one = time.perf_counter()
    if not no_trim:
        kept, trim_shift = trim_silence(x, SAMPLE_RATE)
        trim_tail = max(0.0, duration - trim_shift - len(kept) / SAMPLE_RATE)
        x = kept
    kept_duration = len(x) / SAMPLE_RATE     # the trimmed query, query seconds
    y = shift_pitch(x, pitch)
    qf = logmel_bank(y)
    qduration = len(y) / SAMPLE_RATE   # length of what is actually searched

    t_one = time.perf_counter()
    rep = report(label, index, qf, max(m.shape[0] for m in qf.values()),
                 duration, trim_shift, top_n, verbose,
                 export_min_sim=export_min_sim,
                 export_min_margin=export_min_margin,
                 query_duration=qduration, kept_duration=kept_duration,
                 query_x=y, query_trim_tail=trim_tail,
                 equiv_tol=equiv_tol,
                 equiv_min_sim=equiv_min_sim,
                 equiv_gap=equiv_gap, pitch=pitch, quiet=True)
    t_search = time.perf_counter() - t_one
    return rep, t_search


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
    ap.add_argument("--index-dir", default=None, metavar="DIR",
                    help="where to keep the on-disk index (by default "
                         "./cache/<collection>, next to the query folder)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the index from scratch, do not use the cache")
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
                         "at a later shift more often")
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
        s_base, names, s_root = resolve_sources(
            args.search, recursive=not args.no_recurse)

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
        idx_dir = args.index_dir or default_index_dir(s_base)
        if args.rebuild_index and os.path.isdir(idx_dir):
            shutil.rmtree(idx_dir, ignore_errors=True)
        if args.verbose:
            print(f"Coarse index: {idx_dir}", flush=True)

        # --- what the index of this collection will cost, before it exists ---
        # The plan reads the manifest and the headers of the missing files and
        # writes NOTHING: a search over a big collection should be able to say
        # "this needs 3.4 GB" while the user can still do something about it.
        # It is said out loud, because on a large collection the estimate is
        # itself a long silence — every file of it is asked whether the index
        # already holds it, and how long it is.
        print("Estimating size of index to build (or update)...", flush=True)
        coarse = CoarseIndex(s_base, idx_dir)
        plan = coarse.plan(names)
        for i, line in enumerate(format_plan(plan)):
            print(f"{'PLAN      :' if i == 0 else '            '}{line}")
        if not plan.fits:
            raise NoSpaceError(plan)
        if plan.is_big:
            if not confirm_large_index(args, plan):
                print("STOPPED   : the index was not built — it was "
                      "confirmed away")
                return 3
        ensure_writable_dir(idx_dir,
                            "the index of the collection (the cache folder)")

        t0 = time.perf_counter()
        fresh, cached = coarse.build(names, jobs=args.jobs, verbose=args.verbose)
        t_index = time.perf_counter() - t0
        total_audio = sum(e.duration for e in coarse.entries)
        index = SearchIndex(folder=s_base, sources=[], coarse=coarse)
        # a file that could not be read was counted as work but left no index
        # entry: the line must not claim it was indexed
        skipped = getattr(coarse, "skipped", [])
        indexed = fresh - len(skipped)
        print(f"INDEX     : build in {t_index:.2f} s — {len(coarse.entries)} "
              f"file(s), {fmt_duration(total_audio)} of audio")
        print(f"            {idx_dir}, {coarse.bytes_on_disk/1024/1024:.1f} MB "
              f"({indexed} recomputed, {cached} from cache), "
              f"hop {COARSE_HOP/SAMPLE_RATE*1000:.0f} ms, "
              f"{COARSE_BANDS} bands")
        if skipped:
            print(f"            {len(skipped)} file(s) ignored, they could "
                  f"not be read: "
                  + ", ".join(os.path.basename(s) for s in skipped[:5])
                  + (" ..." if len(skipped) > 5 else ""))

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

        # The sweep is query-major: take one query file, try every pitch shift on it
        # against the WHOLE collection, and only when the shifts run out move on
        # to the next file. Each query is reported and exported as soon as its
        # own shifts are done.
        results: List[Tuple[str, Report]] = []
        t_search_all = 0.0
        attempts = 0
        for label, x in queries:
            if x is None:
                continue
            print(f"\nQUERY : {label}")
            # what the query itself is, before any pitch is tried: the length
            # does not depend on the shift (trimming happens before it, on the
            # original), and the frame count is the one of the first shift the
            # sweep tries. Both cost a few milliseconds to work out here, which
            # is the price of saying it before the table instead of after it.
            raw_s = len(x) / SAMPLE_RATE
            kept_x = trim_silence(x, SAMPLE_RATE)[0] if not args.no_trim else x
            kept_s = len(kept_x) / SAMPLE_RATE
            qf0 = logmel_bank(shift_pitch(kept_x, pitches[0]))
            n_frames0 = max(m.shape[0] for m in qf0.values())
            line = f"         duration {raw_s:.3f} s"
            if kept_s < raw_s - 1e-6:
                line += f" -> {kept_s:.3f} s after trimming"
            print(line + f", {n_frames0} analysis frames")
            best: Optional[Report] = None
            t_search_sum = 0.0
            tried = 0
            # every candidate of every shift, so the report can say which shift
            # each of them came from
            swept: Dict[Tuple[str, float], Tuple[Candidate, float]] = {}
            name_w = 0
            skipped = False
            for pi, pitch in enumerate(pitches):
                if lazy and len(pitches) > 1 and best is not None \
                        and best.confident:
                    skipped = True
                    continue
                # one query must never stop the others: whatever goes wrong
                # here is reported and the loop moves on
                try:
                    rep, t_search = search_query(
                        label, x, index, args.top, args.verbose,
                        pitch=pitch,
                        export_min_sim=args.export_min_sim,
                        export_min_margin=args.export_min_margin,
                        equiv_tol=args.equiv_tol,
                        equiv_min_sim=args.equiv_min_sim,
                        equiv_gap=args.equiv_gap, no_trim=args.no_trim)
                except QUERY_ERRORS as exc:
                    print(f"    {fmt_semitones(pitch):>5} st: ERROR {exc} "
                          f"(this attempt only, the rest continues)")
                    failed.append((label, str(exc)))
                    continue
                attempts += 1
                tried += 1
                t_search_sum += t_search
                t_search_all += t_search
                for c in rep.final[:max(args.top, 1)]:
                    key = (c.file_name, round(c.exact_frame * HOP / SAMPLE_RATE, 3))
                    old = swept.get(key)
                    if old is None or c.similarity > old[0].similarity:
                        swept[key] = (c, pitch)
                better = best is None or rep.score > best.score
                if better:
                    best = rep
                    mark = "  <- best so far" + \
                        (" [confident]" if rep.confident else "")
                else:
                    mark = ""
                # the name column is as wide as the one the candidate list of
                # this query uses, so the two read the same; it is measured on
                # the first shift that runs, and the header is printed with it
                if not name_w:
                    name_w = name_column_width(index, rep.final)
                    # the rows below are the best of each pitch, not the final
                    # answer — the answer comes after every pitch has been tried
                    print("    Best matches for each pitch (might be off; "
                          "you'll be able to listen to good enough results "
                          "for this query after all pitches are checked):")
                    print(f"    {'pitch':>6}  {'source':<{name_w}s} "
                          f"{'position':>9}  {'time spent':>10}  "
                          f"{'r':>6}  {'frames':>6}", flush=True)
                print(f"    {fmt_semitones(pitch):>5} st  "
                      f"{fit_name(rep.best.file_name, name_w):<{name_w}s} "
                      f"{fmt_time(rep.offset):>9s}  "
                      f"{fmt_elapsed(t_search):>10}  "
                      f"{rep.score:6.3f}  "
                      f"{rep.best.frame_support*100:3.0f}%{mark}",
                      flush=True)
            # one line about the shifts that were not tried at all: a line per
            # skipped shift buries the table under them on a wide sweep
            if skipped:
                print(f"    Next pitches were skipped, "
                      f"{fmt_semitones(best.pitch)} st was already confident "
                      f"(use non-lazy mode to try all pitches anyway)")
            if best is None:
                continue
            # hand the candidates of every shift to the report
            best.sweep_candidates = (
                sorted(swept.values(),
                       key=lambda t: -t[0].similarity)[:max(args.top, 1)]
                if len(pitches) > 1 else [])
            print()
            print_report(best, index, args.top, args.verbose)
            export_report(best, run_dir, unsure_dir, not args.no_export,
                          s_base, args.unsure_min_sim)
            print(f"  TIME     : index search {fmt_elapsed(t_search_sum)} over "
                  f"{tried} pitch(es)")
            results.append((label, best))

        if len(results) > 1:
            print("\nSummary   (one line per query: where it was found, at "
                  "which pitch shift, and how similar that was)")
            w = max(len(os.path.basename(lbl)) for lbl, _ in results)
            # the header repeats the layout of the lines below it, arrow and
            # all, so every word stands exactly above its column
            print(f"  {'query':<{w}s} -> {'source':<46} {'position':>9}  "
                  f"{'shift':>5}   r")
            for label, rep in results:
                extra = (f" (+{len(rep.hits)-1} equivalent)"
                         if len(rep.hits) > 1 else "")
                mark = "" if rep.exportable else "  [unsure]"
                pitch = f"{fmt_semitones(rep.pitch):>5} st"
                print(f"  {os.path.basename(label):<{w}s} -> "
                      f"{fit_name(rep.best.file_name, 46):<46} "
                      f"{fmt_time(rep.offset):>9s}  {pitch}  "
                      f"r={rep.best.similarity:.3f}{extra}{mark}")

        if failed:
            print(f"\n{len(failed)} query/queries could not be processed:")
            for label, why in failed:
                print(f"  {os.path.basename(label)}: {why}")

        # the timing summary: the index is built once, the queries run against it
        total = time.perf_counter() - t_run
        if attempts:
            sweeps = (f"x{len(pitches)} pitches" if len(pitches) > 1
                      else f"{len(queries)} quer(y/ies)")
            print(f"\nTOTAL     : index {t_index:.2f} s (once) + "
                  f"query reading {t_read:.2f} s + search "
                  f"{t_search_all:.2f} s for {attempts} attempt(s) = "
                  f"{sweeps} "
                  f"(avg {t_search_all/attempts:.2f} s) = {total:.2f} s")
        else:
            print(f"\nTOTAL     : index {t_index:.2f} s + query reading "
                  f"{t_read:.2f} s = {total:.2f} s")
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

