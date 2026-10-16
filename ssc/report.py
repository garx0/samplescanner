"""The result of one query, and the text a human reads."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ssc.audio import source_span_of_query, source_start_of_query
from ssc.config import (EQUIV_GAP, EQUIV_MIN_SIM, EQUIV_TOL,
                        EXPORT_MIN_MARGIN, EXPORT_MIN_SIM, HOP,
                        MAX_REPORTED_OCCURRENCES, ORIG_PITCH_SUBDIR,
                        SAMPLE_RATE)
from ssc.format import fmt_time
from ssc.index import SearchIndex
from ssc.pitch import fmt_semitones
from ssc.search import (Candidate, _margins, export_decision, find_match,
                        is_confident, verdict)

@dataclass
class Hit:
    """One confirmed match: a source file, the moment in it and the score."""
    file_name: str
    offset: float
    similarity: float


@dataclass
class Report:
    """Result for one query at one pitch shift."""
    label: str
    query_name: str
    best: Candidate
    offset: float
    duration: float          # length of the searched signal (source seconds)
    margin: float
    confident: bool
    verdict_text: str
    exportable: bool = False
    export_reason: str = ""
    # Every confirmed match. Normally just the best one; several entries when
    # the collection holds equivalent copies of the same recording.
    hits: List[Hit] = field(default_factory=list)
    # --- what the report printing needs; kept here so that a pitch sweep can
    # --- search many shifts silently and print only the one that wins
    pitch: float = 0.0
    raw_duration: float = 0.0      # length of the query file itself
    # what the exported fragment has to cover: the WHOLE query, measured in the
    # source timeline. It is longer than `duration` whenever trimming or the
    # pitch shift made the searched signal shorter than the query file.
    export_duration: float = 0.0
    kept_duration: float = 0.0     # the trimmed query, in query seconds
    n_frames: int = 0              # analysis frames of the searched signal
    trim_shift: float = 0.0
    trim_tail: float = 0.0         # what trimming removed from the END, in s
    looping: bool = False
    duplicated: bool = False
    margin_local: float = 0.0
    margin_file: float = 0.0
    equivalents: List[Candidate] = field(default_factory=list)
    runner: Optional[Candidate] = None
    final: List[Candidate] = field(default_factory=list)
    # candidates of a whole pitch sweep: (candidate, the shift it came from).
    # Empty unless more than one shift was tried.
    sweep_candidates: List[Tuple[Candidate, float]] = field(
        default_factory=list)
    # the lines that named the fragments when they were written, so that a match
    # found half way through a sweep can be named again in its verdict without
    # saying that it had been written down before — the reader of a verdict
    # should not have to have been watching the log. See print_saved.
    exported_lines: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """How good this attempt is; used to pick the best of several pitches."""
        return self.best.similarity


def analyse_query(label: str, index: SearchIndex,
                  qfeats: Dict[Tuple[int, int], np.ndarray],
                  n_frames_query: int, duration: float, trim_shift: float,
                  top_n: int, pitch: float = 0.0,
                  export_min_sim: float = EXPORT_MIN_SIM,
                  export_min_margin: float = EXPORT_MIN_MARGIN,
                  query_duration: Optional[float] = None,
                  kept_duration: Optional[float] = None,
                  query_x: Optional[np.ndarray] = None,
                  query_trim_tail: float = 0.0,
                  equiv_tol: float = EQUIV_TOL,
                  equiv_min_sim: float = EQUIV_MIN_SIM,
                  equiv_gap: float = EQUIV_GAP,
                  coarse_result: Optional[Tuple[List, List]] = None) -> Report:
    """
    Searches one query (already trimmed and pitch-shifted) and works out
    everything the report needs. Prints nothing — see print_report.

    `coarse_result` is the shortlist of candidates, already computed together
    with the other queries of the same walk over the index — see
    coarse_search_many.
    """
    best, final, _per_file = find_match(
        index, qfeats, top_n=top_n,
        duration=query_duration or duration, query_x=query_x,
        coarse_result=coarse_result)
    margin_local, margin_file, equivalents = _margins(
        best, final, equiv_tol, equiv_min_sim, equiv_gap)
    # when a repeat was found there is no ambiguity inside the file — the
    # confidence is judged by the margin relative to the OTHER files
    looping = bool(best.occurrences) and len(best.occurrences) > 1
    # several files hold the same recording: they are one answer, not rivals,
    # so the margin is judged against the files that are NOT equivalent
    duplicated = len(equivalents) > 1
    margin_eff = margin_file if (looping or duplicated) \
        else min(margin_local, margin_file)

    if duplicated:
        verdict_text = "FOUND (confident)"
        exportable = True
        export_reason = (f"{len(equivalents)} equivalent sources "
                         f"(r spread {equivalents[-1].similarity - equivalents[0].similarity:.3f}, "
                         f"same recording)")
    else:
        verdict_text = verdict(best.similarity, margin_eff, best.frame_support)
        exportable, export_reason = export_decision(
            best.similarity, margin_eff, best.frame_support,
            min_sim=export_min_sim, min_margin=export_min_margin)

    # the position of the match is where the TRIMMED, resampled query starts;
    # the reported offset is where the query file itself starts (see
    # source_start_of_query), and the fragment has to cover the whole query
    hits = [Hit(file_name=c.file_name,
                offset=source_start_of_query(
                    c.exact_frame * HOP / SAMPLE_RATE, trim_shift, pitch),
                similarity=c.similarity)
            for c in (equivalents if duplicated else [best])]
    return Report(
        label=label, query_name=label, best=best,
        offset=source_start_of_query(
            best.exact_frame * HOP / SAMPLE_RATE, trim_shift, pitch),
        duration=query_duration or duration, margin=margin_eff,
        confident=exportable if duplicated else
        is_confident(best.similarity, margin_eff, best.frame_support),
        verdict_text=verdict_text,
        exportable=exportable, export_reason=export_reason,
        hits=hits, pitch=pitch, raw_duration=duration, n_frames=n_frames_query,
        trim_shift=trim_shift, looping=looping, duplicated=duplicated,
        export_duration=source_span_of_query(duration, pitch),
        kept_duration=(kept_duration if kept_duration is not None
                       else (query_duration or duration)),
        trim_tail=query_trim_tail,
        margin_local=margin_local, margin_file=margin_file,
        equivalents=equivalents,
        runner=next((c for c in final if c is not best), None),
        final=final)


def print_report(rep: Report, index: SearchIndex, top_n: int = 5,
                 verbose: bool = False, with_query: bool = True) -> None:
    """Prints what analyse_query worked out, exactly as it always was.

    `with_query` is False when the caller has already named the query above
    something of its own - the table of shifts, in a sweep - and does not want
    the name twice.
    """
    best = rep.best
    trim_shift = rep.trim_shift
    print()
    if with_query:
        print(f"QUERY  : {rep.label}")
    # NOT FOUND means nothing was found. Printing a source, a position, a
    # similarity and a margin for it would be describing a moment the program
    # itself says is not the sample — so the report is the verdict alone. What
    # the query is was already said above the table of its pitches, where the
    # sweep starts and the length does not depend on the shift yet.
    found_any = not rep.verdict_text.startswith("NOT FOUND")
    if found_any:
        print(f"  SOURCE  : {best.file_name}")
        # the position the search settled on, the same one the candidate list
        # shows, and next to it the position the exported fragment starts from
        # — the two differ by the sub-frame refinement and by the trimmed head
        found = source_start_of_query(best.best_frame * HOP / SAMPLE_RATE,
                                      trim_shift, rep.pitch)
        print(f"  OFFSET  : {fmt_time(found)}   "
              f"(exported fragment starts from {fmt_time(rep.offset)})")
        # what was done to the query, right after where it was found: the shift
        # is what decides where the fragment has to be cut from and how it
        # sounds
        if abs(rep.pitch) > 1e-9:
            print(f"  PITCH  : {fmt_semitones(rep.pitch)} st (relative to "
                  f"query) => the source was sampled at "
                  f"{fmt_semitones(-rep.pitch)} st (see repitched fragment of "
                  f"the source in the results folder, and at original pitch in "
                  f"{ORIG_PITCH_SUBDIR} subfolder)")
        else:
            print("  PITCH  : 0 st (see fragment of the source in the results "
                  "folder)")
        print(f"  SIMILARITY: r={best.similarity:.3f}   "
              f"frame support={best.frame_support*100:.0f}%")
        if rep.looping or rep.duplicated:
            why = "repeats within the file" if rep.looping else \
                "equivalent copies of the same recording"
            print(f"  MARGIN  : to the best non-equivalent moment "
                  f"{rep.margin_file:+.3f} ({why} do not count)")
        else:
            print(f"  MARGIN  : to the best moment in the same file "
                  f"{rep.margin_local:+.3f}, to the best in other files "
                  f"{rep.margin_file:+.3f}")

    print(f"  VERDICT : {rep.verdict_text}")
    if rep.duplicated:
        width = name_column_width(index, rep.equivalents)
        print(f"  EQUIVALENT: {len(rep.equivalents)} files hold the same "
              f"recording — every one of them is correct, every one of them "
              f"is exported:")
        for i, c in enumerate(rep.equivalents):
            t = source_start_of_query(
                c.exact_frame * HOP / SAMPLE_RATE, trim_shift, rep.pitch)
            mark = "  <-- best" if i == 0 else ""
            print(f"             {fit_name(c.file_name, width)} "
                  f"{fmt_time(t):>10s}  r={c.similarity:.3f}{mark}")
    if found_any and best.occurrences and len(best.occurrences) > 1:
        shown = ", ".join(
            fmt_time(source_start_of_query(t, trim_shift, rep.pitch))
            for t, _ in best.occurrences[:MAX_REPORTED_OCCURRENCES])
        more = (f" and {len(best.occurrences)-MAX_REPORTED_OCCURRENCES} more"
                if len(best.occurrences) > MAX_REPORTED_OCCURRENCES else "")
        print(f"  REPEAT  : this fragment repeats every ~{best.period:.2f} s "
              f"({len(best.occurrences)} equal-quality occurrences).")
        print(f"            earliest: {fmt_time(rep.offset)}; all: {shown}{more}")

    # The margin was small, so two moments are hard to tell apart: say so, and name
    # the competitor the decision was actually made against. With a sweep the
    # candidate list below is the best of EVERY pitch, while this one is the
    # runner-up of the winning pitch alone — which is why it can be absent from
    # that list and score below all of it. The margin is printed with it, so the
    # two numbers can be checked against each other.
    runner = rep.runner
    if found_any and rep.looping:
        pass                       # the repeat explains why there are many
    elif found_any and rep.duplicated:
        pass                       # the duplicates are listed above
    elif found_any and runner is not None \
            and min(rep.margin_local, rep.margin_file) < 0.10:
        print(f"  ! WARNING: the margin is only "
              f"{min(rep.margin_local, rep.margin_file):+.3f}. Closest "
              f"competitor: "
              f"{runner.file_name} @ "
              f"{fmt_time(source_start_of_query(runner.exact_frame*HOP/SAMPLE_RATE, trim_shift, rep.pitch))} "
              f"(r={runner.similarity:.3f}) — worth listening to both.")

    if verbose:
        rows = rep.sweep_candidates or [(c, rep.pitch)
                                       for c in rep.final[:max(top_n, 1)]]
        width = name_column_width(index, [c for c, _ in rows])
        # the position belongs to the shift it was found at, so it is printed
        # with that shift too — otherwise the two do not go together
        print("  candidates:"
              + ("   (position and shift of each candidate — the best of "
                 "EVERY pitch, so it can differ from the competitor named "
                 "above)" if rep.sweep_candidates else ""))
        for c, cp in rows:
            mark = "  <-- selected" if c is best else ""
            tag = "same file" if c.file_name == best.file_name else "other file"
            sh = f"{fmt_semitones(cp):>5} st" if rep.sweep_candidates else "      "
            print(f"    {fit_name(c.file_name, width)} {sh} "
                  f"{fmt_time(source_start_of_query(c.best_frame*HOP/SAMPLE_RATE, trim_shift, cp)):>10s}  "
                  f"r={c.similarity:.3f}  frames={c.frame_support*100:3.0f}%  "
                  f"z={c.fused_z:5.2f}  [{tag}]{mark}")


def report(label: str, index: SearchIndex,
           qfeats: Dict[Tuple[int, int], np.ndarray],
           n_frames_query: int, duration: float, trim_shift: float,
           top_n: int, verbose: bool,
           export_min_sim: float = EXPORT_MIN_SIM,
           export_min_margin: float = EXPORT_MIN_MARGIN,
           query_duration: Optional[float] = None,
           kept_duration: Optional[float] = None,
           query_x: Optional[np.ndarray] = None,
           query_trim_tail: float = 0.0,
           equiv_tol: float = EQUIV_TOL,
           equiv_min_sim: float = EQUIV_MIN_SIM,
           equiv_gap: float = EQUIV_GAP,
           pitch: float = 0.0, quiet: bool = False,
           coarse_result: Optional[Tuple[List, List]] = None) -> Report:
    """Searches one query and prints the report unless quiet is set."""
    rep = analyse_query(
        label, index, qfeats, n_frames_query, duration, trim_shift, top_n,
        pitch=pitch, export_min_sim=export_min_sim,
        export_min_margin=export_min_margin,
        query_duration=query_duration, kept_duration=kept_duration,
        query_x=query_x, query_trim_tail=query_trim_tail,
        equiv_tol=equiv_tol, equiv_min_sim=equiv_min_sim,
        equiv_gap=equiv_gap, coarse_result=coarse_result)
    if not quiet:
        print_report(rep, index, top_n, verbose)
    return rep


def name_column_width(index: SearchIndex,
                      final: Optional[List[Candidate]] = None) -> int:
    """Width of the file name column: adapts to subfolders in the paths."""
    names = [s.name for s in index.sources]
    if index.coarse is not None:
        names += [e.name for e in index.coarse.entries]
    if final:
        names += [c.file_name for c in final]
    longest = max((len(n) for n in names), default=32)
    return max(32, min(longest, 70))


def fit_name(name: str, width: int) -> str:
    """Truncates a long path (e.g. 2.12 - Artist - Title.flac) to a width."""
    if len(name) <= width:
        return name.ljust(width)
    if width <= 5:
        return name[:width]
    root, ext = os.path.splitext(name)
    tail = root[-(width - 3 - len(ext)):] + ext if len(ext) < width - 3 else name[-(width - 3):]
    return ("..." + tail).ljust(width)

