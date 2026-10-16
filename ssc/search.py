"""The precise pass: candidates, similarity, margins, the decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ssc.config import (CANDIDATES_PER_FILE, CANDIDATES_PER_MEMBER,
                        CONF_MARGIN, CONF_SIM, CONF_SUPPORT, DECISIVE_MARGIN,
                        DECISIVE_SIM, ENERGY_FLOOR, EPS, EQUIV_GAP,
                        EQUIV_MIN_SIM, EQUIV_TOL, EXPORT_MIN_MARGIN,
                        EXPORT_MIN_SIM, HOP, LOCAL_EXCLUSION, LOCAL_REFINE,
                        LOOP_SEPARATION, LOOP_TOL, LOOP_Z_FRACTION,
                        MAX_EQUIVALENTS, MIN_REPETITION_COUNT, PEAK_EXCLUSION,
                        SAMPLE_RATE)
from ssc.features import (BANK, build_views, frame_correlations,
                          profile_from_prepared, similarity_at)
from ssc.index import SearchIndex, SourceIndex, coarse_search, fine_verify

# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    file_name: str
    frame: int
    fused_z: float            # z-sum of the bank (wide search)
    similarity: float = 0.0   # cosine similarity (the final metric)
    frame_support: float = 0.0
    best_frame: int = 0       # refined offset (whole frames)
    exact_frame: float = 0.0  # same with sub-frame interpolation
    refined: bool = False
    # repetitiveness: if a break/loop plays around, the same material gives
    # several equal-quality positions. In that case we report the earliest.
    period: Optional[float] = None        # repeat period, s (None = no repeat)
    occurrences: Optional[List[Tuple[float, float]]] = None  # [(time, r), ...]


def _fit_profile(prof: np.ndarray, n: int) -> np.ndarray:
    """
    Brings a profile onto the common source frame grid. Frame k starts at the
    same sample k*HOP in every resolution, so the profiles are aligned at the
    start; only their length differs (a larger n_fft yields fewer frames at
    the tail). Missing frames are extended with the last value.
    """
    if len(prof) == n:
        return prof
    if len(prof) > n:
        return prof[:n]
    return np.concatenate([prof, np.full(n - len(prof), prof[-1])])


def energy_gate(energy: np.ndarray, window: int, n_out: int,
                floor: float = ENERGY_FLOOR) -> np.ndarray:
    """
    A mask of n_out frames: the window [k, k+window) is not quieter than
    `floor` times the median frame energy of the file. energy may be shorter
    than n_out (different n_fft give different frame counts) — in that case
    the missing frames are treated as passing.
    """
    n = len(energy)
    med = float(np.median(energy)) + EPS
    cs = np.concatenate([[0.0], np.cumsum(energy)])
    idx = np.arange(n)
    end = np.minimum(idx + window, n)
    wsum = (cs[end] - cs[idx]) / np.maximum(end - idx, 1)
    gate = wsum >= floor * med
    if n_out > n:
        gate = np.concatenate([gate, np.ones(n_out - n, dtype=bool)])
    return gate[:n_out]


def scan_file(src: SourceIndex,
              qviews: Dict[Tuple[int, int, bool], np.ndarray],
              rviews: Dict[Tuple[int, int, bool], np.ndarray],
              keep_profiles: bool = False
              ) -> Tuple[np.ndarray, Optional[List[Tuple[str, np.ndarray]]]]:
    """
    Sum of the z-profiles of the bank for one source file (length = n_frames).
    If keep_profiles, also returns the profiles of the individual features —
    they are needed for candidate selection (see _candidates).

    Prepared sources (centring, padding, cumulative sums) come from the
    SourceIndex cache: they are identical for every query, and recomputing
    them per query was the most expensive part of the search.
    """
    acc = np.zeros(src.n_frames)
    used = 0.0
    parts: List[Tuple[str, np.ndarray]] = []
    gates: Dict[bool, np.ndarray] = {}
    for n_fft, n_mels, delta, weight in BANK:
        key = (n_fft, n_mels, delta)
        T = qviews[key].shape[0]
        gate = gates.get(delta)
        if gate is None:
            gate = gates[delta] = energy_gate(src.energy, T, src.n_frames)
        prep = src.prepared_view(key)
        prof = _fit_profile(profile_from_prepared(qviews[key], prep),
                            src.n_frames)
        prof = np.where(gate, prof, -np.inf)      # quiet windows are excluded
        finite = np.isfinite(prof)
        if finite.sum() < 2:
            continue
        sd = prof[finite].std()
        if sd < 1e-9:
            continue
        mu = prof[finite].mean()
        acc += weight * np.where(finite, (prof - mu) / sd, 0.0)
        used += weight
        if keep_profiles:
            parts.append((f"{n_fft}x{n_mels}{'+d' if delta else ''}", prof))
    if used > 0:
        acc /= used
    acc = np.where(np.isfinite(acc), acc, -np.inf)
    return acc, (parts if keep_profiles else None)


def _candidates(fused: np.ndarray,
                parts: List[Tuple[str, np.ndarray]]) -> List[int]:
    """
    Candidates = peaks of the combined profile plus peaks of each individual
    feature (with neighbouring suppression). The latter is needed for recall:
    when the features disagree, the true offset may not reach the top of the
    combined profile.
    """
    picked = _pick_peaks(fused, CANDIDATES_PER_FILE)
    for _name, prof in parts:
        for k in _pick_peaks(prof, CANDIDATES_PER_MEMBER):
            if all(abs(k - p) > PEAK_EXCLUSION for p in picked):
                picked.append(k)
    picked.sort(key=lambda k: -fused[k] if np.isfinite(fused[k]) else np.inf)
    return picked


def _pick_peaks(score: np.ndarray, count: int) -> List[int]:
    """Local maxima with neighbour suppression over ±PEAK_EXCLUSION."""
    finite = np.isfinite(score)
    if not finite.any():
        return []
    masked = np.where(finite, score, -np.inf)
    order = np.argsort(masked)[::-1]
    picked: List[int] = []
    for k in order:
        if not np.isfinite(masked[k]) or masked[k] <= 0:
            break
        if all(abs(int(k) - p) > PEAK_EXCLUSION for p in picked):
            picked.append(int(k))
        if len(picked) >= count:
            break
    return picked


def _refine(qviews: Dict[Tuple[int, int, bool], np.ndarray],
            rviews: Dict[Tuple[int, int, bool], np.ndarray],
            k: int) -> Tuple[int, float, float]:
    """Refines the offset by +-LOCAL_REFINE frames and computes the final metrics."""
    best = (-2.0, k)
    # band means and scales are computed once over the whole file so that
    # similarity_at agrees with profile_from_prepared
    stats = {key: (R.mean(axis=0), R.std(axis=0) + EPS)
             for key, R in rviews.items()}
    for probe in range(max(0, k - LOCAL_REFINE), k + LOCAL_REFINE + 1):
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

    # fraction of frames with a strong correlation — on the finest bank feature
    n_fft, n_mels, delta = BANK[0][0], BANK[0][1], BANK[0][2]
    Q = qviews[(n_fft, n_mels, delta)]
    win = rviews[(n_fft, n_mels, delta)][best_k: best_k + Q.shape[0]]
    if win.shape[0] == Q.shape[0]:
        _per_frame, support = frame_correlations(Q, win)
    else:
        support = 0.0
    return best_k, similarity, support


def find_match(index: SearchIndex,
               qfeats: Dict[Tuple[int, int], np.ndarray],
               top_n: int = 5, duration: float = 0.0,
               query_x: Optional[np.ndarray] = None,
               coarse_result: Optional[Tuple[List, List]] = None
               ) -> Tuple[Candidate, List[Candidate], Dict[str, np.ndarray]]:
    if index.coarse is not None:
        if query_x is None:
            raise RuntimeError("searching the on-disk index requires the raw "
                               "query signal")
        return _find_match_disk(index, qfeats, top_n, duration, query_x,
                                coarse_result)
    qviews = build_views(qfeats)

    final: List[Candidate] = []
    per_file_score: Dict[str, np.ndarray] = {}
    for src in index.sources:
        rviews = getattr(src, "views", None)
        if rviews is None:
            rviews = src.views = build_views(src.feats)
        fused, parts = scan_file(src, qviews, rviews, keep_profiles=True)
        per_file_score[src.name] = fused
        for k in _candidates(fused, parts):
            bk, sim, sup = _refine(qviews, rviews, k)
            final.append(Candidate(file_name=src.name, frame=k,
                                   fused_z=float(fused[k]), best_frame=bk,
                                   similarity=sim, frame_support=sup,
                                   refined=bk != k))
    if not final:
        raise RuntimeError("Could not compute the match profiles. "
                           "The query is possibly too short.")
    final.sort(key=lambda c: -c.similarity)
    # sub-frame refinement: parabolic interpolation of the peak of the
    # combined profile gives accuracy better than the HOP step (~5.8 ms)
    for c in final:
        c.exact_frame = refine_subframe(per_file_score[c.file_name], c.frame)
    _detect_repetition(final[0], per_file_score, index, qviews)
    return final[0], final, per_file_score


def _find_match_disk(index: SearchIndex,
                     qfeats: Dict[Tuple[int, int], np.ndarray],
                     top_n: int, duration: float, query_x: np.ndarray,
                     coarse_result: Optional[Tuple[List, List]] = None
                     ) -> Tuple[Candidate, List[Candidate], Dict[str, np.ndarray]]:
    """
    Search over the on-disk index: a coarse shortlist of candidates, then a
    precise check of each by re-reading a window from the original file. The
    result is converted to the same shape as the in-memory search so that the
    report, the verdict and repeat detection behave identically.

    `coarse_result` is that shortlist, already computed: a sweep that answers
    several queries in one walk over the index hands it in rather than have it
    walked again. It is the same list the walk would have produced here.
    """
    duration = duration or max(
        (m.shape[0] for m in qfeats.values()), default=1) * HOP / SAMPLE_RATE
    if coarse_result is None:
        coarse_result = coarse_search(index.coarse, query_x, qfeats, duration)
    fine, _rough = coarse_result
    if not fine:
        raise RuntimeError("The coarse search found no candidates — the query "
                           "is possibly too short.")
    final: List[Candidate] = []
    for c in fine:
        final.append(Candidate(
            file_name=c.file_name,
            frame=int(round(c.refined_time * SAMPLE_RATE / HOP)),
            fused_z=c.coarse_z, similarity=c.similarity,
            frame_support=c.frame_support, best_frame=int(round(
                c.refined_time * SAMPLE_RATE / HOP)),
            exact_frame=c.refined_time * SAMPLE_RATE / HOP))
    final.sort(key=lambda c: -c.similarity)
    best = final[0]
    if best.similarity > 0:
        _detect_repetition_disk(best, final, index, qfeats, duration)
    return best, final, {}


def _detect_repetition_disk(best: Candidate, final: List[Candidate],
                            index: SearchIndex,
                            qfeats: Dict[Tuple[int, int], np.ndarray],
                            duration: float) -> None:
    """
    Repeat detection over the coarse index. The coarse pass shortlists only a
    few candidates, so that alone is not enough: once the period is known we
    measure the EARLIER occurrences of the same material directly (each one is
    a single precise check) and take the earliest of the equal-quality ones,
    just as a person would.
    """
    entry = index.by_name(best.file_name)
    same = sorted((c for c in final if c.file_name == best.file_name),
                  key=lambda c: -c.similarity)
    top = best.similarity
    if top <= 0:
        return
    occ = [(c.exact_frame * HOP / SAMPLE_RATE, c.similarity) for c in same
           if c.similarity >= top - 0.05]
    if len(occ) < MIN_REPETITION_COUNT:
        # even two or three close moments are enough to suggest a period;
        # further we check the earlier occurrences
        if len(occ) < 2:
            return
    times = sorted(t for t, _ in occ)
    gaps = np.diff(np.asarray(times))
    periods = [float(g) for g in np.unique(np.round(gaps, 2)) if g > 0.05]
    period = None
    for p in periods:
        phase = np.mod(np.asarray(times), p)
        base = np.median(phase)
        aligned = np.abs(((phase - base + p / 2) % p) - p / 2)
        if int((aligned <= LOOP_TOL).sum()) >= MIN_REPETITION_COUNT:
            period = p
            break
    if period is None and periods:
        period = min(periods)          # period not confirmed — take the smallest
    if period is None:
        return

    # measure the earlier occurrences on the found lattice: walk back from the
    # winner by one period at a time and check each such moment
    anchor = best.exact_frame * HOP / SAMPLE_RATE
    qviews_full = build_views(qfeats)
    known = [t for t, _ in occ]
    checked = 0
    t = anchor - period
    while t > 0.2 and checked < 24:
        if any(abs(x - t) <= LOOP_TOL for x in known):
            t -= period
            continue
        tt, sim, _sup = fine_verify(entry, t, duration, qviews_full)
        checked += 1
        if sim >= top - 0.05:
            occ.append((tt, sim))
            known.append(tt)
        t -= period
    occ.sort()
    if len(occ) < MIN_REPETITION_COUNT:
        return
    # keep only the truly equal-quality ones and take the earliest of them
    best_occ = [(t, r) for t, r in occ if r >= max(r for _, r in occ) - 0.05]
    best.period = period
    best.occurrences = best_occ
    first_t = best_occ[0][0]
    best.exact_frame = first_t * SAMPLE_RATE / HOP
    best.best_frame = int(round(first_t * SAMPLE_RATE / HOP))
    for c in final:
        if c.file_name == best.file_name and c is not best:
            c.similarity = min(c.similarity, best.similarity - 1e-6)


def _detect_repetition(best: Candidate,
                       per_file_score: Dict[str, np.ndarray],
                       index: SearchIndex,
                       qviews: Dict[Tuple[int, int, bool], np.ndarray]) -> None:
    """
    If the same fragment repeats in a track (a drum break, a loop), the
    profile has many equal-quality peaks with one common period. We find that
    period and all occurrences, and report the earliest of them — the same as
    a person searching for "where did they take it from" would do.
    """
    fused = per_file_score.get(best.file_name)
    if fused is None:
        return
    finite = np.isfinite(fused)
    if not finite.any():
        return
    zmax = float(fused[finite].max())
    if zmax <= 0:
        return
    thr = LOOP_Z_FRACTION * zmax
    idx = np.flatnonzero(finite & (fused >= thr))
    if len(idx) < MIN_REPETITION_COUNT:
        return

    sep = max(1, int(round(LOOP_SEPARATION * SAMPLE_RATE / HOP)))
    times: List[float] = []
    last = -10 ** 9
    for k in idx:
        if k - last > sep:
            times.append(refine_subframe(fused, int(k)) * HOP / SAMPLE_RATE)
            last = int(k)
    if len(times) < MIN_REPETITION_COUNT:
        return

    # look for the period: the smallest gap between neighbouring moments
    t = np.sort(np.asarray(times))
    periods = np.unique(np.round(np.diff(t), 3))
    periods = periods[periods > 0.05]
    if len(periods) == 0:
        return
    best_period, best_hits = None, 0
    for p in periods:
        phase = np.mod(t, p)
        # place all moments on a single lattice with step p
        base = np.median(phase)
        aligned = np.abs(((phase - base + p / 2) % p) - p / 2)
        hits = int((aligned <= LOOP_TOL).sum())
        if hits > best_hits or (hits == best_hits and best_period and p < best_period):
            best_hits, best_period = hits, float(p)
    if best_period is None or best_hits < MIN_REPETITION_COUNT:
        return

    base = float(np.median(np.mod(t, best_period)))
    # keep only the moments that fall on the lattice
    aligned_t = []
    for ti in t:
        off = ((ti - base) % best_period)
        if off > best_period / 2:
            off -= best_period
        if abs(off) <= LOOP_TOL:
            aligned_t.append(float(ti))
    if len(aligned_t) < MIN_REPETITION_COUNT:
        return

    # check the similarity and keep only the truly equal-quality moments
    src = index.by_name(best.file_name)
    rviews = src.views if src.views is not None else build_views(src.feats)
    stats = {key: (R.mean(axis=0), R.std(axis=0) + EPS)
             for key, R in rviews.items()}
    occ = []
    for ti in aligned_t:
        k = int(round(ti * SAMPLE_RATE / HOP))
        num = den = 0.0
        for n_fft, n_mels, delta, weight in BANK:
            key = (n_fft, n_mels, delta)
            if k >= rviews[key].shape[0]:
                continue
            rb, gs = stats[key]
            s = similarity_at(qviews[key], rviews[key], k, rb, gs)
            if s > -1.0:
                num += weight * s
                den += weight
        if den > 0:
            occ.append((ti, num / den))
    if len(occ) < MIN_REPETITION_COUNT:
        return
    occ.sort(key=lambda x: -x[1])
    top_r = occ[0][1]
    occ = [(ti, r) for ti, r in occ if r >= top_r - 0.05]
    occ.sort()
    best.occurrences = occ
    best.period = best_period
    best.exact_frame = occ[0][0] * SAMPLE_RATE / HOP


def refine_subframe(fused: np.ndarray, k: int) -> float:
    """
    Sub-frame offset refinement by parabolic interpolation of the peak of the
    combined profile. Returns the offset in frames (float).
    """
    if k <= 0 or k >= len(fused) - 1:
        return float(k)
    y0, y1, y2 = float(fused[k - 1]), float(fused[k]), float(fused[k + 1])
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return float(k)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 1.0:
        return float(k)
    return float(k) + delta


def verdict(sim: float, margin: float, support: float) -> str:
    """
    Overall assessment. The thresholds come from measurements on synthetic
    cases (--selftest): real matches show a similarity of 0.45-0.9 with a
    margin of 0.15-0.4, while false candidates sit around 0.2-0.35 and their
    margin is almost always below 0.05.

    The absolute similarity drops together with the quality of the sample's
    processing (strong EQ, reverb, noise), so a large margin over every other
    candidate counts as sufficient grounds for a confident verdict even when
    the similarity is moderate and the frame support is low.
    """
    if sim >= CONF_SIM and margin >= CONF_MARGIN and support >= CONF_SUPPORT:
        return "FOUND (confident)"
    if margin >= DECISIVE_MARGIN and sim >= DECISIVE_SIM:
        return "FOUND (confident)"
    if sim >= 0.42 and margin >= 0.05:
        return "FOUND (probable)"
    if sim >= 0.32:
        return "WEAK MATCH — worth checking manually"
    return "NOT FOUND — the source is most likely not in the search folder"


def is_confident(sim: float, margin: float, support: float) -> bool:
    """"Sufficient confidence" — the same threshold as "FOUND (confident)"."""
    return verdict(sim, margin, support) == "FOUND (confident)"


def export_decision(sim: float, margin: float, support: float,
                    min_sim: float = EXPORT_MIN_SIM,
                    min_margin: float = EXPORT_MIN_MARGIN
                    ) -> Tuple[bool, str]:
    """
    Decision on exporting the fragment. The threshold is lower than the one
    for the word "confident": cutting a fragment is almost always useful
    (it is seconds of audio), whereas deciding "the source is found" from a
    single match is not. The main criterion here is therefore the margin over
    the other candidates.

    Returns (should_export, explanation).
    """
    if is_confident(sim, margin, support):
        return True, "confident match"
    if sim >= min_sim and margin >= min_margin:
        return True, (f"margin {margin:+.3f} and similarity {sim:.3f} "
                      f"(export threshold: r>={min_sim:.2f}, margin>={min_margin:.2f})")
    why = []
    if sim < min_sim:
        why.append(f"similarity {sim:.3f} < {min_sim:.2f}")
    if margin < min_margin:
        why.append(f"margin {margin:+.3f} < {min_margin:.2f}")
    return False, "; ".join(why) or "not enough margin over the other candidates"


def _equivalent_sources(best: Candidate, final: List[Candidate],
                        tol: float = EQUIV_TOL,
                        min_sim: float = EQUIV_MIN_SIM,
                        gap: float = EQUIV_GAP) -> List[Candidate]:
    """
    Candidates that are as good as the best one and therefore cannot be told
    from it: the same recording in a different mix / edit, i.e. several copies
    in the collection. Only the best moment of each file is considered.

    Returns [best] alone when there is no group, so the caller can simply test
    len(...) > 1. Never fires on a weak best (below min_sim two unrelated
    places scoring alike is just noise) and never when the top candidates form
    an unbroken run — that means real ambiguity, not duplicates.
    A non-positive tol switches the rule off.
    """
    if tol <= 0.0 or best.similarity < min_sim:
        return [best]
    per_file: Dict[str, Candidate] = {best.file_name: best}
    for c in final:
        cur = per_file.get(c.file_name)
        if cur is None or c.similarity > cur.similarity:
            per_file[c.file_name] = c
    ranked = sorted(per_file.values(), key=lambda c: -c.similarity)
    if len(ranked) < 2:
        return [best]

    group = [best]
    for c in ranked[1:]:
        if (c.similarity < best.similarity - tol
                or c.similarity < min_sim
                or len(group) >= MAX_EQUIVALENTS):
            break
        group.append(c)
    if len(group) == 1:
        return [best]
    # the group has to stand apart from everything it did not absorb,
    # otherwise these are simply indistinguishable candidates
    rest = ranked[len(group):]
    if rest and rest[0].similarity > best.similarity - gap:
        return [best]
    return group


def margin_score(similarity: float, support: float) -> float:
    """
    What the margin is measured on: the similarity, scaled down when the
    candidate explains too little of the query.

    The similarity alone is a weak separator once a sample has been processed:
    it is an average over the frames, so a candidate that matches a handful of
    loud frames can tie with the right one. The frame support says exactly that
    — how much of the query is explained at all — and the two numbers together
    separate much better than either. Measured on a 23-query set: for a query
    whose leading frames were spoiled by a vinyl crackle the winner led the
    runner-up by only +0.045 in r (10% support against 43%); with the support
    folded in the same comparison reads +0.163.

    The scaling starts exactly at CONF_SUPPORT, the point below which the
    program already refuses to call a match confident, so a candidate that is
    not taken seriously on its own does not get to compete at face value either.
    Nothing above the threshold is touched, which is why no verdict that was
    confident before can become weaker: measured over 42 queries of five test
    collections, 5 verdicts improved and none got worse, and every group of
    equivalent files stayed intact.
    """
    if support >= CONF_SUPPORT:
        return similarity
    return similarity * (support / CONF_SUPPORT)


def _margins(best: Candidate, final: List[Candidate],
             equiv_tol: float = EQUIV_TOL,
             equiv_min_sim: float = EQUIV_MIN_SIM,
             equiv_gap: float = EQUIV_GAP
             ) -> Tuple[float, float, List[Candidate]]:
    """
    Similarity margins: to the best moment in the same file and in the other
    files. Both sides are compared on margin_score() — see there for why the
    frame support belongs in this comparison. If a repeat was found (a
    loop/break playing around), the equal-quality occurrences of the very same
    fragment are NOT counted as competitors — otherwise the margin would come
    out as zero and the program would complain about an ambiguity that does not
    exist. The same holds for equivalent copies of one recording in different
    files: they are returned as the third value instead of eating the margin.
    """
    occ_t = [t for t, _ in (best.occurrences or [])]

    def near_occurrence(frame: float) -> bool:
        if not occ_t:
            return False
        t = frame * HOP / SAMPLE_RATE
        return any(abs(t - o) < LOCAL_EXCLUSION for o in occ_t)

    equivalents = _equivalent_sources(best, final, equiv_tol, equiv_min_sim,
                                      equiv_gap)
    if len(equivalents) > 1:
        # take the best moment of every equivalent file out of the competitors
        best_per_file = {c.file_name: c for c in equivalents}
        equiv_files = set(best_per_file)
        equivalents = sorted(best_per_file.values(), key=lambda c: -c.similarity)
    else:
        equiv_files = {best.file_name}

    excl = int(round(LOCAL_EXCLUSION * SAMPLE_RATE / HOP))
    others = [c for c in final
              if c.file_name != best.file_name and c.file_name not in equiv_files]
    same = [c for c in final
            if c.file_name == best.file_name
            and abs(c.best_frame - best.best_frame) > excl
            and not near_occurrence(c.exact_frame)]
    m_file = margin_score(best.similarity, best.frame_support) - max(
        (margin_score(c.similarity, c.frame_support) for c in others),
        default=-1.0)
    m_local = margin_score(best.similarity, best.frame_support) - max(
        (margin_score(c.similarity, c.frame_support) for c in same),
        default=-1.0)
    return m_local, m_file, equivalents

