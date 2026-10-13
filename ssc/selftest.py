"""The self-test: synthetic queries with a known source."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import fftconvolve
from ssc.audio import _decode_slice, trim_silence
from ssc.config import BANK, EPS, HOP, SAMPLE_RATE
from ssc.features import logmel_bank, similarity_at
from ssc.format import fmt_time
from ssc.index import SearchIndex, build_index
from ssc.search import _margins, build_views, find_match, verdict

# --------------------------------------------------------------------------- #
# Self-test: synthetic queries with a known source
# --------------------------------------------------------------------------- #


def _biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Direct form II (transposed) biquad — a plain loop over samples."""
    y = np.zeros(len(x))
    z1 = z2 = 0.0
    a0 = a[0]
    a1, a2 = a[1] / a0, a[2] / a0
    b0, b1, b2 = b[0] / a0, b[1] / a0, b[2] / a0
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def _peaking(x, sr, f0, gain_db, q=1.0):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    return _biquad(x, np.array([1 + alpha * A, -2 * cw, 1 - alpha * A]),
                   np.array([1 + alpha / A, -2 * cw, 1 - alpha / A]))


def _lowpass(x, sr, f0, q=0.707):
    w0 = 2.0 * math.pi * f0 / sr
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    return _biquad(x, np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]),
                   np.array([1 + alpha, -2 * cw, 1 - alpha]))


def _highpass(x, sr, f0, q=0.707):
    w0 = 2.0 * math.pi * f0 / sr
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    return _biquad(x, np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]),
                   np.array([1 + alpha, -2 * cw, 1 - alpha]))


def _reverb(x, sr, rng, rt60=0.6, mix=0.4):
    n = max(16, int(rt60 * sr))
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    ir[: max(1, int(0.005 * sr))] = 0.0
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2)) + EPS
    wet = fftconvolve(x, ir)[: len(x)]
    wet /= np.max(np.abs(wet)) + EPS
    out = (1 - mix) * (x / (np.max(np.abs(x)) + EPS)) + mix * wet
    return out / (np.max(np.abs(out)) + EPS) * 0.9


def degrade(x: np.ndarray, sr: int, rng: np.random.Generator,
            severity: str = "normal") -> np.ndarray:
    """
    A random set of artifacts, like a real sample inside someone else's track.

    severity:
      "mild"   — EQ and level only (a realistic minimum);
      "normal" — EQ + reverb + noise 6-22 dB + saturation (the default);
      "harsh"  — adds noise down to 0 dB and heavy reverb.
    """
    y = x.astype(np.float64).copy()
    y /= np.max(np.abs(y)) + EPS
    if severity == "mild":
        lo_gain, lo_cut, lo_snr, lo_rev, lo_drive, lo_vinyl = -12, 0, 999, 0, 0, 0
    elif severity == "harsh":
        lo_gain, lo_cut, lo_snr, lo_rev, lo_drive, lo_vinyl = -16, 1, -8, 1, 1, 1
    else:
        lo_gain, lo_cut, lo_snr, lo_rev, lo_drive, lo_vinyl = -14, 1, 6, 1, 1, 1

    style = int(rng.integers(0, 3))
    if style == 0:                                   # wide EQ from peaking filters
        for _ in range(int(rng.integers(2, 5))):
            f0 = float(np.exp(rng.uniform(math.log(120), math.log(7000))))
            y = _peaking(y, sr, f0, float(rng.uniform(lo_gain, 14)),
                         q=float(rng.uniform(0.7, 3.0)))
    elif style == 1 and lo_cut:                       # band-limited spectrum
        y = _highpass(y, sr, float(rng.uniform(70, 260)))
        y = _lowpass(y, sr, float(np.exp(rng.uniform(math.log(1200),
                                                      math.log(9000)))))
    else:                                            # level only
        y = y * 10.0 ** float(rng.uniform(-14, 6)) / 20.0

    if rng.random() < lo_rev:
        hi_mix = 0.6 if severity == "harsh" else 0.45
        y = _reverb(y, sr, rng, rt60=float(rng.uniform(0.2, 1.1)),
                    mix=float(rng.uniform(0.15, hi_mix)))
    if rng.random() < (1.0 if severity != "mild" else 0.0):
        snr = float(rng.uniform(lo_snr if lo_snr < 999 else 30,
                                22 if severity != "harsh" else 8))
        noise = _lowpass(rng.standard_normal(len(y)), sr, 4000.0)
        noise *= np.sqrt(float(np.mean(y ** 2)) /
                         (float(np.mean(noise ** 2)) + EPS))
        y = y + noise * 10.0 ** (-snr / 20.0)
    if rng.random() < 0.4 * lo_drive:                # soft saturation
        drive = float(rng.uniform(1.5, 6.0))
        y = np.tanh(y * drive) / math.tanh(drive)
    if rng.random() < 0.3 * lo_vinyl:                # "vinyl": clicks and hiss
        clicks = np.zeros(len(y))
        idx = rng.integers(0, len(y), size=int(rng.uniform(1, 6)))
        clicks[idx] = rng.standard_normal(idx.size) * 0.3
        y = y + clicks + _lowpass(rng.standard_normal(len(y)) * 0.01, sr, 6000.0)

    return (y / (np.max(np.abs(y)) + EPS) * 0.95).astype(np.float32)


def _synthetic_noise(rng: np.random.Generator, dur: float = 0.6) -> np.ndarray:
    """A deliberately "unrelated" signal: a sequence of retuning tones with
    noisy modulation. Used for the strict negative test."""
    n = int(dur * SAMPLE_RATE)
    y = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.uniform(0.03, 0.12) * SAMPLE_RATE)
        end = min(i + seg, n)
        f0 = float(np.exp(rng.uniform(math.log(60), math.log(6000))))
        f1 = f0 * float(rng.uniform(0.6, 2.5))
        k = np.arange(end - i) / SAMPLE_RATE
        y[i:end] += np.sin(2 * np.pi * (f0 * k + (f1 - f0) * k * k / (2 * max(k[-1], 1e-6))))
        i = end
    y += 0.3 * rng.standard_normal(n)
    return (y / (np.max(np.abs(y)) + EPS) * 0.9).astype(np.float32)


def similarity_at_time(index_: SearchIndex, file_name: str,
                       qviews: Dict[Tuple[int, int, bool], np.ndarray],
                       t_sec: float) -> float:
    """Similarity of the query with the source at a given moment (for checks)."""
    src = index_.by_name(file_name)
    rviews = src.views if src.views is not None else build_views(src.feats)
    k = int(round(t_sec * SAMPLE_RATE / HOP))
    stats = {key: (R.mean(axis=0), R.std(axis=0) + EPS)
             for key, R in rviews.items()}
    num = den = 0.0
    for n_fft, n_mels, delta, weight in BANK:
        key = (n_fft, n_mels, delta)
        if k >= rviews[key].shape[0]:
            continue
        r_bar, gstd = stats[key]
        s = similarity_at(qviews[key], rviews[key], k, r_bar, gstd)
        if s > -1.0:
            num += weight * s
            den += weight
    return num / den if den > 0 else -1.0


def selftest(search_folder: str, n_cases: int = 12, seed: int = 0,
             severity: str = "normal", verbose: bool = False,
             names: Optional[Sequence[str]] = None) -> int:
    rng = np.random.default_rng(seed)
    print("=== Self-test ===")
    print(f"Sources: {search_folder}   distortion mode: {severity}")
    print("Positive cases: the query is cut from a random place in a track and "
          "distorted (EQ, reverb, noise, saturation, \"vinyl\"). It is expected "
          "that this very file is found; the moment — either accurate to 50 ms, "
          "or one of the occurrences, if the fragment repeats in the track "
          "(break/loop).")
    print("Negative cases: (1) a query that is definitely not in the collection "
          "(a synthetic signal) — the program must not report a confident "
          "match; (2) the same query without its file — this check is "
          "informational: the collection may hold other versions of the same "
          "track with the same sample, and that is not an error.\n",
          flush=True)

    index = build_index(search_folder, verbose=verbose, names=names)
    ok = total = 0
    fails: List[str] = []
    pos_r, neg_r, pos_m, neg_m = [], [], [], []

    def run_case(index_, degraded):
        """Common run of one case: the candidate, the margins and the query views."""
        trimmed, shift = trim_silence(degraded, SAMPLE_RATE)
        qf = logmel_bank(trimmed)
        best, final, _ = find_match(index_, qf)
        m_local, m_file, _equiv = _margins(best, final)
        return best, m_local, m_file, shift, build_views(qf)

    for case in range(n_cases):
        src = index.sources[case % len(index.sources)]
        clip_len = float(rng.uniform(0.25, 0.9))
        max_start = src.duration - clip_len - 0.5
        if max_start <= 0.5:
            continue
        start = float(rng.uniform(0.5, max_start))
        raw = _decode_slice(os.path.join(search_folder, src.name), start, clip_len)
        if len(raw) < SAMPLE_RATE // 4:
            continue
        y = degrade(raw, SAMPLE_RATE, rng, severity=severity)

        # --- positive case: the source is in the index ---
        best, m_local, m_file, shift, qviews = run_case(index, y)
        truth = start + shift
        got = best.exact_frame * HOP / SAMPLE_RATE + shift
        err = abs(got - truth)
        # the fragment may repeat (break/loop) — then any of the equal-quality
        # occurrences fits, not only the original one
        equivalent = bool(best.occurrences) and any(
            abs(t + shift - truth) <= 0.05 for t, _ in best.occurrences)
        looping = bool(best.occurrences and len(best.occurrences) > 1)
        same_file = best.file_name == src.name
        # if the track holds another equal-quality occurrence of the same
        # material, picking it instead of the original is not an error: we
        # check that the true place was among the variants with a comparable
        # similarity
        sim_truth = (similarity_at_time(index, src.name, qviews, truth)
                     if same_file else -1.0)
        alt_ok = sim_truth >= best.similarity - 0.05
        good = same_file and (err <= 0.05 or equivalent or alt_ok)
        margin_eff = m_file if looping else min(m_local, m_file)
        total += 1
        ok += int(good)
        pos_r.append(best.similarity)
        pos_m.append(margin_eff)
        note = "same file" if same_file else best.file_name[:20]
        if equivalent:
            note += f", equivalent occurrence (repeat ~{best.period:.2f} s)"
        elif alt_ok and err > 0.05:
            note += f", equal-quality other place (s@truth={sim_truth:.3f})"
        print(f"  [+] [{'OK ' if good else 'FAIL'}] {src.name[:30]:<30s} "
              f"truth={fmt_time(truth):>9s} found={fmt_time(got):>9s} "
              f"D={err*1000:7.1f} ms  r={best.similarity:.3f} "
              f"frames={best.frame_support*100:3.0f}%  "
              f"margin={margin_eff:+.3f}  {note}", flush=True)
        if not good:
            fails.append(f"POSITIVE: {src.name} @ {fmt_time(truth)} -> "
                         f"{best.file_name} @ {fmt_time(got)} (r={best.similarity:.3f}, "
                         f"similarity at the true point {sim_truth:.3f})")

        # --- informational case: the same query, but the file is not in the index ---
        # not counted as an error: the collection may hold other versions of the
        # same track with the same sample (e.g. vocal and dub versions)
        if case % 3 == 0 and len(index.sources) > 1:
            reduced = SearchIndex(
                folder=index.folder,
                sources=[s for s in index.sources if s.name != src.name])
            best2, ml2, mf2, _sh, _qv = run_case(reduced, y)
            neg_r.append(best2.similarity)
            neg_m.append(min(ml2, mf2))
            print(f"  [i] (file absent) r={best2.similarity:.3f} "
                  f"margin={min(ml2, mf2):+.3f} -> matched {best2.file_name[:34]}",
                  flush=True)

    # --- strict negative case: the signal is not in the collection at all ---
    for _ in range(3):
        y = _synthetic_noise(rng, dur=float(rng.uniform(0.3, 1.5)))
        best, m_local, m_file, _sh, _qv = run_case(index, y)
        margin_eff = min(m_local, m_file)
        confident = best.similarity >= 0.55 and margin_eff >= 0.12
        total += 1
        ok += int(not confident)
        neg_r.append(best.similarity)
        neg_m.append(margin_eff)
        print(f"  [-] [{'OK ' if not confident else 'FAIL'}] "
              f"synthetic signal (not in the collection): "
              f"r={best.similarity:.3f} margin={margin_eff:+.3f} "
              f"-> {verdict(best.similarity, margin_eff, best.frame_support)}",
              flush=True)
        if confident:
            fails.append(f"NEGATIVE: on a foreign signal got r={best.similarity:.3f} "
                         f"margin={margin_eff:+.3f} — that is a false match")

    print(f"\nTotal: {ok}/{total}")
    if pos_r:
        print(f"  similarity of real matches: min={min(pos_r):.3f} "
              f"median={sorted(pos_r)[len(pos_r)//2]:.3f} max={max(pos_r):.3f}")
        print(f"  margin of real matches:     min={min(pos_m):+.3f} "
              f"median={sorted(pos_m)[len(pos_m)//2]:+.3f}")
    if neg_r:
        print(f"  similarity on foreign signals: max={max(neg_r):.3f} "
              f"median={sorted(neg_r)[len(neg_r)//2]:.3f}")
    for line in fails:
        print(f"  {line}")
    return 0 if total and ok == total else 1

