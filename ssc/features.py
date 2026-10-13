"""Spectrograms, and the per-band correlation profiles built from them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import fftconvolve
from ssc.config import (BANK, CHUNK_FRAMES, COARSE_BANDS, COARSE_HOP,
                        COARSE_NFFT, EPS, FMAX, FMIN, FRAME_SUPPORT_THRESHOLD,
                        HOP, MIN_STD_FLOOR, MIN_STD_RATIO, REF_TAIL_PAD,
                        SAMPLE_RATE)

# --------------------------------------------------------------------------- #
# Features: log-mel spectrograms at several resolutions
# --------------------------------------------------------------------------- #


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank (n_mels, n_fft//2+1)."""
    freqs = rfftfreq(n_fft, 1.0 / sr)
    pts = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        left, center, right = pts[i], pts[i + 1], pts[i + 2]
        rise = (freqs - left) / max(center - left, 1e-9)
        fall = (right - freqs) / max(right - center, 1e-9)
        fb[i] = np.clip(np.minimum(rise, fall), 0.0, None)
    return fb


_FB_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def filterbank(n_mels: int, n_fft: int) -> np.ndarray:
    key = (n_mels, n_fft)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filterbank(n_mels, n_fft, SAMPLE_RATE)
    return _FB_CACHE[key]


def logmel_bank(x: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Computes log-mel spectrograms for every resolution of the bank.
    Returns {(n_fft, n_mels): matrix (frames, bands)}.
    Computed in blocks so the whole spectrogram is never held in memory.
    """
    x = np.asarray(x, dtype=np.float32)
    # resolutions: everything in the bank plus the coarse one (disk index)
    wanted = {(n_fft, m) for n_fft, m, _, _ in BANK}
    wanted.add((COARSE_NFFT, COARSE_BANDS))
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for n_fft in sorted({f for f, _ in wanted}):
        mels = sorted({m for f, m in wanted if f == n_fft})
        fbs = {m: filterbank(m, n_fft).astype(np.float32) for m in mels}
        n_frames = 1 + max(0, len(x) - n_fft) // HOP
        mats = {m: np.empty((n_frames, m), dtype=np.float32) for m in mels}
        window = np.hanning(n_fft).astype(np.float32)
        base = np.arange(n_fft)[None, :]
        for f0 in range(0, n_frames, CHUNK_FRAMES):
            n = min(CHUNK_FRAMES, n_frames - f0)
            s0 = f0 * HOP
            seg = x[s0: s0 + (n - 1) * HOP + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            idx = base + HOP * np.arange(n)[:, None]
            power = np.abs(rfft(seg[idx] * window, axis=1)) ** 2
            for m in mels:
                mats[m][f0: f0 + n] = np.log(power @ fbs[m].T + 1e-10)
        for m in mels:
            out[(n_fft, m)] = mats[m]
    return out


def frame_center(x: np.ndarray) -> np.ndarray:
    """Subtracts the across-band mean from every frame: removes the overall level."""
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=1, keepdims=True)


def time_delta(x: np.ndarray) -> np.ndarray:
    """
    First difference along time, with the first frame "repeated" so the frame
    count stays the same (needed to align the offset grids of the different
    feature sets of the bank).
    """
    x = np.asarray(x, dtype=np.float64)
    return np.diff(x, axis=0, prepend=x[:1])


def query_view(x: np.ndarray, delta: bool) -> np.ndarray:
    v = time_delta(x) if delta else x
    return frame_center(v)


# --------------------------------------------------------------------------- #
# Time-axis correlation inside each band
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #




@dataclass
class PreparedRef:
    """
    Everything about a source that can be computed ONCE, independently of the
    query: the centred features (for the FFT convolution) and their cumulative
    sums of squares (for the sliding window energy). Recomputing this per
    query was the most expensive part of the search.
    """
    n_frames: int
    pad: int
    bands: int
    R: np.ndarray      # float32 (n_frames+pad, bands), already centred
    C2: np.ndarray     # float64 (n_frames+pad+1, bands), cumulative sums of R²
    global_std: np.ndarray


def prepare_ref(R: np.ndarray, pad: int = REF_TAIL_PAD) -> PreparedRef:
    """
    Prepares a source for repeated searching: centring across bands, zero
    padding (i.e. padding with the file mean) and cumulative sums of squares.

    Zero padding is equivalent to padding with any constant: the covariance
    contains sum_t (Q[t]-q_bar)=0, so the tail does not affect the result.
    """
    n, B = R.shape
    R32 = np.ascontiguousarray(R, dtype=np.float32)
    R32 -= R32.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Rp = np.zeros((n + pad, B), dtype=np.float32)
    Rp[:n] = R32
    sq = np.zeros((n + pad, B), dtype=np.float64)
    np.multiply(R32, R32, out=sq[:n], dtype=np.float64)
    c2 = np.zeros((n + pad + 1, B), dtype=np.float64)
    np.cumsum(sq, axis=0, out=c2[1:])
    return PreparedRef(n_frames=n, pad=pad, bands=B, R=Rp, C2=c2,
                       global_std=R32.std(axis=0, dtype=np.float64) + EPS)


def profile_from_prepared(Q: np.ndarray, prep: PreparedRef) -> np.ndarray:
    """
    profile[k] — cosine similarity of the query with the source window that
    starts at frame k; the profile length is prep.n_frames.

    For every band b:
        cos_b(k) = sum_t (Q[t,b]-q_bar_b)*(R[k+t,b]-r_bar_b)
                   / (T * q_std_b * w_rms_b(k))
    q_bar_b is the query mean over time, r_bar_b the mean of the WHOLE source
    (not of the window), w_rms_b the RMS of the window relative to r_bar_b.
    Dividing by q_std·w_rms removes a fixed EQ (scaling a band by a constant)
    and partly the noise floor added in the log domain. The window mean is
    deliberately NOT subtracted (unlike Pearson correlation): one free
    parameter — the band level inside the window — is left, which makes the
    feature markedly more discriminative on short queries.

    Band weight = min(query std, window RMS): flat (quiet) bands produce
    neither false peaks nor false zeros.
    """
    T, B = Q.shape
    n = prep.n_frames
    if T == 0 or B == 0 or prep.pad < T - 1:
        return np.zeros(max(n, 0))
    q_mean = Q.mean(axis=0, dtype=np.float64)
    q_std = Q.std(axis=0, dtype=np.float64)
    Qc = np.ascontiguousarray(Q - q_mean, dtype=np.float32)

    s2 = prep.C2[T:T + n] - prep.C2[:n]
    w_rms = np.sqrt(np.maximum(s2 / T, 0.0))

    cov = fftconvolve(prep.R, Qc[::-1], mode="valid", axes=0)[:n]
    denom = T * q_std[None, :] * w_rms
    hi = np.maximum(q_std[None, :], w_rms)
    informative = (np.minimum(q_std[None, :], w_rms) / (hi + EPS) > MIN_STD_RATIO) \
        & (w_rms > MIN_STD_FLOOR * prep.global_std[None, :])

    corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float64),
                     where=informative & (denom > 1e-9))
    np.clip(corr, -1.0, 1.0, out=corr)
    corr *= informative

    weight = np.where(informative, np.minimum(q_std[None, :], w_rms), 0.0)
    wsum = weight.sum(axis=1)
    out = np.divide((corr * weight).sum(axis=1), wsum, out=np.zeros(n),
                    where=wsum > EPS)
    return np.clip(out, -1.0, 1.0)


def band_corr_profile(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """One-off version of profile_from_prepared (prepares the source on the fly)."""
    T, B = Q.shape
    if T == 0 or B == 0:
        return np.zeros(max(R.shape[0], 0))
    pad = max(REF_TAIL_PAD, T - 1)
    return profile_from_prepared(Q, prepare_ref(R, pad))


def frame_correlations(Q: np.ndarray, win: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Correlations for every (frame, band) of one window, and the fraction of
    frames whose across-band average correlation exceeds
    FRAME_SUPPORT_THRESHOLD.
    """
    q_mean, q_std = Q.mean(axis=0), Q.std(axis=0)
    w_mean, w_std = win.mean(axis=0), win.std(axis=0)
    ok = (np.minimum(q_std, w_std) / (np.maximum(q_std, w_std) + EPS) > MIN_STD_RATIO) \
        & (w_std > MIN_STD_FLOOR * (w_std.max() + EPS))
    denom = np.where(ok, q_std * w_std, 1.0)
    C = np.clip((Q - q_mean) * (win - w_mean) / denom, -1.0, 1.0) * ok
    w = np.where(ok, np.minimum(q_std, w_std), 0.0)   # per-band weight, shared
    per_frame = (C * w).sum(axis=1) / max(float(w.sum()), EPS)
    return per_frame, float((per_frame > FRAME_SUPPORT_THRESHOLD).mean())


def similarity_at(Q: np.ndarray, R: np.ndarray, k: int,
                  r_bar: Optional[np.ndarray] = None,
                  global_std: Optional[np.ndarray] = None) -> float:
    """
    Cosine similarity for the exact window R[k:k+T] — the same quantity and
    formula as in profile_from_prepared, but for a single offset (the values
    agree; there is a consistency test). r_bar is the across-band mean of R
    over the WHOLE file; if not given it is computed from R.
    """
    T = Q.shape[0]
    win = R[k:k + T]
    if win.shape[0] < T:
        return -1.0
    if r_bar is None:
        r_bar = R.mean(axis=0)
        if global_std is None:
            global_std = R.std(axis=0) + EPS
    q_mean, q_std = Q.mean(axis=0), Q.std(axis=0)
    rc = win - r_bar
    w_rms = np.sqrt((rc * rc).sum(axis=0) / T)   # RMS, not std: the window is not centred
    ok = (np.minimum(q_std, w_rms) / (np.maximum(q_std, w_rms) + EPS) > MIN_STD_RATIO) \
        & (w_rms > MIN_STD_FLOOR * global_std)
    cov = ((Q - q_mean) * rc).sum(axis=0)
    # A band can be perfectly flat (q_std or w_rms exactly 0, e.g. a constant
    # or a digital-silent band of a very short query). Dividing by zero there
    # would raise "invalid value encountered in divide" and, worse, produce a
    # NaN that survives the multiplication by the zero weight. So the division
    # is done only where the band is usable; elsewhere corr stays 0 — which is
    # what the weight 0 was asking for anyway.
    denom = T * q_std * w_rms
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=ok & (denom > 0))
    np.clip(corr, -1.0, 1.0, out=corr)
    corr *= ok
    w = np.where(ok, np.minimum(q_std, w_rms), 0.0)
    tot = w.sum()
    # -1 means "this feature set says nothing here" — the caller drops it from
    # the weighted average. It must NOT be a 0.0: a feature set that cannot be
    # measured at all (a query shorter than its n_fft has a single frame, so no
    # band varies and every weight is gated off) would otherwise count as "no
    # similarity" and drag a good match down. A query of 95 ms against a bank
    # with n_fft=2048 has four such sets out of six, and reporting 0.0 for them
    # halves the score of a match that is really there.
    return float((corr * w).sum() / tot) if tot > EPS else -1.0


def coarse_logmel(x: np.ndarray, sr: int = SAMPLE_RATE
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarse features for the on-disk index: log-mel (n_fft=COARSE_NFFT,
    COARSE_HOP step, COARSE_BANDS bands) plus the energy of each frame, used
    to filter out quiet places. Computed in blocks, so memory scales with the
    block, not with the file.
    """
    x = np.asarray(x, dtype=np.float32)
    n = 1 + max(0, len(x) - COARSE_NFFT) // COARSE_HOP
    if n <= 0:
        return np.zeros((1, COARSE_BANDS), np.float32), np.zeros(1, np.float32)
    fb = filterbank(COARSE_BANDS, COARSE_NFFT).astype(np.float32)
    win = np.hanning(COARSE_NFFT).astype(np.float32)
    out = np.empty((n, COARSE_BANDS), dtype=np.float32)
    power_sum = np.empty(n, dtype=np.float64)
    base = np.arange(COARSE_NFFT)[None, :]
    step = max(1, CHUNK_FRAMES // 4)
    for f0 in range(0, n, step):
        k = min(step, n - f0)
        seg = x[f0 * COARSE_HOP: f0 * COARSE_HOP + (k - 1) * COARSE_HOP + COARSE_NFFT]
        if len(seg) < COARSE_NFFT:
            seg = np.pad(seg, (0, COARSE_NFFT - len(seg)))
        idx = base + COARSE_HOP * np.arange(k)[:, None]
        power = np.abs(rfft(seg[idx] * win, axis=1)) ** 2
        out[f0: f0 + k] = np.log(power @ fb.T + 1e-10)
        power_sum[f0: f0 + k] = power[:, 1:].sum(axis=1)
    return out, power_sum.astype(np.float32)


def build_views(feats: Dict[Tuple[int, int], np.ndarray]
                ) -> Dict[Tuple[int, int, bool], np.ndarray]:
    """Views of a signal for every element of the bank."""
    views: Dict[Tuple[int, int, bool], np.ndarray] = {}
    for n_fft, n_mels, delta, _w in BANK:
        views[(n_fft, n_mels, delta)] = query_view(feats[(n_fft, n_mels)], delta)
    return views

