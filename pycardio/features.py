"""Cardiorespiratory features, computed the same way for data and for the model.

Every function takes *times in seconds* (respiration sample times and
heartbeat times) instead of sample indices, so the data and the simulated
signals go through identical code.

Definitions (unchanged in meaning from the 2018 analysis):

* **Breath cycle / inspiration onset**: troughs of the respiration
  (lung-volume) signal. **Expiration onset**: peaks.
* **RSA**: each RR interval is assigned the respiratory phase of its midpoint
  within the expiration-onset to expiration-onset cycle (peak to peak), and
  ``RRI(phase) = c + A cos(2 pi phase) + B sin(2 pi phase)`` is fitted.
* **CVC (cardioventilatory coupling)**: the phase in (0, 1] of every heartbeat
  within its inspiration-onset to inspiration-onset cycle (trough to trough).
* **Breath timing**: inspiration = trough to the next highest point; expiration
  = that point to the next trough.
* **DFA**: detrended fluctuation analysis of the RR-interval series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

__all__ = [
    "RSAFit",
    "Features",
    "smooth",
    "respiratory_extrema",
    "rsa_points",
    "fit_rsa",
    "cvc_phases",
    "breath_timing",
    "beat_phase",
    "dfa",
    "extract_features",
]

# Plausibility limits (seconds). Values outside are treated as artefacts.
MIN_RRI, MAX_RRI = 0.25, 4.0
MIN_CYCLE, MAX_CYCLE = 1.0, 10.0


# --------------------------------------------------------------------------
# Signal conditioning
# --------------------------------------------------------------------------


def smooth(x: np.ndarray, width: int, window: str = "nuttall") -> np.ndarray:
    """Zero-phase smoothing with a normalised window of ``width`` samples.

    Uses reflected padding, so the edges are not pulled towards zero. (The
    2018 code convolved with zero padding, which biased the first and last
    half-window of every recording.)
    """
    x = np.asarray(x, dtype=float)
    if width <= 1:
        return x.copy()
    w = signal.get_window(window, width, fftbins=False)
    w /= w.sum()
    pad = min(width, x.size - 1)
    xp = np.pad(x, pad, mode="reflect")
    return signal.fftconvolve(xp, w, mode="same")[pad:-pad]


def respiratory_extrema(
    time: np.ndarray,
    resp: np.ndarray,
    min_separation: float = 1.0,
    rel_prominence: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of respiration peaks (expiration onsets) and troughs (inspiration onsets).

    Extrema closer than ``min_separation`` seconds are merged (the largest one
    is kept), and extrema whose prominence is below ``rel_prominence`` times
    the 5-95 % range of the signal are ignored, which rejects small wiggles
    without depending on the absolute signal scale.
    """
    time = np.asarray(time, dtype=float)
    resp = np.asarray(resp, dtype=float)
    if resp.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    fs = 1.0 / np.median(np.diff(time))
    distance = max(1, int(round(min_separation * fs)))
    lo, hi = np.percentile(resp, [5, 95])
    prominence = rel_prominence * (hi - lo) if hi > lo else None
    peaks, _ = signal.find_peaks(resp, distance=distance, prominence=prominence)
    troughs, _ = signal.find_peaks(-resp, distance=distance, prominence=prominence)
    return peaks, troughs


# --------------------------------------------------------------------------
# RSA
# --------------------------------------------------------------------------


def _locate(events: np.ndarray, x: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """For each x, the index j of the cycle [events[j], events[j+1]) containing it.

    Returns (j, valid) where valid marks x that fall inside some cycle.
    """
    j = np.searchsorted(events, x, side=side) - 1
    valid = (j >= 0) & (j < events.size - 1)
    return np.where(valid, j, 0), valid


def rsa_points(
    beat_times: np.ndarray,
    peak_times: np.ndarray,
    min_rri: float = MIN_RRI,
    max_rri: float = MAX_RRI,
    max_cycle: float = MAX_CYCLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Respiratory phase and length of each RR interval.

    Returns ``(phase, rri, midpoint_times)``. Phase 0 is expiration onset.
    """
    beat_times = np.asarray(beat_times, dtype=float)
    peak_times = np.asarray(peak_times, dtype=float)
    if beat_times.size < 2 or peak_times.size < 2:
        empty = np.array([])
        return empty, empty, empty
    rri = np.diff(beat_times)
    mid = 0.5 * (beat_times[1:] + beat_times[:-1])
    j, valid = _locate(peak_times, mid, side="right")
    cycle = peak_times[np.minimum(j + 1, peak_times.size - 1)] - peak_times[j]
    valid &= (rri >= min_rri) & (rri <= max_rri) & (cycle <= max_cycle) & (cycle > 0)
    phase = (mid[valid] - peak_times[j[valid]]) / cycle[valid]
    return phase, rri[valid], mid[valid]


@dataclass
class RSAFit:
    """Least-squares fit ``rri = c + A cos(2 pi phase) + B sin(2 pi phase)``.

    ``cov`` is the covariance of (c, A, B), estimated from the residual
    variance. Successive RR intervals are correlated, so it is somewhat
    optimistic; it is used here only to weight the comparison with the model.
    """

    coef: np.ndarray  # (c, A, B), seconds
    cov: np.ndarray  # 3x3
    n: int
    resid_sd: float

    @property
    def amplitude(self) -> float:
        """Peak RSA amplitude Q = sqrt(A^2 + B^2) (seconds)."""
        return float(np.hypot(self.coef[1], self.coef[2]))

    @property
    def amplitude_se(self) -> float:
        """Standard error of Q by the delta method."""
        a, b = self.coef[1], self.coef[2]
        q = np.hypot(a, b)
        if q == 0:
            return float(np.sqrt(0.5 * (self.cov[1, 1] + self.cov[2, 2])))
        g = np.array([a / q, b / q])
        return float(np.sqrt(g @ self.cov[1:, 1:] @ g))

    @property
    def peak_phase(self) -> float:
        """Phase (0-1) at which the fitted RRI is longest."""
        return float(np.mod(np.arctan2(self.coef[2], self.coef[1]) / (2 * np.pi), 1.0))

    def __call__(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=float)
        c, a, b = self.coef
        return c + a * np.cos(2 * np.pi * phase) + b * np.sin(2 * np.pi * phase)


def fit_rsa(phase: np.ndarray, rri: np.ndarray) -> RSAFit | None:
    """Fit the first-order Fourier RSA model by ordinary least squares.

    Replaces the 2018 ``curve_fit`` call, which had no intercept, fixed the
    residual scale by hand and combined the parameter variances with a formula
    that was not the variance of anything.
    """
    phase = np.asarray(phase, dtype=float)
    rri = np.asarray(rri, dtype=float)
    n = phase.size
    if n < 4:
        return None
    X = np.column_stack([np.ones(n), np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)])
    coef, *_ = np.linalg.lstsq(X, rri, rcond=None)
    resid = rri - X @ coef
    dof = max(n - 3, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return RSAFit(coef=coef, cov=cov, n=n, resid_sd=float(np.sqrt(s2)))


# --------------------------------------------------------------------------
# CVC, breath timing, synchrogram
# --------------------------------------------------------------------------


def cvc_phases(
    beat_times: np.ndarray, trough_times: np.ndarray, max_cycle: float = MAX_CYCLE
) -> np.ndarray:
    """Phase in (0, 1] of each heartbeat within its breath (inspiration onset to
    inspiration onset), sorted ascending. Breaths longer than ``max_cycle``
    are skipped."""
    beat_times = np.asarray(beat_times, dtype=float)
    trough_times = np.asarray(trough_times, dtype=float)
    if trough_times.size < 2 or beat_times.size == 0:
        return np.array([])
    j, valid = _locate(trough_times, beat_times, side="left")  # t0 < beat <= t1
    t0 = trough_times[j]
    t1 = trough_times[np.minimum(j + 1, trough_times.size - 1)]
    valid &= (t1 - t0 <= max_cycle) & (t1 > t0)
    return np.sort((beat_times[valid] - t0[valid]) / (t1[valid] - t0[valid]))


def breath_timing(
    time: np.ndarray,
    resp: np.ndarray,
    troughs: np.ndarray,
    min_cycle: float = MIN_CYCLE,
    max_cycle: float = MAX_CYCLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Inspiration and expiration durations (s) of each plausible breath."""
    insp, exp_ = [], []
    for a, b in zip(troughs[:-1], troughs[1:]):
        dur = time[b] - time[a]
        if not (min_cycle <= dur <= max_cycle) or b - a < 2:
            continue
        k = a + int(np.argmax(resp[a : b + 1]))
        insp.append(time[k] - time[a])
        exp_.append(time[b] - time[k])
    return np.asarray(insp, dtype=float), np.asarray(exp_, dtype=float)


def beat_phase(
    beat_times: np.ndarray,
    trough_times: np.ndarray,
    min_cycle: float = MIN_CYCLE,
    max_cycle: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Beat times and their phase in the breath (inspiration onset = 0), for
    synchrograms. Returned in time order."""
    beat_times = np.asarray(beat_times, dtype=float)
    trough_times = np.asarray(trough_times, dtype=float)
    if trough_times.size < 2:
        return np.array([]), np.array([])
    j, valid = _locate(trough_times, beat_times, side="right")
    t0 = trough_times[j]
    t1 = trough_times[np.minimum(j + 1, trough_times.size - 1)]
    dur = t1 - t0
    valid &= (dur >= min_cycle) & (dur <= max_cycle)
    return beat_times[valid], (beat_times[valid] - t0[valid]) / dur[valid]


# --------------------------------------------------------------------------
# DFA
# --------------------------------------------------------------------------


def dfa(
    x: np.ndarray, scales: Sequence[int] | None = None, order: int = 1
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detrended fluctuation analysis (Peng et al. 1995).

    The series is mean-centred and integrated into a profile, cut into
    non-overlapping windows of ``n`` samples (from both ends, so no data is
    wasted), each window is detrended with a polynomial of degree ``order``,
    and ``F(n)`` is the RMS of all residuals. Returns ``(scales, F, alpha)``
    where ``alpha`` is the slope of log F against log n.

    The 2018 C version did not integrate the series, skipped every other
    window, normalised the running sum inside the loop, and wrote one element
    past the end of its output buffer.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if scales is None:
        scales = np.arange(4, 17)  # alpha_1, the standard short-term exponent
    scales = np.asarray([int(s) for s in scales if order + 2 <= int(s) <= x.size // 2], dtype=int)
    if scales.size < 2:
        raise ValueError("series too short for the requested DFA scales")
    profile = np.cumsum(x - x.mean())
    F = np.empty(scales.size)
    for i, n in enumerate(scales):
        m = profile.size // n
        segs = np.concatenate(
            [profile[: m * n].reshape(m, n), profile[profile.size - m * n :].reshape(m, n)]
        )
        t = np.arange(n, dtype=float)
        V = np.vander(t, order + 1)
        # least-squares detrending of every window at once
        coef, *_ = np.linalg.lstsq(V, segs.T, rcond=None)
        resid = segs.T - V @ coef
        F[i] = np.sqrt(np.mean(resid**2))
    alpha = float(np.polyfit(np.log(scales), np.log(F), 1)[0])
    return scales, F, alpha


# --------------------------------------------------------------------------
# All features
# --------------------------------------------------------------------------


@dataclass
class Features:
    """Summary of a recording or a simulation."""

    duration: float  # seconds of signal analysed
    beat_times: np.ndarray
    rri: np.ndarray  # all plausible RR intervals (s)
    rsa_phase: np.ndarray
    rsa_rri: np.ndarray
    rsa: RSAFit | None
    cvc: np.ndarray  # sorted beat phases in the breath
    insp: np.ndarray  # inspiration durations (s)
    exp: np.ndarray  # expiration durations (s)
    n_breaths: int
    dfa_alpha1: float = float("nan")
    meta: dict = field(default_factory=dict)

    # ---- derived quantities ----
    @property
    def rri_mean(self) -> float:
        return float(np.mean(self.rri)) if self.rri.size else float("nan")

    @property
    def heart_rate(self) -> float:
        """Mean heart rate (beats per second)."""
        return 1.0 / self.rri_mean if self.rri.size else float("nan")

    @property
    def resp_rate(self) -> float:
        """Breaths per second."""
        return self.n_breaths / self.duration if self.duration > 0 else float("nan")

    @property
    def rotation_number(self) -> float:
        """Heartbeats per breath. (The 2018 code divided the mean RR interval
        by the breathing rate, which has units of s^2 and is not this.)"""
        return self.heart_rate / self.resp_rate

    def summary(self) -> dict[str, float]:
        out = {
            "duration_s": self.duration,
            "n_beats": int(self.beat_times.size),
            "n_breaths": self.n_breaths,
            "rri_mean_s": self.rri_mean,
            "rri_sd_s": float(np.std(self.rri, ddof=1)) if self.rri.size > 1 else float("nan"),
            "heart_rate_bpm": 60.0 * self.heart_rate,
            "resp_rate_per_min": 60.0 * self.resp_rate,
            "rotation_number": self.rotation_number,
            "insp_mean_s": float(np.mean(self.insp)) if self.insp.size else float("nan"),
            "exp_mean_s": float(np.mean(self.exp)) if self.exp.size else float("nan"),
            "dfa_alpha1": self.dfa_alpha1,
        }
        if self.rsa is not None:
            out.update(
                rsa_amplitude_s=self.rsa.amplitude,
                rsa_amplitude_se_s=self.rsa.amplitude_se,
                rsa_peak_phase=self.rsa.peak_phase,
            )
        return out

    # ---- persistence (plain arrays, no pickle) ----
    def to_npz(self, path) -> None:
        arrays = {k: np.asarray(v) for k, v in asdict(self).items() if k not in ("rsa", "meta")}
        if self.rsa is not None:
            arrays["rsa_coef"] = self.rsa.coef
            arrays["rsa_cov"] = self.rsa.cov
            arrays["rsa_n"] = np.asarray(self.rsa.n)
            arrays["rsa_resid_sd"] = np.asarray(self.rsa.resid_sd)
        arrays["meta_keys"] = np.asarray(list(self.meta.keys()), dtype=str)
        arrays["meta_values"] = np.asarray([str(v) for v in self.meta.values()], dtype=str)
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path) -> "Features":
        with np.load(path, allow_pickle=False) as z:
            rsa = None
            if "rsa_coef" in z:
                rsa = RSAFit(
                    coef=z["rsa_coef"],
                    cov=z["rsa_cov"],
                    n=int(z["rsa_n"]),
                    resid_sd=float(z["rsa_resid_sd"]),
                )
            meta = dict(zip(z["meta_keys"].tolist(), z["meta_values"].tolist()))
            return cls(
                duration=float(z["duration"]),
                beat_times=z["beat_times"],
                rri=z["rri"],
                rsa_phase=z["rsa_phase"],
                rsa_rri=z["rsa_rri"],
                rsa=rsa,
                cvc=z["cvc"],
                insp=z["insp"],
                exp=z["exp"],
                n_breaths=int(z["n_breaths"]),
                dfa_alpha1=float(z["dfa_alpha1"]),
                meta=meta,
            )


def _segment_points(time, resp, beat_times, min_separation, rel_prominence):
    peaks, troughs = respiratory_extrema(time, resp, min_separation, rel_prominence)
    in_seg = (beat_times >= time[0]) & (beat_times <= time[-1])
    beats = beat_times[in_seg]
    rri = np.diff(beats)
    rri = rri[(rri >= MIN_RRI) & (rri <= MAX_RRI)]
    phase, rsa_rri, _ = rsa_points(beats, time[peaks])
    cvc = cvc_phases(beats, time[troughs])
    insp, exp_ = breath_timing(time, resp, troughs)
    return dict(
        beats=beats,
        rri=rri,
        phase=phase,
        rsa_rri=rsa_rri,
        cvc=cvc,
        insp=insp,
        exp=exp_,
        n_breaths=max(int(troughs.size) - 1, 0),
        duration=float(time[-1] - time[0]),
    )


def extract_features(
    time: np.ndarray,
    resp: np.ndarray,
    beat_times: np.ndarray,
    segments: Iterable[tuple[float, float]] | None = None,
    min_separation: float = 1.0,
    rel_prominence: float = 0.1,
    compute_dfa: bool = True,
) -> Features:
    """Compute all features from a respiration signal and heartbeat times.

    ``segments`` optionally restricts the analysis to (start, end) time
    windows in seconds (e.g. artefact-free stretches); points from all
    segments are pooled. RR intervals are never formed across segments.
    """
    time = np.asarray(time, dtype=float)
    resp = np.asarray(resp, dtype=float)
    beat_times = np.sort(np.asarray(beat_times, dtype=float))
    if segments is None:
        segments = [(time[0], time[-1])]
    parts = []
    for start, end in segments:
        sel = (time >= start) & (time <= end)
        if sel.sum() < 3:
            continue
        parts.append(
            _segment_points(time[sel], resp[sel], beat_times, min_separation, rel_prominence)
        )
    if not parts:
        raise ValueError("no usable signal in the requested segments")

    def cat(key):
        return np.concatenate([p[key] for p in parts]) if parts else np.array([])

    phase, rsa_rri, rri = cat("phase"), cat("rsa_rri"), cat("rri")
    alpha = float("nan")
    if compute_dfa:
        # DFA needs a contiguous series; use the longest segment.
        longest = max(parts, key=lambda p: p["beats"].size)
        series = np.diff(longest["beats"])
        series = series[(series >= MIN_RRI) & (series <= MAX_RRI)]
        if series.size >= 64:
            alpha = dfa(series)[2]
    return Features(
        duration=float(sum(p["duration"] for p in parts)),
        beat_times=cat("beats"),
        rri=rri,
        rsa_phase=phase,
        rsa_rri=rsa_rri,
        rsa=fit_rsa(phase, rsa_rri),
        cvc=np.sort(cat("cvc")),
        insp=cat("insp"),
        exp=cat("exp"),
        n_breaths=int(sum(p["n_breaths"] for p in parts)),
        dfa_alpha1=alpha,
    )
