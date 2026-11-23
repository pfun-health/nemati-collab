"""Feature extraction on synthetic signals with known answers."""

import numpy as np
import pytest

from pycardio import features as F

FS = 250.0


def breathing(duration=600.0, t_insp=1.5, t_exp=2.5):
    """Triangle-wave respiration: rises for t_insp, falls for t_exp. Troughs at k*T."""
    t = np.arange(0, duration, 1 / FS)
    T = t_insp + t_exp
    ph = np.mod(t, T)
    resp = np.where(ph < t_insp, ph / t_insp, 1 - (ph - t_insp) / t_exp)
    return t, resp, T


def test_extrema_and_breath_timing():
    t, resp, T = breathing()
    peaks, troughs = F.respiratory_extrema(t, resp)
    np.testing.assert_allclose(np.diff(t[peaks]), T, atol=2 / FS)
    np.testing.assert_allclose(np.mod(t[troughs], T), 0, atol=2 / FS)
    insp, exp_ = F.breath_timing(t, resp, troughs)
    np.testing.assert_allclose(insp, 1.5, atol=2 / FS)
    np.testing.assert_allclose(exp_, 2.5, atol=2 / FS)


def test_extrema_ignore_small_wiggles():
    t, resp, T = breathing()
    noisy = resp + 0.01 * np.sin(2 * np.pi * 3.1 * t)
    peaks, _ = F.respiratory_extrema(t, noisy)
    assert abs(peaks.size - round(600 / T)) <= 1


def test_rsa_fit_recovers_known_modulation():
    t, resp, T = breathing()
    peak_times = np.arange(1.5, 600, T)  # expiration onsets
    c, A, B = 0.9, 0.05, -0.03

    def rri_at(mid):
        ph = np.mod(mid - 1.5, T) / T
        return c + A * np.cos(2 * np.pi * ph) + B * np.sin(2 * np.pi * ph)

    beats = [2.0]
    while beats[-1] < 595:
        rr = c
        for _ in range(50):  # fixed point: the interval depends on its own midpoint
            rr = rri_at(beats[-1] + rr / 2)
        beats.append(beats[-1] + rr)
    beats = np.array(beats)
    phase, rri, _ = F.rsa_points(beats, peak_times)
    fit = F.fit_rsa(phase, rri)
    np.testing.assert_allclose(fit.coef, [c, A, B], atol=1e-9)
    np.testing.assert_allclose(fit.amplitude, np.hypot(A, B), atol=1e-9)
    assert fit.resid_sd < 1e-9
    # the fitted curve peaks where A cos + B sin is largest
    np.testing.assert_allclose(fit.peak_phase, np.mod(np.arctan2(B, A) / (2 * np.pi), 1))


def test_rsa_standard_errors_are_calibrated():
    rng = np.random.default_rng(0)
    est, se = [], []
    for _ in range(300):
        ph = rng.random(400)
        rri = 1.0 + 0.04 * np.cos(2 * np.pi * ph) + 0.05 * rng.standard_normal(400)
        fit = F.fit_rsa(ph, rri)
        est.append(fit.coef)
        se.append(np.sqrt(np.diag(fit.cov)))
    ratio = np.std(est, axis=0) / np.mean(se, axis=0)
    np.testing.assert_allclose(ratio, 1.0, atol=0.15)


def test_rsa_rejects_artefact_intervals():
    peak_times = np.arange(0, 100, 4.0)
    beats = np.array([1.0, 2.0, 3.0, 9.0, 10.0, 10.1, 11.0])  # a 6 s gap and a 0.1 s interval
    _, rri, _ = F.rsa_points(beats, peak_times)
    np.testing.assert_allclose(rri, [1.0, 1.0, 1.0, 0.9])


def test_cvc_phase_locked_and_uniform():
    troughs = np.arange(0, 400, 4.0)
    locked = troughs[:-1] + 0.3 * 4.0
    np.testing.assert_allclose(F.cvc_phases(locked, troughs), 0.3)
    rng = np.random.default_rng(1)
    uniform = np.sort(rng.uniform(0, 396, 5000))
    ph = F.cvc_phases(uniform, troughs)
    assert ph.size == 5000 and ph.min() > 0 and ph.max() <= 1
    D = np.max(np.abs(np.arange(1, ph.size + 1) / ph.size - ph))
    assert D < 0.03


def test_cvc_boundaries_and_long_cycles():
    troughs = np.array([0.0, 4.0, 8.0, 30.0, 34.0])  # 8 -> 30 is too long
    beats = np.array([-1.0, 0.0, 4.0, 6.0, 20.0, 32.0, 40.0])
    # 0.0 is not after any trough; 4.0 closes the first breath (phase 1)
    np.testing.assert_allclose(F.cvc_phases(beats, troughs), [0.5, 0.5, 1.0])


def test_beat_phase_for_synchrogram():
    troughs = np.arange(0, 40, 4.0)
    t, ph = F.beat_phase(np.array([1.0, 5.0, 10.0]), troughs)
    np.testing.assert_allclose(t, [1.0, 5.0, 10.0])
    np.testing.assert_allclose(ph, [0.25, 0.25, 0.5])


@pytest.mark.parametrize("kind,expected", [("white", 0.5), ("brown", 1.5)])
def test_dfa_scaling_exponents(kind, expected):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2**14)
    if kind == "brown":
        x = np.cumsum(x)
    scales = np.unique(np.logspace(np.log10(16), np.log10(512), 12).astype(int))
    _, Fn, alpha = F.dfa(x, scales)
    assert np.all(np.diff(Fn) > 0)
    assert abs(alpha - expected) < 0.08


def test_dfa_removes_linear_trend():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4096)
    _, f0, _ = F.dfa(x, [16, 32, 64], order=2)
    _, f1, _ = F.dfa(x + 0.01 * np.arange(x.size), [16, 32, 64], order=2)
    np.testing.assert_allclose(f0, f1, rtol=1e-6)


def test_smooth_preserves_constants_and_edges():
    x = np.full(1000, 3.0)
    np.testing.assert_allclose(F.smooth(x, 250), 3.0, atol=1e-12)
    ramp = np.linspace(0, 1, 1000)
    s = F.smooth(ramp, 51)
    np.testing.assert_allclose(s[100:-100], ramp[100:-100], atol=1e-10)


def test_extract_features_and_roundtrip(tmp_path):
    t, resp, T = breathing()
    beats = np.arange(0.4, 599, 0.8)
    f = F.extract_features(t, resp, beats)
    np.testing.assert_allclose(f.rri_mean, 0.8)
    np.testing.assert_allclose(f.heart_rate, 1.25)
    np.testing.assert_allclose(f.resp_rate, 1 / T, rtol=0.02)
    np.testing.assert_allclose(f.rotation_number, T / 0.8, rtol=0.02)  # beats per breath
    f.meta["subject"] = "synthetic"
    f.to_npz(tmp_path / "f.npz")
    g = F.Features.from_npz(tmp_path / "f.npz")
    np.testing.assert_array_equal(g.cvc, f.cvc)
    np.testing.assert_array_equal(g.rsa.coef, f.rsa.coef)
    assert g.meta["subject"] == "synthetic" and g.n_breaths == f.n_breaths


def test_segments_do_not_bridge_gaps():
    t, resp, _ = breathing()
    beats = np.arange(0.4, 599, 0.8)
    f = F.extract_features(t, resp, beats, segments=[(0, 200), (300, 500)])
    assert f.rri.max() < 1.0  # no RR interval spanning 200 -> 300
    np.testing.assert_allclose(f.duration, 400, atol=0.1)
