"""Tests of the compiled model against an independent reference."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import pycardio as pc
from pycardio import model as m


def reference_rhs(y, P):
    """Independent NumPy transcription of the 2018 equations (resp_odes.c).
    y[12] is the early-I adaptation."""
    eiAd = y[12]

    def f(v):
        return float(np.clip((v - P["vlow"]) / (P["vhigh"] - P["vlow"]), 0.0, 1.0))

    def ninf(v):
        return 1 / (1 + np.exp((v - P["v12n"]) / P["kn"]))

    def mpinf(v):
        return 1 / (1 + np.exp((v - P["v12mp"]) / P["kmp"]))

    def hpinf(v):
        return 1 / (1 + np.exp((v - P["v12hp"]) / P["khp"]))

    def tauinf(v):
        return P["tauhp"] / np.cosh((v - P["v12hp"]) / P["khp"])

    def beat(phi):
        return ((1 + np.sin(phi * 2 * np.pi)) / 2) ** 20

    gK, gL, gE, gI, gAD = P["gKdr"], P["gL"], P["gE"], P["gI"], P["gAD"]
    EK, EL, Ee, Ei = P["EK"], P["EL"], P["Ee"], P["Ei"]
    dy = np.zeros(13)
    dy[12] = (f(y[1]) - eiAd) / P["tad_ei"]
    dy[0] = (
        -gK * ninf(y[0]) ** 4 * (y[0] - EK)
        - gL * (y[0] - EL)
        - P["gNaP"] * mpinf(y[0]) * y[5] * (y[0] - P["ENa"])
        - gE * P["d1"] * (y[0] - Ee)
        - gI * (P["b31"] * f(y[2]) + P["b41"] * f(y[3])) * (y[0] - Ei)
    ) / P["C"]
    dy[1] = (
        -gK * ninf(y[1]) ** 4 * (y[1] - EK)
        - gL * (y[1] - EL)
        - gAD * eiAd * (y[1] - EK)
        - gE * (P["d2"] + P["a12"] * f(y[0])) * (y[1] - Ee)
        - gI * (P["b32"] * f(y[2]) + P["b42"] * f(y[3]) + P["hb2"] * y[8] + P["st2"]) * (y[1] - Ei)
    ) / P["C"]
    dy[2] = (
        -gK * ninf(y[2]) ** 4 * (y[2] - EK)
        - gL * (y[2] - EL)
        - gAD * y[6] * (y[2] - EK)
        - gE * P["d3"] * (y[2] - Ee)
        - gI * (P["b23"] * f(y[1]) + P["b43"] * f(y[3])) * (y[2] - Ei)
    ) / P["C"]
    dy[3] = (
        -gK * ninf(y[3]) ** 4 * (y[3] - EK)
        - gL * (y[3] - EL)
        - gAD * y[7] * (y[3] - EK)
        - gE * (P["d4"] + P["hb"] * y[8] + P["st4"] + P["baro"] * y[10]) * (y[3] - Ee)
        - gI * (P["b24"] * f(y[1]) + P["b34"] * f(y[2])) * (y[3] - Ei)
    ) / P["C"]
    dy[4] = (
        -gK * ninf(y[4]) ** 4 * (y[4] - EK)
        - gL * (y[4] - EL)
        - gE * (P["d5"] + P["a15"] * f(y[0])) * (y[4] - Ee)
        - gI * (P["b25"] * f(y[1]) + P["b35"] * f(y[2]) + P["b45"] * f(y[3])) * (y[4] - Ei)
    ) / P["C"]
    dy[5] = (hpinf(y[0]) - y[5]) / tauinf(y[0])
    dy[6] = (f(y[2]) - y[6]) / P["tad"]
    dy[7] = (f(y[3]) - y[7]) / P["tad"]
    dy[8] = (f(y[4]) - y[8]) / P["taul"]
    dy[9] = P["hr"] + P["rsa1"] * f(y[3]) + P["rsa2"] * f(y[4]) - P["db"] * y[11]
    dy[10] = (beat(y[9]) + P["sym"] * f(y[4]) - P["sb"] * y[11] - (y[10] - P["po"])) / P["taup"]
    dy[11] = (y[10] - y[11]) / P["taupp"]
    return dy


# every coupling switched on, so each term of the equations is exercised
RICH = dict(baro=0.1, sym=0.2, db=0.0007, sb=0.1, hb2=0.05, st2=0.02, st4=0.03, b34=0.1, b31=0.05)


def test_parameter_table():
    assert len(set(pc.PARAM_NAMES)) == len(pc.PARAM_NAMES) == 57
    # values of the 2018 parameters.py / parameters.h
    for name, value in dict(
        hr=0.00084, rsa1=0.0005, rsa2=0.00035, d2=0.564, gL=2.8, b42=0.42, tauhp=4000.0, vlow=-50.0
    ).items():
        assert pc.DEFAULT_PARAMS[name] == value
    assert all(pc.PARAM_DOCS[n] for n in pc.PARAM_NAMES)


def test_unknown_parameter_raises():
    with pytest.raises(KeyError):
        pc.make_params({"not_a_parameter": 1.0})
    with pytest.raises(KeyError):
        pc.simulate({"gl": 3.0}, 1.0)


def test_rhs_matches_reference():
    rng = np.random.default_rng(0)
    P = dict(pc.DEFAULT_PARAMS) | RICH
    for _ in range(200):
        y = np.concatenate(
            [
                rng.uniform(-80, 0, 5),
                rng.uniform(0, 1, 4),
                rng.uniform(0, 5, 1),
                rng.uniform(0, 2, 2),
                rng.uniform(0, 1, 1),
            ]
        )
        np.testing.assert_allclose(m.rhs(y, P), reference_rhs(y, P), rtol=1e-12, atol=1e-14)


def test_integrator_matches_scipy():
    """Without noise the model is a plain ODE; compare with SciPy's DOP853
    integrating the independent reference right-hand side."""
    P = dict(pc.DEFAULT_PARAMS) | RICH | dict(rs1=0.0)
    y0 = m.default_initial_state()
    T = 8000.0  # ms, about two breaths
    ref = solve_ivp(
        lambda t, y: reference_rhs(y, P),
        (0, T),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        t_eval=np.arange(0, T + 1, 50.0),
    )
    sim = pc.simulate(P, T / 1000, rtol=1e-9, atol=1e-11)
    got = np.column_stack(
        [
            sim["lung_volume"],
            sim["heart_phase"],
            sim["bp"],
            sim["bp_smooth"],
            sim["preii_nap"],
            sim["earlyi_iad"],
        ]
    )[::50]
    want = ref.y[[8, 9, 10, 11, 5, 12]].T
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-6)


def test_tolerance_convergence():
    P = {"rs1": 0.0}
    ref = pc.simulate(P, 20, rtol=1e-11, atol=1e-13)
    # a coarse output grid, so the tolerance (not the grid) limits the step
    ref = pc.simulate(P, 20, dt=50, rtol=1e-11, atol=1e-13)
    err = [
        np.abs(
            pc.simulate(P, 20, dt=50, rtol=r, atol=r * 1e-2)["heart_phase"] - ref["heart_phase"]
        ).max()
        for r in (1e-3, 1e-5, 1e-7)
    ]
    assert err[0] > err[1] > err[2]
    assert err[2] < 1e-5


def test_pure_pacemaker_period():
    """With no respiratory or baroreflex coupling the RR interval is 1/hr."""
    sim = pc.simulate(dict(hr=0.001, rsa1=0, rsa2=0, db=0), 30)
    rr = np.diff(sim.beat_times)
    np.testing.assert_allclose(rr, 1.0, atol=1e-9)
    assert sim.beat_times.size == 30


def test_output_grid_is_honoured():
    P = {"rs1": 0.0}
    a = pc.simulate(P, 5, dt=1.0, rtol=1e-9, atol=1e-11)
    b = pc.simulate(P, 5, dt=0.5, rtol=1e-9, atol=1e-11)
    assert b.t.size == 2 * (a.t.size - 1) + 1
    np.testing.assert_allclose(b.t[::2], a.t)
    np.testing.assert_allclose(b["heart_phase"][::2], a["heart_phase"], atol=1e-7)
    c = pc.simulate(P, 5, dt=10.0, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(c["heart_phase"], a["heart_phase"][::10], atol=1e-7)


def test_seed_reproducibility():
    P = {"rs1": 0.01}
    a, b, c = (pc.simulate(P, 20, seed=s) for s in (3, 3, 4))
    assert np.array_equal(a.states, b.states)
    assert not np.array_equal(a.states, c.states)


def test_noise_statistics_independent_of_step():
    """Isolate the SDE d(eiAd) = -eiAd/tau dt + sigma dW (early-I silenced):
    the stationary variance sigma^2 tau / 2 must not depend on the output step."""
    tau, rs1 = 50.0, 0.02
    P = dict(rs1=rs1, tad_ei=tau, d2=0.0, a12=0.0, hr=0.001)
    expected = (rs1**2 / 12) * tau / 2
    for dt in (0.5, 1.0, 2.0):
        sim = pc.simulate(P, 400, dt=dt, transient=2, seed=1)
        assert sim["earlyi"].max() == 0.0  # the drift target stays at zero
        var = sim["earlyi_iad"].var()
        # exact drift + noise increment every dt: var = sigma^2 dt / (1 - exp(-2 dt / tau))
        exact = (rs1**2 / 12) * dt / (1 - np.exp(-2 * dt / tau))
        np.testing.assert_allclose(var, exact, rtol=0.08)
        np.testing.assert_allclose(var, expected, rtol=0.12)


def test_continuation_matches_single_run():
    P = {"rs1": 0.0}
    full = pc.simulate(P, 10)
    first = pc.simulate(P, 5)
    second = pc.simulate(P, 5, y0=first.y_final)
    np.testing.assert_allclose(second.states[-1], full.states[-1], atol=1e-5)


def test_transient_is_dropped():
    P = {"rs1": 0.0}
    full = pc.simulate(P, 12)
    late = pc.simulate(P, 10, transient=2)
    assert late.t[0] == 0.0
    np.testing.assert_allclose(late.states, full.states[2000:], atol=0)


def test_solver_failure_is_reported():
    with pytest.raises(pc.SimulationError):
        pc.simulate({"C": 0.0}, 1.0)


def test_default_model_is_physiological():
    f = pc.simulate(duration=120, transient=10).features()
    assert 40 < 60 * f.heart_rate < 100
    assert 8 < 60 * f.resp_rate < 25
    assert f.rsa is not None and 0.01 < f.rsa.amplitude < 0.3
