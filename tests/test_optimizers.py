"""CMA-ES, MCMC, cost functions and the fitting layer."""

import numpy as np
import pytest

import pycardio as pc
from pycardio import costs
from pycardio.cmaes import cmaes
from pycardio.mcmc import adaptive_metropolis, ess, split_rhat


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def test_cmaes_sphere():
    res = cmaes(lambda x: float(np.sum((x - 0.3) ** 2)), np.zeros(5), 0.5, maxfev=4000, seed=0)
    assert res.fun < 1e-12
    np.testing.assert_allclose(res.x, 0.3, atol=1e-5)


def test_cmaes_rosenbrock():
    res = cmaes(rosenbrock, np.full(4, -1.0), 0.5, maxfev=20000, seed=1, tolfun=1e-14)
    np.testing.assert_allclose(res.x, 1.0, atol=1e-4)


def test_cmaes_ill_conditioned_ellipsoid():
    scale = 10 ** np.linspace(0, 6, 6)
    res = cmaes(
        lambda x: float(np.sum(scale * x**2)), np.ones(6), 0.5, maxfev=20000, seed=2, tolfun=1e-16
    )
    assert res.fun < 1e-8


def test_cmaes_respects_bounds_and_finds_boundary_optimum():
    seen = []

    def f(x):
        seen.append(x.copy())
        return float(np.sum((x - 1.5) ** 2))  # optimum outside the box

    res = cmaes(f, np.full(3, 0.5), 0.3, bounds=(np.zeros(3), np.ones(3)), maxfev=3000, seed=3)
    seen = np.array(seen)
    assert seen.min() >= 0 and seen.max() <= 1
    np.testing.assert_allclose(res.x, 1.0, atol=1e-3)


def test_cmaes_is_reproducible_and_handles_inf():
    def f(x):
        return np.inf if x[0] < 0 else float(np.sum(x**2))

    a = cmaes(f, np.ones(2), 0.5, maxfev=600, seed=5)
    b = cmaes(f, np.ones(2), 0.5, maxfev=600, seed=5)
    assert a.fun == b.fun and np.array_equal(a.x, b.x)
    assert np.isfinite(a.fun)


def test_cmaes_evaluates_starting_point_when_budget_is_below_population():
    res = cmaes(lambda x: float(np.sum((x - 1) ** 2)), np.array([0.0]), maxfev=3, seed=0)
    assert res.fun == pytest.approx(1.0)
    assert res.nfev == 1


def test_adaptive_metropolis_recovers_gaussian():
    mean = np.array([0.5, 0.4])
    cov = np.array([[0.004, 0.003], [0.003, 0.0045]])
    prec = np.linalg.inv(cov)

    def logp(x):
        d = x - mean
        return -0.5 * d @ prec @ d

    chains = []
    for seed in range(4):
        s, _, acc = adaptive_metropolis(
            logp, mean + 0.05, 6000, 2000, rng=np.random.default_rng(seed)
        )
        assert 0.15 < acc < 0.6
        chains.append(s)
    chains = np.stack(chains)
    flat = chains.reshape(-1, 2)
    np.testing.assert_allclose(flat.mean(0), mean, atol=0.006)
    np.testing.assert_allclose(np.cov(flat.T), cov, rtol=0.2, atol=2e-4)
    assert np.all(split_rhat(chains) < 1.03)
    assert np.all(ess(chains) > 800)


def test_adaptive_metropolis_rejects_out_of_bounds_start_before_logp():
    def logp(_):
        pytest.fail("logp must not be evaluated for an out-of-bounds start")

    with pytest.raises(ValueError, match="starting point is outside bounds"):
        adaptive_metropolis(logp, np.array([1.1]), 10, 5)


def test_adaptive_metropolis_rejects_negative_warmup():
    with pytest.raises(ValueError, match="n_warmup must be non-negative"):
        adaptive_metropolis(lambda _: 0.0, np.array([0.5]), 10, -1)


def test_rhat_detects_disagreeing_chains():
    rng = np.random.default_rng(0)
    chains = rng.standard_normal((4, 500, 1))
    chains[0] += 5
    assert split_rhat(chains)[0] > 1.5


def test_ess_of_independent_and_correlated_draws():
    rng = np.random.default_rng(1)
    iid = rng.standard_normal((2, 4000, 1))
    assert 6000 < ess(iid)[0] < 10000
    ar = np.zeros((2, 4000, 1))
    for i in range(1, 4000):
        ar[:, i] = 0.9 * ar[:, i - 1] + rng.standard_normal((2, 1))
    # theoretical ESS factor (1 - rho) / (1 + rho) = 1/19
    np.testing.assert_allclose(ess(ar)[0], 8000 / 19, rtol=0.35)


def test_ess_retains_positive_pairs_for_oscillating_autocorrelation():
    rng = np.random.default_rng(42)
    ar = np.zeros((2, 4000, 1))
    for i in range(1, ar.shape[1]):
        ar[:, i] = -0.8 * ar[:, i - 1] + rng.standard_normal((2, 1))
    assert ess(ar)[0] > ar.shape[0] * ar.shape[1] * 1.5


@pytest.fixture(scope="module")
def target():
    return pc.simulate(dict(hr=0.0009, rsa1=0.0006), 90, transient=10, seed=0).features()


def test_costs_zero_for_identical_and_grow_with_distance(target):
    for feat in pc.FEATURES:
        assert pc.total_cost(target, target, feat) == pytest.approx(0.0, abs=1e-9)
    near = pc.simulate(dict(hr=0.00091, rsa1=0.0006), 90, transient=10).features()
    far = pc.simulate(dict(hr=0.0012, rsa1=0.0001), 90, transient=10).features()
    for feat in ("rri", "rsa", "cvc"):
        assert 0 < pc.total_cost(target, near, feat) < pc.total_cost(target, far, feat)


def test_cost_is_chi_square_scaled():
    """Mean of the RRI cost between independent samples of one population is ~1."""
    rng = np.random.default_rng(0)

    def fake(n):
        f = pc.simulate(duration=20).features()
        f.rri = 1.0 + 0.05 * rng.standard_normal(n)
        return f

    vals = [costs.cost_rri(fake(300), fake(100)) for _ in range(300)]
    assert 0.8 < np.mean(vals) < 1.25


def test_cvc_cost_is_the_two_sample_ks_statistic(target):
    from scipy.stats import ks_2samp

    other = pc.simulate(dict(hr=0.001), 90, transient=10).features()
    D = ks_2samp(target.cvc, other.cvc).statistic
    n, k = target.cvc.size, other.cvc.size
    assert costs.cost_cvc(target, other) == pytest.approx(n * k / (n + k) * D**2)


def test_degenerate_model_output_gives_infinite_cost(target):
    silent = pc.simulate(dict(hr=1e-6, rsa1=0, rsa2=0), 30).features()  # no heartbeats
    assert pc.total_cost(target, silent, "rsa") == np.inf


def test_problem_coordinates(target):
    p = pc.FitProblem(
        target,
        free={"hr": (0.0005, 0.0015), "taup": (500, 3000)},
        start={"hr": 0.001},
        fixed={"d2": 0.6},
    )
    assert p.names == ("hr", "taup")
    np.testing.assert_allclose(p.u0, [0.5, 0.5])
    np.testing.assert_allclose(p.from_unit(p.to_unit([0.0007, 800])), [0.0007, 800])
    assert p.params([0, 1]) == {"d2": 0.6, "hr": 0.0005, "taup": 3000.0}
    with pytest.raises(KeyError):
        pc.FitProblem(target, free={"nope": None})


def test_objective_is_deterministic(target):
    p = pc.FitProblem(target, free={"hr": None, "rs1": (0, 0.01)}, duration=30)
    u = np.array([0.4, 0.5])
    assert p(u) == p(u)


@pytest.mark.parametrize("method", ["cmaes", "differential_evolution", "nelder-mead"])
def test_fit_recovers_known_parameters(target, method):
    problem = pc.FitProblem(
        target,
        free={"hr": (0.0006, 0.0012), "rsa1": (0.0002, 0.001)},
        feature="rsa",
        start={"hr": 0.00075, "rsa1": 0.0004},
        duration=90,
        transient=10,
    )
    res = pc.fit(problem, method=method, maxfev=400, seed=0, verbose=False)
    assert res.cost < 1.0
    assert res.params["hr"] == pytest.approx(0.0009, rel=0.02)
    assert res.params["rsa1"] == pytest.approx(0.0006, rel=0.1)


def test_mcmc_on_the_model(target):
    problem = pc.FitProblem(
        target,
        free={"hr": (0.0008, 0.001)},
        feature="rri",
        fixed={"rsa1": 0.0006},
        duration=90,
        transient=10,
    )
    start = problem.to_unit(np.array([0.0009]))
    res = pc.sample(problem, n_samples=150, n_warmup=100, n_chains=2, start=start, seed=0)
    assert res.chains.shape == (2, 150, 1)
    s = res.summary()["hr"]
    assert s["q2.5"] < 0.0009 < s["q97.5"]
    assert s["sd"] < 2e-5


def test_options_for_another_method_are_ignored(target):
    problem = pc.FitProblem(target, free={"hr": (0.0008, 0.001)}, feature="rri", duration=30)
    with pytest.warns(UserWarning, match="mutation"):
        res = pc.fit(problem, method="cmaes", maxfev=20, verbose=False, mutation=(0.5, 1.0))
    assert np.isfinite(res.cost)


def test_cost_does_not_reward_a_noisier_model():
    """The spread in the denominator comes from the recording. If it came from
    the model, a model with the wrong mean could score well just by being noisy."""
    rng = np.random.default_rng(0)

    def with_rri(mean, sd, n):
        f = pc.simulate(duration=20).features()
        f.rri = mean + sd * rng.standard_normal(n)
        return f

    data = with_rri(1.0, 0.05, 2000)
    quiet_wrong = with_rri(1.05, 0.05, 300)
    noisy_wrong = with_rri(1.05, 0.50, 300)
    # with the model's own spread in the denominator the noisy one would score ~3
    assert costs.cost_rri(data, quiet_wrong) > 50
    assert costs.cost_rri(data, noisy_wrong) > 30

    ph = rng.random(2000)
    d = pc.simulate(duration=20).features()
    d.rsa = pc.features.fit_rsa(
        ph, 1 + 0.02 * np.cos(2 * np.pi * ph) + 0.03 * rng.standard_normal(2000)
    )
    phm = rng.random(300)

    def model_rsa(amp, sd):
        m = pc.simulate(duration=20).features()
        m.rsa = pc.features.fit_rsa(
            phm, 1 + amp * np.cos(2 * np.pi * phm) + sd * rng.standard_normal(300)
        )
        return m

    good = costs.cost_rsa(d, model_rsa(0.02, 0.03))
    big_and_noisy = costs.cost_rsa(d, model_rsa(0.16, 0.30))
    assert good < 15 < 200 < big_and_noisy
