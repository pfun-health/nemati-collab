"""Posterior sampling of model parameters by adaptive Metropolis MCMC.

Target: ``log p(u) = -cost(u) / 2`` on the unit cube (a uniform prior over the
parameter bounds), where ``cost`` is the chi-square-scaled discrepancy from
:mod:`pycardio.costs`.

Proposal: Gaussian random walk whose covariance is learned during warm-up
from the chain history (Haario, Saksman & Tamminen 2001), scaled by
``2.38^2 / d`` (Roberts & Rosenthal 2001). Adaptation stops when warm-up ends,
so the retained samples come from a fixed, reversible Metropolis kernel.

Several chains run in parallel threads. The split-R-hat (Gelman et al. 2013;
rank-free version) and a basic effective sample size are reported so
convergence can be checked.

This replaces the 2018 ``Sampler``, which (a) accepted a candidate only if its
"likelihood" ``1/cost`` was at least the current one, i.e. it was a hill
climber, not an MCMC sampler; (b) ignored the off-diagonal proposal
covariance; and (c) built each model run out of four independent short
simulations concatenated end to end.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fitting import FitProblem

__all__ = ["MCMCResult", "adaptive_metropolis", "sample", "split_rhat", "ess"]


@dataclass
class MCMCResult:
    names: tuple[str, ...]
    chains: np.ndarray  # (n_chains, n_samples, d), physical units
    logp: np.ndarray  # (n_chains, n_samples)
    acceptance: np.ndarray  # per chain, post warm-up
    rhat: np.ndarray  # per parameter
    ess: np.ndarray  # per parameter (all chains)
    meta: dict = field(default_factory=dict)

    @property
    def samples(self) -> np.ndarray:
        return self.chains.reshape(-1, self.chains.shape[-1])

    def summary(self) -> dict[str, dict[str, float]]:
        s = self.samples
        q = np.percentile(s, [2.5, 50, 97.5], axis=0)
        return {
            n: {
                "mean": float(s[:, i].mean()),
                "sd": float(s[:, i].std(ddof=1)),
                "q2.5": float(q[0, i]),
                "median": float(q[1, i]),
                "q97.5": float(q[2, i]),
                "rhat": float(self.rhat[i]),
                "ess": float(self.ess[i]),
            }
            for i, n in enumerate(self.names)
        }

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            names=np.asarray(self.names),
            chains=self.chains,
            logp=self.logp,
            acceptance=self.acceptance,
            rhat=self.rhat,
            ess=self.ess,
        )


def adaptive_metropolis(
    logp: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_samples: int,
    n_warmup: int,
    init_scale: float | np.ndarray = 0.05,
    rng: np.random.Generator | None = None,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    adapt_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One adaptive-Metropolis chain.

    Returns ``(samples, logp_values, acceptance_rate)`` for the post-warm-up
    part. Proposals outside ``bounds`` are rejected (zero prior density).
    """
    rng = rng or np.random.default_rng()
    x = np.array(x0, dtype=float)
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative")
    if bounds is not None and not np.all((x >= bounds[0]) & (x <= bounds[1])):
        raise ValueError("starting point is outside bounds")
    d = x.size
    lp = logp(x)
    if not np.isfinite(lp):
        raise ValueError("log density is not finite at the starting point")
    sd = 2.38**2 / d
    eps = 1e-10
    cov = np.diag(np.broadcast_to(np.asarray(init_scale, dtype=float) ** 2, (d,)))
    chol = np.linalg.cholesky(cov)
    adapt_start = adapt_start if adapt_start is not None else max(2 * d, n_warmup // 4)

    # running mean / covariance of the chain (Welford)
    mean = x.copy()
    m2 = np.zeros((d, d))
    out = np.empty((n_samples, d))
    out_lp = np.empty(n_samples)
    accepted = 0
    total = n_warmup + n_samples
    for i in range(total):
        prop = x + chol @ rng.standard_normal(d)
        inside = bounds is None or np.all((prop >= bounds[0]) & (prop <= bounds[1]))
        if inside:
            lp_prop = logp(prop)
            if np.log(rng.random()) < lp_prop - lp:
                x, lp = prop, lp_prop
                if i >= n_warmup:
                    accepted += 1
        # adaptation (warm-up only)
        if i < n_warmup:
            k = i + 2
            delta = x - mean
            mean += delta / k
            m2 += np.outer(delta, x - mean)
            if i >= adapt_start:
                emp = m2 / (k - 1)
                prop_cov = sd * (emp + eps * np.eye(d))
                try:
                    chol = np.linalg.cholesky(prop_cov)
                except np.linalg.LinAlgError:
                    pass
        else:
            out[i - n_warmup] = x
            out_lp[i - n_warmup] = lp
    return out, out_lp, accepted / max(n_samples, 1)


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-R-hat for (n_chains, n, d) arrays (Gelman et al., BDA3)."""
    c, n, d = chains.shape
    half = n // 2
    if half < 2:
        return np.full(d, np.nan)
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    m = split.shape[0]
    means = split.mean(axis=1)
    W = split.var(axis=1, ddof=1).mean(axis=0)
    B = half * means.var(axis=0, ddof=1)
    var_hat = (half - 1) / half * W + B / half
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / W) if m > 1 else np.full(d, np.nan)


def ess(chains: np.ndarray) -> np.ndarray:
    """Effective sample size from the averaged autocorrelation (Geyer's
    initial-positive-sequence truncation)."""
    c, n, d = chains.shape
    out = np.empty(d)
    for j in range(d):
        x = chains[:, :, j] - chains[:, :, j].mean(axis=1, keepdims=True)
        f = np.fft.rfft(x, n=2 * n, axis=1)
        acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :n].mean(axis=0)
        if acf[0] <= 0:
            out[j] = np.nan
            continue
        rho = acf / acf[0]
        s = 0.0
        for t in range(0, n - 1, 2):
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            s += pair
        out[j] = c * n / max(-1.0 + 2.0 * s, 1e-12)
    return out


def sample(
    problem: FitProblem,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    n_chains: int = 4,
    start: np.ndarray | None = None,
    init_scale: float = 0.05,
    seed: int | None = 0,
    jitter: float = 0.02,
) -> MCMCResult:
    """Sample the posterior of the problem's free parameters.

    ``start`` (unit-cube coordinates) defaults to the problem's start point;
    a good choice is the optimum from :func:`pycardio.fitting.fit`. Each chain
    starts from a small random perturbation of it so R-hat is meaningful.
    """
    root = np.random.default_rng(seed)
    child_seeds = root.integers(0, 2**63, size=n_chains)
    u0 = np.asarray(problem.u0 if start is None else start, dtype=float)

    def logp(u):
        return -0.5 * problem(u)

    def run(k):
        rng = np.random.default_rng(child_seeds[k])
        for _ in range(100):  # find a finite starting point near u0
            x0 = np.clip(u0 + jitter * rng.standard_normal(u0.size), 0, 1)
            if np.isfinite(logp(x0)):
                break
        else:
            raise ValueError("could not find a starting point with finite cost")
        return adaptive_metropolis(logp, x0, n_samples, n_warmup, init_scale, rng)

    with ThreadPoolExecutor(max_workers=n_chains) as pool:
        results = list(pool.map(run, range(n_chains)))
    chains_u = np.stack([r[0] for r in results])
    chains = problem.lower + chains_u * (problem.upper - problem.lower)
    return MCMCResult(
        names=problem.names,
        chains=chains,
        logp=np.stack([r[1] for r in results]),
        acceptance=np.array([r[2] for r in results]),
        rhat=split_rhat(chains),
        ess=ess(chains),
        meta={"n_warmup": n_warmup, "feature": problem.feature, "seed": seed},
    )
