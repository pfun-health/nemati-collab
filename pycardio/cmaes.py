"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

Follows N. Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772,
2016): weighted recombination, cumulative step-size adaptation, rank-one and
rank-mu covariance updates, and the h_sigma stall correction.

This replaces the C implementation in ``c_fit-model/``, which was never
finished and did not compile.

Box constraints are handled by evaluating the objective at the projection of
each candidate onto the box and adding a quadratic penalty on the distance to
it, so the distribution is pushed back inside while the ranking stays
informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

__all__ = ["CMAESResult", "cmaes"]


@dataclass
class CMAESResult:
    x: np.ndarray  # best point found
    fun: float  # objective at x
    nfev: int
    nit: int
    message: str
    mean: np.ndarray  # final distribution mean
    sigma: float
    cov: np.ndarray  # final covariance matrix C (the search covariance is sigma^2 C)
    history: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.message != "maxfev"


def cmaes(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma0: float = 0.3,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    popsize: int | None = None,
    maxfev: int = 2000,
    tolx: float = 1e-8,
    tolfun: float = 1e-10,
    seed: int | np.random.Generator | None = None,
    map_fn: Callable[[Callable, Iterable], Iterable] = map,
    callback: Callable[[dict], bool | None] | None = None,
    verbose: bool = False,
) -> CMAESResult:
    """Minimise ``fun`` starting from ``x0`` with step size ``sigma0``.

    ``map_fn`` evaluates a population (e.g. ``ThreadPoolExecutor().map``).
    ``callback`` receives a dict of the generation's state; returning True
    stops the run.
    """
    rng = np.random.default_rng(seed)
    m = np.array(x0, dtype=float)
    N = m.size
    if bounds is not None:
        lo, hi = (np.broadcast_to(np.asarray(b, dtype=float), (N,)) for b in bounds)
        m = np.clip(m, lo, hi)
    sigma = float(sigma0)

    # --- strategy parameters (Hansen 2016, Table 1) ---
    lam = popsize or 4 + int(3 * np.log(N))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))

    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    C = np.eye(N)
    invsqrtC = np.eye(N)
    eigen_gap = max(1, int(lam / (c1 + cmu) / N / 10))

    best_x, best_f = m.copy(), np.inf
    nfev = 0
    gen = 0
    hist_len = 10 + int(np.ceil(30 * N / lam))
    recent_best: list[float] = []
    history = {"best": [], "sigma": [], "mean": []}
    message = "maxfev"

    def penalised(x):
        if bounds is None:
            return x, 0.0
        xc = np.clip(x, lo, hi)
        return xc, float(np.sum((x - xc) ** 2))

    while nfev + lam <= maxfev:
        gen += 1
        z = rng.standard_normal((lam, N))
        y = z @ (B * D).T  # y_k ~ N(0, C)
        x = m + sigma * y
        projected = [penalised(xi) for xi in x]
        fvals = np.array(list(map_fn(fun, [p[0] for p in projected])), dtype=float)
        nfev += lam
        dist2 = np.array([p[1] for p in projected])
        fvals = np.where(np.isnan(fvals), np.inf, fvals)
        fpen = fvals + 1e3 * (1.0 + np.abs(np.where(np.isfinite(fvals), fvals, 0.0))) * dist2

        order = np.argsort(fpen, kind="stable")
        i0 = int(np.argmin(fvals))
        if fvals[i0] < best_f:
            best_f, best_x = float(fvals[i0]), projected[i0][0].copy()

        # --- recombination ---
        y_sel = y[order[:mu]]
        y_w = w @ y_sel
        m = m + sigma * y_w

        # --- step-size path and h_sigma ---
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
        ps_norm = np.linalg.norm(ps)
        hsig = ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (N + 1)

        # --- covariance update ---
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        dh = (1 - hsig) * cc * (2 - cc)
        rank_mu = (y_sel.T * w) @ y_sel
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + dh * C) + cmu * rank_mu

        # --- step-size update ---
        sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))

        if gen % eigen_gap == 0:
            C = np.triu(C) + np.triu(C, 1).T
            evals, B = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-300)
            D = np.sqrt(evals)
            invsqrtC = (B / D) @ B.T

        history["best"].append(best_f)
        history["sigma"].append(sigma)
        history["mean"].append(m.copy())
        recent_best.append(float(fvals[order[0]]))
        if verbose:
            print(f"gen {gen:4d}  nfev {nfev:6d}  best {best_f:.6g}  sigma {sigma:.3g}")

        state = dict(gen=gen, nfev=nfev, best_x=best_x, best_f=best_f, mean=m, sigma=sigma)
        if callback is not None and callback(state):
            message = "callback"
            break

        # --- termination ---
        if sigma * D.max() < tolx:
            message = "tolx"
            break
        if len(recent_best) >= hist_len:
            window = recent_best[-hist_len:]
            if np.all(np.isfinite(window)) and max(window) - min(window) < tolfun:
                message = "tolfun"
                break
        if D.max() / D.min() > 1e7:
            message = "condition"
            break

    return CMAESResult(
        x=best_x,
        fun=best_f,
        nfev=nfev,
        nit=gen,
        message=message,
        mean=m,
        sigma=sigma,
        cov=C,
        history=history,
    )
