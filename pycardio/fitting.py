"""Fit model parameters to the features of a recording.

Design choices (see CHANGES-AI.md for the reasoning):

* **Normalised coordinates.** Optimisers work on ``u`` in the unit cube; each
  free parameter is mapped linearly onto its bounds. The raw parameters span
  seven orders of magnitude (``hr`` ~ 1e-3, ``taup`` ~ 1e3), which made the
  2018 step sizes and simplex moves badly scaled.
* **Common random numbers.** Every evaluation uses the same noise seed, so the
  objective is a deterministic function of the parameters. A noisy objective
  breaks the convergence assumptions of every optimiser used here.
* **Parallel evaluation.** The C solver releases the GIL, so populations
  (CMA-ES, differential evolution) are evaluated on a thread pool.
"""

from __future__ import annotations

import threading
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import optimize

from .cmaes import cmaes
from .costs import FEATURES, resolve_feature, total_cost
from .features import Features
from .model import DEFAULT_BOUNDS, DEFAULT_PARAMS, SimulationError, simulate

__all__ = ["FitProblem", "FitResult", "fit", "METHODS"]

METHODS = ("cmaes", "differential_evolution", "nelder-mead", "powell")
METHOD_ALIASES = {
    "genetic_alg": "differential_evolution",
    "de": "differential_evolution",
    "minimize": "nelder-mead",
    "basinhopping": "cmaes",
}


# Optimiser options accepted per method (configs may carry options for another).
_METHOD_OPTIONS = {
    "cmaes": {"sigma0", "popsize", "tolx", "tolfun"},
    "differential_evolution": {
        "popsize",
        "mutation",
        "recombination",
        "strategy",
        "tol",
        "atol",
        "init",
    },
    "nelder-mead": {"options"},
    "powell": {"options"},
}


@dataclass
class FitProblem:
    """Everything needed to score a parameter vector against a recording.

    Parameters
    ----------
    target : features of the recording
    free : free parameter -> (lower, upper) bound. ``None`` bounds fall back
        to :data:`DEFAULT_BOUNDS`, then to +/-50 % of the start value.
    feature : 'rri', 'resp', 'rsa' or 'cvc' (2018 names are accepted too)
    start : initial values of the free parameters (default: box centre)
    fixed : values for other parameters (default: model defaults)
    duration : simulated seconds kept per evaluation
    transient : seconds simulated first and discarded
    seed : noise seed used for every evaluation
    """

    target: Features
    free: Mapping[str, tuple[float, float] | None]
    feature: str = "rsa"
    start: Mapping[str, float] | None = None
    fixed: Mapping[str, float] = field(default_factory=dict)
    duration: float = 120.0
    transient: float = 10.0
    seed: int = 0
    dt: float = 1.0
    fs: float = 250.0

    def __post_init__(self):
        self.feature = resolve_feature(self.feature)
        self.names = tuple(sorted(self.free))
        unknown = [n for n in (*self.names, *self.fixed) if n not in DEFAULT_PARAMS]
        if unknown:
            raise KeyError(f"unknown model parameters: {unknown}")
        start = dict(self.start or {})
        lo, hi = [], []
        for n in self.names:
            b = self.free[n] or DEFAULT_BOUNDS.get(n)
            if b is None:
                v = start.get(n, self.fixed.get(n, DEFAULT_PARAMS[n]))
                b = (v - 0.5 * abs(v), v + 0.5 * abs(v))
            if not b[1] > b[0]:
                raise ValueError(f"empty bounds for {n}: {b}")
            lo.append(float(b[0]))
            hi.append(float(b[1]))
        self.lower, self.upper = np.array(lo), np.array(hi)
        x0 = [start.get(n, 0.5 * (a + b)) for n, a, b in zip(self.names, lo, hi)]
        self.u0 = np.clip(self.to_unit(np.array(x0, dtype=float)), 0.0, 1.0)
        self.nfev = 0
        self._lock = threading.Lock()

    # ---- coordinate maps ----
    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(u, dtype=float) * (self.upper - self.lower)

    def params(self, u: np.ndarray) -> dict[str, float]:
        """Full parameter overrides for unit-cube point ``u``."""
        p = dict(self.fixed)
        p.update(zip(self.names, self.from_unit(u).tolist()))
        return p

    # ---- evaluation ----
    def model_features(self, params: Mapping[str, float]) -> Features:
        sim = simulate(params, self.duration, dt=self.dt, seed=self.seed, transient=self.transient)
        return sim.features(fs=self.fs, compute_dfa=False)

    def cost_params(self, params: Mapping[str, float]) -> float:
        with self._lock:
            self.nfev += 1
        try:
            feats = self.model_features(params)
        except (SimulationError, ValueError):
            return np.inf
        return total_cost(self.target, feats, self.feature)

    def __call__(self, u: np.ndarray) -> float:
        """Objective on the unit cube (points outside are clipped)."""
        return self.cost_params(self.params(np.clip(u, 0.0, 1.0)))


@dataclass
class FitResult:
    params: dict[str, float]  # best free-parameter values
    cost: float
    method: str
    nfev: int
    elapsed: float
    message: str
    problem: FitProblem = field(repr=False)
    raw: object = field(default=None, repr=False)

    def full_params(self) -> dict[str, float]:
        p = dict(self.problem.fixed)
        p.update(self.params)
        return p

    def to_dict(self) -> dict:
        return {
            "subject_feature": self.problem.feature,
            "method": self.method,
            "cost": self.cost,
            "params": self.params,
            "fixed": dict(self.problem.fixed),
            "bounds": {
                n: [a, b]
                for n, a, b in zip(self.problem.names, self.problem.lower, self.problem.upper)
            },
            "nfev": self.nfev,
            "elapsed_s": self.elapsed,
            "message": self.message,
            "duration": self.problem.duration,
            "transient": self.problem.transient,
            "seed": self.problem.seed,
        }


def fit(
    problem: FitProblem,
    method: str = "cmaes",
    maxfev: int = 2000,
    seed: int | None = 0,
    workers: int | None = None,
    verbose: bool = True,
    **options,
) -> FitResult:
    """Minimise the problem's cost.

    Methods: ``cmaes`` (default; robust for these noisy-looking, rugged
    landscapes), ``differential_evolution``, ``nelder-mead``, ``powell``.
    ``workers`` threads evaluate populations in parallel (default: CPU count).
    Extra ``options`` go to the underlying optimiser.
    """
    if method.lower() == "basinhopping":
        warnings.warn("basinhopping is no longer supported; using cmaes", stacklevel=2)
    method = METHOD_ALIASES.get(method.lower(), method.lower())
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    allowed = _METHOD_OPTIONS[method]
    dropped = sorted(k for k in options if k not in allowed)
    if dropped:
        warnings.warn(f"options {dropped} do not apply to {method} and are ignored", stacklevel=2)
    options = {k: v for k, v in options.items() if k in allowed}
    d = len(problem.names)
    t0 = _time.perf_counter()
    problem.nfev = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if method == "cmaes":
            res = cmaes(
                problem,
                problem.u0,
                sigma0=options.pop("sigma0", 0.25),
                bounds=(np.zeros(d), np.ones(d)),
                maxfev=maxfev,
                seed=seed,
                map_fn=pool.map,
                verbose=verbose,
                **options,
            )
            u, f, msg = res.x, res.fun, res.message
        elif method == "differential_evolution":
            popsize = options.pop("popsize", 10)
            maxiter = max(1, maxfev // (popsize * d) - 1)
            res = optimize.differential_evolution(
                problem,
                [(0.0, 1.0)] * d,
                x0=problem.u0,
                popsize=popsize,
                maxiter=maxiter,
                seed=seed,
                polish=False,
                updating="deferred",
                workers=pool.map,
                disp=verbose,
                **options,
            )
            u, f, msg = res.x, res.fun, res.message
        else:
            scipy_method = "Nelder-Mead" if method == "nelder-mead" else "Powell"
            opts = {"maxfev": maxfev, "disp": verbose, **options.pop("options", {})}
            if scipy_method == "Nelder-Mead":
                opts.setdefault("xatol", 1e-4)
                opts.setdefault("fatol", 1e-3)
            res = optimize.minimize(
                problem,
                problem.u0,
                method=scipy_method,
                bounds=[(0.0, 1.0)] * d,
                options=opts,
                **options,
            )
            u, f, msg = res.x, res.fun, str(res.message)
    params = dict(zip(problem.names, problem.from_unit(np.clip(u, 0, 1)).tolist()))
    return FitResult(
        params=params,
        cost=float(f),
        method=method,
        nfev=problem.nfev,
        elapsed=_time.perf_counter() - t0,
        message=str(msg),
        problem=problem,
        raw=res,
    )


def available_features() -> tuple[str, ...]:
    return tuple(FEATURES)
