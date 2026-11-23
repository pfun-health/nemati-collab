"""Discrepancy measures between data features and model features.

Each term is scaled like a chi-square statistic: a squared difference divided
by the variance of that difference, where the variance combines the
uncertainty of the data estimate *and* of the model estimate (both are
averages over finite, noisy records). So ``0.5 * cost`` can be read as a
Gaussian negative log-likelihood (the CVC term is the one exception; see
:func:`cost_cvc`), and the terms can be added.

This replaces the 2018 cost functions, which compared RRI deviations
element-wise between unaligned data and model arrays, divided by an
arbitrarily repeated variance, scored the CVC distributions against the
identity line, and multiplied the components together.
"""

from __future__ import annotations

import numpy as np

from .features import Features

__all__ = [
    "cost_rri",
    "cost_resp",
    "cost_rsa",
    "cost_cvc",
    "total_cost",
    "FEATURES",
    "FEATURE_ALIASES",
]

# Minimum amount of model output needed before a comparison is meaningful.
MIN_POINTS = 8


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n))


def _z2(data: np.ndarray, model: np.ndarray) -> float:
    if data.size < 2 or model.size < MIN_POINTS:
        return np.inf
    md, sd = _mean_se(data)
    mm, sm = _mean_se(model)
    var = sd**2 + sm**2
    if not np.isfinite(var) or var <= 0:
        return np.inf
    return (md - mm) ** 2 / var


def cost_rri(data: Features, model: Features) -> float:
    """Squared z-score of the mean RR interval."""
    return _z2(data.rri, model.rri)


def cost_resp(data: Features, model: Features) -> float:
    """Squared z-scores of the mean inspiration and expiration durations."""
    return _z2(data.insp, model.insp) + _z2(data.exp, model.exp)


def cost_rsa(data: Features, model: Features) -> float:
    """Mahalanobis distance between the fitted RSA coefficients (c, A, B).

    The intercept c carries the mean RR interval, (A, B) the amplitude and
    phase of the respiratory modulation.
    """
    if data.rsa is None or model.rsa is None or model.rsa.n < MIN_POINTS:
        return np.inf
    d = data.rsa.coef - model.rsa.coef
    S = data.rsa.cov + model.rsa.cov
    # tiny ridge relative to the diagonal guards against a singular sum
    S = S + np.eye(3) * 1e-12 * max(np.trace(S), 1e-300)
    try:
        return float(d @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        return np.inf


def cost_cvc(data: Features, model: Features) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between the CVC phase
    distributions, scaled as ``n_eff * D^2`` with ``n_eff = n m / (n + m)``.

    This is a distribution distance, not a likelihood, so for sampling it
    acts as a pseudo-likelihood term.
    """
    x, y = data.cvc, model.cvc
    if x.size < MIN_POINTS or y.size < MIN_POINTS:
        return np.inf
    grid = np.concatenate([x, y])
    Fx = np.searchsorted(x, grid, side="right") / x.size
    Fy = np.searchsorted(y, grid, side="right") / y.size
    D = float(np.max(np.abs(Fx - Fy)))
    n_eff = x.size * y.size / (x.size + y.size)
    return n_eff * D**2


# Feature name -> list of cost terms that make it up.
FEATURES = {
    "rri": (cost_rri,),
    "resp": (cost_resp,),
    "rsa": (cost_rsa,),
    "cvc": (cost_cvc, cost_rsa, cost_resp),
}

# Names used by the 2018 configs.
FEATURE_ALIASES = {
    "rri_ave": "rri",
    "ave_resp": "resp",
    "exp_time": "resp",
    "insp_time": "resp",
}


def resolve_feature(name: str) -> str:
    name = FEATURE_ALIASES.get(name.lower(), name.lower())
    if name not in FEATURES:
        raise ValueError(f"unknown feature {name!r}; choose from {sorted(FEATURES)}")
    return name


def total_cost(data: Features, model: Features, feature: str) -> float:
    """Sum of the cost terms that define ``feature``. Non-finite -> inf."""
    total = 0.0
    for term in FEATURES[resolve_feature(feature)]:
        c = term(data, model)
        if not np.isfinite(c):
            return np.inf
        total += c
    return float(total)
