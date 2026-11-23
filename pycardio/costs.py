"""Discrepancy measures between data features and model features.

Each term is scaled like a chi-square statistic: a squared difference divided
by the variance that difference would have if the model produced data with the
recording's statistics. Both estimates are averages over finite records, so
both sample sizes count, but the *spread* is always taken from the recording:
``var = s_data^2 * (1/n_data + 1/n_model)``.

Taking the spread from the model as well (the obvious "sum of the two squared
standard errors") is a trap: the optimiser can then lower the cost by making
the model noisier instead of making it fit. An earlier version of this module
did exactly that and produced a "best fit" with eight times the recording's
RSA.

``0.5 * cost`` can be read as a Gaussian negative log-likelihood (the CVC term
is the one exception; see :func:`cost_cvc`), and the terms can be added.

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


def _z2(data: np.ndarray, model: np.ndarray) -> float:
    """Squared z-score of the difference in means, with the recording's spread."""
    if data.size < 2 or model.size < MIN_POINTS:
        return np.inf
    diff = float(np.mean(data) - np.mean(model))
    var = float(np.var(data, ddof=1)) * (1.0 / data.size + 1.0 / model.size)
    if not np.isfinite(var) or var <= 0:
        return 0.0 if diff == 0 else np.inf
    return diff**2 / var


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
    # covariance of the difference, with the recording's residual spread for
    # both records: cov_data scales as 1/n, so the model's share is n_d/n_m of it
    S = data.rsa.cov * (1.0 + data.rsa.n / model.rsa.n)
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
