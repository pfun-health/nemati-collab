"""Run configuration files (JSON).

Current format::

    {
      "subject": "f1y02",
      "database": "fantasia",
      "data_dir": "~/Public",
      "feature": "rsa",                    # rri | resp | rsa | cvc
      "method": "cmaes",                   # cmaes | differential_evolution | nelder-mead | powell
      "free":  {"hr": [0.00075, 0.0009], "rsa1": null},   # null -> default bounds
      "start": {"hr": 0.00077},            # optional
      "fixed": {"d2": 0.62, "d4": 0.43},   # optional, any model parameter
      "duration": 120, "transient": 10, "seed": 0,
      "maxfev": 2000,
      "options": {},                       # passed to the optimiser
      "mcmc": {"n_samples": 2000, "n_warmup": 1000, "n_chains": 4}
    }

The 2018 configs (``subject_name``, ``opt_feature``, ``freeParams``,
``bounds``, ``defaultParams``, ``simtime``, ``ga_args``, ...) are read and
translated; keys with no equivalent are reported in ``ignored``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .costs import resolve_feature
from .model import DEFAULT_PARAMS

__all__ = ["RunConfig", "load_config"]

_LEGACY_IGNORED = {"LIB_PATH", "nr_p", "miniter", "tol", "basin_args", "min_args", "cmaes_args"}


@dataclass
class RunConfig:
    subject: str
    feature: str = "rsa"
    database: str = "fantasia"
    data_dir: str | None = None
    method: str = "cmaes"
    free: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    start: dict[str, float] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    duration: float = 120.0
    transient: float = 10.0
    seed: int = 0
    maxfev: int = 2000
    options: dict[str, Any] = field(default_factory=dict)
    mcmc: dict[str, Any] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    source: str | None = None

    def problem(self, target=None):
        """Build a :class:`~pycardio.fitting.FitProblem` (loads the subject's
        features unless ``target`` is given)."""
        from .data import subject_features
        from .fitting import FitProblem

        if target is None:
            target = subject_features(self.subject, self.database, self.data_dir)
        return FitProblem(
            target=target,
            free=self.free,
            feature=self.feature,
            start=self.start,
            fixed=self.fixed,
            duration=self.duration,
            transient=self.transient,
            seed=self.seed,
        )


def _from_legacy(d: dict) -> dict:
    out: dict[str, Any] = {"ignored": sorted(k for k in d if k in _LEGACY_IGNORED)}
    out["subject"] = d["subject_name"]
    out["database"] = d.get("database", "fantasia")
    if d.get("path"):
        out["data_dir"] = d["path"]
    out["feature"] = d.get("opt_feature", "rsa")
    out["method"] = d.get("method", "cmaes")
    start = dict(d.get("freeParams") or {})
    bounds = d.get("bounds") or {}
    out["start"] = start
    out["free"] = {k: (tuple(bounds[k]) if k in bounds else None) for k in start}
    fixed = {}
    for k, v in (d.get("defaultParams") or {}).items():
        if k in start:
            continue
        if k in DEFAULT_PARAMS:
            fixed[k] = v
        else:
            out["ignored"].append(f"defaultParams.{k}")
    out["fixed"] = fixed
    if "simtime" in d:
        out["duration"] = float(d["simtime"])
    out["transient"] = 0.0  # the 2018 runs kept the start-up transient
    options: dict[str, Any] = {}
    ga = d.get("ga_args") or {}
    if out["method"] in ("genetic_alg", "differential_evolution"):
        for k in ("popsize", "mutation", "recombination"):
            if k in ga:
                options[k] = tuple(ga[k]) if isinstance(ga[k], list) else ga[k]
        if ga.get("seed") is not None:
            out["seed"] = int(ga["seed"])
        if "maxiter" in d:
            pop = options.get("popsize", 15)
            out["maxfev"] = int(d["maxiter"]) * pop * max(len(start), 1) + pop * max(len(start), 1)
    elif "maxiter" in d:
        out["maxfev"] = int(d["maxiter"]) * 10 * max(len(start), 1)
    out["options"] = options
    return out


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    d = json.loads(path.read_text())
    if "subject_name" in d:
        d = _from_legacy(d)
    known = set(RunConfig.__dataclass_fields__)
    extra = sorted(set(d) - known)
    kwargs = {k: v for k, v in d.items() if k in known}
    kwargs.setdefault("ignored", []).extend(extra)
    kwargs["free"] = {
        k: (tuple(v) if v is not None else None) for k, v in kwargs.get("free", {}).items()
    }
    cfg = RunConfig(**kwargs, source=str(path))
    cfg.feature = resolve_feature(cfg.feature)
    unknown = [k for k in (*cfg.free, *cfg.fixed) if k not in DEFAULT_PARAMS]
    if unknown:
        raise KeyError(f"{path}: unknown model parameters {unknown}")
    return cfg
