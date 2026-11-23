"""Build the JSON behind the Fantasia results dashboard.

Usage:
    python examples/fantasia_dashboard_data.py OUT.json [--data-dir DIR] [--fit SUBJECT]

For every subject: summary statistics, the fitted RSA curve with binned means,
and the CVC phase histogram. For one subject (default f2y03): a CMA-ES fit of
the model's cardiac parameters to the RSA feature, with the model's features
alongside the data's.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np

import pycardio as pc

N_RSA_BINS = 20
N_CVC_BINS = 40


def rnd(x, n=5):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, n) for v in x]
    x = float(x)
    return None if not np.isfinite(x) else float(f"{x:.{n}g}")


def curves(f: pc.Features) -> dict:
    edges = np.linspace(0, 1, N_RSA_BINS + 1)
    idx = np.clip(np.digitize(f.rsa_phase, edges) - 1, 0, N_RSA_BINS - 1)
    c = f.rsa.coef[0]
    binned = [
        rnd(1000 * (f.rsa_rri[idx == k].mean() - c), 4) if np.any(idx == k) else None
        for k in range(N_RSA_BINS)
    ]
    hist, _ = np.histogram(f.cvc, bins=N_CVC_BINS, range=(0, 1))
    # Kolmogorov-Smirnov distance of the CVC phases from a uniform distribution
    n = f.cvc.size
    ecdf_hi = np.arange(1, n + 1) / n
    ks = float(max(np.max(ecdf_hi - f.cvc), np.max(f.cvc - (ecdf_hi - 1 / n))))
    return {
        "rsa_coef_ms": rnd(1000 * f.rsa.coef, 5),
        "rsa_binned_ms": binned,
        "cvc_density": rnd(hist / hist.sum() * N_CVC_BINS, 4),
        "cvc_ks": rnd(ks, 4),
    }


def subject_entry(name: str, f: pc.Features) -> dict:
    s = f.summary()
    return {
        "id": name,
        "group": "young" if name[2] == "y" else "old",
        "minutes": rnd(s["duration_s"] / 60, 4),
        "n_beats": s["n_beats"],
        "n_breaths": s["n_breaths"],
        "hr_bpm": rnd(s["heart_rate_bpm"], 4),
        "rri_sd_ms": rnd(1000 * s["rri_sd_s"], 4),
        "resp_per_min": rnd(s["resp_rate_per_min"], 4),
        "beats_per_breath": rnd(s["rotation_number"], 4),
        "insp_s": rnd(s["insp_mean_s"], 4),
        "exp_s": rnd(s["exp_mean_s"], 4),
        "rsa_ms": rnd(1000 * s["rsa_amplitude_s"], 4),
        "rsa_se_ms": rnd(1000 * s["rsa_amplitude_se_s"], 3),
        "rsa_peak_phase": rnd(s["rsa_peak_phase"], 3),
        "dfa_alpha1": rnd(s["dfa_alpha1"], 4),
        "used_ranges": f.meta.get("used_ranges") in (True, "True"),
        **curves(f),
    }


def example_fit(name: str, target: pc.Features, maxfev: int) -> dict:
    free = {"hr": (0.0006, 0.0012), "rsa1": (0.0, 0.0015), "rsa2": (0.0, 0.0014)}
    problem = pc.FitProblem(target, free=free, feature="rsa", duration=300, transient=10)
    res = pc.fit(problem, method="cmaes", maxfev=maxfev, seed=0, verbose=False)
    model = pc.simulate(res.full_params(), 300, transient=10, seed=0).features()
    other_seeds = [
        pc.total_cost(
            target, pc.simulate(res.full_params(), 300, transient=10, seed=s).features(), "rsa"
        )
        for s in (1, 2, 3)
    ]
    return {
        "subject": name,
        "feature": "rsa",
        "method": "cmaes",
        "nfev": res.nfev,
        "elapsed_s": rnd(res.elapsed, 3),
        "cost": rnd(res.cost, 4),
        "cost_other_seeds": rnd(other_seeds, 4),
        "start_cost": rnd(problem(problem.u0), 4),
        "params": {k: rnd(v, 5) for k, v in res.params.items()},
        "bounds": {k: list(v) for k, v in free.items()},
        "model_seconds": 300,
        "model": {
            "hr_bpm": rnd(60 * model.heart_rate, 4),
            "resp_per_min": rnd(60 * model.resp_rate, 4),
            "rsa_ms": rnd(1000 * model.rsa.amplitude, 4),
            **curves(model),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fit", default="f2y03")
    ap.add_argument("--maxfev", type=int, default=1500)
    a = ap.parse_args()

    subjects, feats, failed = [], {}, {}
    for name in pc.list_subjects("fantasia", a.data_dir):
        try:
            feats[name] = pc.subject_features(name, "fantasia", a.data_dir)
            subjects.append(subject_entry(name, feats[name]))
        except Exception as exc:  # report, do not hide, subjects that fail
            failed[name] = f"{type(exc).__name__}: {exc}"
        print(name, "ok" if name in feats else failed[name], flush=True)

    out = {
        "generated": date.today().isoformat(),
        "pycardio_version": pc.__version__,
        "n_rsa_bins": N_RSA_BINS,
        "n_cvc_bins": N_CVC_BINS,
        "subjects": subjects,
        "failed": failed,
        "fit": example_fit(a.fit, feats[a.fit], a.maxfev) if a.fit in feats else None,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {a.out}: {len(subjects)} subjects, {len(failed)} failed")


if __name__ == "__main__":
    main()
