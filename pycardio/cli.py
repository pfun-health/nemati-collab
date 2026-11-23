"""Command-line interface: ``pycardio <command> ...`` (see ``pycardio -h``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump(obj, path: Path | None = None) -> None:
    text = json.dumps(obj, indent=2, default=_json_default)
    if path is None:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"wrote {path}")


def _parse_params(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items or []:
        key, _, val = item.partition("=")
        if not _:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        out[key.strip()] = float(val)
    return out


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------


def cmd_download(a):
    from .data import download

    print(f"downloaded to {download(a.database, a.data_dir)}")


def cmd_features(a):
    from .data import list_subjects, subject_features

    subjects = a.subjects or list_subjects(a.database, a.data_dir)
    table = {}
    for s in subjects:
        f = subject_features(
            s,
            a.database,
            a.data_dir,
            refresh=a.refresh,
            **({"use_ranges": True} if a.ranges else {}),
        )
        table[s] = f.summary() | {"meta": f.meta}
    _dump(table, Path(a.out) if a.out else None)


def cmd_simulate(a):
    from .model import simulate

    sim = simulate(_parse_params(a.param), a.duration, dt=a.dt, seed=a.seed, transient=a.transient)
    feats = sim.features()
    _dump({"params": _parse_params(a.param), "seed": a.seed, "features": feats.summary()})
    if a.save:
        np.savez_compressed(a.save, t=sim.t, states=sim.states, beat_times=sim.beat_times)
        print(f"wrote {a.save}")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import plot_traces

        _savefig(plot_traces(sim, (0, min(a.duration, 30.0))), Path(a.plot))


def cmd_fit(a):
    from .config import load_config
    from .fitting import fit

    cfg = load_config(a.config)
    if cfg.ignored:
        print(f"note: ignoring config keys {cfg.ignored}", file=sys.stderr)
    method = a.method or cfg.method
    maxfev = a.maxfev or cfg.maxfev
    problem = cfg.problem()
    res = fit(
        problem,
        method=method,
        maxfev=maxfev,
        seed=cfg.seed,
        workers=a.workers,
        verbose=not a.quiet,
        **cfg.options,
    )
    out = res.to_dict() | {"subject": cfg.subject, "config": cfg.source}
    _dump(out, Path(a.out) if a.out else Path("results") / f"fit_{cfg.subject}_{cfg.feature}.json")


def cmd_sample(a):
    from .config import load_config
    from .mcmc import sample

    cfg = load_config(a.config)
    problem = cfg.problem()
    start = None
    if a.start_from:
        best = json.loads(Path(a.start_from).read_text())["params"]
        start = problem.to_unit(np.array([best[n] for n in problem.names]))
    kw = {"n_samples": 2000, "n_warmup": 1000, "n_chains": 4} | cfg.mcmc
    for k in ("n_samples", "n_warmup", "n_chains"):
        if getattr(a, k):
            kw[k] = getattr(a, k)
    res = sample(problem, start=start, seed=cfg.seed, **kw)
    stem = Path(a.out) if a.out else Path("results") / f"mcmc_{cfg.subject}_{cfg.feature}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    res.save(stem.with_suffix(".npz"))
    _dump({"summary": res.summary(), "acceptance": res.acceptance}, stem.with_suffix(".json"))


def cmd_plot(a):
    import matplotlib

    matplotlib.use("Agg")
    from .data import load_recording, subject_features
    from .model import simulate
    from .plotting import comparison_figure, plot_synchrogram

    out = Path(a.out)
    data = subject_features(a.subject, a.database, a.data_dir)
    model = None
    params = _parse_params(a.param)
    if a.fit:
        fitres = json.loads(Path(a.fit).read_text())
        params = {**fitres.get("fixed", {}), **fitres["params"], **params}
    if params or a.fit:
        sim = simulate(params, a.duration, seed=a.seed, transient=10.0)
        model = sim.features()
        t, v = sim.resampled()
        _savefig(
            plot_synchrogram(t, v, sim.beat_times, title="model").figure,
            out / f"{a.subject}_synchrogram_model.png",
        )
    _savefig(comparison_figure(data, model, title=a.subject), out / f"{a.subject}_rsa_cvc.png")
    rec = load_recording(a.subject, a.database, a.data_dir)
    from .features import smooth

    resp = smooth(np.nan_to_num(rec.resp), int(rec.fs))
    _savefig(
        plot_synchrogram(rec.time, resp, rec.beat_times, title=f"{a.subject} data").figure,
        out / f"{a.subject}_synchrogram_data.png",
    )


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pycardio", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp):
        sp.add_argument("--database", default="fantasia")
        sp.add_argument(
            "--data-dir",
            default=None,
            help="folder containing the database folder (default $PYCARDIO_DATA_DIR or ~/Public)",
        )

    sp = sub.add_parser("download", help="download a PhysioNet database")
    data_args(sp)
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("features", help="compute and print features of recordings")
    sp.add_argument("subjects", nargs="*", help="subject IDs (default: all)")
    sp.add_argument(
        "--ranges", action="store_true", help="analyse only the hand-picked epochs in .rng files"
    )
    sp.add_argument("--refresh", action="store_true", help="recompute cached features")
    sp.add_argument("--out", help="write JSON here instead of stdout")
    data_args(sp)
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("simulate", help="run the model once")
    sp.add_argument("--param", "-p", action="append", metavar="NAME=VALUE")
    sp.add_argument("--duration", type=float, default=120.0, help="seconds (default 120)")
    sp.add_argument("--transient", type=float, default=10.0, help="seconds dropped first")
    sp.add_argument("--dt", type=float, default=1.0, help="output step in ms")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--save", help="save raw output to this .npz")
    sp.add_argument("--plot", help="save a trace figure here")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("fit", help="fit parameters to a subject (config file)")
    sp.add_argument("config")
    sp.add_argument(
        "--method", choices=["cmaes", "differential_evolution", "nelder-mead", "powell"]
    )
    sp.add_argument("--maxfev", type=int)
    sp.add_argument("--workers", type=int, default=None, help="threads (default: all CPUs)")
    sp.add_argument("--out")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("sample", help="MCMC posterior sampling (config file)")
    sp.add_argument("config")
    sp.add_argument("--start-from", help="fit result JSON to start the chains from")
    sp.add_argument("--n-samples", type=int)
    sp.add_argument("--n-warmup", type=int)
    sp.add_argument("--n-chains", type=int)
    sp.add_argument("--out", help="output path stem")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("plot", help="figures for a subject, optionally with a model fit")
    sp.add_argument("subject")
    sp.add_argument("--fit", help="fit result JSON")
    sp.add_argument("--param", "-p", action="append", metavar="NAME=VALUE")
    sp.add_argument("--duration", type=float, default=300.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="figs")
    data_args(sp)
    sp.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
