"""Config parsing (including every 2018 config in the repo), data helpers, CLI."""

import json
from pathlib import Path

import numpy as np
import pytest

import pycardio as pc
from pycardio import data
from pycardio.cli import main
from pycardio.config import load_config

CONFIGS = sorted((Path(__file__).parent.parent / "config").glob("*.json"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.feature in pc.FEATURES
    assert cfg.free and set(cfg.free) <= set(pc.PARAM_NAMES)
    assert set(cfg.fixed) <= set(pc.PARAM_NAMES)
    assert not set(cfg.free) & set(cfg.fixed)
    target = pc.simulate(duration=20).features()
    problem = cfg.problem(target)
    assert np.all(problem.upper > problem.lower)
    assert np.all((problem.u0 >= 0) & (problem.u0 <= 1))


def test_legacy_translation(tmp_path):
    legacy = {
        "LIB_PATH": "/somewhere",
        "path": "~/Public",
        "database": "fantasia",
        "subject_name": "f1y02",
        "opt_feature": "ave_resp",
        "method": "genetic_alg",
        "freeParams": {"d2": 0.6, "d4": 0.5},
        "bounds": {"d2": [0.5, 0.7]},
        "maxiter": 10,
        "simtime": 60,
        "nr_p": 4,
        "ga_args": {"popsize": 8, "seed": None, "mutation": [0.5, 1.0], "recombination": 0.7},
        "defaultParams": {"gL": 2.9, "hr": 0.001, "d2": 0.1, "bogus": 1},
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(legacy))
    cfg = load_config(p)
    assert (cfg.subject, cfg.feature, cfg.duration) == ("f1y02", "resp", 60.0)
    assert cfg.free == {"d2": (0.5, 0.7), "d4": None}
    assert cfg.fixed == {"gL": 2.9, "hr": 0.001}  # constants are now real parameters
    assert cfg.options == {"popsize": 8, "mutation": (0.5, 1.0), "recombination": 0.7}
    assert "LIB_PATH" in cfg.ignored and "defaultParams.bogus" in cfg.ignored


def test_unknown_parameter_in_config_is_an_error(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"subject": "x", "free": {"nope": [0, 1]}}))
    with pytest.raises(KeyError):
        load_config(p)


def test_finite_runs():
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert data._finite_runs(mask, 2) == [(0, 2), (3, 6)]
    assert data._finite_runs(mask, 1) == [(0, 2), (3, 6), (8, 9)]


def test_recording_features_handles_nans_and_ranges():
    fs = 250.0
    t = np.arange(0, 400, 1 / fs)
    resp = np.sin(2 * np.pi * t / 4.0)
    resp[int(150 * fs) : int(160 * fs)] = np.nan  # a gap in the middle
    beats = np.arange(0.5, 399, 0.8)
    rec = data.Recording("syn", fs, t, resp, beats, bp=None, ranges=None)
    f = data.recording_features(rec)
    assert f.meta["n_segments"] == 2
    np.testing.assert_allclose(f.duration, 390, atol=0.1)  # nothing before the gap is lost
    np.testing.assert_allclose(f.resp_rate, 0.25, rtol=0.03)
    rec.ranges = np.array([[0, int(100 * fs)]])
    g = data.recording_features(rec)
    np.testing.assert_allclose(g.duration, 100, atol=0.1)
    assert g.meta["used_ranges"]


def test_bundled_range_files_are_found(tmp_path):
    r = data._find_ranges("f1y02", tmp_path, "fantasia")
    assert r is not None and r.shape[1] == 2 and np.all(r[:, 1] > r[:, 0])


def test_cli_simulate(tmp_path, capsys):
    out = tmp_path / "sim.npz"
    assert (
        main(
            [
                "simulate",
                "-p",
                "hr=0.001",
                "--duration",
                "20",
                "--save",
                str(out),
                "--plot",
                str(tmp_path / "t.png"),
            ]
        )
        == 0
    )
    assert "heart_rate_bpm" in capsys.readouterr().out
    assert np.load(out)["states"].shape == (20001, 13)
    assert (tmp_path / "t.png").stat().st_size > 0


def test_cli_fit_and_sample_end_to_end(tmp_path, monkeypatch):
    """Whole workflow against a cached 'recording' made from the model itself."""
    db = tmp_path / "fantasia" / "pycardio-cache"
    db.mkdir(parents=True)
    pc.simulate(dict(hr=0.00095), 60, transient=10).features().to_npz(db / "syn.npz")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "subject": "syn",
                "data_dir": str(tmp_path),
                "feature": "rri",
                "method": "nelder-mead",
                "free": {"hr": [0.0007, 0.0012]},
                "duration": 60,
                "maxfev": 60,
                "mcmc": {"n_samples": 60, "n_warmup": 40, "n_chains": 2},
            }
        )
    )
    fit_out = tmp_path / "fit.json"
    assert main(["fit", str(cfg), "--out", str(fit_out), "--quiet"]) == 0
    res = json.loads(fit_out.read_text())
    assert res["params"]["hr"] == pytest.approx(0.00095, rel=0.01)
    assert (
        main(["sample", str(cfg), "--start-from", str(fit_out), "--out", str(tmp_path / "m")]) == 0
    )
    summary = json.loads((tmp_path / "m.json").read_text())["summary"]["hr"]
    assert summary["mean"] == pytest.approx(0.00095, rel=0.02)
