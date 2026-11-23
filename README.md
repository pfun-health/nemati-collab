# nemati-collab / pycardio

Model-based analysis of cardiorespiratory coupling: a neural model of the
respiratory central pattern generator coupled to heart rate and blood
pressure, fitted to respiratory sinus arrhythmia (RSA) and cardioventilatory
coupling (CVC) measured in human recordings (PhysioNet Fantasia).

> **History.** The model and analysis were written in 2017-2018 in the Molkov
> Lab, Neuroscience Institute, Georgia State University. In November 2026 the
> code was overhauled with Claude Code: the model equations, parameter values
> and feature definitions are the original ones; the numerics, statistics,
> packaging and tests around them were rewritten. [CHANGES-AI.md](CHANGES-AI.md)
> lists exactly what is original, what was changed and why, and what has not
> been verified. The pre-overhaul code is commit `bad301d`.

## Install

Requirements: Python 3.10+ and a C compiler. Nothing else (GSL and the WFDB C
library are no longer needed).

```bash
uv sync            # or: pip install -e .
uv run pytest      # 67 tests, about a minute
```

## Quick start

```bash
# 1. Get the data (about 650 MB) into ~/Public/fantasia
uv run pycardio download

# 2. Features of a recording
uv run pycardio features f1y02

# 3. Run the model
uv run pycardio simulate -p hr=0.0009 -p rsa1=0.0006 --plot figs/traces.png

# 4. Fit the model to a subject (2018 config files work as they are)
uv run pycardio fit config/fantasia_f2y03_rsa_ga.json --method cmaes --maxfev 1500

# 5. Posterior sampling around the fit, and figures
uv run pycardio sample config/fantasia_f2y03_rsa_ga.json --start-from results/fit_f2y03_rsa.json
uv run pycardio plot f2y03 --fit results/fit_f2y03_rsa.json
```

Results for all 40 recordings and an example fit are shown at
<https://pfun.one/examples/physionet-fantasia>; its data file is produced by
`examples/fantasia_dashboard_data.py`.

Data location: `--data-dir`, else `$PYCARDIO_DATA_DIR`, else `~/Public`; the
database folder (`fantasia/`) is expected inside it.

From Python:

```python
import pycardio as pc

data = pc.subject_features("f1y02")               # features of a recording
print(data.summary())

sim = pc.simulate({"hr": 0.0009}, duration=120, transient=10, seed=0)
model = sim.features()                             # the same features from the model
pc.total_cost(data, model, "rsa")                  # chi-square-scaled mismatch

problem = pc.FitProblem(
    data,
    free={"hr": (0.0006, 0.0012), "rsa1": None, "rsa2": None},   # None -> default bounds
    feature="rsa",
    fixed={"d2": 0.62, "d4": 0.43},
)
fit = pc.fit(problem, method="cmaes", maxfev=1500)
chains = pc.sample(problem, start=problem.to_unit(list(fit.params.values())))
print(fit.params, chains.summary())
```

## What it computes

**Model** (`pycardio/_c/model.c`). Five neural populations (pre-I/I, early-I,
aug-E, post-I, ramp-I) with persistent-sodium and adaptation currents generate
the breathing rhythm; ramp-I drives lung volume; post-I and ramp-I modulate a
heart-phase oscillator (RSA); each heartbeat injects a pulse into a lumped
blood-pressure variable, which feeds back to post-I and the heart
(baroreflex). Thirteen state variables, 57 parameters (`pc.PARAM_DOCS`), time
in ms. Noise enters through the early-I adaptation.

**Features** (`pycardio/features.py`), identical for data and model:

| Feature | Definition |
|---|---|
| `rri` | RR intervals |
| `resp` | inspiration (trough to peak) and expiration (peak to trough) durations |
| `rsa` | RR interval vs respiratory phase from expiration onset, fitted with `c + A cos 2πφ + B sin 2πφ` |
| `cvc` | distribution of heartbeat phase within the breath from inspiration onset |
| also | heart rate, breathing rate, rotation number (beats per breath), DFA α₁ |

**Cost** (`pycardio/costs.py`). Squared differences divided by the variance the
difference would have if the model had the recording's variability (both
sample sizes count; the spread always comes from the recording): z-scores for `rri` and
`resp`, a Mahalanobis distance on the Fourier coefficients for `rsa`, and a
two-sample Kolmogorov-Smirnov statistic plus the `rsa` and `resp` terms for
`cvc`.

**Fitting** (`pycardio/fitting.py`, `cmaes.py`, `mcmc.py`). CMA-ES (default),
differential evolution, Nelder-Mead or Powell on bound-normalised parameters,
with a fixed noise seed so the objective is deterministic; adaptive Metropolis
MCMC with R-hat and effective sample size for uncertainty.

## Config files

```json
{
  "subject": "f1y02",
  "feature": "rsa",
  "method": "cmaes",
  "free":  {"hr": [0.00075, 0.0009], "rsa1": null, "rsa2": null},
  "start": {"hr": 0.00077},
  "fixed": {"d2": 0.62, "d4": 0.43},
  "duration": 120, "transient": 10, "seed": 0, "maxfev": 2000,
  "mcmc": {"n_samples": 2000, "n_warmup": 1000, "n_chains": 4}
}
```

The 2018-format files in `config/` are translated on load; keys with no
equivalent (`LIB_PATH`, `nr_p`, `tol`, ...) are reported and ignored.

## Layout

```
pycardio/
├── _c/model.c, model_params.h   the model and its integrator (C, no dependencies)
├── model.py                      simulate(), parameters, Simulation
├── features.py                   RSA, CVC, breath timing, DFA
├── costs.py                      data-model discrepancy
├── cmaes.py, fitting.py          optimisation
├── mcmc.py                       posterior sampling
├── data.py                       PhysioNet loading, hand-picked epochs, caching
├── config.py, cli.py, plotting.py
└── data/fantasia/*.rng           hand-picked epochs per subject (opt-in)
config/                           run configurations (2018 format, still readable)
tests/                            pytest suite (no dataset needed)
```

## Limitations

See the "What has and has not been verified" section of
[CHANGES-AI.md](CHANGES-AI.md). In short: no 2018 result has been reproduced
with the new code, and only one subject has been fitted (three parameters
to three RSA numbers, which shows the machinery works, not that the model
explains the subject).

## Data and references

- Iyengar N. et al. (1996), Fantasia database, PhysioNet,
  <https://physionet.org/content/fantasia/>.
- Hansen N. (2016), *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772.
- Haario H., Saksman E., Tamminen J. (2001), *An adaptive Metropolis
  algorithm*, Bernoulli 7(2).
- Peng C.-K. et al. (1995), *Quantification of scaling exponents and crossover
  phenomena in nonstationary heartbeat time series*, Chaos 5(1).
