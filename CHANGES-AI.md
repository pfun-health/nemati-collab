# What changed in the AI-assisted overhaul

This file separates the original work from the AI-assisted rewrite, so that
anyone reading the repository can tell which is which.

| | |
|---|---|
| **Original work** | Everything up to and including commit `bad301d` ("a few bugfixes by hand for old-times sake"), on `main`. Written 2017-2018 by Robbie Capps (with William Barnett on the model equations) in the Molkov Lab, Georgia State University, and touched up by hand in November 2026. The README at that commit notes that Claude Code had already been used for packaging conveniences (`pyproject.toml`, `uv.lock`, moving shell scripts into `scripts/`). |
| **AI-assisted overhaul** | Every commit after `bad301d` on the `overhaul` branch, made on 2026-11-23 by Claude (Anthropic) through Claude Code, at the author's request. Those commits carry a `Co-Authored-By: Claude` trailer. |

To see the original code: `git show bad301d:<path>` or `git checkout bad301d`.
To see exactly what the overhaul changed: `git diff bad301d overhaul`.

## What is still the original science

- **The model equations.** The five-population respiratory CPG, heart-phase
  oscillator and blood-pressure loop in `pycardio/_c/model.c` are the 2018
  equations, term for term. A test (`tests/test_model.py`) checks the C code
  against an independent NumPy transcription of the 2018 source to 1e-12.
- **All parameter values and initial conditions** (`parameters.h`,
  `parameters.py`).
- **The feature definitions**: RSA as RR interval against respiratory phase
  from expiration onset with a first-order Fourier fit; CVC as heartbeat phase
  within the breath from inspiration onset; inspiration/expiration durations;
  the heartbeat marker on the model's cardiac pulse.
- **The overall approach**: extract the same features from data and model,
  score the mismatch, optimise, then sample.
- **The configs** in `config/` and the hand-picked epochs (`*.rng`), which
  are unchanged apart from the `.rng` files moving into the package.

## Bugs found in the original code

These are defects in the code as it stood at `bad301d`, each fixed in the
overhaul. Severity is my judgement of the effect on results.

### Affected results

| # | Where (at `bad301d`) | Problem | Fix |
|---|---|---|---|
| 1 | `process_data/db.py` `calc_descriptive` | "Rotation number" was computed as mean RR interval (s) divided by breathing rate (1/s), i.e. s², not beats per breath. `hr` held the mean RR interval but was printed as "bps". | `Features.rotation_number` = heart rate / breathing rate. |
| 2 | `c_process-data/calc_rsa.c` + `db.py` | The C loop fills `nr_beats - 1` rows; the last row stayed zero and passed the `> -998` filter, so every RSA dataset contained one spurious point (phase 0, RRI 0, time 0). | Vectorised NumPy implementation; no sentinel values. |
| 3 | `c_process-data/calc_dfa.c` | Not DFA: the series was never integrated; the loop advanced `i` twice so every other window was skipped; the running sum was divided by N inside the loop; and `F[max_N - min_N]` was written one past the end of the buffer (heap overflow). | `features.dfa`: standard DFA (Peng et al. 1995). Checked on white noise (alpha = 0.5), a random walk (1.5), and real data (about 1.0 for young healthy subjects). |
| 4 | `fit_model/cost_functions.py` `CF_rsa` | Compared the data's fitted RSA curve (sorted by phase) with the model's raw, unsorted RRI deviations element by element after decimating one array to the other's length. The two arrays were not aligned in phase, so the cost did not measure RSA mismatch. | `costs.cost_rsa`: Mahalanobis distance between the fitted Fourier coefficients, with the covariance of their difference. |
| 5 | `cost_functions.py` `CF_cvc` | `Dmax` compared a CDF with the latency values themselves; the components were multiplied together; the rotation-number term was computed and then overwritten. | `costs.cost_cvc`: two-sample Kolmogorov-Smirnov statistic scaled by effective sample size, added to the RSA and breath-timing terms. Verified against `scipy.stats.ks_2samp`. |
| 6 | `db.py` `calc_RSA` | `rri_var = (sum of parameter variances / Q²) / sqrt(n)` is not the variance of any quantity; `curve_fit` had no intercept and a hand-set sigma. | Ordinary least squares with intercept; covariance from the residuals; amplitude standard error by the delta method. A test confirms the standard errors are calibrated. |
| 7 | `model/model_output.py` | `rri_times[rri_times > rri_dev.min()]` compared times with RRI deviations, and filtered the four arrays with different masks. | Removed; features come from one code path. |
| 8 | `fit_model/adaptive_stepsize.py` `RandomDisplacement` | `x += normal(x, stepsize)` adds a draw *centred on x*, so every basin-hopping step roughly doubled the parameters. | Module removed; see "Optimisers". |
| 9 | `sample_model/sample.py` | The "sampler" accepted a candidate only if `1/cost` was at least the current value: a hill climber, not MCMC, so its output was not a posterior sample. `SamplerMH` accepted on `L_new / L >= U` with `L = 1/cost`, which is not a Metropolis ratio for any stated likelihood. Off-diagonal proposal covariance was ignored. | `mcmc.py`: adaptive Metropolis (Haario et al. 2001) with `log p = -cost/2`, adaptation frozen after warm-up, split-R-hat and ESS reported. Tested against a known correlated Gaussian. |
| 10 | `sample.py` `_runModel` | Each "simulation" was four independent quarter-length runs (same initial condition, so four copies of the start-up transient) concatenated as one time series. | One continuous simulation; parallelism is across evaluations instead. |
| 11 | `c_model/resp_odes.c` | `srand(time(0))` on every call: runs were not reproducible, and evaluations started within the same second shared a noise sequence. The objective was therefore random, which none of the optimisers account for. | Seedable xoshiro256** generator; fits use one fixed seed (common random numbers), so the objective is deterministic. |
| 12 | `db.py` `check_for_..._nans` | On any NaN in the respiration signal, *everything before the last NaN* was discarded; the index correction was also applied to the BP *values* as if they were indices. | NaNs split the record into segments; nothing else is lost. |
| 13 | `fit_model/optimize.py` `covar` | Eigenvalues of the covariance matrix reported as the *squares* of its singular values. | Removed (covariance comes from CMA-ES or the MCMC chain). |
| 14 | `optimize.py` `ga_acceptance` | "Metropolis-Hastings" acceptance used the ratio of two negative log-likelihoods. | Removed. |

### Did not run, or silently did nothing

| # | Where | Problem |
|---|---|---|
| 15 | `c_fit-model/*` (CMA-ES in C) | Never compiled: undefined identifiers (`M`, `C_trui1`), `char` compared with string literals, `rt->costs[];`, declarations like `double *result->x = ...`, missing semicolons, `setup()` assigning to locals that shadow the globals, a stack array returned through a struct. The hand fix in `bad301d` added a write through an uninitialised pointer (`mat_flat`). Replaced by `pycardio/cmaes.py`. |
| 16 | `c_utils/cvc_utils.c` | Did not compile after `bad301d` (`DistTable` used without `struct`); its comparator returned `floor(difference)`, which is not a valid ordering for `qsort`. Unused by anything that worked; removed. |
| 17 | `c_model/resp_odes.c` | No longer compiles with GCC 14+ (incompatible function-pointer type is now an error). |
| 18 | `model/parameters.py` `setParams` | `setattr` on a ctypes struct with an unknown name just creates a Python attribute. Configs that set model constants (`gL`, `C`, `b23`, ...) under `defaultParams` were silently ignored. (The values in the shipped configs equal the compiled constants, so no past result changed.) Unknown names now raise; all 57 constants are real parameters. |
| 19 | `resp_odes.c` `integrate` | The time-step argument was ignored (output was always 1 ms apart, with `te` in seconds against a model in ms); a solver error printed a message and left zeros in the output. |
| 20 | `load_data/load_data.py` | `np.load(path)[()]` fails on any NumPy from the last several years (`allow_pickle`). |
| 21 | `optimize.py` | `print_callback` read `self.utils.data['cdf']`, never set (KeyError on the first accepted basin-hopping step); `if not self.fun` treated a cost of 0 as unset; the `cmaes` branch called a tuple. |
| 22 | `sample.py`, `load_data/_physionet_db.py`, `model/pyresp.py` | References to names that do not exist (`CF_simple`, `get_data`, `mkmodel.MkModel`). |
| 23 | `Dockerfile` | Referenced a `src/` directory that does not exist, downloaded a 2017 MKL installer from a dead URL, and had `RUN` nested inside a `RUN` continuation. |
| 24 | `tests/` | Not tests: scripts with hard-coded home-directory paths that needed the dataset and wrote figures. No automated checks existed. |

One oddity I left alone: `config/fantasia_f2y03_rsa_ga.json` has an upper
bound of `0.0086` for `hr`, ten times the others; it looks like a typo for
`0.00086` but I did not change your configs.

## What was rewritten

### Model (`pycardio/_c/`, `pycardio/model.py`)

- **Integrator.** GSL's `rk2` driver replaced by a self-contained
  Dormand-Prince 5(4) integrator with standard error control. No GSL, no
  WFDB C library, no `python-config`: a C compiler is the only build
  requirement. It agrees with SciPy's DOP853 to 2e-6 over two breaths and is
  about five times faster than the old build (120 s of model time in about
  0.14 s).
- **How close to the old output?** Against the original binary (noise off, 60 s):
  the same number of bursts in every population, burst onsets within 16 ms,
  heart phase within 0.003 cycles. That difference is the old solver's error:
  the new one changes by less than 0.007 between tolerances of 1e-6 and 1e-10.
- **Noise.** The early-I adaptation is now a state variable integrated with
  the others, with noise added as `sigma * sqrt(dt) * N(0,1)`,
  `sigma = rs1 / sqrt(12)`. At the 1 ms step the old code always used, this
  has the same variance as the old uniform kick, so existing `rs1` values keep
  their meaning; unlike before, the statistics no longer change with the step.
  (The old kick was uniform; the new one is Gaussian with equal variance.)
- **Parameters.** One X-macro table in `model_params.h` generates the C
  struct, the defaults and the names; Python builds its ctypes struct from the
  compiled table, so the two cannot drift apart.
- **Heartbeat detection.** The old sample-by-sample threshold on the pulse is
  solved for exactly (phase `0.75 + arccos(0.98)/2pi`) and beat times are
  interpolated, instead of being rounded to the 4 ms grid.

### Features (`pycardio/features.py`)

- The C routines for RSA and CVC were O(beats x breaths) loops; they are now
  `searchsorted` lookups in NumPy.
- Everything works on times in seconds, not sample indices, so data and model
  share one code path (the old code had two copies that had diverged).
- Respiratory extrema use `scipy.signal.find_peaks` with a minimum separation
  and a prominence threshold relative to the signal range, instead of
  `argrelmax` with a fixed 250-sample order. **This is a change in method**:
  breath counts can differ slightly from 2018.
- Plausibility limits are applied uniformly: RR intervals 0.25-4 s, breaths
  1-10 s. (2018: RR <= 4 s for RSA, breaths <= 10 s for CVC only.)
- Smoothing reflects at the edges instead of zero-padding.
- The `.rng` files (epochs picked by eye with `select_epochs.py`) can be
  applied with `use_ranges=True` / `--ranges`. As in 2018, the default
  analyses the whole record: for several subjects the picked epochs cover
  only a few minutes.
- Features are saved as plain `.npz` arrays instead of pickled dicts.

### Cost functions (`pycardio/costs.py`)

Every term is a squared difference over the variance that difference would
have if the model had the recording's variability:
`s_data² (1/n_data + 1/n_model)`. Both sample sizes count, but the spread is
always the recording's, so a model cannot score better by being noisier.
Terms add, and half the total is a Gaussian negative log-likelihood (the KS
term is a distance, not a likelihood; this is stated in the code).

### Optimisers (`pycardio/cmaes.py`, `pycardio/fitting.py`)

- CMA-ES written from Hansen's 2016 tutorial; tested on sphere, Rosenbrock, an
  ellipsoid with condition number 1e6, and a boundary optimum.
- Optimisation runs on the unit cube. The raw parameters span seven orders of
  magnitude (`hr` about 1e-3, `taup` about 1e3).
- Populations are evaluated on a thread pool (the C solver releases the GIL).
- `differential_evolution`, `nelder-mead` and `powell` from SciPy remain
  available. `basinhopping` with the custom step class is gone; configs that
  ask for it run CMA-ES with a warning.
- A 10 s start-up transient is now discarded by default in new configs.
  Translated 2018 configs keep `transient = 0` to match what they did.

### Everything else

- `wfdb==1.3.7` (2018) replaced by current `wfdb`, which removes the pins on
  `numpy<2`, `pandas<2`, `scipy<1.14`. Tested on NumPy 2.4 / SciPy 1.17 / Python
  3.11 locally and 3.10, 3.12, 3.13 in CI.
- `google-cloud-storage` dropped: `_zip_db.py` only listed blobs in a
  `sepsis-data` bucket and was otherwise commented out.
- One CLI (`pycardio download | features | simulate | fit | sample | plot`)
  replaces `build.sh`, `scripts/*.sh` and the scripts under `tests/`.
- 2018 config files load unchanged (`pycardio/config.py` translates them and
  reports the keys it ignores).
- 67 automated tests; GitHub Actions workflow replacing `.gitlab-ci.yml`.

## Removed files and where their function went

| Removed | Replacement |
|---|---|
| `pycardio/c_model/` | `pycardio/_c/model.c`, `model_params.h` |
| `pycardio/c_process-data/` | `pycardio/features.py` |
| `pycardio/c_fit-model/`, `c_utils/` | `pycardio/cmaes.py` |
| `pycardio/model/` | `pycardio/model.py` |
| `pycardio/process_data/`, `load_data/` | `pycardio/features.py`, `pycardio/data.py` |
| `pycardio/fit_model/` | `pycardio/costs.py`, `pycardio/fitting.py` |
| `pycardio/sample_model/` | `pycardio/mcmc.py` |
| `tests/test_model.py`, `plot_result.py`, `scripts/phase_time_plot.py` | `pycardio simulate`, `pycardio plot`, `pycardio/plotting.py` |
| `tests/test_optimize*.py`, `scripts/optimize.sh` | `pycardio fit` |
| `tests/test_sampler*.py`, `plot_sampler.py`, `scripts/sample*.sh` | `pycardio sample` |
| `tests/test_process_data.py`, `test_compute_all_stats.py`, `test_dfa.py` | `pycardio features` |
| `scripts/process*.py`, `select_epochs.py`, `tests/test_baro.py`, `test_bpi.py` | **Not ported.** Exploratory BP-latency and epoch-selection analyses built on a second copy of the old `DB` class. Recover with `git show bad301d:scripts/process_bp.py` etc. |
| `fantasia/*.rng` | moved to `pycardio/data/fantasia/` |
| `build.sh`, `scripts/build.sh` | `uv sync` / `pip install .` |

## What has and has not been verified

Verified in this session:

- All 67 tests pass locally and in CI (Python 3.10, 3.12, 3.13); `ruff` is clean.
- All 40 Fantasia records were downloaded from PhysioNet and run through the
  loader and feature extraction without a failure. Group medians come out as
  expected from the literature (RSA amplitude about 31 ms in the young group
  and 10 ms in the elderly group; DFA alpha-1 near 1).
- A fit of `f2y03` (three cardiac parameters to the RSA feature, CMA-ES,
  931 evaluations, 49 s) reproduces the recording's mean RR interval and RSA
  amplitude and phase essentially exactly, and stays there under three other
  noise seeds. Note what this is: three parameters matched to three numbers.
  It shows the optimiser and cost work; it is not evidence that the model
  explains the subject. Quantities that were not fitted do not match
  (breathing rate 13.6 vs 17.4 per minute; the model shows almost no
  cardioventilatory phase locking where the recording shows some).
- The 2018 config `fantasia_f2y03_rsa_ga.json` loads and runs unmodified
  through `pycardio fit`, `sample` and `plot`.

Not verified:

- **No 2018 result was reproduced.** The saved outputs of the original
  pipeline are not in the repository, the cost functions are different by
  design, and several original components did not run. Fitted parameter
  values from this code should not be expected to match 2018 values.
- Only one subject and one feature (`rsa`) have been fitted. The `cvc` and
  `resp` fits, and MCMC beyond short smoke runs, are untested on real data.
- RSA standard errors assume independent residuals. Successive RR intervals
  are correlated, so on real data they are somewhat too small, and posterior
  widths from `pycardio sample` inherit that.
- The Dockerfile was written but not built.
- Windows and macOS builds are untested.

## Two mistakes of mine, caught before release

Both were found when the first all-subject run looked wrong, and both are
fixed in the commits on this branch.

1. **The first cost functions could be gamed.** They divided by the sum of the
   data's and the *model's* squared standard errors. An optimiser can then
   lower the cost by making the model noisier: the first example fit "won"
   with an RSA of 158 ms against 20 ms in the recording. The variance now
   always comes from the recording (see "Cost functions" above), and a test
   pins this.
2. **The `.rng` files were first applied by default** on my assumption that
   they marked all artefact-free signal. They are epochs picked by eye, and
   for some subjects cover only a few minutes (1.2 of 120 for `f2o05`). They
   are opt-in again, as in 2018.

## A file I deleted by mistake

While clearing the emptied `fantasia/` folder I also removed
`fantasia/bp_list.txt`, an untracked, git-ignored file, so it is not in git
history. I had seen only its first three lines (`f2y09.dat`, `f2o05.dat`,
`f2o01.dat`). I regenerated it from the PhysioNet headers as the 20 records
that contain a BP channel (all of `f2*`), in sorted order. If the original was
a hand-picked subset or its order mattered, that information is lost.
