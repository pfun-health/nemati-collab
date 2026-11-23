"""Python interface to the compiled cardiorespiratory model.

The ODE right-hand side and its integrator are in ``pycardio/_c/model.c``.
This module loads the shared library with :mod:`ctypes`, builds the parameter
struct from the table compiled into the library (so the two can never drift
out of sync), and wraps the output in a :class:`Simulation`.

Units: the C model works in milliseconds. Everything returned to Python is in
seconds unless a name says otherwise.

ctypes releases the GIL while ``pc_simulate`` runs, so independent simulations
can run in parallel threads.
"""

from __future__ import annotations

import ctypes as ct
import glob
import importlib.machinery
import os
import shutil
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

__all__ = [
    "PARAM_NAMES",
    "PARAM_DOCS",
    "DEFAULT_PARAMS",
    "DEFAULT_BOUNDS",
    "OUTPUT_COLUMNS",
    "BEAT_PHASE",
    "Parameters",
    "Simulation",
    "SimulationError",
    "make_params",
    "rhs",
    "simulate",
]

_PKG_DIR = Path(__file__).resolve().parent

# Names of the 13 output columns, in order.
OUTPUT_COLUMNS = (
    "preii",  # firing rate of pre-I/I
    "earlyi",  # firing rate of early-I
    "auge",  # firing rate of aug-E
    "posti",  # firing rate of post-I
    "rampi",  # firing rate of ramp-I
    "preii_nap",  # NaP inactivation (pre-I/I)
    "earlyi_iad",  # adaptation (early-I, stochastic)
    "auge_iad",  # adaptation (aug-E)
    "posti_iad",  # adaptation (post-I)
    "lung_volume",
    "heart_phase",  # cycles
    "bp",
    "bp_smooth",
)

# Phase (mod 1) at which a heartbeat is registered: where the pulse
# ((1 + sin 2*pi*phi)/2)^2 rises through 1e-4. This is the threshold the 2018
# code used, now solved exactly instead of being detected sample by sample.
BEAT_PHASE = 0.75 + float(np.arccos(0.98)) / (2.0 * np.pi)

_ERRORS = {
    1: "step size underflow (the system is too stiff or the parameters are invalid)",
    2: "non-finite state (the solution diverged)",
    3: "exceeded the maximum number of steps in one output interval",
    4: "invalid arguments",
}


class SimulationError(RuntimeError):
    """Raised when the ODE solver fails."""


# --------------------------------------------------------------------------
# Library loading
# --------------------------------------------------------------------------


def _find_library() -> Path | None:
    env = os.environ.get("PYCARDIO_LIB")
    if env:
        return Path(env)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _PKG_DIR / f"_libmodel{suffix}"
        if candidate.exists():
            return candidate
    hits = sorted(glob.glob(str(_PKG_DIR / "_libmodel*.so")))
    return Path(hits[0]) if hits else None


def _compile_fallback() -> Path:
    """Compile the model into a cache directory (for source checkouts that
    were never installed). Needs a C compiler; the model has no other C
    dependencies."""
    src = _PKG_DIR / "_c" / "model.c"
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pycardio"
    cache.mkdir(parents=True, exist_ok=True)
    stamp = int(max(src.stat().st_mtime, (_PKG_DIR / "_c" / "model_params.h").stat().st_mtime))
    target = cache / f"libmodel-{stamp}.so"
    if target.exists():
        return target
    cc = (sysconfig.get_config_var("CC") or "cc").split()[0]
    if shutil.which(cc) is None:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise ImportError(
            "The compiled model library was not found and no C compiler is available. "
            "Install the package with `uv sync` or `pip install -e .`."
        )
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        tmp_target = Path(tmp) / target.name
        cmd = [cc, "-O3", "-std=c11", "-shared", "-fPIC", str(src), "-o", str(tmp_target), "-lm"]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp_target, target)
    return target


def _load_library() -> ct.CDLL:
    path = _find_library()
    if path is None:
        path = _compile_fallback()
    lib = ct.CDLL(str(path))
    lib.pc_param_count.restype = ct.c_int
    lib.pc_param_struct_size.restype = ct.c_size_t
    lib.pc_param_name.restype = ct.c_char_p
    lib.pc_param_name.argtypes = [ct.c_int]
    lib.pc_param_doc.restype = ct.c_char_p
    lib.pc_param_doc.argtypes = [ct.c_int]
    lib.pc_param_default.restype = ct.c_double
    lib.pc_param_default.argtypes = [ct.c_int]
    lib.pc_n_state.restype = ct.c_int
    lib.pc_n_output.restype = ct.c_int
    return lib


_lib = _load_library()

PARAM_NAMES: tuple[str, ...] = tuple(
    _lib.pc_param_name(i).decode() for i in range(_lib.pc_param_count())
)
PARAM_DOCS: dict[str, str] = {n: _lib.pc_param_doc(i).decode() for i, n in enumerate(PARAM_NAMES)}
_DEFAULTS: dict[str, float] = {n: _lib.pc_param_default(i) for i, n in enumerate(PARAM_NAMES)}


def _defaults() -> dict[str, float]:
    return dict(_DEFAULTS)


DEFAULT_PARAMS: Mapping[str, float] = MappingProxyType(_defaults())

# Plausible search ranges for the parameters that are usually fitted.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "hr": (0.00045, 0.002),
    "rsa1": (0.0, 0.0015),
    "rsa2": (0.0, 0.0014),
    "d2": (0.45, 0.75),
    "d4": (0.35, 0.65),
    "d5": (0.45, 0.75),
    "baro": (0.0, 0.5),
    "po": (0.0, 1.0),
    "taup": (500.0, 3000.0),
    "db": (0.0, 0.002),
    "rs1": (0.0, 0.01),
}


class Parameters(ct.Structure):
    """ctypes mirror of the C ``Parameters`` struct (all doubles)."""

    _fields_ = [(name, ct.c_double) for name in PARAM_NAMES]


if ct.sizeof(Parameters) != _lib.pc_param_struct_size():  # pragma: no cover
    raise ImportError("Parameter struct layout mismatch between Python and the C library")
_N_STATE = _lib.pc_n_state()
_N_OUT = _lib.pc_n_output()
assert _N_OUT == len(OUTPUT_COLUMNS)

_dbl_arr = np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS")
_lib.pc_simulate.restype = ct.c_int
_lib.pc_simulate.argtypes = [
    ct.POINTER(Parameters),
    ct.c_double,  # dt (ms)
    ct.c_int,  # n_out
    _dbl_arr,  # out
    ct.c_uint64,  # seed
    ct.c_double,  # rtol
    ct.c_double,  # atol
    ct.c_void_p,  # y0 (nullable)
    ct.c_void_p,  # y_final (nullable)
    ct.POINTER(ct.c_long),  # n_steps
]
_lib.pc_rhs.restype = None
_lib.pc_rhs.argtypes = [ct.POINTER(Parameters), _dbl_arr, _dbl_arr]
_lib.pc_default_initial_state.restype = None
_lib.pc_default_initial_state.argtypes = [_dbl_arr]


def make_params(overrides: Mapping[str, float] | None = None, **kwargs: float) -> Parameters:
    """Build a :class:`Parameters` struct from the defaults plus overrides.

    Unknown parameter names raise :class:`KeyError`. (The 2018 code set them
    as plain Python attributes, which the C code never saw.)
    """
    values = _defaults()
    for source in (overrides or {}), kwargs:
        for key, val in source.items():
            if key not in values:
                raise KeyError(f"unknown model parameter {key!r}")
            values[key] = float(val)
    return Parameters(**values)


def params_dict(p: Parameters) -> dict[str, float]:
    return {name: getattr(p, name) for name in PARAM_NAMES}


def default_initial_state() -> np.ndarray:
    """The 13 state variables; the last one is the early-I adaptation."""
    y0 = np.empty(_N_STATE)
    _lib.pc_default_initial_state(y0)
    return y0


def rhs(y: np.ndarray, params: Mapping[str, float] | Parameters | None = None) -> np.ndarray:
    """Evaluate the deterministic right-hand side dy/dt (per ms) at state ``y`` (13 values)."""
    p = params if isinstance(params, Parameters) else make_params(params)
    y = np.ascontiguousarray(y, dtype=float)
    if y.shape != (_N_STATE,):
        raise ValueError(f"state must have shape ({_N_STATE},)")
    dy = np.empty(_N_STATE)
    _lib.pc_rhs(ct.byref(p), y, dy)
    return dy


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class Simulation:
    """Output of :func:`simulate`.

    Attributes
    ----------
    t : sample times in seconds (after the transient was dropped, starting at 0)
    states : (n, 13) array; columns are named in :data:`OUTPUT_COLUMNS`
    params : the full parameter set used
    seed : noise seed
    y_final : final state (13 values); pass as ``y0`` to continue
    n_steps : accepted solver steps
    """

    t: np.ndarray
    states: np.ndarray
    params: dict[str, float]
    seed: int
    y_final: np.ndarray
    n_steps: int

    def __getitem__(self, name: str) -> np.ndarray:
        return self.states[:, OUTPUT_COLUMNS.index(name)]

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0]) if self.t.size > 1 else float("nan")

    @cached_property
    def beat_times(self) -> np.ndarray:
        """Heartbeat times (s): where the heart phase crosses k + BEAT_PHASE,
        located by linear interpolation between samples."""
        phi = self["heart_phase"]
        k = np.floor(phi - BEAT_PHASE)
        idx = np.flatnonzero(np.diff(k) > 0)
        target = k[idx + 1] + BEAT_PHASE
        frac = (target - phi[idx]) / (phi[idx + 1] - phi[idx])
        return self.t[idx] + frac * (self.t[idx + 1] - self.t[idx])

    def resampled(self, fs: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
        """Lung volume resampled at ``fs`` Hz (to match the data's sampling)."""
        step = 1.0 / fs
        tq = np.arange(self.t[0], self.t[-1] + 1e-12, step)
        return tq, np.interp(tq, self.t, self["lung_volume"])

    def features(self, fs: float = 250.0, **kwargs):
        """Extract the same features as from data (see :mod:`pycardio.features`)."""
        from .features import extract_features

        time, resp = self.resampled(fs)
        return extract_features(time, resp, self.beat_times, **kwargs)


def simulate(
    params: Mapping[str, float] | Parameters | None = None,
    duration: float = 120.0,
    *,
    dt: float = 1.0,
    seed: int = 0,
    transient: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    y0: np.ndarray | None = None,
) -> Simulation:
    """Integrate the model.

    Parameters
    ----------
    params : overrides of :data:`DEFAULT_PARAMS` (dict) or a full struct
    duration : simulated time (s) kept in the output
    dt : output step (ms)
    seed : seed of the noise generator; equal seeds give identical runs
    transient : time (s) simulated first and dropped from the output
    rtol, atol : error tolerances of the Dormand-Prince 5(4) solver
    y0 : initial state (13 values, see ``model.c``); defaults to the model's
    """
    p = params if isinstance(params, Parameters) else make_params(params)
    if duration <= 0 or dt <= 0 or transient < 0:
        raise ValueError("duration and dt must be positive and transient non-negative")
    n_drop = int(round(transient * 1000.0 / dt))
    n_keep = int(round(duration * 1000.0 / dt)) + 1
    n_out = n_drop + n_keep
    out = np.empty((n_out, _N_OUT))
    y_final = np.empty(_N_STATE)
    n_steps = ct.c_long(0)
    if y0 is not None:
        y0 = np.ascontiguousarray(y0, dtype=float)
        if y0.shape != (_N_STATE,):
            raise ValueError(f"y0 must have shape ({_N_STATE},)")
    status = _lib.pc_simulate(
        ct.byref(p),
        float(dt),
        n_out,
        out,
        int(seed) % (1 << 64),
        float(rtol),
        float(atol),
        y0.ctypes.data if y0 is not None else None,
        y_final.ctypes.data,
        ct.byref(n_steps),
    )
    if status != 0:
        raise SimulationError(_ERRORS.get(status, f"solver error {status}"))
    t = np.arange(n_keep) * (dt / 1000.0)
    return Simulation(
        t=t,
        states=out[n_drop:],
        params=params_dict(p),
        seed=int(seed),
        y_final=y_final,
        n_steps=int(n_steps.value),
    )
