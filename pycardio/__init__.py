"""pycardio: model-based analysis of cardiorespiratory coupling.

Typical use::

    import pycardio as pc

    data = pc.subject_features("f1y02")            # features of a recording
    sim = pc.simulate({"hr": 0.0009}, duration=120)  # run the model
    model = sim.features()                           # same features, from the model
    pc.total_cost(data, model, "rsa")                # discrepancy

    problem = pc.FitProblem(data, free={"hr": None, "rsa1": None, "rsa2": None})
    result = pc.fit(problem, method="cmaes", maxfev=1500)
"""

from .costs import FEATURES, total_cost
from .data import list_subjects, load_recording, recording_features, subject_features
from .features import Features, extract_features
from .fitting import FitProblem, FitResult, fit
from .mcmc import sample
from .model import (
    DEFAULT_BOUNDS,
    DEFAULT_PARAMS,
    PARAM_DOCS,
    PARAM_NAMES,
    Simulation,
    SimulationError,
    make_params,
    simulate,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_PARAMS",
    "FEATURES",
    "Features",
    "FitProblem",
    "FitResult",
    "PARAM_DOCS",
    "PARAM_NAMES",
    "Simulation",
    "SimulationError",
    "extract_features",
    "fit",
    "list_subjects",
    "load_recording",
    "make_params",
    "recording_features",
    "sample",
    "simulate",
    "subject_features",
    "total_cost",
]
