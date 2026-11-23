"""Figures comparing data and model features."""

from __future__ import annotations

import numpy as np

from .features import Features, beat_phase, respiratory_extrema

__all__ = ["plot_rsa", "plot_cvc", "plot_synchrogram", "plot_traces", "comparison_figure"]

DATA_COLOR = "#222222"
MODEL_COLOR = "#2a6fdb"


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_rsa(data: Features | None = None, model: Features | None = None, ax=None):
    """RR-interval deviation against respiratory phase, with the Fourier fits."""
    ax = _ax(ax)
    phi = np.linspace(0, 1, 200)
    for f, color, label in ((data, DATA_COLOR, "data"), (model, MODEL_COLOR, "model")):
        if f is None or f.rsa is None:
            continue
        c = f.rsa.coef[0]
        ax.plot(f.rsa_phase, f.rsa_rri - c, ".", ms=2, alpha=0.15, color=color)
        ax.plot(
            phi,
            f.rsa(phi) - c,
            color=color,
            lw=2,
            label=f"{label}: Q = {1000 * f.rsa.amplitude:.0f} ms",
        )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("respiratory phase (expiration onset = 0)")
    ax.set_ylabel("RR interval - mean (s)")
    ax.set_title("Respiratory sinus arrhythmia")
    ax.legend(loc="best", frameon=False)
    return ax


def plot_cvc(data: Features | None = None, model: Features | None = None, ax=None):
    """Empirical CDFs of the heartbeat phase within the breath."""
    ax = _ax(ax)
    for f, color, label in ((data, DATA_COLOR, "data"), (model, MODEL_COLOR, "model")):
        if f is None or f.cvc.size == 0:
            continue
        phase = np.sort(f.cvc)
        x = np.concatenate(([0], phase, [1]))
        y = np.concatenate(([0], np.arange(1, phase.size + 1) / phase.size, [1]))
        ax.step(x, y, where="post", color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, label="uniform")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("beat phase in breath (inspiration onset = 0)")
    ax.set_ylabel("CDF")
    ax.set_title("Cardioventilatory coupling")
    ax.legend(loc="best", frameon=False)
    return ax


def plot_synchrogram(time, resp, beat_times, ax=None, title="Synchrogram"):
    """Phase of each heartbeat in its breath against time."""
    ax = _ax(ax)
    _, troughs = respiratory_extrema(time, resp)
    t, ph = beat_phase(beat_times, np.asarray(time)[troughs])
    ax.plot(t, ph, "k.", ms=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("beat phase in breath")
    ax.set_title(title)
    return ax


def plot_traces(sim, t_window: tuple[float, float] | None = None):
    """Neuron firing rates and cardiovascular outputs of a simulation."""
    import matplotlib.pyplot as plt

    t = sim.t
    sel = slice(None) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    rows = [
        ("preii", "pre-I/I"),
        ("earlyi", "early-I"),
        ("auge", "aug-E"),
        ("posti", "post-I"),
        ("rampi", "ramp-I"),
        ("lung_volume", "lung volume"),
        ("bp", "BP (a.u.)"),
    ]
    fig, axes = plt.subplots(len(rows) + 1, 1, sharex=True, figsize=(9, 11))
    for ax, (col, label) in zip(axes, rows):
        ax.plot(t[sel], sim[col][sel], lw=0.8)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    bt = sim.beat_times
    keep = (bt >= t[sel][0]) & (bt <= t[sel][-1])
    rr = np.diff(bt)
    axes[-1].plot(bt[1:][keep[1:]], 60.0 / rr[keep[1:]], ".-", lw=0.8)
    axes[-1].set_ylabel("HR (bpm)", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def comparison_figure(data: Features, model: Features, title: str = ""):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_rsa(data, model, ax=a)
    plot_cvc(data, model, ax=b)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
