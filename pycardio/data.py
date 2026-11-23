"""Loading PhysioNet recordings (Fantasia) and turning them into features.

The Fantasia database (Iyengar et al. 1996, https://physionet.org/content/fantasia/)
has 120-minute recordings of ECG, respiration and, for some subjects, blood
pressure, sampled at 250 Hz, with beat annotations in ``.ecg`` files.

Data directory layout: ``<data_dir>/fantasia/f1y01.hea``, ``.dat``, ``.ecg``.
The default ``data_dir`` is ``$PYCARDIO_DATA_DIR`` or ``~/Public`` (the 2018
default).

Selected epochs: ``<subject>.rng`` files list (start, end) sample ranges that
were picked by eye in 2018 (``select_epochs.py``). For some subjects they cover
almost the whole record, for others only a few minutes, so they are **not**
applied unless asked for (``use_ranges=True``). They are looked up in the
database folder first, then among the files shipped with the package
(``pycardio/data/fantasia``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import Features, extract_features, smooth

__all__ = [
    "Recording",
    "default_data_dir",
    "download",
    "list_subjects",
    "load_recording",
    "recording_features",
    "subject_features",
]

_BUNDLED = Path(__file__).resolve().parent / "data"


def default_data_dir() -> Path:
    return Path(os.environ.get("PYCARDIO_DATA_DIR", Path.home() / "Public")).expanduser()


def download(database: str = "fantasia", data_dir: str | Path | None = None) -> Path:
    """Download a PhysioNet database (about 650 MB for Fantasia)."""
    import wfdb

    dest = Path(data_dir or default_data_dir()).expanduser() / database
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(database, dl_dir=str(dest))
    return dest


def list_subjects(database: str = "fantasia", data_dir: str | Path | None = None) -> list[str]:
    folder = Path(data_dir or default_data_dir()).expanduser() / database
    return sorted(p.stem for p in folder.glob("*.hea"))


@dataclass
class Recording:
    subject: str
    fs: float
    time: np.ndarray  # s
    resp: np.ndarray
    beat_times: np.ndarray  # s
    bp: np.ndarray | None
    ranges: np.ndarray | None  # (k, 2) sample ranges of clean signal

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def _find_ranges(subject: str, folder: Path, database: str) -> np.ndarray | None:
    for candidate in (folder / f"{subject}.rng", _BUNDLED / database / f"{subject}.rng"):
        if candidate.exists():
            r = np.loadtxt(candidate, ndmin=2)
            return r.astype(np.int64) if r.size else None
    return None


def load_recording(
    subject: str, database: str = "fantasia", data_dir: str | Path | None = None
) -> Recording:
    """Read one subject's respiration, blood pressure and beat annotations."""
    import wfdb

    folder = Path(data_dir or default_data_dir()).expanduser() / database
    base = str(folder / subject)
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "ecg")
    names = [n.upper() for n in rec.sig_name]
    if "RESP" not in names:
        raise ValueError(f"{subject}: no RESP signal (signals: {rec.sig_name})")
    resp = rec.p_signal[:, names.index("RESP")].astype(float)
    bp = rec.p_signal[:, names.index("BP")].astype(float) if "BP" in names else None
    fs = float(rec.fs)
    time = np.arange(rec.sig_len) / fs
    samples = np.asarray(ann.sample)
    samples = samples[(samples >= 0) & (samples < rec.sig_len)]
    return Recording(
        subject=subject,
        fs=fs,
        time=time,
        resp=resp,
        beat_times=samples / fs,
        bp=bp,
        ranges=_find_ranges(subject, folder, database),
    )


def _finite_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive-exclusive index runs where ``mask`` is True, at least ``min_len`` long."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    runs = list(zip(edges[::2], edges[1::2]))
    return [(int(a), int(b)) for a, b in runs if b - a >= min_len]


def recording_features(
    rec: Recording,
    smooth_window: float = 1.0,
    use_ranges: bool = False,
    min_segment: float = 30.0,
) -> Features:
    """Features of a recording.

    The respiration signal is smoothed with a ``smooth_window``-second Nuttall
    window (as in 2018). Missing samples (NaN) split the recording into
    separate segments instead of discarding everything before the last NaN,
    which is what the 2018 code did. If ``use_ranges`` and a ``.rng`` file
    exists, only the listed stretches are analysed.
    """
    finite = np.isfinite(rec.resp)
    if use_ranges and rec.ranges is not None:
        allowed = np.zeros_like(finite)
        for a, b in rec.ranges:
            allowed[max(a, 0) : min(b, finite.size)] = True
        finite &= allowed
    using = use_ranges and rec.ranges is not None
    # hand-picked epochs can be short; keep any that hold a few breaths
    runs = _finite_runs(finite, int((10.0 if using else min_segment) * rec.fs))
    if not runs:
        raise ValueError(f"{rec.subject}: no clean segment of at least {min_segment} s")
    resp = np.full_like(rec.resp, np.nan)
    width = max(1, int(round(smooth_window * rec.fs)))
    for a, b in runs:
        resp[a:b] = smooth(rec.resp[a:b], width)
    segments = [(rec.time[a], rec.time[b - 1]) for a, b in runs]
    feats = extract_features(rec.time, resp, rec.beat_times, segments=segments)
    feats.meta.update(
        subject=rec.subject,
        fs=rec.fs,
        n_segments=len(runs),
        used_ranges=bool(use_ranges and rec.ranges is not None),
    )
    if rec.bp is not None:
        bp = rec.bp[finite]
        feats.meta["bp_mean"] = float(np.nanmean(bp)) if bp.size else float("nan")
    return feats


def subject_features(
    subject: str,
    database: str = "fantasia",
    data_dir: str | Path | None = None,
    cache: bool = True,
    refresh: bool = False,
    **kwargs,
) -> Features:
    """Features of one subject, cached as ``<db>/pycardio-cache/<subject>.npz``."""
    folder = Path(data_dir or default_data_dir()).expanduser() / database
    cache_file = folder / "pycardio-cache" / f"{subject}.npz"
    if cache and cache_file.exists() and not refresh and not kwargs:
        return Features.from_npz(cache_file)
    feats = recording_features(load_recording(subject, database, data_dir), **kwargs)
    if cache and not kwargs:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        feats.to_npz(cache_file)
    return feats
