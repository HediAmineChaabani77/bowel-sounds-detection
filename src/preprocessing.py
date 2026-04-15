"""Audio I/O, denoising, bandpass, label parsing and frame target building."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .config import (
    BP_HIGH,
    BP_LOW,
    BP_ORDER,
    C2I,
    FT,
    LABEL_MAP,
    NC,
    NON_TARGET,
    ONSET_SIGMA,
    SR,
)


def load_audio(path: str | Path, target_sr: int = SR) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def noise_reduce(x: np.ndarray, sr: int = SR) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=x, sr=sr).astype(np.float32)


def rms_normalize(x: np.ndarray, target_rms: float = 0.02) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    if rms < 1e-9:
        return x
    return (x * (target_rms / rms)).astype(np.float32)


def bandpass(
    x: np.ndarray,
    sr: int = SR,
    low: float = BP_LOW,
    high: float = BP_HIGH,
    order: int = BP_ORDER,
) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def parse_labels(path: str | Path) -> List[Tuple[float, float, str]]:
    events: List[Tuple[float, float, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            raw = parts[2].strip().lower()
            if raw in NON_TARGET:
                continue
            cls = LABEL_MAP.get(raw)
            if cls is None:
                continue
            if end > start:
                events.append((start, end, cls))
    events.sort(key=lambda e: e[0])
    return events


def build_targets(
    events: List[Tuple[float, float, str]],
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-frame SED target (n_frames, NC) and onset target (n_frames,).

    The onset target is a Gaussian bump placed on the onset frame of every
    single-burst event, with standard deviation ``ONSET_SIGMA`` frames.
    """
    sed = np.zeros((n_frames, NC), dtype=np.float32)
    onset = np.zeros((n_frames,), dtype=np.float32)
    for start, end, cls in events:
        c = C2I[cls]
        f0 = max(0, int(round(start / FT)))
        f1 = min(n_frames, int(round(end / FT)) + 1)
        if f1 <= f0:
            continue
        sed[f0:f1, c] = 1.0
        if cls == "b":
            center = f0
            lo = max(0, center - int(4 * ONSET_SIGMA))
            hi = min(n_frames, center + int(4 * ONSET_SIGMA) + 1)
            idx = np.arange(lo, hi)
            bump = np.exp(-0.5 * ((idx - center) / ONSET_SIGMA) ** 2).astype(np.float32)
            onset[lo:hi] = np.maximum(onset[lo:hi], bump)
    return sed, onset
