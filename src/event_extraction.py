"""Frame probability -> event list post-processing.

Includes peak-picking for single-burst events, contiguous-run extraction
for multiple-burst and harmonic events, median filtering, and a validation
grid search for per-class thresholds and filter sizes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from .config import C2I, CLASSES, FT, MIN_DUR, NC, ONSET_SIGMA


def _extract_b(onset_scores: np.ndarray, threshold: float, min_dur_s: float) -> List[Tuple[float, float]]:
    smoothed = gaussian_filter1d(onset_scores, sigma=ONSET_SIGMA)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=5)
    half = max(1, int(round(min_dur_s / FT / 2)))
    events: List[Tuple[float, float]] = []
    for p in peaks:
        start = max(0, p - half) * FT
        end = (p + half) * FT
        events.append((float(start), float(end)))
    return events


def _extract_runs(
    scores: np.ndarray,
    threshold: float,
    filt_size: int,
    min_dur_s: float,
) -> List[Tuple[float, float]]:
    if filt_size > 1:
        scores = median_filter(scores, size=filt_size)
    mask = (scores >= threshold).astype(np.uint8)
    events: List[Tuple[float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j < n and mask[j] == 1:
            j += 1
        start = i * FT
        end = j * FT
        if (end - start) >= min_dur_s:
            events.append((float(start), float(end)))
        i = j
    return events


def extract_events(
    sed_probs: np.ndarray,
    onset_probs: np.ndarray,
    thresholds: Dict[str, float],
    filt_sizes: Dict[str, int],
) -> List[Tuple[float, float, str]]:
    """Extract events from per-frame probabilities.

    ``sed_probs``: ``(n_frames, NC)``.
    ``onset_probs``: ``(n_frames,)``.
    """
    events: List[Tuple[float, float, str]] = []
    for b in _extract_b(onset_probs, thresholds["b"], MIN_DUR["b"]):
        events.append((b[0], b[1], "b"))
    for cls in ("mb", "h"):
        c = C2I[cls]
        runs = _extract_runs(sed_probs[:, c], thresholds[cls], filt_sizes[cls], MIN_DUR[cls])
        for start, end in runs:
            events.append((start, end, cls))
    events.sort(key=lambda e: e[0])
    return events


def tune_on_val(
    val_sed_probs: np.ndarray,
    val_onset_probs: np.ndarray,
    val_events: List[Tuple[float, float, str]],
    score_fn: Callable[[List[Tuple[float, float, str]], List[Tuple[float, float, str]]], Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    filt_grid = [1, 5, 11, 21]
    thr_grid = np.round(np.arange(0.05, 0.86, 0.05), 2).tolist()
    best_thr: Dict[str, float] = {"b": 0.5, "mb": 0.5, "h": 0.5}
    best_filt: Dict[str, int] = {"b": 1, "mb": 1, "h": 1}
    for cls in CLASSES:
        best_score = -1.0
        for filt in filt_grid:
            for thr in thr_grid:
                thresholds = dict(best_thr)
                filt_sizes = dict(best_filt)
                thresholds[cls] = float(thr)
                filt_sizes[cls] = int(filt)
                preds = extract_events(val_sed_probs, val_onset_probs, thresholds, filt_sizes)
                metrics = score_fn(preds, val_events)
                f1 = metrics.get(f"f1_{cls}", 0.0)
                if f1 > best_score:
                    best_score = f1
                    best_thr[cls] = float(thr)
                    best_filt[cls] = int(filt)
    return best_thr, best_filt


def predict_full_mel(model, feat: np.ndarray, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Run a frame-level model over a full log-mel feature stream.

    ``feat``: ``(n_mels, n_frames)``. Returns ``sed_probs`` and ``onset_probs``.
    """
    import torch
    model.eval()
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        sed_logits, onset_logits = model(x)
    sed_probs = torch.sigmoid(sed_logits).squeeze(0).cpu().numpy()
    onset_probs = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
    return sed_probs, onset_probs


def predict_full_raw(model, waveform: np.ndarray, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Run a waveform-input (pretrained encoder) model on a full clip."""
    import torch
    model.eval()
    x = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        sed_logits, onset_logits = model(x)
    sed_probs = torch.sigmoid(sed_logits).squeeze(0).cpu().numpy()
    onset_probs = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
    return sed_probs, onset_probs
