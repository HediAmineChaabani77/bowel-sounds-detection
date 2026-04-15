"""Feature extraction: log-mel, MFCC, and PCEN.

All parameters match those used in the notebooks so cached features can be
re-used across training runs.
"""

from __future__ import annotations

import librosa
import numpy as np

from .config import FMAX, FMIN, HOP, N_FFT, N_MELS, SR


def compute_mel(x: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return logmel


def compute_mfcc(x: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=N_MELS)
    return mfcc.astype(np.float32)


def compute_pcen(x: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0,
    )
    pcen = librosa.pcen(
        mel * (2 ** 31),
        sr=sr,
        hop_length=HOP,
        gain=0.98,
        bias=2.0,
        power=0.5,
        time_constant=0.4,
    )
    return pcen.astype(np.float32)


def normalize_feat(feat: np.ndarray) -> np.ndarray:
    """Per-utterance zero-mean unit-variance normalization."""
    mu = feat.mean()
    sd = feat.std() + 1e-6
    return ((feat - mu) / sd).astype(np.float32)
