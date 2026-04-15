"""PyTorch datasets for frame-level and waveform models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NC, SEG_DUR, SEG_HOP, SR, TARGET_NF


def temporal_split(
    n_frames: int,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[slice, slice, slice]:
    """Return contiguous train/val/test slices over the frame axis."""
    r_tr, r_va, _ = ratios
    i1 = int(round(n_frames * r_tr))
    i2 = int(round(n_frames * (r_tr + r_va)))
    return slice(0, i1), slice(i1, i2), slice(i2, n_frames)


def segment_frame_indices(n_frames: int, seg_len: int = TARGET_NF, hop_len: int = 25 * int(SEG_HOP)) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if n_frames <= seg_len:
        out.append((0, n_frames))
        return out
    start = 0
    while start + seg_len <= n_frames:
        out.append((start, start + seg_len))
        start += hop_len
    if out[-1][1] < n_frames:
        out.append((n_frames - seg_len, n_frames))
    return out


def segment_samples(n_samples: int, sr: int = SR) -> List[Tuple[int, int]]:
    seg_len = int(SEG_DUR * sr)
    hop_len = int(SEG_HOP * sr)
    out: List[Tuple[int, int]] = []
    if n_samples <= seg_len:
        out.append((0, n_samples))
        return out
    start = 0
    while start + seg_len <= n_samples:
        out.append((start, start + seg_len))
        start += hop_len
    if out[-1][1] < n_samples:
        out.append((n_samples - seg_len, n_samples))
    return out


def compute_pos_weight(sed_train: np.ndarray) -> torch.Tensor:
    """Class-frequency-based positive weight clipped to ``[1, 100]``."""
    pos = sed_train.sum(axis=0) + 1e-6
    neg = (sed_train.shape[0] - pos) + 1e-6
    w = np.clip(neg / pos, 1.0, 100.0)
    return torch.tensor(w, dtype=torch.float32)


@dataclass
class Segment:
    feat: np.ndarray
    sed: np.ndarray
    onset: np.ndarray


class CachedFeatDS(Dataset):
    """Frame-level SED dataset backed by pre-computed features.

    Applies mixup (``beta=0.3``, ``p=0.3``) and SpecAugment during training.
    """

    def __init__(
        self,
        segments: List[Segment],
        training: bool = False,
        mixup_alpha: float = 0.3,
        mixup_prob: float = 0.3,
        freq_mask: int = 9,
        time_mask: int = 17,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        self.segments = segments
        self.training = training
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __len__(self) -> int:
        return len(self.segments)

    def _spec_augment(self, feat: np.ndarray) -> np.ndarray:
        n_mels, n_frames = feat.shape
        feat = feat.copy()
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask + 1)
            if f > 0 and n_mels - f > 0:
                f0 = np.random.randint(0, n_mels - f)
                feat[f0:f0 + f, :] = feat.mean()
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask + 1)
            if t > 0 and n_frames - t > 0:
                t0 = np.random.randint(0, n_frames - t)
                feat[:, t0:t0 + t] = feat.mean()
        return feat

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        feat = seg.feat
        sed = seg.sed
        onset = seg.onset

        if self.training:
            feat = self._spec_augment(feat)
            if np.random.rand() < self.mixup_prob:
                j = np.random.randint(0, len(self.segments))
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                other = self.segments[j]
                feat = lam * feat + (1.0 - lam) * other.feat
                sed = lam * sed + (1.0 - lam) * other.sed
                onset = lam * onset + (1.0 - lam) * other.onset

        return (
            torch.from_numpy(feat.astype(np.float32)),
            torch.from_numpy(sed.astype(np.float32)),
            torch.from_numpy(onset.astype(np.float32)),
        )


class RawDS(Dataset):
    """Waveform dataset for pretrained encoders (Path A)."""

    def __init__(self, waveforms: List[np.ndarray], seds: List[np.ndarray], onsets: List[np.ndarray]):
        assert len(waveforms) == len(seds) == len(onsets)
        self.waveforms = waveforms
        self.seds = seds
        self.onsets = onsets

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.waveforms[idx].astype(np.float32)),
            torch.from_numpy(self.seds[idx].astype(np.float32)),
            torch.from_numpy(self.onsets[idx].astype(np.float32)),
        )
