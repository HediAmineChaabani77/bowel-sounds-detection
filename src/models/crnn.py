"""CRNN backbone for frame-level sound event detection.

Conv-BN-ReLU stack followed by BiGRU with a parallel SED head and onset head.
Reference: Cakir et al. "Convolutional Recurrent Neural Networks for
Polyphonic Sound Event Detection" (2017).

Parameter count at N_MELS=64, NC=3: 1,101,987.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC


class CB(nn.Module):
    """Conv2d + BatchNorm + ReLU + MaxPool(freq) building block."""

    def __init__(self, c_in: int, c_out: int, pool_f: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(pool_f, 1)) if pool_f > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class CRNN(nn.Module):
    """Frame-level CRNN for SED.

    Input: ``(B, 1, n_mels, n_frames)``.
    Output: ``sed_logits (B, T, NC)``, ``onset_logits (B, T)``.
    """

    def __init__(self, n_mels: int = 64, n_classes: int = NC, rnn_hidden: int = 128) -> None:
        super().__init__()
        self.block1 = CB(1, 64, pool_f=2)
        self.block2 = CB(64, 128, pool_f=2)
        self.block3 = CB(128, 128, pool_f=2)
        self.block4 = CB(128, 128, pool_f=2)
        feat_dim = 128 * (n_mels // 16)
        self.rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.dropout = nn.Dropout(0.3)
        self.sed_head = nn.Linear(2 * rnn_hidden, n_classes)
        self.onset_head = nn.Linear(2 * rnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        sed_logits = self.sed_head(x)
        onset_logits = self.onset_head(x).squeeze(-1)
        return sed_logits, onset_logits
