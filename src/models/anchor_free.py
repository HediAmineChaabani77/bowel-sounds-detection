"""Anchor-free detector for bowel sound events.

A 1D FCOS-style detector that emits per-frame classification and
centerness predictions. Parameter count at N_MELS=64, NC=3: 1,102,759.

Reference: Tian et al. "FCOS: Fully Convolutional One-Stage Object
Detection" (ICCV 2019), adapted to 1D audio frame grids.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC
from .crnn import CB


class AnchorFreeDetector(nn.Module):
    def __init__(self, n_mels: int = 64, n_classes: int = NC, rnn_hidden: int = 128) -> None:
        super().__init__()
        self.b1 = CB(1, 64, pool_f=2)
        self.b2 = CB(64, 128, pool_f=2)
        self.b3 = CB(128, 128, pool_f=2)
        self.b4 = CB(128, 128, pool_f=2)
        feat_dim = 128 * (n_mels // 16)
        self.rnn = nn.GRU(feat_dim, rnn_hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        d = 2 * rnn_hidden
        self.cls_head = nn.Linear(d, n_classes)
        self.center_head = nn.Linear(d, 1)
        self.onset_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.b4(self.b3(self.b2(self.b1(x))))
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.rnn(x)
        cls_logits = self.cls_head(x)
        center_logits = self.center_head(x).squeeze(-1)
        sed_logits = cls_logits + center_logits.unsqueeze(-1)
        onset_logits = self.onset_head(x).squeeze(-1)
        return sed_logits, onset_logits
