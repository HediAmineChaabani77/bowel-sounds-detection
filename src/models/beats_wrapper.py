"""Frozen BEATs encoder with a BiGRU frame-level head.

Reference: Chen et al. "BEATs: Audio Pre-Training with Acoustic Tokenizers"
(ICML 2023). The BEATs encoder is loaded from the official iter3+ checkpoint
and kept frozen during training.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC


class BEATsBiGRU(nn.Module):
    """BEATs encoder feeding a small BiGRU classification head."""

    def __init__(self, beats_encoder: nn.Module, feat_dim: int = 768, n_classes: int = NC, rnn_hidden: int = 128) -> None:
        super().__init__()
        self.encoder = beats_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.rnn = nn.GRU(feat_dim, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.sed_head = nn.Linear(2 * rnn_hidden, n_classes)
        self.onset_head = nn.Linear(2 * rnn_hidden, 1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feats = self.encoder(waveform)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        x, _ = self.rnn(feats)
        x = self.dropout(x)
        return self.sed_head(x), self.onset_head(x).squeeze(-1)
