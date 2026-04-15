"""Conformer-based frame-level SED model.

Reference: Gulati et al. "Conformer: Convolution-augmented Transformer for
Speech Recognition" (Interspeech 2020).
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC


class _FFN(nn.Module):
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, expansion * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pointwise1 = nn.Conv1d(d, 2 * d, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d, d, kernel_size=kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d, d, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x).transpose(1, 2)
        y = self.glu(self.pointwise1(y))
        y = self.act(self.bn(self.depthwise(y)))
        y = self.pointwise2(y)
        y = y.transpose(1, 2)
        return self.drop(y)


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int = 4, ff_expansion: int = 4, kernel: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = _FFN(d, ff_expansion, dropout)
        self.attn_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = _ConvModule(d, kernel, dropout)
        self.ffn2 = _FFN(d, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        y = self.attn_ln(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.attn_drop(a)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConformerSED(nn.Module):
    """Conformer-based frame-level SED network."""

    def __init__(self, n_mels: int = 64, n_classes: int = NC, d_model: int = 128, n_layers: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        flat = 64 * (n_mels // 4)
        self.linear = nn.Linear(flat, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model) for _ in range(n_layers)])
        self.sed_head = nn.Linear(d_model, n_classes)
        self.onset_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.linear(x)
        for block in self.blocks:
            x = block(x)
        return self.sed_head(x), self.onset_head(x).squeeze(-1)
