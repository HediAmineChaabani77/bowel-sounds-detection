"""YOLO-style 1D audio event detector.

CNN backbone followed by Transformer layers feeding per-frame objectness,
classification, and regression (start, end offset) heads. Parameter count
at N_MELS=64, NC=3: 1,244,262.

Reference: Kalahasty et al. "YOLO for Bowel Sounds" (Sensors 2025).
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, pool: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.MaxPool2d((pool, 1)) if pool > 1 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLOAudio(nn.Module):
    def __init__(self, n_mels: int = 64, n_classes: int = NC, d_model: int = 128, n_transformer: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(1, 32, pool=2),
            _ConvBlock(32, 64, pool=2),
            _ConvBlock(64, 128, pool=2),
            _ConvBlock(128, 128, pool=2),
        )
        flat = 128 * (n_mels // 16)
        self.proj = nn.Linear(flat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer)
        self.obj_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, n_classes)
        self.reg_head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj(x)
        x = self.transformer(x)
        return {
            "obj": self.obj_head(x).squeeze(-1),
            "cls": self.cls_head(x),
            "reg": self.reg_head(x),
        }
