"""BowelRCNN: proposal + refine network for bowel sound event detection.

Reference: Matynia and Nowak "BowelRCNN" (arXiv:2504.08659, 2025).

Parameter count at N_MELS=64, NC=3: 3,232,678.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import NC


class ProposalNet(nn.Module):
    """CNN backbone plus per-frame objectness and coarse regression."""

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.feat_dim = 128 * (n_mels // 16)
        self.obj_head = nn.Conv1d(self.feat_dim, 1, 1)
        self.reg_head = nn.Conv1d(self.feat_dim, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        b, c, f, t = x.shape
        feat = x.permute(0, 1, 2, 3).reshape(b, c * f, t)
        obj = self.obj_head(feat).squeeze(1)
        reg = self.reg_head(feat).permute(0, 2, 1)
        return feat, obj, reg


class RefineNet(nn.Module):
    """Recurrent classification head operating on proposal features."""

    def __init__(self, feat_dim: int, n_classes: int = NC, rnn_hidden: int = 128) -> None:
        super().__init__()
        self.rnn = nn.GRU(feat_dim, rnn_hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.dropout = nn.Dropout(0.3)
        self.cls_head = nn.Linear(2 * rnn_hidden, n_classes)
        self.reg_head = nn.Linear(2 * rnn_hidden, 2)

    def forward(self, feat_tc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.rnn(feat_tc)
        x = self.dropout(x)
        return self.cls_head(x), self.reg_head(x)


class BowelRCNN(nn.Module):
    """Two-stage event detector: proposal then refine."""

    def __init__(self, n_mels: int = 64, n_classes: int = NC, rnn_hidden: int = 128) -> None:
        super().__init__()
        self.proposal = ProposalNet(n_mels=n_mels)
        self.refine = RefineNet(self.proposal.feat_dim, n_classes=n_classes, rnn_hidden=rnn_hidden)

    def forward(self, x: torch.Tensor) -> dict:
        feat, obj, reg_coarse = self.proposal(x)
        feat_tc = feat.transpose(1, 2)
        cls_logits, reg_refine = self.refine(feat_tc)
        return {
            "obj": obj,
            "cls": cls_logits,
            "reg_coarse": reg_coarse,
            "reg_refine": reg_refine,
        }
