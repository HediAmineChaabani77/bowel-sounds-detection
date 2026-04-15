"""Model zoo for bowel sound detection.

All 9 architectures described in the project documentation:

Frame-level from scratch:
  CRNN, DualStreamHPSS, ConformerSED, AnchorFreeDetector

Frame-level pretrained:
  BEATsBiGRU, ASTBiGRU, HuBERTBiGRU

Direct event detection:
  YOLOAudio, BowelRCNN
"""

from .anchor_free import AnchorFreeDetector
from .ast_wrapper import ASTBiGRU
from .beats_wrapper import BEATsBiGRU
from .bowel_rcnn import BowelRCNN
from .conformer import ConformerBlock, ConformerSED
from .crnn import CB, CRNN
from .hpss import DualStreamHPSS
from .hubert_wrapper import HuBERTBiGRU
from .yolo_audio import YOLOAudio

__all__ = [
    "AnchorFreeDetector",
    "ASTBiGRU",
    "BEATsBiGRU",
    "BowelRCNN",
    "CB",
    "CRNN",
    "ConformerBlock",
    "ConformerSED",
    "DualStreamHPSS",
    "HuBERTBiGRU",
    "YOLOAudio",
]
