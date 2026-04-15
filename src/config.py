"""Global configuration constants for the bowel sound detection pipeline.

All values match the configuration used in the Colab notebooks so that
results can be reproduced exactly from the checkpoints saved to Drive.
"""

from __future__ import annotations

SEED: int = 42

SR: int = 16000
BP_LOW: float = 60.0
BP_HIGH: float = 3000.0
BP_ORDER: int = 4

N_FFT: int = 512
HOP: int = 160
N_MELS: int = 64
FMIN: float = 60.0
FMAX: float = 3000.0

SEG_DUR: float = 4.0
SEG_HOP: float = 1.0
TARGET_NF: int = 400
FT: float = HOP / SR

CLASSES: tuple[str, ...] = ("b", "mb", "h")
C2I: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NC: int = len(CLASSES)

LABEL_MAP: dict[str, str] = {
    "b": "b",
    "mb": "mb",
    "h": "h",
    "burst": "b",
    "single": "b",
    "single_burst": "b",
    "multiple": "mb",
    "multi": "mb",
    "multiple_burst": "mb",
    "harmonic": "h",
}

NON_TARGET: frozenset[str] = frozenset({"n", "noise", "other", "silence"})

ONSET_SIGMA: float = 2.0

MIN_DUR: dict[str, float] = {"b": 0.020, "mb": 0.050, "h": 0.100}

COLLARS: dict[str, dict[str, float]] = {
    "b": {"t_collar": 0.050, "percentage_of_length": 1.00},
    "mb": {"t_collar": 0.100, "percentage_of_length": 0.20},
    "h": {"t_collar": 0.200, "percentage_of_length": 0.20},
}
