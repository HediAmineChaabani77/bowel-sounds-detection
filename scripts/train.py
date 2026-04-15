"""Training CLI for the bowel sound detection models.

Example:
    python scripts/train.py --model crnn --features logmel \\
        --audio-dir data/ --checkpoint-dir checkpoints/ \\
        --epochs 60 --batch-size 32 --lr 2e-3 --patience 6 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a bowel sound SED model.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "crnn",
            "crnn_pcen",
            "hpss",
            "conformer",
            "anchor_free",
            "beats",
            "ast",
            "hubert",
            "yolo_audio",
            "bowel_rcnn",
        ],
        help="Model architecture to train.",
    )
    p.add_argument("--features", type=str, default="logmel", choices=["logmel", "mfcc", "pcen", "raw"])
    p.add_argument("--audio-dir", type=Path, required=True, help="Directory containing .wav and .txt label files.")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] model={args.model} features={args.features} "
        f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} device={args.device}"
    )
    print(
        "[train] Reference implementation: training loops live in the three notebooks "
        "under notebooks/. This CLI is a thin wrapper around src/ modules."
    )


if __name__ == "__main__":
    main()
