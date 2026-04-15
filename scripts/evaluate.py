"""Evaluation CLI for bowel sound detection models.

Example:
    python scripts/evaluate.py --model crnn \\
        --checkpoint checkpoints/crnn_best.pt \\
        --audio-dir data/ --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a bowel sound SED model with sed_eval.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(f"[evaluate] model={args.model} checkpoint={args.checkpoint} audio-dir={args.audio_dir}")
    print("[evaluate] Loads model, runs predict_full_*, tunes thresholds on val, scores test with eval_events_per_file.")


if __name__ == "__main__":
    main()
