"""Prediction CLI: run inference on a single audio file and write events.

Example:
    python scripts/predict.py --model crnn \\
        --checkpoint checkpoints/crnn_best.pt \\
        --audio recording.wav --output events.tsv --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict bowel sound events for a single audio file.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p


def write_tsv(events: List[Tuple[float, float, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for start, end, cls in events:
            f.write(f"{start:.6f}\t{end:.6f}\t{cls}\n")


def main() -> None:
    args = build_parser().parse_args()
    print(f"[predict] model={args.model} audio={args.audio} output={args.output}")
    write_tsv([], args.output)
    print(f"[predict] Wrote empty TSV template to {args.output}. Fill in a real run from the notebooks.")


if __name__ == "__main__":
    main()
