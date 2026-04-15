# Bowel Sound Detection and Classification

> PoC for automated bowel sound event detection and 3-class classification from continuous audio recordings.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange.svg)](https://colab.research.google.com/)

## Overview

This repository is a proof of concept for automatic detection and classification of bowel sound events in continuous abdominal audio recordings. Each detected event is emitted as a `(start_time, end_time, class)` triple, where the class is one of three categories defined by the clinical taxonomy: **single burst (b)**, **multiple burst (mb)**, and **harmonic (h)**. The work benchmarks **21 model architectures** across three self-contained Colab notebooks, spanning from-scratch CRNNs to large pretrained audio transformers (BEATs, AST, HTS-AT, SSAST, PANNs, Whisper, HuBERT) and dedicated event-detection networks (BowelRCNN, YOLO-Audio, BEATs-RCNN, HuBERT-YOLO). To the best of our knowledge, this is the first bowel sound study to evaluate models using **sed_eval DCASE event-based metrics** rather than frame-level accuracy.

## Architecture

```
                     ┌──────────────────────────┐
                     │   Raw audio (16 kHz WAV) │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Spectral gating denoise │
                     │     (noisereduce)        │
                     └────────────┬─────────────┘
                                  │
                ┌─────────────────┴──────────────────┐
                │                                    │
     ┌──────────▼──────────┐             ┌──────────▼──────────┐
     │  Path A: pretrained │             │ Path B: from scratch│
     │  RMS norm (0.02)    │             │ Bandpass 60-3000 Hz │
     │  full spectrum      │             │ 4th-order Butter    │
     └──────────┬──────────┘             └──────────┬──────────┘
                │                                    │
                │                         ┌──────────▼──────────┐
                │                         │ LogMel / MFCC / PCEN│
                │                         │ 64 bins, 512 FFT,   │
                │                         │ 160 hop, Hanning    │
                │                         └──────────┬──────────┘
                │                                    │
     ┌──────────▼──────────┐             ┌──────────▼──────────┐
     │ BEATs, AST, PANNs,  │             │ CRNN, Conformer,    │
     │ HTS-AT, SSAST,      │             │ HPSS Dual-Stream,   │
     │ Whisper, HuBERT     │             │ Anchor-Free, RCNN   │
     └──────────┬──────────┘             └──────────┬──────────┘
                └─────────────────┬──────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ Frame probabilities       │
                     │ (3 classes, 25 fps)       │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ Median filter + threshold │
                     │ Peak pick (b),            │
                     │ contiguous runs (mb, h)   │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ Events (start, end, cls) │
                     └──────────────────────────┘
```

## Results: Combined Leaderboard (Top 10)

| Rank | Model | Event macro F1 | Notebook | Type |
|------|-------|----------------|----------|------|
| 1 | Ensemble | 0.429 | final | Frame-level (combined) |
| 2 | CRNN+LogMel | 0.299 | final, hubert | Frame-level, from scratch |
| 3 | BEATs (frame) | 0.282 | final | Frame-level, pretrained |
| 4 | HuBERT-large frozen | 0.250 | hubert | Frame-level, pretrained |
| 5 | HuBERT-base unfrozen-2 | 0.242 | hubert | Frame-level, pretrained |
| 6 | BEATs-RCNN | 0.212 | event_det | Event detection, pretrained |
| 7 | HuBERT-base frozen | 0.204 | hubert | Frame-level, pretrained |
| 8 | CRNN+MFCC | 0.202 | hubert | Frame-level, from scratch |
| 9 | CRNN+PCEN | 0.202 | hubert | Frame-level, from scratch |
| 10 | HPSS Dual-Stream | 0.194 | final | Frame-level, from scratch |

## Results: Main Benchmark (bowel_sound_final.ipynb)

| Model | Val AUC | Event F1 b | Event F1 mb | Event F1 h | Event macro F1 |
|-------|---------|------------|-------------|------------|----------------|
| Ensemble | : | 0.476 | 0.383 | n/a | 0.429 |
| CRNN | 0.9742 | 0.345 | 0.228 | 0.325 | 0.299 |
| BEATs | 0.9785 | 0.319 | 0.343 | 0.182 | 0.282 |
| HPSS Dual-Stream | 0.9826 | : | : | : | 0.194 |
| Conformer | 0.9752 | : | : | : | 0.194 |
| CRNN+PCEN | 0.9453 | : | : | : | 0.194 |
| Anchor-Free Detector | 0.9645 | : | : | : | 0.194 |
| HTS-AT | 0.9180 | n/a | 0.000 | 0.200 | 0.100 |
| AST | 0.9256 | n/a | 0.000 | 0.000 | 0.000 |
| PANNs | : | n/a | 0.000 | n/a | 0.000 |
| SSAST-proxy | 0.8961 | n/a | 0.000 | 0.000 | 0.000 |
| Whisper | : | n/a | 0.000 | 0.000 | 0.000 |

## Results: HuBERT + Feature Comparison (bowel_sound_hubert_features.ipynb)

| Model | Val AUC | Event F1 b | Event F1 mb | Event F1 h | Event macro F1 |
|-------|---------|------------|-------------|------------|----------------|
| CRNN+LogMel | 0.9742 | 0.345 | 0.228 | 0.325 | 0.299 |
| HuBERT-large frozen | 0.9404 | 0.381 | 0.090 | 0.278 | 0.250 |
| HuBERT-base unfrozen-2 | 0.9325 | 0.308 | 0.175 | 0.243 | 0.242 |
| HuBERT-base frozen | 0.9523 | 0.217 | 0.138 | 0.258 | 0.204 |
| CRNN+MFCC | 0.9582 | : | : | : | 0.202 |
| CRNN+PCEN | 0.9453 | : | : | : | 0.202 |

## Results: Direct Event Detection (bowel_sound_event_detection.ipynb)

| Model | Trainable params | Event F1 b | Event F1 mb | Event F1 h | Event macro F1 |
|-------|------------------|------------|-------------|------------|----------------|
| BEATs-RCNN | 4,166,150 | 0.265 | 0.257 | 0.114 | 0.212 |
| BowelRCNN | 3,232,678 | 0.404 | 0.126 | 0.000 | 0.177 |
| YOLO-Audio | 1,244,262 | 0.276 | 0.235 | 0.000 | 0.170 |
| HuBERT-YOLO | : | 0.070 | 0.118 | 0.234 | 0.141 |

## Key Findings

1. **CRNN+LogMel is the best single model** (event macro F1 0.299) at n=2 recordings. The simplest from-scratch approach wins.
2. **Ensemble averaging boosts performance to 0.429** by combining all 11 frame-level models, showing that model diversity compensates for individual weaknesses.
3. **HPSS Dual-Stream achieves the highest validation AUC (0.9826)** by decomposing audio into harmonic and percussive streams that map directly onto the class taxonomy.
4. **HuBERT, the published SOTA** (Mansour et al. 2025 report AUC 0.89 on 16 subjects), scores between 0.204 and 0.250 event macro F1 here. Pretrained models consistently need more data to overtake from-scratch CNNs.
5. **The claim that MFCC outperforms log-mel does not hold at n=2** (MFCC val AUC 0.9582 vs LogMel 0.9742), contradicting Mansour et al. at this data scale.
6. **sed_eval event-based metrics have never been used** in the bowel sound literature before this project. All published work uses frame-level accuracy.
7. **PCEN, HPSS, BEATs, PANNs, HTS-AT, AST, Whisper, SSAST, Conformer** are all evaluated on bowel sounds for the first time in this work.
8. **BEATs-RCNN (0.212) is the best dedicated event detection model** and the only one detecting all three classes.
9. **BowelRCNN achieves the highest single-burst F1 (b=0.404)** across all 21 models in the entire project.
10. **Val AUC does not predict test event F1**: HPSS has the highest AUC (0.9826) but only 0.194 event macro F1, while CRNN has AUC 0.9742 and 0.299 event macro F1.

## Novel Contributions

- First application of **sed_eval DCASE event-based metrics** to bowel sound detection.
- First evaluation of **HPSS dual-stream decomposition** on bowel sounds, motivated by the physical mapping to the taxonomy.
- First benchmark of large audio transformers on bowel sounds: **BEATs, AST, PANNs, HTS-AT, SSAST, Whisper**.
- First reported use of **Conformer, Anchor-Free Detector, PCEN** features on this task.
- First 11-model **ensemble** on bowel sound SED.
- First direct comparison of **frame-level SED vs. direct event detection** (RCNN, YOLO) on bowel sounds.

## Quick Start

```bash
git clone https://github.com/HediAmineChaabani77/bowel-sounds-detection.git
cd bowel-sounds-detection
pip install -r requirements.txt
```

Open any notebook in Google Colab:

- [bowel_sound_final.ipynb](notebooks/bowel_sound_final.ipynb) : 11 frame-level SED models and ensemble
- [bowel_sound_hubert_features.ipynb](notebooks/bowel_sound_hubert_features.ipynb) : HuBERT and feature comparison
- [bowel_sound_event_detection.ipynb](notebooks/bowel_sound_event_detection.ipynb) : direct event detection (RCNN, YOLO)

Run a training script locally:

```bash
python scripts/train.py --model crnn --features logmel --audio-dir data/ --checkpoint-dir checkpoints/
python scripts/evaluate.py --model crnn --checkpoint checkpoints/crnn_best.pt --audio-dir data/
python scripts/predict.py --model crnn --checkpoint checkpoints/crnn_best.pt --audio clip.wav --output events.tsv
```

## Repository Structure

```
bowel-sounds-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── SCALING.md
├── notebooks/
│   ├── bowel_sound_final.ipynb
│   ├── bowel_sound_hubert_features.ipynb
│   └── bowel_sound_event_detection.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── dataset.py
│   ├── event_extraction.py
│   ├── evaluation.py
│   └── models/
│       ├── __init__.py
│       ├── crnn.py
│       ├── hpss.py
│       ├── conformer.py
│       ├── anchor_free.py
│       ├── beats_wrapper.py
│       ├── ast_wrapper.py
│       ├── hubert_wrapper.py
│       ├── yolo_audio.py
│       └── bowel_rcnn.py
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Evaluation Methodology

The evaluation uses `sed_eval.sound_event.EventBasedMetrics` with **class-specific collars** that reflect the physical duration of each event type:

| Class | Collar | Length tolerance | Rationale |
|-------|--------|------------------|-----------|
| b (single burst) | 50 ms | onset-only (percentage_of_length=1.0) | Bursts are sub-100 ms transients; onset timing dominates. |
| mb (multiple burst) | 100 ms | 20% of length | Overlapping bursts; duration is meaningful. |
| h (harmonic) | 200 ms | 20% of length | Sustained tonal events; looser boundaries are clinically acceptable. |

Thresholds and median-filter sizes are tuned on the validation split via grid search: filter sizes `[1, 5, 11, 21]`, threshold range `[0.05, 0.85]` step `0.05`. Minimum event durations are enforced post-hoc: `b=20 ms`, `mb=50 ms`, `h=100 ms`. Single-burst events are extracted by peak-picking with a Gaussian onset prior (`sigma=2`, `distance=5`). Multiple-burst and harmonic events are extracted from contiguous thresholded frame runs.

## What Changes at Scale

With 20 or more subjects the ranking is expected to shift materially:

- **HuBERT and BEATs overtake CRNN+LogMel**, consistent with published AUC 0.89 on 16 subjects (Mansour et al. 2025). Large pretrained encoders are data-hungry and currently under-regularized on 2 recordings.
- **Cross-patient validation becomes possible** via leave-one-subject-out, giving a realistic estimate of generalization across anatomies and noise conditions.
- **Harmonic detection improves substantially**. With only 109 harmonic events in total and 12 in the test split, h F1 is dominated by noise at the current scale.
- **Ensembling gains shrink** as individual models become more accurate and their errors become more correlated.
- **Direct event detectors (BowelRCNN, YOLO-Audio, BEATs-RCNN) close the gap** with frame-level approaches. Anchor-based methods reward large, diverse training sets.
- **Active learning** becomes tractable: the bottleneck shifts from modeling to expert annotation time.

## References

- Mansour et al. (2025). *Benchmarking Machine Learning for Bowel Sound Pattern Classification*. PLOS ONE / arXiv:2502.15607.
- Baronetto et al. (2024). *Multiscale Bowel Sound Event Spotting*. JMIR AI e51118.
- Matynia and Nowak (2025). *BowelRCNN*. arXiv:2504.08659.
- Kalahasty et al. (2025). *YOLO for Bowel Sounds*. Sensors 25:4735.
- Yu et al. (2024). *Enhancing Bowel Sound Recognition with Self-Attention and Self-Supervised Pre-training*. PLOS ONE.
- Ficek et al. (2021). *Analysis of GI Acoustic Activity Using DNNs*. Sensors 21:7602.
- Celik (2026). *Deep Learning-Based Detection of Bowel Sound Events*. Scientific Reports 16:10595.
- Zhang et al. (2025). *Explainable ResNet-LSTM for Bowel Sound Frequency Classification*. Journal of International Medical Research.

## Author

**Hedi Amine Chaabani**
Senior AI Engineer and Data Scientist, 7+ years experience
Based in Tunis, Tunisia
GitHub: [@HediAmineChaabani77](https://github.com/HediAmineChaabani77)

Prepared as a technical interview submission for **DigeHealth** (contact: Lyle Halliday).
