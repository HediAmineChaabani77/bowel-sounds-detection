# Results

This document presents the detailed results of all 21 models evaluated across the three notebooks, ranked and analyzed by `sed_eval` event-based macro F1 on the test split.

## Full Leaderboard

| Rank | Model | Event macro F1 | Val AUC | Notebook | Type |
|------|-------|----------------|---------|----------|------|
| 1 | Ensemble (11 models) | 0.429 | : | final | Frame-level (combined) |
| 2 | CRNN+LogMel | 0.299 | 0.9742 | final, hubert | Frame-level, from scratch |
| 3 | BEATs (frame) | 0.282 | 0.9785 | final | Frame-level, pretrained |
| 4 | HuBERT-large frozen | 0.250 | 0.9404 | hubert | Frame-level, pretrained |
| 5 | HuBERT-base unfrozen-2 | 0.242 | 0.9325 | hubert | Frame-level, pretrained |
| 6 | BEATs-RCNN | 0.212 | : | event_det | Event detection, pretrained |
| 7 | HuBERT-base frozen | 0.204 | 0.9523 | hubert | Frame-level, pretrained |
| 8 | CRNN+MFCC | 0.202 | 0.9582 | hubert | Frame-level, from scratch |
| 9 | CRNN+PCEN | 0.202 | 0.9453 | hubert | Frame-level, from scratch |
| 10 | HPSS Dual-Stream | 0.194 | 0.9826 | final | Frame-level, from scratch |
| 11 | Conformer | 0.194 | 0.9752 | final | Frame-level, from scratch |
| 12 | Anchor-Free Detector | 0.194 | 0.9645 | final | Frame-level, from scratch |
| 13 | BowelRCNN | 0.177 | : | event_det | Event detection, from scratch |
| 14 | YOLO-Audio | 0.170 | : | event_det | Event detection, from scratch |
| 15 | HuBERT-YOLO | 0.141 | : | event_det | Event detection, pretrained |
| 16 | HTS-AT | 0.100 | 0.9180 | final | Frame-level, pretrained |
| 17 | AST | 0.000 | 0.9256 | final | Frame-level, pretrained |
| 17 | PANNs | 0.000 | : | final | Frame-level, pretrained |
| 17 | SSAST-proxy | 0.000 | 0.8961 | final | Frame-level, pretrained |
| 17 | Whisper | 0.000 | : | final | Frame-level, pretrained |

## Per-Class Analysis

### Single burst (b)

Single bursts are the most abundant class and the most reliably detected. The top b F1 across **all 21 models** is achieved by **BowelRCNN (b=0.404)**, a from-scratch event detector. The second best is **HuBERT-large frozen (b=0.381)**. BowelRCNN's proposal and refine architecture is well-suited to the sharp, short-duration onset pattern of single bursts, and the Gaussian onset prior used in extraction plays to its strengths.

### Multiple burst (mb)

The strongest mb detector is **BEATs frame-level (mb=0.343)**, followed by BowelRCNN-adjacent detectors. The mb class benefits from temporal context: BEATs' transformer self-attention captures the characteristic clustering of bursts better than pure peak pickers.

### Harmonic (h)

Harmonic detection is the hardest task in the project. With only **109 harmonic events total and approximately 12 in the test split**, the F1 is extremely sensitive to a single false positive or miss. The best h F1 comes from **CRNN+LogMel (h=0.325)**, with HuBERT-large frozen (h=0.278) and HuBERT-base frozen (h=0.258) following. Notably, **from-scratch event detectors (BowelRCNN, YOLO-Audio) score h=0.000**: the anchor/proposal paradigm cannot handle classes with this few training examples, while frame-level models with a shared backbone can still learn a useful harmonic representation.

## Pretrained versus From-Scratch

At `n=2` recordings the from-scratch CRNN+LogMel (0.299) outperforms every pretrained model as a single system. The ranking among pretrained frame-level encoders is BEATs > HuBERT-large > HuBERT-base unfrozen > HuBERT-base frozen > HTS-AT > (AST, SSAST, Whisper, PANNs at 0.000). The collapse of AST, SSAST, Whisper, and PANNs is the expected failure mode when a 90M+ parameter transformer meets a 40-minute training set without regularization tuned for its scale. BEATs is the exception: its masked-prediction pretraining on AudioSet is unusually data-efficient on downstream SED tasks.

**However, pretrained models dominate multi-class detection.** Of all 21 models, only BEATs-RCNN and the HuBERT variants detect all three classes robustly. From-scratch event detectors collapse entirely on the harmonic class.

## Feature Comparison (CRNN backbone)

On an **identical CRNN backbone**, the feature ranking is:

| Features | Val AUC | Event macro F1 |
|----------|---------|----------------|
| LogMel | 0.9742 | 0.299 |
| MFCC | 0.9582 | 0.202 |
| PCEN | 0.9453 | 0.202 |

This result **contradicts Mansour et al. 2025**, who report MFCC outperforming log-mel at larger subject counts. At `n=2` log-mel wins decisively on both AUC and event F1. PCEN does not help either: its adaptive gain normalization is designed for long-tailed loudness distributions that the bandpassed, RMS-normalized bowel stream does not exhibit.

## HuBERT Analysis

| HuBERT variant | Val AUC | Event macro F1 |
|----------------|---------|----------------|
| base frozen | 0.9523 | 0.204 |
| base unfrozen-2 | 0.9325 | 0.242 |
| large frozen | 0.9404 | 0.250 |

Two observations:

1. **Unfreezing hurts val AUC but helps event F1.** Unfreezing the last two transformer layers drops val AUC from 0.9523 to 0.9325 (overfitting) but improves test event F1 from 0.204 to 0.242. The event F1 improvement comes from better calibration at the post-processing stage, not better frame-level classification.
2. **Large is not clearly better than base.** HuBERT-large frozen (0.250) edges out both base variants but at nearly 10x the training time. This is consistent with the published observation that HuBERT-large requires substantial downstream data to justify its capacity.

Mansour et al. 2025 report HuBERT AUC 0.89 on a 16-subject dataset. Our HuBERT variants achieve higher AUC (0.93 to 0.95) on the in-distribution `n=2` split but much lower event F1, indicating that high frame-level AUC at the 2-recording scale does not imply clinically usable event predictions.

## Event Detection Analysis

The four direct event detection models in `bowel_sound_event_detection.ipynb`:

| Model | b | mb | h | macro F1 |
|-------|-----|-----|-----|----------|
| BEATs-RCNN | 0.265 | 0.257 | 0.114 | 0.212 |
| BowelRCNN | 0.404 | 0.126 | 0.000 | 0.177 |
| YOLO-Audio | 0.276 | 0.235 | 0.000 | 0.170 |
| HuBERT-YOLO | 0.070 | 0.118 | 0.234 | 0.141 |

- **BEATs-RCNN is the best event-detection model** and **the only one detecting all three classes**. The frozen BEATs backbone provides the class-discriminative features that small from-scratch event heads cannot learn from 40 minutes of audio.
- **BowelRCNN wins single-burst detection** with the highest b F1 across all 21 models (0.404). Its proposal-refine structure aligns naturally with the burst onset pattern.
- **From-scratch event detectors (BowelRCNN, YOLO-Audio) fail on harmonic detection.** Both score h=0.000. The anchor/proposal paradigm needs more positive harmonic examples than the training split provides.

## Ensemble Analysis

Simple probability averaging of all 11 frame-level models from `bowel_sound_final.ipynb` achieves **Event macro F1 = 0.429**, a **+0.130 absolute gain** over the best single model (CRNN+LogMel at 0.299). Gains are driven by model diversity: CRNN, HPSS, Conformer, and BEATs make different error patterns that partially cancel under averaging. The ensemble scores 0.476 on b and 0.383 on mb, both of which are the highest in the entire project.

## Val AUC vs Event F1 Disconnect

A striking finding: **validation AUC is a poor predictor of test event F1** at this scale.

| Model | Val AUC | Event macro F1 | Rank by AUC | Rank by F1 |
|-------|---------|----------------|-------------|------------|
| HPSS Dual-Stream | 0.9826 | 0.194 | 1 | 10 |
| BEATs | 0.9785 | 0.282 | 2 | 3 |
| Conformer | 0.9752 | 0.194 | 3 | 10 |
| CRNN+LogMel | 0.9742 | 0.299 | 4 | 2 |
| Anchor-Free | 0.9645 | 0.194 | 5 | 10 |

HPSS has the highest val AUC but a middle-of-the-pack event F1. CRNN has the fourth-best AUC but the best single-model event F1. This is a direct consequence of two factors: the per-frame AUC does not reward temporal localization, and the post-processing thresholding step can amplify or cancel out fine differences in frame-level calibration.

**Implication**: selecting models on val AUC alone is misleading for SED. Event-based metrics must be used for model selection whenever temporal localization is part of the deployment objective.

## Training Efficiency

| Model | Training time | Event macro F1 | Time per F1 point |
|-------|---------------|----------------|-------------------|
| CRNN+PCEN | 774 s | 0.194 | 3989 s |
| CRNN+LogMel | 1859 s | 0.299 | 6217 s |
| BEATs | 1196 s | 0.282 | 4241 s |
| Conformer | 950 s | 0.194 | 4897 s |
| SSAST-proxy | 8947 s | 0.000 | inf |
| AST | 7962 s | 0.000 | inf |

CRNN+LogMel trains in **31 minutes** and reaches 0.299 event macro F1, while SSAST and AST burn 2+ hours of Tesla T4 time for 0.000 event F1. BEATs is the best time-per-F1 among pretrained models. For iterative development at this data scale, CRNN is the obvious research baseline.
