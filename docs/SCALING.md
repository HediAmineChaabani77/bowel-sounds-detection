# Scaling Roadmap

This document outlines the path from the current 2-recording PoC to a production-grade bowel sound detection system.

## Performance Expectations with 20+ Recordings

At `n=2` the ranking is dominated by inductive bias: small from-scratch CRNNs beat large pretrained transformers because the transformers have not seen enough data to specialize. At `n=20` subjects and beyond, the ranking is expected to invert:

- **HuBERT and BEATs overtake CRNN+LogMel**. Mansour et al. 2025 report HuBERT AUC 0.89 on 16 subjects, which would translate to materially better event F1 than the 0.250 achieved here. Large self-supervised encoders become the clear winners once they see enough clinical data to adapt their pretrained representations.
- **Harmonic detection becomes tractable**. The current h F1 is dominated by stochastic noise on ~12 test events. Scaling to 20+ subjects yields hundreds of harmonic events, enough to train dedicated event detection heads without class collapse.
- **Ensembling gains shrink**. At `n=2` the 11-model ensemble gains +0.130 absolute F1 because individual models are noisy and their errors are uncorrelated. As each model becomes more accurate, its errors become more correlated with the best model, reducing the ensemble's incremental value. A single HuBERT-large finetuned on 20+ subjects will likely match or beat the current ensemble.

## Cross-Patient Validation

At `n=2` only temporal splits within each recording are possible, which does not test generalization across anatomies, microphone placements, body mass, or gut pathology. At `n=20+` the validation protocol should be:

- **Leave-one-subject-out (LOSO) cross-validation**. The gold standard for small clinical datasets. Report mean and standard deviation across folds.
- **Held-out site validation** once multi-center data is available. Different recording environments (acoustic, microphone hardware) introduce distribution shift that LOSO within a single site does not catch.

## Active Learning

With expert annotation time as the dominant cost, active learning becomes central:

- **Uncertainty sampling**: train an initial model on a small labeled set, run it on unlabeled recordings, and flag the high-entropy frames for expert review.
- **Diversity sampling**: select frames that are both uncertain and dissimilar to already-labeled frames using embedding distance on HuBERT features.
- **Weak labeling**: allow experts to mark "there is at least one harmonic event in this 30-second window" and use MIL-style losses to localize it.

A pipeline that uses 1 hour of expert time to label the most informative 5 minutes of audio should outperform 1 hour of random annotation by a factor of 3-5x at harmonic detection.

## Real-Time Inference

CRNN at 1.1M parameters is lightweight enough for real-time inference on commodity hardware:

- **Latency**: with 4-second windows and 1-second hop, the system produces updated event predictions every 1 second. Inference on CRNN is under 50 ms on CPU, under 10 ms on a small edge GPU.
- **Causal variant**: replacing the BiGRU with a unidirectional GRU removes the 2-second look-ahead and enables truly online inference with a 4-second warm-up.
- **Streaming buffer**: a circular buffer of 4 seconds of 16 kHz audio (128 KB) is the only state needed. Suitable for embedded deployment.

## Deployment Targets

- **Edge wearable** (CRNN, YOLO-Audio, BowelRCNN). Target: Raspberry Pi Zero 2 W, Nordic nRF52840, or equivalent Cortex-M class with DSP. Model size under 5 MB after quantization. Battery-friendly duty cycling driven by activity detection.
- **Mobile companion** (CRNN, Conformer, small HuBERT-base). Runs on smartphone via ONNX Runtime or Core ML. Model size 5 to 50 MB. Power budget allows full-precision inference.
- **Cloud backend** (BEATs, HuBERT-large, ensemble). Batch inference on stored recordings for clinical review. GPU inference at hundreds of hours of audio per hour of wall-clock time. Model size unconstrained.

## Model Compression

- **Post-training quantization** to int8 gives a 4x size reduction with under 1% AUC loss for CRNN. Larger models (BEATs, HuBERT) need quantization-aware training.
- **Structured pruning** of the BiGRU and final dense layers can remove 30 to 50% of CRNN parameters with minimal F1 loss.
- **Knowledge distillation** from the 11-model ensemble (F1 0.429) into a single CRNN student is the single highest-leverage compression play. The student learns the ensemble's calibrated probabilities directly, capturing most of the ensemble gain in a 1.1M-parameter model.

## Clinical Validation

- **IRB approval** for prospective data collection at each participating site.
- **Multi-center study** with at least three hospitals to capture patient diversity and recording-environment diversity.
- **Reference standard**: double-annotation by two gastroenterology experts with a third adjudicator on disagreements. Inter-rater agreement should be reported per class.
- **Regulatory path**: FDA 510(k) or EU MDR Class IIa, depending on intended use. Likely positioned as a software as a medical device (SaMD) for decision support rather than primary diagnosis.

## Data Pipeline

- **Continuous recording ingestion**: secure upload from wearable to cloud over HTTPS with per-recording metadata (subject ID, start time, device, sampling rate, battery state).
- **Automated QC**: reject recordings with excessive clipping, low RMS, DC offset, or missing samples. Flag recordings with atypical spectral profiles for manual review.
- **Annotation tooling**: web-based waveform and spectrogram viewer with keyboard shortcuts for each event class, confidence scoring, and review queues driven by the active learning module.
- **Versioned datasets**: every model release is tied to an immutable dataset snapshot (DVC, LakeFS, or equivalent) so results are reproducible.
- **Drift monitoring**: track the distribution of predicted event rates per class per site. Alert on sudden shifts that indicate either clinical change or data pipeline regression.
