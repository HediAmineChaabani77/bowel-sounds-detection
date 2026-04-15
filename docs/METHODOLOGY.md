# Methodology

This document describes the full technical methodology used in the bowel sound detection PoC.

## 1. Problem Formulation

The task is **sound event detection (SED)** over continuous abdominal audio: given a waveform, the system must emit a set of temporally-localized events of the form `(start_time, end_time, class)`. The class vocabulary has three entries:

- `b` : single burst
- `mb` : multiple burst
- `h` : harmonic

This is a 3-class temporal localization problem. The ground-truth annotations are provided as tab-separated label files with one event per line.

## 2. Data

Two annotated WAV recordings at 16 kHz are used:

| Recording | Duration | Events | b | mb | h |
|-----------|----------|--------|---|----|---|
| AS_1.wav | 2212 s | 1712 | 858 | 774 | 80 |
| 23M74M.wav | 300 s | 514 | 255 | 230 | 29 |
| **Total** | **2512 s** | **2226** | **1113** | **1004** | **109** |

A **temporal 70/15/15 split** is applied **within each recording** to build train, validation, and test sets. The temporal split (rather than a file-level split) is necessary because the dataset contains only two files: a file-level split would place the rare harmonic class almost entirely on one side. The test set contains approximately 12 harmonic events, which is the principal source of variance in the reported F1 scores.

## 3. Preprocessing Pipeline

Raw audio at 16 kHz passes through the following steps:

1. **Noise reduction**: spectral gating via the `noisereduce` library to attenuate stationary background noise.
2. **Path split**: the audio is then routed along one of two paths depending on the downstream model.

### Path A: full-spectrum for pretrained encoders

Used for BEATs, AST, PANNs, HTS-AT, SSAST, Whisper, HuBERT. The audio is RMS-normalized to a target of 0.02 and passed to the encoder at full bandwidth, because these models were pretrained on AudioSet / speech corpora that span the full spectrum and bandpassing breaks their input assumptions.

### Path B: bandpassed for from-scratch models

Used for CRNN, CRNN+PCEN, HPSS Dual-Stream, Conformer, Anchor-Free, BowelRCNN, YOLO-Audio. A **4th-order Butterworth bandpass filter from 60 Hz to 3000 Hz** is applied, removing cardiac and respiratory low-frequency energy and high-frequency content outside the bowel sound band.

### Feature extraction (Path B only)

From the bandpassed waveform, three alternative representations are computed using `librosa`:

- **Log-mel spectrogram**: `n_fft=512`, `hop_length=160`, `win_length=512`, Hanning window, 64 mel bins, `fmin=60`, `fmax=3000`, followed by `librosa.power_to_db`.
- **MFCC**: 64 MFCC coefficients computed from the mel spectrogram above.
- **PCEN**: per-channel energy normalization with `gain=0.98`, `bias=2`, `power=0.5`, `time_constant=0.4`.

### Segmentation

Each stream is segmented into **4-second windows with 1-second hop** (75% overlap). With `hop=160` and `sr=16000`, each window contains `TARGET_NF = 400` frames, yielding a 25 fps frame grid.

## 4. Model Architectures

### A. From-scratch frame-level

- **CRNN** (`1,101,987` params). Stacked conv blocks (Conv-BN-ReLU, the `CB` block) followed by BiGRU and a per-frame classification head with a parallel onset head. Trained 1859 s on Tesla T4, early stop at epoch 20.
- **CRNN+PCEN**. Same architecture as CRNN but fed PCEN features. `1,101,987` params, 774 s, early stop at epoch 5.
- **HPSS Dual-Stream**. `librosa.effects.hpss` decomposes the audio into harmonic and percussive components. Each component passes through an independent CRNN-style branch and the streams are fused before the classification head. The decomposition maps physically onto the taxonomy: percussive energy drives `b` and `mb`, harmonic energy drives `h`. Early stop epoch 13, 2103 s.
- **Conformer**. Stack of ConformerBlocks (self-attention, depthwise conv, macaron FFN) on log-mel. Early stop epoch 17, 950 s.
- **Anchor-Free Detector**. FCOS-inspired 1D detector with per-frame classification and centerness. `1,102,759` params, early stop epoch 7, 949 s.

### B. Pretrained frame-level

All pretrained encoders are used with frozen weights (unless noted) and a lightweight BiGRU plus classification head on top.

- **BEATs**. Microsoft BEATs iter3+ checkpoint, early stop epoch 13, 1196 s.
- **AST**. Audio Spectrogram Transformer, early stop epoch 14, 7962 s.
- **PANNs**. CNN14 pretrained on AudioSet.
- **HTS-AT**. Hierarchical Token-Semantic Audio Transformer, early stop epoch 23, 3999 s.
- **SSAST-proxy**. Self-Supervised AST, early stop epoch 15, 8947 s.
- **Whisper**. OpenAI Whisper encoder repurposed as a frame-level feature extractor.
- **HuBERT-base frozen**. `986,883` trainable parameters in the head, 3921 s, 30 epochs (no early stop), best AUC 0.9523.
- **HuBERT-base unfrozen-2**. Last two transformer layers unfrozen, 3985 s, early stop epoch 23, best AUC 0.9325.
- **HuBERT-large frozen**. 5375 s, early stop epoch 12, best AUC 0.9404.

### C. From-scratch event detection

- **BowelRCNN** (`3,232,678` params). Proposal network generates candidate regions, a refine network classifies and regresses boundaries. 686 s, early stop epoch 2, best val loss 0.5357.
- **YOLO-Audio** (`1,244,262` params). CNN backbone followed by Transformer layers feeding per-frame objectness, classification, and regression heads. 573 s, early stop epoch 3, best val loss 0.9999.

### D. Pretrained event detection

- **BEATs-RCNN** (`4,166,150` trainable params, BEATs frozen). RCNN-style detection on top of BEATs features. 1290 s, early stop epoch 12, best val loss 0.1732.
- **HuBERT-YOLO**. YOLO-style detection head on HuBERT features. 3524 s, early stop epoch 18, best val loss 0.4408.

## 5. Training Procedure

All models share the following training setup:

- **Optimizer**: AdamW with `weight_decay=0.01`.
- **Learning rate**: `2e-3` for from-scratch models, `3e-5` to `5e-4` for pretrained encoders.
- **Scheduler**: `CosineAnnealingLR`.
- **Early stopping**: patience 6 for frame-level models, patience 8 for event detection models.
- **Mixed precision**: `bfloat16` via `torch.amp.autocast`.
- **Augmentation**:
    - **Mixup** with `beta=0.3`, applied with probability 0.3.
    - **SpecAugment**: two frequency masks up to 9 bins, two time masks up to 17 frames.
- **Positive weights**: computed per class from training frequencies and clipped to `[1, 100]` to avoid collapse on the rare harmonic class.

### Losses

- **Frame-level models** use `DualLoss = MSE(sigmoid(onset_logits), onset_target) + BCEWithLogits(sed_channels, sed_target)`. The MSE onset branch provides a sharp temporal prior for burst detection, while the BCE SED head carries class identity.
- **Event detection models** use `BCE(objectness) + BCE(classification on positive frames) + 0.5 * SmoothL1(normalized regression targets)`.

### Checkpointing

All model checkpoints are saved to Google Drive after every model completes training so runs are resumable across Colab sessions.

## 6. Event Extraction Post-Processing

After the model produces per-frame class probabilities (3 channels at 25 fps), events are extracted as follows:

1. **Median filter** the probability streams with a window size selected from `[1, 5, 11, 21]` per class.
2. **Threshold** each stream at a value selected from `[0.05, 0.85]` in steps of `0.05`.
3. **Class-specific extraction**:
    - `b` (single burst): **peak picking** on the onset head with Gaussian smoothing (`sigma=2`) and a minimum peak distance of 5 frames.
    - `mb`, `h`: extract **contiguous runs of above-threshold frames** and convert each run into a single event.
4. **Minimum duration filter**: discard events shorter than `b=20 ms`, `mb=50 ms`, `h=100 ms`.

The median filter size and threshold are tuned per class on the validation set by grid search, maximizing the sed_eval event F1 for that class.

## 7. Evaluation Protocol

Metrics are computed with `sed_eval.sound_event.EventBasedMetrics`. The evaluation is **per-file** and then averaged with `safe_mean` to avoid NaN contamination when a class is absent from the test split of a file.

### Collar configuration

| Class | `t_collar` | `percentage_of_length` | Rationale |
|-------|------------|------------------------|-----------|
| b | 0.050 s | 1.00 (onset-only) | Single bursts are sub-100 ms transients. Boundary matching is ill-defined; onset timing is the clinically relevant quantity. |
| mb | 0.100 s | 0.20 | Multiple-burst trains have meaningful duration. A 20% length tolerance accounts for annotator variance on the trailing boundary. |
| h | 0.200 s | 0.20 | Harmonic events are sustained tonal sounds of variable length. A 200 ms collar and 20% tolerance reflect the looser clinical definition of onset and offset. |

The per-class collar is a conscious departure from the uniform collar used in most DCASE challenges. It is justified by the fact that the three target events differ in duration by more than an order of magnitude. Applying the same collar to all classes would either over-reward harmonic detection (too loose for bursts) or under-reward it (too tight for harmonics).

### Ensemble

The final frame-level ensemble averages the probability outputs of all 11 models from `bowel_sound_final.ipynb` (pre-thresholding), then runs the same extraction and scoring pipeline. No stacking or learned weighting is used.
