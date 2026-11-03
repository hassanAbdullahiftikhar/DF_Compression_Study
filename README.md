# What Survives Compression?

Updated 2026-10-28. **Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); DFDM + Celeb-DF second-dataset study, external test E10 and the I-JEPA probe extension X8 (Amendments 7-8) completed on 28 October 2026; manuscript work remains.**

A controlled study of deepfake detection and four-way, closed-set
manipulation-method attribution under additional JPEG compression.
The reference crops come from an already H.264 C23-compressed
FaceForensics++ mirror.

## Completed work

- Literature review and qualified novelty assessment prepared.
- Main core inventory: 5,000 videos across real, Deepfakes, Face2Face,
  FaceSwap and NeuralTextures.
- Main cohort: 39,898 valid crops from 4,999 videos, in 500 provenance groups.
- All nine main RGB runs completed: reference, JPEG augmentation and paired
  training at seeds 42, 43 and 44, with 20 epochs per run.
- Selected RGB checkpoints and validation operating thresholds recorded.
- Six reference-trained probes completed: pooled RGB, FFT, block DCT,
  Haar wavelet, Laplacian residual and frozen DINOv2-small.
- Probe validation evaluated at reference/Q80/Q40, with frozen thresholds.
- Group bootstrap: 2,000 draws across 70 validation provenance groups.
- Six Q40 known-quality adaptation probes and shuffled-label diagnostics completed.
- Q80 known-quality adaptation: **complete, validation only**.
- Complete-eight-frame sensitivity on validation completed; its definition is frozen for test.
- Registered controls completed on validation (gate G1):
  - nine control RGB runs (C1-DET, C1-ATT, C2-BIN at three seeds);
  - the C2 source readout;
  - face-masked border probes, a metadata/geometry probe and a content-matched decomposition.
- [Comparison register](research/comparison_register.md) v1.0
  frozen 2026-10-23, before any classifier test scoring.
- Gates G2–G4 completed on 24 October 2026:
  - G2: all 49 registered models' scorers reproduced their stored validation predictions;
  - G3: model-freeze manifest and register Amendment 1 recorded;
  - G4: one test-scoring pass at six conditions and one registered analysis.

The validation cohort contains 699 videos (559 fake); the test cohort contains
700 videos in 70 provenance groups.

## Held-out test results (registered analysis, gate G4)

Each registered model was scored once on the official test split at reference and JPEG
Q95/Q80/Q60/Q40/Q20, after the gates above. Q95, Q60 and Q20 played no part in any
selection. Intervals are 95% provenance-group bootstrap intervals (2,000 paired draws,
seed 4242). They condition on the validation-selected models and thresholds and omit
selection and training-seed uncertainty. Full results, including every registered
outcome, are in [main_results_test.json](research/main_results_test.json).

### RGB quality curves

Test balanced accuracy in percent, mean ± sample SD over seeds 42/43/44 (descriptive).
Detection uses each model's frozen validation threshold; attribution is fake-only.

| Model | Ref | Q95 | Q80 | Q60 | Q40 | Q20 |
|---|---:|---:|---:|---:|---:|---:|
| Detection: Reference training | 92.11 ± 2.00 | 91.79 ± 2.73 | 92.89 ± 1.21 | 85.45 ± 0.86 | 76.01 ± 5.84 | 58.72 ± 6.57 |
| Detection: JPEG augmentation | 94.46 ± 0.18 | 94.17 ± 0.21 | 93.15 ± 1.05 | 92.05 ± 0.47 | 91.01 ± 1.58 | 80.09 ± 0.98 |
| Detection: Paired, weight 1 | 95.15 ± 0.54 | 94.79 ± 0.49 | 94.40 ± 0.58 | 93.39 ± 0.56 | 91.82 ± 0.27 | 79.67 ± 3.12 |
| Attribution: Reference training | 98.39 ± 0.36 | 98.27 ± 0.37 | 98.04 ± 0.47 | 95.36 ± 2.32 | 91.13 ± 2.60 | 67.56 ± 3.30 |
| Attribution: JPEG augmentation | 98.21 ± 0.18 | 98.21 ± 0.18 | 97.86 ± 0.18 | 97.38 ± 0.72 | 96.90 ± 0.92 | 94.05 ± 1.49 |
| Attribution: Paired, weight 1 | 98.21 ± 0.36 | 98.21 ± 0.47 | 97.80 ± 0.74 | 97.02 ± 0.81 | 96.61 ± 0.82 | 94.35 ± 0.45 |

### Primary contrast (RQ2): Δ(Q20) = S_attr − S_det for reference-trained models

S is chance-adjusted retention relative to the test reference condition
(1 = no loss, 0 = chance).

**P1-RGB (F1).** Registered reading: **attribution retained more than detection**.
A direction is claimed only when all three seed intervals exclude zero with the same sign.

| Seed | S_det(Q20) | S_attr(Q20) | Δ(Q20) |
|---|---:|---:|---:|
| 42 | 0.31 [0.26, 0.37] | 0.61 [0.55, 0.66] | 0.29 [0.23, 0.35] |
| 43 | 0.26 [0.21, 0.31] | 0.61 [0.56, 0.65] | 0.35 [0.29, 0.40] |
| 44 | 0.03 [0.02, 0.05] | 0.53 [0.48, 0.57] | 0.49 [0.45, 0.53] |

**P1-PROBE (F4).** Two-sided bootstrap p, with Holm correction over the fixed six probes.

| Probe | Ref detection BA | Ref attribution BA | Δ(Q20) | p | Holm bound | Supported |
|---|---:|---:|---:|---:|---:|---|
| FFT | 53.30 | 42.14 | undefined | 1.000 | 0.0167 | no |
| Pooled RGB | 59.11 | 46.79 | 0.07 [-0.15, 0.26] | 0.508 | 0.0125 | no |
| DCT | 60.62 | 46.96 | -0.25 [-0.48, -0.06] | 0.015 | 0.0083 | no |
| Wavelet | 56.79 | 34.46 | 0.05 (no interval) | 1.000 | 0.0500 | no |
| Residual | 56.79 | 39.46 | 0.04 (no interval) | 1.000 | 0.0250 | no |
| DINOv2 | 72.32 | 75.00 | 0.06 [-0.12, 0.20] | 0.437 | 0.0100 | no |

No probe contrast survives Holm correction.
Δ is undefined for FFT: a test reference BA is below chance + 5 points.
Intervals are unavailable for wavelet and residual, because more than 5% of
bootstrap draws had an undefined retention. Several probes are close to chance
even at the reference condition, so they carry little information about retention.

**Exploratory context (not part of the registered contrast).** Detection BA is measured at a
fixed validation operating point, but attribution is a threshold-free argmax. At Q20 the F1 models
flag 39%, 22% and 3% of fake videos (seeds 42, 43 and 44). Their mean
detection AUROC falls only from 98.38% to 75.91%.
Chance-adjusted detection AUROC retention (0.54) is well above the mean fixed-threshold
detection retention (0.20). Much of the detection loss is therefore a shift of scores past
the validation threshold, not a loss of ranking. The P1-RGB reading concerns detection at the frozen
operating point. Extension X1a below compares the two tasks like for like, both by AUROC. It keeps a
large gap in the same direction. (The AUROC retention above is not directly comparable with the
attribution BA retention of 0.58.)

**S1: Δ at the other held-out qualities (F1).**

| Seed | Δ(Q95) | Δ(Q60) |
|---|---:|---:|
| 42 | -0.00 [-0.04, 0.04] | 0.19 [0.14, 0.23] |
| 43 | -0.01 [-0.03, 0.00] | 0.05 [-0.03, 0.12] |
| 44 | 0.03 [-0.01, 0.08] | 0.11 [0.01, 0.18] |

### Compression-aware training (RQ3, S2)

Registered reading: **no reliable benefit over augmentation**. Paired-minus-augmentation (F3 − F2)
attribution at Q20 does not exclude zero positively in any seed. JPEG augmentation (F2) improves
on reference training (F1) at Q20 in every seed, for both detection and attribution.
Differences are in percentage points of balanced accuracy.

| Seed | F3 − F2 attribution, Q20 | F3 − F2 attribution, ref | F3 − F2 detection, Q20 | F2 − F1 attribution, Q20 | F2 − F1 detection, Q20 |
|---|---:|---:|---:|---:|---:|
| 42 | -0.9 [-2.5, 0.9] | 0.2 [-0.7, 1.1] | 2.3 [0.5, 4.0] | 26.3 [22.0, 30.7] | 15.2 [12.8, 17.9] |
| 43 | 1.4 [-0.5, 3.6] | 0.0 [-0.9, 0.9] | 1.4 [-0.5, 3.2] | 23.4 [19.8, 27.0] | 19.2 [16.6, 21.6] |
| 44 | 0.4 [-1.8, 2.5] | -0.2 [-1.3, 0.9] | -5.0 [-7.1, -2.8] | 29.8 [26.3, 33.4] | 29.7 [27.2, 32.0] |

### Known-quality adaptation (S3)

Matched-quality heads minus reference-trained heads, evaluated at the matched quality.
Attribution BA differences have intervals; AUROC differences are point estimates.

| Probe | Attribution, F5 − F4 at Q80 | AUROC Δ at Q80 | Attribution, F6 − F4 at Q40 | AUROC Δ at Q40 |
|---|---:|---:|---:|---:|
| FFT | 2.5 [-0.7, 5.7] | +0.4 | 6.1 [2.7, 9.3] | +1.4 |
| Pooled RGB | 1.3 [-0.7, 3.2] | +0.0 | 2.1 [0.2, 4.1] | -0.4 |
| DCT | 17.7 [13.4, 22.0] | +9.8 | 17.0 [13.6, 20.4] | +8.8 |
| Wavelet | 1.4 [-1.1, 3.9] | -0.6 | 3.6 [0.7, 6.4] | +0.4 |
| Residual | -0.2 [-2.0, 1.8] | +0.1 | 0.0 [-2.9, 2.9] | +0.3 |
| DINOv2 | 0.4 [-1.8, 2.5] | -0.1 | 5.7 [2.3, 8.9] | +2.6 |

Matched heads recover 17.7 (Q80) and 17.0 (Q40) points of DCT attribution.
At Q40, attribution also improves, with intervals excluding zero, for
FFT, pooled RGB, wavelet and DINOv2. A reference-trained loss therefore partly reflects the
mismatch between training and test compression. It does not show that the information is absent.

### Task sharing (S4) and source supervision (S5)

S4 compares separate single-task models (C1) with the shared F1 model. The last column is the
separate system's Δ(Q20) minus F1's.

| Seed | C1-DET − F1 detection, ref | C1-DET − F1 detection, Q20 | C1-ATT − F1 attribution, ref | C1-ATT − F1 attribution, Q20 | Separate − shared Δ(Q20) |
|---|---:|---:|---:|---:|---:|
| 42 | -4.6 [-7.8, -1.5] | -11.2 [-13.4, -9.2] | -1.4 [-3.2, 0.2] | 9.1 [5.5, 12.7] | 0.39 [0.32, 0.46] |
| 43 | -1.7 [-4.6, 1.2] | -10.0 [-11.7, -8.2] | 0.5 [-0.7, 2.0] | 11.4 [8.2, 15.0] | 0.39 [0.33, 0.46] |
| 44 | 0.4 [-2.9, 3.7] | 1.7 [0.8, 2.6] | -0.7 [-2.0, 0.4] | 15.4 [11.1, 19.5] | 0.17 [0.11, 0.24] |

At Q20 the separate fake-only attribution model is better than the shared model in every seed. The
separate detector is worse in seeds 42 and 43 and better in seed 44. The separate system's Δ(Q20) is larger than F1's in
every seed, so the P1-RGB direction does not depend on the two tasks sharing capacity.

S5 compares paired training with and without source supervision (C2).

| Seed | C2-R − F3-R attribution, ref | C2-R − F3-R attribution, Q20 | C2-BIN − F3 detection, ref | C2-BIN − F3 detection, Q20 |
|---|---:|---:|---:|---:|
| 42 | -5.9 [-8.4, -3.6] | -24.3 [-27.7, -20.7] | -3.3 [-5.6, -1.1] | -1.5 [-4.2, 1.2] |
| 43 | -7.5 [-10.4, -4.8] | -23.6 [-27.0, -20.2] | -2.3 [-4.3, -0.4] | -4.9 [-7.1, -2.7] |
| 44 | -8.4 [-11.1, -5.7] | -24.6 [-27.9, -21.1] | -5.2 [-7.3, -3.1] | -1.5 [-3.8, 0.8] |

Without the source loss, the same fixed linear readout recovers 5.9–8.4
points less attribution at the reference condition and 23.6–24.6 points
less at Q20. Detection is also lower at the reference condition (2.3–5.2 points).
At Q20 detection is lower, with an interval excluding zero, only in seed 43.

### Shortcut and content controls (S6, S7)

Test balanced accuracy in percent. A result is flagged "material" when the 95% lower bound is
above 30% for attribution (chance 25%) or above 55% for detection (chance 50%).

| Control | Attribution BA | Detection BA | Material flags |
|---|---:|---:|---|
| C3a face-masked FFT (ref) | 29.1 [27.0, 31.2] | 51.5 [48.4, 54.6] | none |
| C3a face-masked Pooled RGB (ref) | 27.0 [25.2, 28.7] | 50.5 [48.3, 52.9] | none |
| C3a face-masked DCT (ref) | 29.6 [27.5, 31.8] | 48.8 [46.1, 51.4] | none |
| C3a face-masked Wavelet (ref) | 30.0 [28.2, 32.0] | 52.7 [50.6, 54.7] | none |
| C3a face-masked Residual (ref) | 25.9 [24.1, 27.7] | 54.1 [50.4, 57.6] | none |
| C3a face-masked DINOv2 (ref) | 30.4 [28.0, 32.9] | 53.6 [51.6, 55.7] | none |
| C3b metadata/geometry (video level) | 35.4 [33.0, 37.9] | 55.0 [52.7, 57.6] | attribution |

None of the face-masked probes is flagged. The video-level metadata/geometry probe is flagged for
**attribution**: file size, frame count and face-box geometry predict
the manipulation method above chance. Under the register, full-crop attribution findings carry
this dataset-level caveat. It does not show that the crop models use that cue.

C3c content-matched decomposition (chance 50%): family BA (identity swap vs. reenactment) and the
two within-family BAs, at ref, Q60 and Q20.

| Model | Family | DF vs FS | F2F vs NT | Family Q60 | DF vs FS Q60 | F2F vs NT Q60 | Family Q20 | DF vs FS Q20 | F2F vs NT Q20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference-trained RGB (seed mean) | 98.93 | 100.00 | 98.57 | 96.85 | 99.64 | 96.55 | 80.48 | 95.12 | 70.71 |
| JPEG-augmented RGB (seed mean) | 98.57 | 100.00 | 99.05 | 98.04 | 99.64 | 98.33 | 96.49 | 99.29 | 95.00 |
| Fake-only RGB (C1-ATT) (seed mean) | 98.51 | 100.00 | 98.57 | 97.32 | 99.88 | 97.02 | 84.05 | 97.86 | 84.17 |

S7 (complete-eight supports) flagged 10 status changes. None of them
changes whether an interval excludes zero; all are sign changes of points near zero. The support set
is 691 complete videos and 66 complete groups.

Exploratory test-crop distortion, averaged over the 5,581 test crops:

| JPEG | Mean bytes | RGB PSNR (dB) | Luma SSIM |
|---|---:|---:|---:|
| Q95 | 14154 | 44.86 | 0.9955 |
| Q80 | 7071 | 41.12 | 0.9876 |
| Q60 | 5035 | 38.56 | 0.9779 |
| Q40 | 4067 | 36.67 | 0.9667 |
| Q20 | 2998 | 33.40 | 0.9368 |

## Post-freeze extensions (register Amendments 2–3)

These analyses were designed **after** the registered test results were seen. Each was pre-registered
in Amendment 2 before it ran, selected on validation only, frozen in Amendment 3 (all nine new models
reproduced their validation predictions exactly) and scored once. They are not registered results and
do not change the readings above. Full results:
[main_results_extensions.json](research/main_results_extensions.json).

### Operating point (X1, X2, X3)

F1 at Q20. Detection BA is shown at the frozen threshold and at a per-quality threshold chosen on
validation at Q20 (X1b). The AUROC columns score both tasks without a threshold; attribution AUROC is
the mean one-vs-rest AUROC over the four methods on fake videos (X1a). The last three columns are Δ
under the registered metric, with per-quality thresholds, and with AUROC for both tasks.

| Seed | Detection BA, frozen | Detection BA, Q20 threshold | Detection AUROC | Attribution AUROC | Δ (registered) | Δ (Q20 threshold) | Δ (AUROC) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 63.9 [61.7, 66.2] | 67.4 [64.4, 70.2] | 74.7 [72.2, 77.3] | 96.1 [95.0, 97.2] | 0.29 [0.23, 0.35] | 0.21 [0.14, 0.28] | 0.42 [0.37, 0.46] |
| 43 | 60.9 [58.9, 62.7] | 70.9 [68.0, 73.6] | 80.3 [77.7, 82.8] | 95.9 [94.8, 96.8] | 0.35 [0.29, 0.40] | 0.11 [0.04, 0.17] | 0.29 [0.25, 0.33] |
| 44 | 51.3 [50.7, 52.1] | 64.6 [61.7, 67.6] | 72.7 [70.3, 75.2] | 93.7 [92.3, 95.0] | 0.49 [0.45, 0.53] | 0.16 [0.08, 0.25] | 0.41 [0.36, 0.45] |

The direction survives both checks. Recalibrating the detection threshold recovers part of the
detection loss. Comparing AUROC with AUROC still shows attribution retained far better: at Q20,
attribution AUROC stays about 94–96% while detection AUROC falls to about 73–80%.

| EfficientNet-B4, Q20 | Detection AUROC | Attribution AUROC | Δ (AUROC) |
|---|---:|---:|---:|
| EB4-F1 seed 42 | 81.9 [79.1, 84.6] | 95.8 [94.7, 96.8] | 0.26 [0.22, 0.31] |
| EB4-F1 seed 43 | 80.6 [77.7, 83.5] | 94.4 [93.2, 95.6] | 0.27 [0.22, 0.31] |
| EB4-F1 seed 44 | 80.5 [77.8, 83.1] | 95.3 [93.9, 96.6] | 0.29 [0.24, 0.33] |

**X2, frozen-encoder Q20 readout.** A linear readout refitted at Q20 on the frozen reference-trained
F1 encoder recovers most of the loss:

| Seed | Native detection BA | Readout detection BA | Native attribution BA | Readout attribution BA |
|---|---:|---:|---:|---:|
| 42 | 63.9 [61.7, 66.2] | 79.3 [76.2, 82.1] | 69.5 [65.2, 73.6] | 88.9 [85.9, 91.8] |
| 43 | 60.9 [58.9, 62.7] | 79.8 [76.8, 82.9] | 69.5 [66.1, 72.9] | 90.0 [87.5, 92.3] |
| 44 | 51.3 [50.7, 52.1] | 78.3 [74.9, 81.6] | 63.7 [60.7, 66.8] | 87.1 [84.5, 89.6] |

The Q20 information is still linearly available in the encoder; the reference-trained heads no longer
read it. This is evidence against irreversible loss for this model, not proof of physical presence.

**X3, intervals pooled over seeds** (training seeds and groups both resampled):

| Contrast | Pooled estimate [95% interval] |
|---|---:|
| F1 Δ(Q20) | 0.38 [0.28, 0.48] |
| EfficientNet-B4 F1 Δ(Q20) | 0.42 [0.36, 0.50] |
| F3 − F2 attribution BA, Q20 (points) | 0.3 [-1.4, 2.1] |
| F3 − F2 detection BA, Q20 (points) | -0.4 [-4.5, 2.9] |
| F2 − F1 attribution BA, Q20 (points) | 26.5 [22.3, 30.7] |
| F2 − F1 detection BA, Q20 (points) | 21.4 [14.9, 28.5] |
| EB4 F2 − F1 attribution BA, Q20 (points) | 19.9 [15.7, 23.8] |
| EB4 F2 − F1 detection BA, Q20 (points) | 23.5 [20.4, 26.6] |

### Whole-video H.264/H.265 re-encoding (X5)

Each test video was re-encoded in memory at six settings. The scheduled frames were re-cropped with
the stored boxes; every reference crop was first reproduced pixel for pixel. Frozen models and
thresholds were used. Values are test BA in percent, seed means.

| Model | H.264 CRF 18 | H.264 CRF 23 | H.264 CRF 32 | H.264 CRF 40 | H.265 CRF 23 | H.265 CRF 32 |
|---|---:|---:|---:|---:|---:|---:|
| Detection: Reference training | 93.81 | 92.29 | 87.50 | 73.54 | 81.64 | 74.67 |
| Detection: JPEG augmentation | 93.10 | 92.38 | 82.17 | 67.56 | 92.38 | 86.93 |
| Detection: Paired | 93.99 | 93.07 | 82.77 | 68.96 | 93.33 | 87.92 |
| Detection: EfficientNet-B4, reference | 95.83 | 94.43 | 86.93 | 73.87 | 80.71 | 69.17 |
| Detection: EfficientNet-B4, augmentation | 95.71 | 95.36 | 87.47 | 73.18 | 93.60 | 83.75 |
| Attribution: Reference training | 98.27 | 97.74 | 90.77 | 69.94 | 96.49 | 87.32 |
| Attribution: JPEG augmentation | 97.92 | 97.62 | 93.45 | 76.67 | 97.26 | 92.08 |
| Attribution: Paired | 98.04 | 97.74 | 94.40 | 78.45 | 96.73 | 92.32 |
| Attribution: EfficientNet-B4, reference | 98.93 | 98.51 | 91.90 | 68.21 | 97.98 | 89.58 |
| Attribution: EfficientNet-B4, augmentation | 98.75 | 98.63 | 94.40 | 78.63 | 98.27 | 91.61 |
| Mean crop PSNR vs. reference (dB) | 41.2 | 39.8 | 35.4 | 31.3 | 39.1 | 35.2 |
| Mean bitrate (kbps) | 1498 | 972 | 367 | 168 | 670 | 227 |

F1 Δ (attribution minus detection retention) per seed:

| Seed | H.264 CRF 18 | H.264 CRF 23 | H.264 CRF 32 | H.264 CRF 40 | H.265 CRF 23 | H.265 CRF 32 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | -0.04 [-0.09, 0.01] | -0.02 [-0.06, 0.02] | 0.06 [0.01, 0.11] | 0.13 [0.07, 0.20] | 0.21 [0.14, 0.28] | 0.33 [0.24, 0.42] |
| 43 | -0.03 [-0.07, 0.00] | 0.00 [-0.05, 0.05] | -0.00 [-0.07, 0.06] | -0.06 [-0.15, 0.02] | 0.17 [0.10, 0.25] | 0.21 [0.14, 0.30] |
| 44 | -0.06 [-0.12, -0.01] | -0.02 [-0.07, 0.02] | -0.04 [-0.14, 0.04] | 0.08 [-0.01, 0.17] | 0.30 [0.22, 0.38] | 0.25 [0.15, 0.34] |

- **H.265:** the JPEG pattern repeats (Δ > 0 in every seed).
- **H.264 at CRF 40, the harshest setting:** attribution falls as far as detection, and Δ changes sign
  across seeds. The direction is not established there.
- **Augmentation does not transfer to H.264 detection:** under H.264 CRF 32/40, JPEG augmentation (F2) has
  lower seed-mean detection BA than reference training (at CRF 40, lower with intervals excluding zero in two
  of three seeds), although it still helps attribution. EfficientNet-B4 augmentation shows no detection gain there.
- Equal CRF across codecs is not equal distortion; compare by PSNR and bitrate.

### EfficientNet-B4 replication (X6)

| Model | Ref | Q95 | Q80 | Q60 | Q40 | Q20 |
|---|---:|---:|---:|---:|---:|---:|
| Detection: EfficientNet-B4, reference | 95.98 | 91.31 | 93.81 | 88.39 | 80.89 | 61.28 |
| Detection: EfficientNet-B4, augmentation | 96.76 | 96.49 | 95.77 | 94.46 | 93.69 | 84.76 |
| Attribution: EfficientNet-B4, reference | 99.23 | 98.93 | 98.10 | 94.82 | 90.54 | 74.76 |
| Attribution: EfficientNet-B4, augmentation | 99.11 | 99.05 | 98.81 | 98.51 | 97.62 | 94.70 |

Under the P1-RGB rule, applied descriptively, the reading for EfficientNet-B4 is
**attribution retained more** (Δ(Q20) 0.49 [0.44, 0.55], 0.38 [0.33, 0.43], 0.40 [0.35, 0.45]).
JPEG augmentation again gives large Q20 gains, so the main findings are not specific to ResNet-18.

### Metadata cue (X4, exploratory)

At the reference condition, no crop model's attribution accuracy differs between videos where the metadata probe is right and where it is wrong (every interval includes zero).
Re-weighting test videos to balance the metadata cue changes attribution BA by under one point. This
does not suggest the crop models rely on the cue at the reference condition. At Q20 the intervals are wide.

### CLIP ViT-L/14 probes (X7, Amendments 4–5)

Frozen OpenAI CLIP ViT-L/14 image embeddings with ridge heads, using the F4/F5/F6 procedures. CLIP ran
deterministically from its TorchScript checkpoint with no added packages, and re-extracting validation
reproduced every stored probability exactly. Test balanced accuracy and AUROC in percent:

| Probe | Ref | Q95 | Q80 | Q60 | Q40 | Q20 |
|---|---:|---:|---:|---:|---:|---:|
| Detection BA: CLIP ViT-L/14 probe | 80.89 | 58.84 | 74.91 | 67.59 | 72.95 | 65.71 |
| Detection BA: DINOv2 probe (F4) | 72.32 | 72.68 | 71.96 | 70.71 | 66.61 | 61.16 |
| Attribution BA: CLIP ViT-L/14 probe | 83.39 | 70.18 | 70.71 | 62.32 | 61.07 | 42.14 |
| Attribution BA: DINOv2 probe (F4) | 75.00 | 73.04 | 71.79 | 67.14 | 62.50 | 53.21 |
| Detection AUROC: CLIP probe | 90.91 | 88.33 | 84.86 | 81.66 | 80.82 | 76.89 |

- **Strongest frozen probe at reference.** CLIP exceeds DINOv2 by
  8.4 [4.3, 12.1] points of attribution BA.
- **Opposite sign under JPEG at the operating point.** CLIP's argmax attribution collapses at Q20, giving
  Δ(Q20) = -0.22 [-0.35, -0.09] (bootstrap p < 0.001; descriptive, outside the Holm family). That is the
  reverse of the RGB models.
- **Positive again without a threshold.** Attribution AUROC (83.4 [81.3, 85.5]) still exceeds
  detection AUROC (76.9 [74.2, 79.6]), so Δ_AUC(Q20) = 0.07 [0.01, 0.14].
  The four methods remain separable; the reference-trained head's decisions drift.
- **Calibration again at Q95.** CLIP detection BA at Q95 is lower than at Q80 while its AUROC barely
  changes, so the frozen threshold misfires at a quality it never saw.
- **Matched heads recover attribution:** 10.2 [7.1, 13.2]
  points at Q80 and 16.2 [12.5, 20.0] at Q40.
- **Codecs:** Δ is near zero at every H.264 setting and positive at H.265 CRF 32
  (0.21 [0.10, 0.32]).

### I-JEPA ViT-H/14 probes (X8, Amendments 7–8)

Frozen I-JEPA ViT-H/14 target-encoder features (Meta's ImageNet-1k checkpoint; mean of the final-layer
tokens) with the same ridge heads and procedures as X7. The encoder is a plain-PyTorch copy loaded with
strict key matching, and re-extracting validation reproduced every stored probability exactly. This
extension was designed after every other result was known. Test balanced accuracy and AUROC in percent:

| Probe | Ref | Q95 | Q80 | Q60 | Q40 | Q20 |
|---|---:|---:|---:|---:|---:|---:|
| Detection BA: I-JEPA ViT-H/14 probe | 72.23 | 69.46 | 73.39 | 66.96 | 58.39 | 50.98 |
| Detection BA: CLIP probe (X7) | 80.89 | 58.84 | 74.91 | 67.59 | 72.95 | 65.71 |
| Detection BA: DINOv2 probe (F4) | 72.32 | 72.68 | 71.96 | 70.71 | 66.61 | 61.16 |
| Attribution BA: I-JEPA ViT-H/14 probe | 82.50 | 82.14 | 80.89 | 73.57 | 64.11 | 48.57 |
| Attribution BA: CLIP probe (X7) | 83.39 | 70.18 | 70.71 | 62.32 | 61.07 | 42.14 |
| Attribution BA: DINOv2 probe (F4) | 75.00 | 73.04 | 71.79 | 67.14 | 62.50 | 53.21 |
| Detection AUROC: I-JEPA probe | 84.58 | 83.29 | 81.11 | 77.48 | 74.43 | 69.69 |

- **Strong attribution at reference.** I-JEPA exceeds DINOv2 by
  7.5 [3.8, 11.4] points of attribution BA and matches CLIP
  (-0.9 [-4.8, 3.2]). Its reference detection is below CLIP's.
- **Same sign as the trained CNNs at the operating point:** Δ(Q20) = 0.37 [0.30, 0.43] (bootstrap
  p < 0.001; descriptive, outside the Holm family). Much of this is a threshold failure: at Q20 the frozen
  threshold labels almost every video one way, so detection BA is
  50.98%.
- **Positive without a threshold too:** Δ_AUC(Q20) = 0.19 [0.10, 0.27], with detection AUROC
  69.7 [67.3, 72.6] and attribution AUROC 84.7 [82.8, 86.6]. This is the
  better-supported reading.
- **The stated expectation was not borne out.** Amendment 7 anticipated smaller JPEG losses from a
  latent-prediction encoder. At Q20, I-JEPA minus DINOv2 is
  -10.2 [-12.8, -7.8] points of detection BA and
  -4.6 [-7.9, -1.4] of attribution BA. I-JEPA still keeps
  more attribution than CLIP (6.4 [2.7, 10.4] points).
- **Matched heads:** 14.6 [10.7, 18.6] points of attribution
  BA at Q40, and 0.2 [-2.5, 3.0] at Q80.
- **Codecs:** as for the trained CNNs, Δ is near zero at every H.264 setting and positive at H.265
  (CRF 23: 0.55 [0.42, 0.67]; CRF 32: 0.51 [0.39, 0.63]).

The sign of a fixed-operating-point detection/attribution contrast therefore depends on the representation.
The threshold-free contrast Δ_AUC(Q20) also depends on it:

- **Attribution retained better** for both trained CNNs, DINOv2 (0.20 [0.13, 0.25]),
  CLIP and I-JEPA.
- **Near zero** for the pooled-RGB, wavelet and residual descriptors.
- **Attribution retained worse** for the FFT (-0.33 [-0.52, -0.16]) and DCT
  (-0.20 [-0.33, -0.07]) descriptors. Their method cues appear tied to
  compression-sensitive frequency statistics, consistent with DCT's recovery under matched heads.

No single direction holds across representations.

## Second dataset: DFDM + Celeb-DF (separate register)

A separate study with its own pre-registered register ([dfdm_register.md](research/dfdm_register.md),
frozen before any crop extraction), its own validation selection, a model freeze (Amendment D-2) and a
single test pass. Full results are in [main_results_dfdm.json](research/main_results_dfdm.json).

**Setup:**
- **Fakes:** DFDM (Jia, Li & Lyu 2022), 430 clips × 5 related autoencoder face-swap generators. The
  reference is lossless H.264 4:4:4.
- **Reals:** Celeb-DF v2, the exact source videos the fakes were made from, re-encoded with DFDM's own
  x264 settings.
- **Split:** identity-disjoint, from the identity graph: test has 16 identities, 95 clips and 570 videos in
  19 groups.
- **Crops:** face boxes come from the real video and are reused for all five fakes, so backgrounds are
  pixel-identical and geometry cannot reveal the generator.
- **Models:** ResNet-18 and EfficientNet-B4, reference-only and JPEG-augmented, 3 seeds each. Attribution
  is five-way (chance 20%).

Test results in percent, seed means:

| Model | Ref | Q80 | Q60 | Q40 | Q20 | H.264 CRF 10 | H.264 CRF 23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribution BA: ResNet-18, reference | 46.81 | 27.93 | 22.39 | 21.75 | 20.70 | 46.53 | 23.58 |
| Attribution BA: ResNet-18, augmentation | 35.58 | 33.96 | 32.49 | 31.93 | 26.25 | 35.86 | 33.68 |
| Attribution BA: EfficientNet-B4, reference | 53.61 | 30.81 | 24.56 | 22.74 | 20.98 | 46.60 | 28.84 |
| Attribution BA: EfficientNet-B4, augmentation | 40.28 | 41.82 | 37.96 | 36.14 | 32.70 | 42.04 | 39.16 |
| Detection BA: ResNet-18, reference | 95.26 | 97.02 | 90.11 | 60.77 | 52.46 | 92.46 | 81.51 |
| Detection BA: ResNet-18, augmentation | 99.51 | 99.09 | 98.25 | 97.02 | 89.09 | 99.47 | 92.39 |
| Detection BA: EfficientNet-B4, reference | 88.60 | 97.79 | 81.96 | 66.56 | 60.46 | 66.14 | 50.49 |
| Detection BA: EfficientNet-B4, augmentation | 99.82 | 99.93 | 99.79 | 98.91 | 91.47 | 100.00 | 95.58 |
| Detection AUROC: ResNet-18, reference | 100.00 | 100.00 | 99.37 | 87.75 | 66.76 | 100.00 | 98.92 |
| Detection AUROC: ResNet-18, augmentation | 100.00 | 99.98 | 99.90 | 99.67 | 98.92 | 99.99 | 99.98 |
| Detection AUROC: EfficientNet-B4, reference | 100.00 | 99.87 | 93.28 | 81.29 | 69.26 | 100.00 | 86.46 |
| Detection AUROC: EfficientNet-B4, augmentation | 100.00 | 100.00 | 100.00 | 99.97 | 99.74 | 100.00 | 99.85 |

**Primary contrast D-P1** (Δ(Q20) = S_attr − S_det) and its threshold-free version:

| Seed | ResNet-18 Δ(Q20) | ResNet-18 Δ_AUC(Q20) | EfficientNet-B4 Δ(Q20) | EfficientNet-B4 Δ_AUC(Q20) |
|---|---:|---:|---:|---:|
| 42 | -0.13 [-0.22, -0.05] | -0.38 [-0.47, -0.30] | -0.20 [-0.34, -0.10] | -0.28 [-0.38, -0.17] |
| 43 | 0.06 [0.01, 0.11] | -0.15 [-0.25, -0.08] | -0.33 [-0.48, -0.21] | -0.29 [-0.37, -0.21] |
| 44 | -0.00 [-0.01, 0.00] | -0.40 [-0.48, -0.34] | -0.20 [-0.33, -0.08] | -0.28 [-0.38, -0.18] |

Registered readings: ResNet-18 — **direction not established**; EfficientNet-B4 —
**attribution retained less than detection**. Threshold-free, attribution separability is lost faster than detection
separability in all six models.

- **Fine-grained attribution is fragile.** Telling five closely related generators apart falls to chance
  by JPEG Q60. At Q20 the reference-trained models assign almost every fake to one or two generators.
  This is the opposite of FF++, where coarse method differences survived.
- **Controls:**
  - The compression-history probe (block-DCT statistics, detection only) reaches
    84.0 [80.6, 87.9]% detection BA at ref. It is **material**: DFDM detection results carry
    a compression-history caveat, because the real faces carry an earlier MPEG-4 encode.
  - The face-masked probe is exactly at chance, so backgrounds carry no signal.
- **JPEG augmentation** keeps detection high under compression. It improves Q20 attribution, but costs
  8–17 points of reference attribution.
- **Frozen DINOv2 and CLIP probes** detect well but cannot attribute DFDM generators, even at ref.

**Synthesis with FF++.** Whether method evidence survives compression depends on how fine the attribution
task is:
- coarse method differences survive (FF++ four-way, and Deepfakes vs. FaceSwap);
- fine distinctions collapse first (Face2Face vs. NeuralTextures within FF++, and DFDM's five autoencoder
  variants).

### External detection test E10 (FF++ Amendment 6)

Frozen FF++ detectors on the 518 official Celeb-DF v2 test videos (124 identity groups). Detection AUROC,
seed means, percent:

| FF++ model | Ref | Q95 | Q80 | Q60 | Q40 | Q20 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18, reference (F1) | 63.32 | 63.66 | 63.42 | 58.53 | 56.87 | 56.11 |
| ResNet-18, augmentation (F2) | 71.95 | 72.60 | 73.28 | 73.01 | 71.52 | 62.58 |
| ResNet-18, paired (F3) | 68.00 | 68.18 | 69.82 | 68.21 | 66.06 | 60.78 |
| EfficientNet-B4, reference | 65.99 | 66.46 | 65.44 | 61.26 | 58.12 | 53.43 |
| EfficientNet-B4, augmentation | 72.08 | 71.98 | 72.36 | 73.20 | 72.70 | 64.96 |

Transfer is weak, as is typical across datasets. JPEG augmentation improves both external AUROC and its
robustness to compression.

## Validation: main RGB results at JPEG Q40

Mean ± sample standard deviation across seeds 42/43/44.
Both columns are balanced accuracy in percent; SD is in percentage points.
These are descriptive seed summaries, not confidence intervals.

| Training approach | Detection | Fake-only attribution |
|---|---:|---:|
| Reference training | 76.72 ± 5.73 | 91.54 ± 2.50 |
| JPEG augmentation | 92.28 ± 1.34 | 97.37 ± 0.10 |
| Paired training, weight 1 | 92.64 ± 0.45 | 97.49 ± 0.31 |

JPEG augmentation improves Q40 validation performance in all three seeds.
The paired approach's incremental advantage is small and mixed.
All per-seed Q40 paired-versus-augmentation confidence intervals include
zero for detection and attribution.

Checkpoint selection used mean task balanced accuracy across validation
reference/Q80/Q40 at threshold 0.5. Each selected model's detection threshold
was subsequently chosen on those validation conditions and frozen.

## Validation: reference-trained representation probes at Q40

Values are validation balanced accuracy in percent, with frozen detection
thresholds. Pooled RGB is a pixel descriptor, separate from the trained ResNet.

| Representation | Detection | Fake-only attribution |
|---|---:|---:|
| FFT | 57.48 | 33.11 |
| Pooled RGB | 60.40 | 45.82 |
| DCT | 50.36 | 25.54 |
| Wavelet | 55.16 | 29.73 |
| Residual | 57.80 | 37.79 |
| DINOv2 | 66.84 | 59.78 |

DINOv2 has the highest absolute Q40 performance among these probes.
The reference-trained DCT probe becomes nearly chance-level under JPEG.
FFT detection retention and wavelet attribution retention have unstable
reference denominators; their affected confidence intervals are unavailable.

The usable probe intervals do not establish a general ordering of detection
versus attribution degradation at Q40. Representation, classifier capacity
and pretraining differ across tracks.

## Validation: known-quality adaptation control at Q80 and Q40

At each quality, new scaling and ridge heads were fitted on training crops
compressed at that quality and selected using validation at the same quality.
Values below are percentages.

| Metric | Reference-trained at Q80 | Q80-trained at Q80 | Reference-trained at Q40 | Q40-trained at Q40 |
|---|---:|---:|---:|---:|
| DCT: Detection AUROC | 53.08 | 63.89 | 52.55 | 62.09 |
| DCT: Attribution balanced accuracy | 28.58 | 44.53 | 25.54 | 39.36 |
| DINOv2: Detection AUROC | 77.68 | 78.61 | 71.40 | 74.58 |
| DINOv2: Attribution balanced accuracy | 70.14 | 70.85 | 59.78 | 65.85 |

At Q80, the reference-trained DCT probe falls close to chance, and
Q80-matched heads restore much of its reference-condition score. For the
other four representations, Q80 adaptation changes AUROC and fake-only
attribution by at most about two percentage points. These differences
have no bootstrap intervals and are descriptive.

DCT's partial recovery supports training/compression mismatch as a
contributor to its earlier failure. These results demonstrate usable label
information on this validation cohort; they do not establish a physical
forensic fingerprint. Threshold selection domains differ between the two
regimes, so AUROC and fake-only attribution are the clearest comparisons.

## Complete-eight-frame sensitivity (validation)

Partial videos were re-examined with frozen thresholds and selections.
On `complete_video` support (691 of
699 videos), the largest change in any model's
detection or attribution balanced accuracy is
0.46 points.
On `complete_group` support (66 of 70 groups),
it is 0.91 points.
RGB paired contrasts changing status: 0 on complete-video support and
5 on complete-group support. Every changed contrast is near zero, and its
interval includes or borders zero on both supports.
The Q40 conclusions above are unchanged.

## Registered controls on validation (gate G1)

These controls are registered in the [comparison register](research/comparison_register.md).
Their test comparisons are in the held-out test section above. Values here are validation percentages
at frozen thresholds: the seed mean over 42/43/44 for RGB models. They are descriptive,
with no intervals.

| Validation quantity | Reference | Q40 |
|---|---:|---:|
| Detection BA: shared F1 / separate C1-DET | 95.41 / 92.53 | 76.72 / 66.49 |
| Attribution BA: shared F1 / separate C1-ATT | 98.51 / 98.39 | 91.54 / 95.12 |
| Detection BA: F3 / binary-only paired C2-BIN | 95.89 / 92.46 | 92.64 / 88.26 |
| Readout R attribution BA: F3-R / C2-R | 98.45 / 92.36 | 97.97 / 87.41 |

- **C1 (task sharing):** a separate detector is weaker than the shared two-head model,
  especially at Q40. A separate fake-only attribution model is at least as good.
- **C2 (source supervision):** removing the source loss from paired training lowers
  detection. It also lowers the source information that a fixed linear readout can recover.

**C3a and C3b (border and metadata shortcut checks).** Attribution BA has chance 25%,
and the shuffle column is the best of five shuffled-label refits.
Detection BA has chance 50%. All values are at the reference condition.

| Probe | Full crop attribution | Face-masked attribution | Best shuffle | Face-masked detection |
|---|---:|---:|---:|---:|
| FFT | 41.84 | 30.04 | 27.18 | 56.65 |
| Pooled RGB | 47.25 | 26.67 | 25.59 | 52.98 |
| DCT | 49.00 | 32.53 | 27.54 | 54.59 |
| Wavelet | 31.12 | 28.60 | 27.36 | 51.56 |
| Residual | 38.48 | 28.62 | 27.36 | 55.15 |
| DINOv2 | 75.49 | 30.40 | 26.65 | 55.22 |
| Metadata/geometry (C3b, video level) | — | 34.51 | 30.76 | 58.63 |

With the face masked, the probes fall close to chance. Video metadata and
face-box geometry carry a modest cue. The registered "material" flags were
judged on test intervals (see above).

**C3c (content-matched decomposition).** Every fake pair ID has all four methods
(139 of
140 validation quadruplets fully usable).
Below are family BA (identity swap vs. reenactment) and the two within-family BAs
(chance 50%), first at the reference condition and then at Q40.

| Model | Family | DF vs. FS | F2F vs. NT | Family Q40 | DF vs. FS Q40 | F2F vs. NT Q40 |
|---|---:|---:|---:|---:|---:|---:|
| Reference-trained RGB (seed mean) | 99.05 | 100.00 | 98.45 | 93.92 | 99.64 | 94.29 |
| JPEG-augmented RGB (seed mean) | 99.17 | 100.00 | 98.21 | 98.33 | 99.88 | 97.50 |
| Reference-trained DINOv2 probe | 79.96 | 91.76 | 83.57 | 72.84 | 85.65 | 70.00 |

For reference-trained RGB, Deepfakes vs. FaceSwap stays near perfect at Q40.
Its losses come from family separation and from Face2Face vs. NeuralTextures.

## Interpretation and limitations

- Validation results are selection-stage evidence and optimistic. Test results come from one
  registered pass. Their intervals condition on the selected models and thresholds and
  exclude selection and training-seed uncertainty.
- The registered P1-RGB reading is that, at Q20, attribution is retained more than detection.
  This is the reverse of the proposal's expectation that method cues would be more fragile. It
  describes detection at a fixed validation operating point. In the post-freeze extensions, the same
  direction holds for the trained CNNs with per-quality thresholds, with AUROC for both tasks, and for
  EfficientNet-B4. It does **not** generalize everywhere:
  - under H.264 re-encoding at CRF 40, attribution falls as far as detection;
  - the frozen CLIP probe reverses the sign at its operating point;
  - the FFT and DCT descriptors retain attribution worse even without a threshold.

  The defensible claim is representation-, codec- and granularity-specific. On DFDM's fine-grained generator
  attribution the direction does not hold: attribution collapses before detection.
- No reference-trained probe contrast survives Holm correction, and several probes are close to
  chance even on reference test crops. The probe track does not establish a representation ranking.
- JPEG augmentation is the clear compression-aware gain. Paired training shows no reliable
  benefit over augmentation on test.
- The metadata/geometry probe is material for attribution on test. A dataset-level cue exists, and
  attribution claims carry that caveat.
- Attribution concerns four known manipulation methods on fake videos.
  It does not identify individual generator checkpoints or unknown methods.
- JPEG is additional compression of C23-derived crops.
- Official splits separate known original IDs/provenance groups.
  Identity/YouTube-level independence and official-byte authenticity
  of the mirror are not established.
- Failures were logged without replacement. One zero-crop validation video,
  Deepfakes/638_640, was excluded; partial videos remain included.
- Classifier failure does not establish irreversible loss of identifying information.
- Broad first-ever novelty claims are unsupported. The candidate contribution
  is a bounded comparative empirical study.

## Remaining work

1. Generate final figures and tables from the registered test aggregate
   (research/main_results_test.json); report every registered outcome,
   including null and negative ones, with validation and test kept separate.
2. Any further error analysis (for example the Q20 detection score shift) is
   exploratory and must be labelled as such; it cannot change registered readings.
3. Second dataset (DFDM + Celeb-DF) and external test E10 are complete; integrate their results
   into the paper as a separately registered replication.
4. Recompression history (E06) and other extensions are outside the frozen core;
   each needs a register amendment written before its own test scoring.
5. Refresh the literature/novelty assessment, write the manuscript and revise
   its claims against the test evidence.

Whole-video H.264/H.265 was added as post-freeze extension X5. DFDM + Celeb-DF were added as a separately registered study.

## Measured environment and feasibility

Current environment: Python 3.11.9, torch 2.7.1+cu128,
torchvision 0.22.1+cu128, CUDA runtime 12.8, NumPy 2.4.6,
Pillow 12.3.0, PyAV 18.1.0 and OpenCV-headless 4.14.0.94.

Implemented JPEG experiments completed on an RTX 5070, Ryzen 7 7700
and nominal 16 GB RAM system. Main RGB runs took approximately
49.1–56.0 minutes each, with 500 MiB peak reserved
PyTorch GPU memory. DINOv2 reference extraction took 123.6
seconds for 34,317 crops, excluding model loading, with 166 MiB peak reserved
GPU memory. System/process RAM peaks were not measured.

The earlier Python 3.12 preprocessing lock is historical.
The current environment is recorded in
[environment_deepfake.json](research/environment_deepfake.json).

## Documentation and evidence

- [Registered test results](research/main_results_test.json)
- [Model-freeze manifest](research/model_freeze_manifest.json)
- [Post-freeze extension results](research/main_results_extensions.json), with the
  [extension](research/extension_freeze_manifest.json), [CLIP](research/clip_freeze_manifest.json) and
  [I-JEPA](research/jepa_freeze_manifest.json) freeze manifests
- [Aggregate validation milestone](research/main_results_validation.json)
- [Current module tracker](research/module_tracker.md)
- [Frozen comparison register](research/comparison_register.md)
- [Experiment register](research/experiment_matrix.md)
- [Experiment scripts and workflow](experiments/README.md)
- [Main cohort inclusion specification](research/main_cohort_specification.md)
- [Initial scope and analysis rules](research/module1_specification.md)
- [Literature review](research/literature_review.md)
- [Protocol](research/experimental_protocol.md)
- [Full roadmap](research/research_roadmap.md)
- [Bibliography](research/references.bib)

Dataset access: [official FaceForensics++](https://github.com/ondyari/FaceForensics)
and the [C23 mirror used here](https://www.kaggle.com/datasets/xdxd003/ff-c23).

Dataset media, crops, raw predictions, feature caches, weights, virtual
environments and local results directories remain excluded from Git.
The aggregate records preserve metrics, confidence intervals,
configuration/resource records and artifact hashes.

Saved hashes describe execution artifact bytes. The snapshot also records
LF-normalized source hashes for comparison with Git's text normalization.
