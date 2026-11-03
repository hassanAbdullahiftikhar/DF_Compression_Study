# Comparison register (frozen before test scoring)

**Version 1.0 — frozen 23 October 2026, before any classifier test scoring.**
The SHA-256 of this file and of the frozen validation records it relies on
are stored in [comparison_register_freeze.json](comparison_register_freeze.json),
written by `experiments/freeze_register.py`. On 23 October 2026 no file in
`experiments/runs/` or `experiments/results/` contained classifier test
predictions. Test crops exist from preprocessing, but no model has scored them.

This register governs what is compared on the test cohort and how.
It does not change any earlier validation selection. Changes after freezing
are allowed only as dated amendments appended at the end (section 11).
Earlier sections are never edited in place.

## 1. Question, scope and labels

- **Detection:** real vs. manipulated face video.
- **Attribution:** closed-set classification of a *fake* video into one of
  four known FF++ manipulation methods (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures). It does not identify generator checkpoints or unknown sources.
- **Reference condition:** crops from an already H.264 C23-compressed
  FF++ mirror. All JPEG conditions are *additional* compression of those crops.
- **Research questions** (from the proposal, narrowed to this scope):
  - RQ1: which representations retain detection and attribution performance under additional JPEG?
  - RQ2: does chance-adjusted attribution retention differ from detection retention?
  - RQ3: does compression-paired training help at held-out qualities while keeping reference attribution?
- A lower score does not show that information was physically or irreversibly
  erased. Matched-quality adaptation is the registered check on that question.
  No first-ever novelty claim is made.

## 2. Test cohort and support

- Official FF++ test split: 700 videos in 70 provenance groups, 5,581 valid crops.
  Every test video has at least one crop. Nine test videos are partial, in four groups.
- **Primary support:** all 700 test videos. Frame probabilities are averaged per video
  (`metrics.aggregate`). Every model and condition uses the same video set.
- **Secondary support (frozen on 23 October 2026 after the validation check):**
  - `complete_video`: videos with all eight scheduled crops.
  - `complete_group`: groups whose ten core videos all qualify.

  Both use the logic and frozen thresholds of `experiments/eight_frame_sensitivity.py`.
  A contrast's status is flagged as changed when its sign flips, or when its 95% interval
  stops (or starts) excluding zero. Validation result: `experiments/results/eight_frame_sensitivity_validation.json`.
- The unit of uncertainty is the provenance group. Frames are not treated as independent.

## 3. Test conditions

Each condition is applied once, in memory, to the reference PNG crop.
Encoder: Pillow 12.3.0 JPEG at quality *q*, 4:2:0 subsampling, `optimize=False`, `progressive=False`.

| Condition | Role in earlier selection | Test role |
|---|---|---|
| ref | RGB checkpoint and threshold selection; probe penalty and threshold selection | Reference denominator |
| jpeg_q95 | Never used | Held out |
| jpeg_q80 | RGB checkpoint and threshold selection; reference-probe thresholds; Q80 known-quality heads | Selection condition (not held out) |
| jpeg_q60 | Never used | Held out; secondary contrast |
| jpeg_q40 | As Q80, plus Q40 known-quality heads | Selection condition (not held out) |
| jpeg_q20 | Never used | Held out; **primary contrast** |

All RGB modes, including reference-trained, chose their checkpoint from
validation ref/Q80/Q40. So "reference-trained" means *trained* without JPEG,
not *selected* without it. Q95, Q60 and Q20 played no part in any selection.

## 4. Frozen models already selected on validation

| ID | Model | Seeds or members | Frozen record |
|---|---|---|---|
| F1 | RGB ResNet-18 (ImageNet init), two heads, reference-only training (two identical reference views, pair weight 0) | 42, 43, 44 | `experiments/results/main_validation_thresholds_seeds42_44.json` |
| F2 | As F1, two views (reference plus a random Q80/Q40 view), pair weight 0. This is the **JPEG augmentation / two-view no-pair** baseline | 42, 43, 44 | same |
| F3 | As F2, projection cosine pair loss weight 1 | 42, 43, 44 | same |
| F4 | Reference-trained class-weighted ridge probes: pooled RGB, FFT, block DCT, Haar wavelet, Laplacian residual, frozen DINOv2-small | 6 | `experiments/results/c23_main_probe_compression_validation/report.json` |
| F5 | Q80 known-quality ridge probes, same six representations | 6 | `experiments/runs/c23_main_matched_q80_probes_validation/report.json` |
| F6 | Q40 known-quality ridge probes | 6 | `experiments/runs/c23_main_matched_q40_probes_validation/report.json` |

- **Thresholds:** RGB and F4 maximize mean video detection balanced accuracy (BA)
  over validation ref/Q80/Q40. F5 and F6 maximize it on their own matched-quality validation.
  Ties go to the threshold closest to 0.5, then the smaller one (tolerance 1e-12).
- **Prediction rule:** a video is called fake when `p_fake >= threshold`.
- Checkpoints, heads, thresholds and penalties are not changed after this point.

## 5. Registered controls (train and freeze on validation before any test scoring)

Each control uses the existing recipe unless stated otherwise:
- ResNet-18 with ImageNet init, 20 epochs, 8 pairs per microbatch.
- AdamW, learning rate 3e-4, weight decay 1e-4, AMP, workers 0.
- Inverse-count class weights.
- Seeds 42/43/44, with validation-only selection.

Training data is the reference manifest, with JPEG generated on demand.

### C1 — Task sharing (E04): shared two-head vs. separate single-task models

- **C1-DET** (3 seeds): F1 recipe with source loss weight 0, so detection loss only.
  - Checkpoint: maximum mean validation detection BA over ref/Q80/Q40 at threshold 0.5.
  - Threshold: F1 rule.
- **C1-ATT** (3 seeds): F1 recipe trained on *fake* training videos only, with detection loss weight 0.
  - Checkpoint: maximum mean validation fake-only attribution BA over ref/Q80/Q40.
  - It has no detection output.
- **Budget difference (accepted and disclosed):** C1-ATT sees about 80% as many
  samples, and therefore fewer optimizer updates, per epoch.
- **Pairing for C1 retention:** C1-DET with C1-ATT of the same seed form the
  "separate" system. Detection comes from C1-DET and attribution from C1-ATT.

### C2 — Source supervision (E08): binary-only paired encoder plus a train-only source readout

- **C2-BIN** (3 seeds): F3 recipe (two views, pair weight 1) with source loss weight 0.
  - Checkpoint: maximum mean validation detection BA over ref/Q80/Q40 at 0.5.
  - Threshold: F1 rule.
- **Readout R:** a frozen-encoder ridge source probe.
  - Features: the 512-d pooled encoder output for train and validation crops at ref, Q80 and Q40.
  - Fit: class-weighted ridge (`probe_validation.weighted_system`) on fake training crops,
    pooled across the three views. Standardization is fitted on those same training features.
  - Penalty: chosen from (0.001, 0.01, 0.1, 1) by mean validation fake-only attribution BA over
    ref/Q80/Q40. Ties go to the smaller penalty.
- R is applied identically to **F3 encoders**, giving F3-R. The registered comparison is
  **C2-R vs. F3-R**: same readout, isolating whether source supervision changes the encoder.
  F3's native source head is reported alongside.

### C3 — Background/border, metadata/geometry, and content controls (E09)

- **C3a Face-masked border probes.**
  - Masking: take each crop's stored YuNet face box (`crop_metadata.jsonl`: `face_xywh`, `crop_box`),
    map it into the 224×224 crop, expand it by 10% of its width and height on each side, clip it,
    and fill it with constant gray (128). For compressed conditions, JPEG is applied first and then the mask.
  - Representations: the same six as F4.
  - Fitting: the F4 procedure (training-only scaling, penalty chosen on reference validation,
    F4 threshold rule), plus five video-level label shuffles as diagnostics.
  - Test conditions: all six; ref is the primary readout.
- **C3b Metadata/geometry probe.** Video-level ridge probe, fitted with the C3a procedure on the 3,600 training videos.
  - Features from `videos_core.csv`: log file size, frame count, width, height, log bytes per pixel-frame.
  - Features from `crop_metadata.jsonl` (mean and SD over valid crops): face-box width/height relative
    to the frame, face-centre x/y, detector confidence, faces detected, crop-box side relative to the frame,
    and the valid-crop count.
  - Not affected by JPEG, so it is evaluated once. The audit already shows method-dependent
    frame-count differences (`dataset_audit_c23.md`).
- **"Material" flag for C3a and C3b:**
  - Attribution: the 95% test interval for BA has its lower bound above 0.30.
  - Detection: the lower bound for BA is above 0.55.

  A material flag adds an "outside-the-face-box or dataset-level cue" caveat to the matching
  full-crop findings. It does not invalidate them, because manipulation traces can extend past the detector box.
- **C3c Content-matched decomposition** (computed from registered predictions, no training).
  - Design check: all 1,000 fake pair IDs exist under all four methods, so content is matched across methods.
  - **Family BA:** identity-swap {Deepfakes, FaceSwap} vs. reenactment {Face2Face, NeuralTextures},
    using summed source probabilities. Chance is 0.5.
  - **Within-family BA:** Deepfakes vs. FaceSwap, and Face2Face vs. NeuralTextures, each restricted to that
    family's true videos and using the two relevant probabilities. Chance is 0.5.
  - Reported for every attribution model at all conditions. This separates identity/semantic swap cues from
    finer method cues. It is descriptive.
- **C4 Preprocessing coverage.**
  - Failures by source and split are already logged.
  - The complete-eight sensitivity (section 2) is the registered preprocessing control.
  - Crops use fixed reference boxes by design. A redetection or crop-drift end-to-end arm is outside the core.

## 6. Primary contrast (RQ2)

**Δ(Q20) = S_attr(Q20) − S_det(Q20)** for *reference-trained* models on test.
- S(q) = (BA(q) − b) / (BA(ref) − b), with b = 0.5 for detection and 0.25 for attribution. It is not clipped.
- S is undefined when the test reference BA is less than b + 0.05.
- Intervals: 2,000 paired group-bootstrap draws (seed 4242, 70 test groups), 95% percentile.
  The denominator is recomputed in every draw.
- An interval is "unavailable" when fewer than 95% of draws are defined.

The contrast is evaluated in two families:
- **P1-RGB (F1, three seeds):** per-seed points and intervals.
  - A direction is claimed only if all three seeds' intervals exclude zero with the same sign.
  - Otherwise the result is "direction not established".
- **P1-PROBE (six F4 probes):**
  - Two-sided bootstrap p = min(1, 2·min(P(Δ*≤0), P(Δ*≥0))).
  - Holm correction across a **fixed family of six**, family-wise α = 0.05.
  - An undefined contrast counts as p = 1. No representation ranking is inferred beyond Holm-supported signs.

## 7. Secondary contrasts (intervals reported; no stand-alone significance claims)

| ID | Contrast | Decision rule or reading |
|---|---|---|
| S1 | Δ(Q60), then Δ(Q95), for F1 and F4 | Same computation as section 6; descriptive severity trend |
| S2 (RQ3) | Per seed: F3 − F2 and F2 − F1, detection BA and attribution BA at Q20, Q60, Q95 and ref | A "paired-learning benefit" is claimed only if F3 − F2 attribution BA at Q20 excludes zero positively in all three seeds **and** the ref attribution BA difference has its lower bound above −0.02 in all three seeds. Otherwise: "no reliable benefit over augmentation" |
| S3 (E02) | F5 − F4 at Q80 and F6 − F4 at Q40: AUROC (point estimate) and attribution BA (interval) | Does adaptation recover performance? It does not establish physical presence or absence of information |
| S4 (E04) | C1 vs. F1, per seed: detection BA (C1-DET − F1), attribution BA (C1-ATT − F1), and Δ(Q20) of the separate system minus Δ(Q20) of F1 | Does the P1-RGB reading depend on shared capacity? |
| S5 (E08) | C2-R − F3-R attribution BA, and C2-BIN − F3 detection BA, per seed, at all conditions | Effect of explicit source supervision on what a fixed readout can recover |
| S6 (E09) | C3a and C3b material flags; C3c family and within-family BA and their Q20/Q60 retention | Shortcut and semantic-cue context for the main findings |
| S7 | Primary support vs. `complete_video` and `complete_group` for section 6 and S2 | Sensitivity only. A status change is reported, never used to select results |

## 8. Exploratory (labelled as such)

- Full quality curves and seed mean ± SD.
- AUROC, EER and macro F1.
- Per-source recall and confusion matrices.
- Detection-gated attribution and coverage, and calibration.
- Test-crop PSNR/SSIM distortion.
- Label-shuffle diagnostics.
- Embedding views (silhouette, Davies–Bouldin, UMAP) as illustration only.

## 9. Test-scoring procedure and gates

1. **G1 — controls frozen.**
   - C1, C2 and C3 are implemented with overwrite protection and run on validation.
   - Their selections, thresholds and hashes are exported to the aggregate milestone.
2. **G2 — scorers verified.**
   - Test scorers are written: `predict_rgb.py` for RGB at ref plus the five JPEG qualities, and probe/readout scorers that apply frozen heads.
   - Before touching test, each scorer must reproduce stored *validation* predictions.
3. **G3 — model freeze manifest.**
   - A register amendment lists every model's checkpoint/head hash and threshold. It must be written before G4.
4. **G4 — one scoring pass.**
   - Each registered model is scored once on test at all six conditions, with overwrite protection.
   - The registered analysis runs once.
   - A rerun is allowed only to fix a documented bug, with the reason and the code change recorded. It must not change selections.
5. **Reporting.** Every registered outcome is reported, including null or negative ones, with validation and test results clearly separated.
6. **Prohibited:**
   - choosing or excluding models, seeds, thresholds, penalties, pair weights, conditions or support because of test results;
   - adding post hoc primary contrasts.

## 10. Outside the frozen core

Each item below is an extension or not planned. It may join only through a dated amendment written
**before** its own test scoring, and its results are labelled as a post-freeze extension.
- E05: whole-video H.264/H.265, including H.265 as a held-out codec.
- E06: repeated JPEG recompression (80→80, 80→40, 40→80) with matched single-pass controls.
- E10: Celeb-DF external detection.
- E11: GenImage generator-family study.
- ADD, QAD, DPL, F3-Net and Perelli et al. reproductions. These are cited as context only; published scores are not compared directly.
- Single-view mixed-quality augmentation, an auxiliary compression-prediction head, a 4:4:4 chroma control, and a resizing arm.
- A common learnable encoder across representation maps. The representation comparison is therefore
  probe-based (fixed descriptors plus linear heads), and this is a stated limitation.
- Pair-weight 0.1 in the main cohort. Pair weight 1 was chosen from the exploratory pilot.

**Mapping to the proposal's Table 3:**

| Proposal protocol | Status here |
|---|---|
| Clean baseline | Ref → ref, with C23-derived references (not clean) |
| Compression degradation | F1, F4 |
| Matched compression | F5, F6 |
| Mixed quality | F2 |
| Cross-quality | F2 and F3 at Q95/Q60/Q20 |
| Source attribution | All models, fake-only |
| Cross-codec | E05, outside the core |
| Recompression | E06, outside the core |

## 11. Amendments

*None yet. Append entries as "Amendment N — date — before/after test scoring — change — reason".*

### Amendment 1 — 24 October 2026 — before test scoring — model freeze (gate G3)

- **Change:** this amendment adopts the G3 model-freeze manifest `research/model_freeze_manifest.json`
  (SHA-256 `ad449b5d05c9ef74122de7f9583bd347960154c1db891d23a25f1f87733919b5`). The manifest lists all 49
  registered models with their checkpoint or head hashes and frozen thresholds:
  - F1–F3, C1-DET, C1-ATT and C2-BIN, three seeds each;
  - F3-R and C2-R;
  - F4, F5, F6 and C3a, six representations each;
  - C3b.

  It also hashes the frozen scorer and analysis code and the G2 verification reports. Test scorers refuse to
  run unless this amendment, the manifest and every hash listed in it still match.
- **Scorer naming:** section 9 names `predict_rgb.py`, but the RGB scorer is `experiments/score_rgb.py`.
  It calls the same `train_paired.predict` at batch 8 for ref plus Q95/Q80/Q60/Q40/Q20, and readout R
  through `fit_source_readout.extract`. `predict_rgb.py` is unchanged. Probes F4–F6, C3a and C3b are scored
  by `experiments/score_probes.py`. The registered analysis is `experiments/analyze_test.py`.
- **G2 outcome:**
  - All 24 RGB and readout models reproduced their stored validation predictions exactly (maximum difference 0).
  - All 25 probe heads reproduced theirs to within 1e-15.
  - One validation video, FaceSwap/982_004 at Q40 under F6_rgb, has a stored score exactly equal to its frozen
    threshold, because thresholds are drawn from observed scores. A rounding-level change flips its `>=` decision.
  - The first probe verification failed on that video alone. The pass rule was then made boundary-aware.
    No threshold, head or prediction changed, and the failed report is kept
    (`experiments/results/scorer_verification_probes_validation_attempt1.json`).
- **Analysis details fixed before scoring (sections 1–10 unchanged):**
  - The C1 "separate system" Δ takes detection from C1-DET and attribution from C1-ATT of the same seed.
  - C3c family and within-family retention use chance 0.5.
  - C3a material flags are read at ref.
  - Label-shuffle diagnostics and embedding views are not computed on test, because shuffled heads were not saved.
  - Test-crop distortion is reported as PSNR (RGB) and Gaussian-window SSIM (BT.601 luma).
- **Dry run:** `analyze_test.py --split val` ran the same code on validation predictions, with Q40 standing in
  for Q20 and Q80 for Q60/Q95. It reproduced 549 recorded validation values exactly. That run checks the code;
  it is not a result.
- **Reason:** section 9 requires the model-freeze manifest before the single test-scoring pass (G4).

### Amendment 2 — 24 October 2026 — after the registered test analysis — pre-registration of post-freeze extensions X1–X6

- **Status:** written after G4 and after the registered test results were seen. Everything below is a
  post-freeze extension (section 10) or an exploratory re-analysis. The registered readings of sections 6–7
  stand unchanged; nothing here replaces or re-scores them.
- **Process:**
  - Validation-only work may start at once.
  - X2, X5 and X6 score test only after Amendment 3, a model-freeze record with their new hashes and
    thresholds. Each is scored once.
  - X1, X3 and X4 re-analyse stored test predictions in one script run.
  - All results are labelled "post-freeze extension" or "exploratory".

**X1 — Operating-point analysis (exploratory; motivated by the test result).**
- **X1a, threshold-free contrast:** Δ_AUC(q) = S_attr^AUC(q) − S_det^AUC(q).
  - Detection AUROC is computed over all videos.
  - Attribution AUROC is the mean over the four sources of the one-vs-rest AUROC on fake videos.
  - Chance is 0.5 for both. S uses the section 6 formula with b = 0.5 and the 0.05 margin.
  - Intervals use the same 2,000 paired group draws (seed 4242).
  - Models: F1 (three seeds), F4 and C1-SEP, at every JPEG condition.
- **X1b, per-quality calibration:**
  - Validation predictions at Q95, Q60 and Q20 are made for F1–F3, C1-DET and C2-BIN with
    `train_paired.predict` at batch 8.
  - For each model and quality, the threshold maximizes validation video detection BA at that quality,
    with the section 4 tie rule.
  - These thresholds are applied to the stored test predictions. Reported per quality: detection BA,
    retention and Δ_cal(q), with intervals. The P1 direction rule is applied descriptively.
- **X1c, score shift:** mean video p_fake for real and for fake test videos at each condition (descriptive).

**X2 — Frozen-encoder Q20 readout (F1, three seeds).**
- Features: 512-d pooled frozen F1 encoder output for training and validation crops at Q20.
- Heads: class-weighted ridge detection and source heads (`probe_validation`), standardized on the
  training features.
- Penalty: from (0.001, 0.01, 0.1, 1) by mean validation detection/source BA at Q20 at threshold 0.5.
  Ties go to the smaller penalty.
- Threshold: maximizes validation video detection BA at Q20.
- Test: all six conditions. The contrast is readout minus native F1 at Q20: detection BA and attribution BA
  with intervals, AUROC as a point estimate, per seed.
- Reading: information that is linearly available in the frozen encoder at Q20. It is not a claim about
  physical presence.

**X3 — Seed-pooled bootstrap.**
- Contrasts: Δ(Q20) of F1, and F3 − F2 and F2 − F1 at Q20 for detection and attribution BA.
- Each draw combines the section 6 group draw with an independent draw of three seeds with replacement
  (seed 4243). The statistic is the mean over the drawn seeds; the point estimate is the three-seed mean.

**X4 — Metadata-cue analysis (exploratory).**
- **(a)** For each attribution model at ref and at Q20: attribution accuracy on fake test videos where C3b's
  argmax is correct, minus that where it is wrong, with a group-bootstrap interval.
- **(b)** Metadata-balanced attribution BA.
  - Weight each fake video by 1/p_C3b(true source).
  - Clip weights at their 99th percentile, then normalize them to mean 1 within each true source.
  - Report the weighted BA minus the unweighted BA.

**X5 — Video-codec extension (E05), with frozen models.**
- **Encoding:** each test video is re-encoded in memory with PyAV 18.1.
  - Settings: libx264 at CRF 18, 23, 32 and 40, and libx265 at CRF 23 and 32.
  - Both codecs: preset medium, GOP 48, three B-frames, yuv420p, four threads. For libx265 these go through
    `x265-params`.
  - Every decoded frame is encoded with sequential timestamps. The output frame count must equal the input's.
- **Crops:** the scheduled frames are decoded at the stored frame indices, cropped with the stored
  `crop_box` and resized to 224 px (Lanczos). There is no redetection.
- **Pipeline check before scoring:** decoding the original files through the same code must reproduce every
  stored test reference crop pixel for pixel.
- **Models:** F1–F3, C1-DET, C1-ATT, C2-BIN, the six F4 probes, and the X6 models after their freeze.
  All use their frozen thresholds.
- **Reporting, per setting:** detection BA, attribution BA, retention relative to the ref crops, Δ, and
  F2 − F1 and F3 − F2, all with intervals. Also mean crop PSNR against the ref crops and mean bitrate.
- These results are descriptive; X5 adds no new primary claim. Equal CRF across codecs is not equal distortion.

**X6 — EfficientNet-B4 replication.**
- **Architecture:** torchvision `efficientnet_b4` with IMAGENET1K_V1 weights; the classifier is replaced by
  the identity, giving 1,792-d features. Heads: detection (2), source (4) and projection (1792→128→128).
- **Input:** 224 px. The native resolution is 380 px; this deviation is disclosed.
- **Recipes:** F1 (reference-only, two identical views, pair weight 0) and F2 (reference plus a random
  Q80/Q40 view, pair weight 0), at seeds 42, 43 and 44.
- **Unchanged rules:** all other training, checkpoint-selection and threshold rules follow section 4.
- **Test:** the six JPEG conditions and the X5 settings.
- **Readings:** the P1-RGB rule at Q20 and the S2-style F2 − F1 contrasts at Q20, applied descriptively.

**Deferred:** a foundation-feature probe (CLIP or larger DINOv2) may be added only by a later amendment
written before its fitting.

### Amendment 3 — 25 October 2026 — before extension test scoring — extension model freeze

- **Change:** this amendment adopts `research/extension_freeze_manifest.json`
  (SHA-256 `0e1b7714de0b15cdbcc90c9fa90f5e68c81fd9dfebe3bf4c9bb62416911aa305`). It freezes:
  - the six EfficientNet-B4 models (X6: F1 and F2 recipes, seeds 42/43/44), with thresholds from the F1 rule;
  - the three X2 Q20 readouts on the frozen F1 encoders;
  - the X1b per-quality validation thresholds;
  - the extension scorer, codec and analysis code.
- **Verification:** all nine extension models reproduced their stored validation predictions exactly
  (maximum difference 0).
- **Process deviation (disclosed):**
  - On 24 October, two copies of the EfficientNet-B4 F2 seed-43 run started at once (21:24 and 21:25)
    after a runner-shell restart.
  - Both were stopped before either finished an epoch. The empty output folder was removed, and the run
    was restarted alone at 21:30.
  - No other run was affected, and the F2 seed-42 run had finished before the overlap.
- **Reason:** Amendment 2 requires this freeze before X2, X5 or X6 touch test.

### Amendment 4 — 25 October 2026 — before fitting — extension X7: CLIP ViT-L/14 probes

- **Status:** post-freeze extension. This is the foundation-feature probe deferred in Amendment 2. It was
  designed after the registered and X1–X6 results were seen, and nothing here changes those readings.
- **Backbone:**
  - Model: OpenAI CLIP ViT-L/14, from the TorchScript checkpoint `ViT-L-14.pt` (SHA-256
    `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`).
  - Loading: `torch.jit.load`, with no extra packages. It runs under `torch.jit.optimized_execution(False)`;
    a check showed repeated calls are identical and outputs do not depend on batch composition.
  - Precision and batch: native fp16 weights, batch 8.
- **Features:**
  - Input: the 224-px study crop, without resizing, using the CLIP mean and standard deviation.
  - Output: the 768-d `encode_image` embedding, cast to float32.
  - JPEG is applied before feature extraction, as in the F4 track.
- **Probes** (the F4/F5/F6 procedures, class-weighted ridge detection and source heads, standardization on
  training features only):
  - **X7-REF:** fitted on training features at ref. Penalty by mean validation detection/source BA at ref.
    Threshold maximizes mean validation video detection BA over ref/Q80/Q40.
  - **X7-Q80 and X7-Q40:** fitted on training features at Q80 or Q40. Penalty and threshold come from
    matched-quality validation.
  - Each probe also gets five video-level label shuffles (seeds 101–105) as diagnostics.
- **Verification:** re-extracting validation features end to end must reproduce the stored validation
  probabilities to within 1e-6.
- **Freeze:** Amendment 5 will record the head hashes and thresholds before any test scoring.
- **Test:** one pass covering the six JPEG conditions and the six X5 codec conditions.
- **Analysis** (descriptive; same 2,000 group draws, seed 4242):
  - Δ(Q20) for X7-REF, with its interval and a two-sided bootstrap p. This test is outside the
    P1-PROBE Holm family.
  - Δ_AUC(Q20) for X7-REF.
  - X7-Q80 − X7-REF at Q80 and X7-Q40 − X7-REF at Q40: attribution BA with an interval, AUROC as a point
    estimate.
  - X7-REF Δ at each codec setting.
  - X7-REF − F4-DINOv2 detection and attribution BA at ref and Q20.

### Amendment 5 — 25 October 2026 — before CLIP test scoring — X7 freeze

- **Change:** this amendment adopts `research/clip_freeze_manifest.json`
  (SHA-256 `1a5b2b09861cc96996a6b9e5a0fbd5b26f5085abf6bedcbcc7e3fc03a41a2664`). It freezes the X7-REF,
  X7-Q80 and X7-Q40 heads and thresholds, the CLIP checkpoint hash, and the X7 scoring and analysis code.
- **Verification:** re-extracting the validation features end to end reproduced every stored validation
  probability exactly (maximum difference 0).
- **Note:** all three probes selected the smallest penalty in the grid (0.001), as several F4–F6 probes did.
  The grid was fixed in advance and is not extended.
- **Reason:** Amendment 4 requires this freeze before X7 touches test.

### Amendment 6 — 28 October 2026 — before scoring — extension E10: external Celeb-DF v2 detection test

- **Status:** post-freeze extension, linked to `research/dfdm_register.md` section 9. It is descriptive
  and changes no FF++ reading.
- **Data:**
  - The 518 official Celeb-DF v2 test videos (178 real, 340 fake), from the authors' official copy.
  - Crops follow the FF++ protocol: YuNet 2023mar, 8 uniform frames, square box with margin 1.3, 224 px
    Lanczos. Of 4,144 scheduled frames, 17 had no face and were logged; every video has crops.
  - Bootstrap groups are target identities, with YouTube-real videos as their own groups: 124 groups.
- **Models:** the frozen FF++ detectors F1, F2 and F3 (ResNet-18) and EB4-F1 / EB4-F2, three seeds each.
  Their checkpoint hashes and thresholds are in the model and extension freeze manifests; nothing is
  refitted.
- **Conditions:** ref and JPEG Q95/Q80/Q60/Q40/Q20, applied in memory.
- **Reported:** detection AUROC and BA at the frozen thresholds, with 2,000 group-bootstrap draws
  (seed 4242). There is no attribution, because Celeb-DF has one manipulation method.
- **Code:** `experiments/celebdf_external.py`, SHA-256 `6da0ff8bdc2c06e9b5cc29a10e41e411bed5656c38d5cc886cafaf95f7df3c08`. It may not change before scoring.

### Amendment 7 — 28 October 2026 — before fitting — extension X8: I-JEPA ViT-H/14 probes

- **Status:** post-freeze extension. It was designed after the registered, X1–X7, DFDM and E10 results were
  seen, and nothing here changes those readings. It adds a third pretrained-encoder family (self-supervised
  latent prediction) next to DINOv2 (F4) and CLIP (X7). Scope is FF++ only, as for X7.
- **Backbone:**
  - Model: Meta's official I-JEPA ViT-H/14, ImageNet-1k, 300 epochs (`IN1K-vit.h.14-300e.pth.tar` from
    `dl.fbaipublicfiles.com/ijepa/`). The **target encoder** is used, as in the I-JEPA paper's evaluations.
  - Loading: `jepa_probes.py --prepare` extracts the target encoder into a state-dict file and records the
    checkpoint and encoder SHA-256 in `experiments/results/ext_jepa_encoder_extraction.json`. The network is a
    plain-PyTorch copy of the I-JEPA ViT loaded with strict key matching; no extra packages.
  - Precision and batch: float32, TF32 off, deterministic kernels, batch 8.
- **Features:**
  - Input: the 224-px study crop, without resizing (16×16 patches of 14 px), with ImageNet mean and standard
    deviation.
  - Output: the mean of the 256 final-layer tokens after the final layer norm, 1280-d.
  - JPEG is applied before feature extraction, as in the F4 and X7 tracks.
- **Probes:** identical to X7 (Amendment 4), renamed:
  - **X8-REF:** fitted on training features at ref; penalty by mean validation detection/source BA at ref;
    threshold maximizes mean validation video detection BA over ref/Q80/Q40.
  - **X8-Q80 and X8-Q40:** fitted at Q80 or Q40, with penalty and threshold from matched-quality validation.
  - Five video-level label shuffles (seeds 101–105) per probe as diagnostics. The penalty grid is the fixed
    F4–F6 grid and is not extended.
- **Verification:** re-extracting validation features end to end must reproduce the stored validation
  probabilities to within 1e-6.
- **Freeze:** Amendment 8 will record the head hashes, thresholds and code hashes before any test scoring.
- **Test:** one pass covering the six JPEG conditions and the six X5 codec conditions.
- **Analysis** (descriptive; same 2,000 group draws, seed 4242):
  - Δ(Q20) for X8-REF, with its interval and a two-sided bootstrap p, outside the P1-PROBE Holm family.
  - Δ_AUC at every condition for X8-REF.
  - X8-Q80 − X8-REF at Q80 and X8-Q40 − X8-REF at Q40: attribution BA with an interval, AUROC as a point.
  - X8-REF Δ at each codec setting.
  - X8-REF − F4-DINOv2 and X8-REF − X7-REF detection and attribution BA at ref and Q20 (paired draws). The
    registered F4-DINOv2 and the X7-REF intervals must be reproduced exactly from their stored predictions.
- **Expectation, stated in advance and not a confirmatory hypothesis:** I-JEPA predicts in latent space rather
  than pixels, so its features may depend less on high-frequency detail than DINOv2 or CLIP. This would show as
  smaller detection and attribution losses under JPEG; no direction for Δ is registered. Whatever the result,
  it is reported with the other probes and marked as a post-freeze extension.

### Amendment 8 — 28 October 2026 — before I-JEPA test scoring — X8 freeze

- **Change:** this amendment adopts `research/jepa_freeze_manifest.json`
  (SHA-256 `7ceee04c938608dcfc94086eda9e352dc4e7cb371226a8aa53a19dbd0096a90e`). It freezes the X8-REF,
  X8-Q80 and X8-Q40 heads and thresholds, the I-JEPA checkpoint hash
  (`0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`), the extracted target-encoder hash
  (`ffc0cd9296785cf095d7fbee7beff629e6ded1e20f95308f7d008563545dc46a`), and the X8 scoring and analysis code.
- **Verification:** re-extracting the validation features end to end reproduced every stored validation
  probability exactly (maximum difference 0). A first verification attempt was stopped by the operating
  environment for low system memory before it wrote anything; it was rerun unchanged.
- **Note:** X8-REF and X8-Q80 selected the smallest penalty in the grid (0.001) and X8-Q40 selected 0.01. The
  grid was fixed in advance and is not extended.
- **Reason:** Amendment 7 requires this freeze before X8 touches test.

### Amendment 9 — 28 October 2026 — after X8 test scoring — disclosure only

- **Record:** the first X8 test-scoring attempt was stopped by the operating environment for low system
  memory before its first condition finished. It wrote no predictions, report or log lines, and no test
  output was seen. The frozen code was then rerun unchanged as a separate process and completed in one pass
  (66 minutes); `ext_jepa_analysis_test.json` came from that pass. Nothing in Amendments 7–8 changed.
