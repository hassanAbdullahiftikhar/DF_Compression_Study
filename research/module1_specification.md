# Module 01: initial experimental specification

**Version 1 — 10 October 2026.** This freezes an initial core design using the downloaded FF++ C23 mirror. No classifier test scores existed when this specification was written. Preprocessing smoke checks can justify documented implementation corrections; changes to scientific comparisons must be logged.

## Research scope

Measure the effect of **additional compression** on real/fake detection and closed-set manipulation-method attribution, starting from already H.264 C23-compressed video. This is not a clean-to-first-compression study. The nominal C23 origin comes from the mirror's description/layout; selected video decoding confirms H.264, not byte equivalence to official files or a per-file CRF history.

RQ1: which tested representations retain useful detection/source-discriminative performance after additional JPEG compression?

RQ2: how does chance-adjusted attribution retention compare with detection retention on the same video/provenance support?

RQ3, following the core comparison: does source-supervised paired learning improve held-out quality compared with matched augmentation and two-view no-pair controls, while retaining reference attribution?

## Data and labels

- Root: `C:/Users/hassa/Desktop/Sem7/MBR/experiments/FaceForensics/FaceForensics++_C23`.
- Core: 1,000 original videos and 1,000 each of Deepfakes, Face2Face, FaceSwap, NeuralTextures: 5,000 videos total.
- Binary labels: real versus fake. Attribution: four fake-source labels only; real samples have no fake-source label.
- Additional DeepFakeDetection and FaceShifter folders are outside the primary four-source experiment. Their existence does not make them compatible extra source classes automatically.
- Official source-pair splits: 720/140/140 original IDs; 360/70/70 pair groups; 3,600/700/700 core videos across the five labels.
- Pilot: seed-42 selection of 72/14/14 complete source-pair groups: 720/140/140 videos; 1,000 videos total. A nested 8/1/1-group extraction smoke test checks the preprocessing implementation.
- Grouping ensures disjoint known original IDs; identity/YouTube-level independence is not established by filenames alone.

## Preprocessing

Sample eight uniformly spaced decoded frame indices, including endpoints, with a declared rounding rule. Freeze YuNet 2023mar at score threshold 0.9 and NMS 0.3 for the initial preprocessing test. Detection runs on aspect-preserving thumbnails with maximum dimension 640; map boxes back to decoded resolution. Select largest face area with confidence/position tie-breaks. Use a square 1.3× face box, fit it within image boundaries, resize to 224² using Lanczos, and save reference PNGs. Do not warp/alignment-transform the face.

Save frame indices/PTS, reference boxes, detector/model hash, confidence, number of faces, color/decoder versions, and failures. A failed extraction is logged, not replaced with a different frame. Define a common paired support set across quality conditions. Examine source/split coverage before main runs. The detector settings are a practical initial choice, not tuned on classifier test performance; any change following the preprocessing pilot requires a new manifest/config version.

The generic manifest condition `ref` means **C23-derived reference crop**, not uncompressed media. Every variant inherits its history.

## Compression and representations

- JPEG evaluation: reference plus Q95/Q80/Q60/Q40/Q20, fixed Pillow/libjpeg settings and 4:2:0 subsampling.
- Mixed-quality training/model selection: reference/Q80/Q40.
- Held-out quality testing: Q95/Q60/Q20. These conditions must not choose a checkpoint or pair weight.
- Reference-only and matched-quality adaptation are distinct protocols. A matched-quality run may train at a quality that is held out in another regime; label that clearly.
- Compare RGB, FFT, block DCT, declared Haar wavelet, declared residual, and frozen DINOv2-small. Cheap descriptors and learned/pretrained encoders belong in transparently identified tracks.
- The initial residual implementation is Laplacian, not SRM; no SRM or F3Net reproduction claim applies without its actual implementation.
- H.264/H.265 and history sweeps are extensions with their own frozen settings and calibration support. They cannot be described as pristine-to-codec tests using C23 inputs.

## Training, selection and evaluation

- Begin with RGB ResNet-18 and inexpensive probes. Use pretrained RGB as a comparison to pretrained DINOv2, rather than attributing pretraining differences entirely to representation.
- Initial RGB settings: eight pairs per microbatch, AdamW at 3e-4, weight decay 1e-4, up to twenty epochs. Adjust memory/throughput settings based on measured training; log adjustments before main runs. These are proposed settings, not an empirically optimized final recipe.
- Use inverse-count detection/source weighting or a declared balanced sampling strategy. The nominal dataset contains four times as many fake videos as real videos.
- Select checkpoints using validation only. The starter's average validation task balanced accuracy is an initial selection rule; changes must be documented before main test scoring.
- Select the binary operating threshold on validation, then freeze it across test conditions.
- Average frame probabilities per video before primary metrics.
- Binary: AUROC, balanced accuracy, macro F1 and descriptive EER. Source: fake-only balanced accuracy, top-1, macro F1, per-source recall/confusion. Report detection-gated attribution separately with coverage.
- Chance-adjusted retention uses balanced accuracy with chance 0.5 for binary and 0.25 for source. Mark it unstable/undefined when reference balanced accuracy is less than 0.05 above chance; always report raw scores.
- Compute paired retention differences with source-pair-group bootstrap, initially 2,000 draws, including denominator variation. Final selected runs use seeds 42/43/44; report seed variability separately from fixed-model sampling uncertainty.

Primary JPEG contrast: source retention minus detection retention at Q20 for reference-trained models on common support. Q60 is a secondary severity comparison. Cross-family comparisons across the six representations require multiple-comparison treatment (e.g. Holm correction) if significance tests are reported. Remaining quality curves and history/codec comparisons are descriptive/exploratory until the extension analysis specification is frozen. Use bootstrap intervals and effect sizes; do not infer significance merely by comparing two unpaired intervals.

## What counts as completion

The core is complete when audited reference/JPEG inputs, all claimed representation implementations, predeclared baseline regimes, video-level predictions, controls, and uncertainty estimates support a bounded conclusion. A method improvement is optional. Neither attribution degrading first nor a positive paired-learning result is required for a useful study. A negative classifier result does not establish that all identifying information is irrecoverable.

## Remaining implementation gates

Complete the full pilot's preprocessing/coverage review, finish the statistical evaluator, install and benchmark GPU training, implement frozen DINOv2 extraction, and then freeze the final run budget before main test evaluation. This specification provides defaults; it does not mark those gates achieved.
