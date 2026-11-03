# Experiment register and planned comparisons

Initial design prepared 10 October 2026; execution status updated 2026-10-28.
Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); DFDM + Celeb-DF second-dataset study, external test E10 and the I-JEPA probe extension X8 (Amendments 7-8) completed on 28 October 2026; manuscript work remains. The family table below retains the initial plan.

## Current execution status

| Family | Current evidence | Remaining |
|---|---|---|
| E00: Pipeline pilot | Completed | Historical evidence retained |
| E01: Reference sensitivity | Six probes and RGB at ref/Q80/Q40 (validation); test at six conditions completed; CLIP (X7) and I-JEPA (X8) probes as labelled extensions | Paper reporting |
| E02: Known-quality adaptation | Q40 and Q80 probes; registered test comparison S3 completed | Paper reporting |
| E03: Mixed-quality generalization | RGB augmentation at three seeds; tested at held-out Q95/Q60/Q20 | Paper reporting |
| E04: Task-sharing | C1 separate models at three seeds; registered test comparison S4 completed | Paper reporting |
| E05: Video coding | Post-freeze extension X5 on test (Amendments 2-3): H.264 CRF 40 breaks the JPEG direction; H.265 repeats it | Paper reporting as a labelled extension |
| E06: Processing history | Generator and engineering checks | Scientific experiments if retained |
| E07: Paired learning | Weight-1 candidate at three seeds; test S2: no reliable benefit over augmentation | Paper reporting |
| E08: Source supervision | C2 binary-only paired model and readout; registered test comparison S5 completed | Paper reporting |
| E09: Explanatory controls | Validation controls; test S6/S7 completed (C3b flagged material for attribution) | Paper reporting |
| E10: External detection | Frozen FF++ detectors on the 518 Celeb-DF v2 test videos (Amendment 6): AUROC about 63–72% | Paper reporting |
| E11: Generator-family study | DFDM five-generator study (dfdm_register.md): fine-grained attribution collapses by Q60 | Paper reporting |

Models, thresholds, penalties and regimes are recorded in
[main_results_validation.json](main_results_validation.json) and frozen in
[model_freeze_manifest.json](model_freeze_manifest.json). Registered test results are in
[main_results_test.json](main_results_test.json).

## Planned families of experiments

| ID | Question | Training domain | Evaluation domain | Models/representations | Priority |
|---|---|---|---|---|---|
| E00 | Does the pipeline work and fit? | Tiny FF++ pilot/reference | Pilot validation; exploratory degraded conditions | RGB ResNet-18; one cheap probe | Required gate |
| E01 | How sensitive are reference-trained models? | FF++ reference | Reference and JPEG 95/80/60/40/20 | Six representation families with declared probes/baselines | Core |
| E02 | Can models adapt to known quality? | Selected JPEG quality | Same quality, disjoint videos/groups | Same chosen representation/model budgets | Core |
| E03 | Does augmentation generalize across quality? | Reference and JPEG 80/40 | Full JPEG sweep; 95/60/20 held out | Same representations/budgets | Core |
| E04 | Does sharing the encoder alter the task comparison? | Same training domain | Same evaluation grid | Two-head versus separate binary/fake-source RGB models | Core control, selected backbone |
| E05 | Which representations retain information under video coding? | Reference/selected H.264 conditions | Withheld H.264 and held-out H.265, calibrated distortion matches | Predeclared representation subset | Video extension |
| E06 | Does processing history matter beyond final distortion? | Single-quality/reference or declared mixed-quality | Repeated JPEG/video histories; single-pass matched controls | Already-trained selected baselines | History extension |
| E07 | Does pairing add value beyond exposure/augmentation? | Same reference/degraded training pairs | Held-out qualities/codecs/histories | Two-view no-pair versus pair weights 0.1/1 | Learning extension |
| E08 | Does explicit source supervision help? | Binary-only paired versus binary+source paired | Same held-out conditions | Common encoder; train-only source probe for binary-only variant | Learning control |
| E09 | What explains the scores? | Declared training subsets | Same paired conditions | Background/border, metadata, shuffle, preprocessing, content controls | Required selected controls |
| E10 | Do detection results transfer to another benchmark? | Frozen FF++-trained model | Celeb-DF v2 reference/degraded | Selected detection models | Optional |
| E11 | Does the pattern extend to generator families? | Separate balanced GenImage subset | Disjoint content/provenance; controlled JPEG | Separate source head/label scope | Optional separate study |

Use reference/Q80/Q40 for model selection in the initial mixed-quality design. Do not use Q95/Q60/Q20 to select the winning model and then describe its score there as untouched held-out evidence. Codec operating-point calibration is a separate validation task; it must not use test classifier performance.

Matched-quality adaptation and held-out-quality evaluation answer different questions. An experimental quality can be held out in one protocol and trained in another; label each regime explicitly so those results cannot be conflated.

## One record per actual training run

Complete this record before launch and append measured outcomes afterwards.

```text
Run ID:
Experiment family:
Question/primary contrast:
Code version/hash:
Config file/hash:
Dataset release and manifest hash:
Train/validation/test group and video counts:
Source label order:
Representation/preprocessing:
Encoder and pretrained checkpoint/hash:
Training conditions:
Validation conditions and checkpoint-selection rule:
Held-out conditions:
Loss weights/class balance:
Seed:
Physical batch, accumulation, epochs and optimizer updates:
Environment and encoder versions:
Estimated storage/compute:
Start/end time:
Measured peak GPU/RAM and elapsed time:
Checkpoint path:
Validation selection and frozen operating threshold:
Prediction paths:
Metric/analysis version:
Missing samples/exclusions:
Primary versus exploratory status:
Deviations and reason:
Result summary:
```

Do not compare runs without documenting different pretraining, input channels, sample presentations, updates, crop support, and extra teacher/quality-estimator costs. Final seeds should be reruns of predeclared comparisons rather than new selection opportunities.
