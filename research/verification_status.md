# Verification record

> **Historical record — status reconciled 23 October 2026.** The table below
> is the pre-experiment verification of 10 October 2026 and is kept unchanged
> as a record. Several rows are superseded; see
> [Status reconciliation](#status-reconciliation-23-october-2026) at the end.
> Current status is maintained in [module_tracker.md](module_tracker.md),
> [main_results_validation.json](main_results_validation.json) and
> [main_results_test.json](main_results_test.json).

Date: 10 October 2026. At that date, no real-data experiments had been run.

| Check | Outcome |
|---|---|
| All twelve proposal titles located | Yes; primary sources/author records linked in the review |
| Complete text access | Most papers; BZNet abstract/university record; Li/Zhu publisher previews only |
| Novelty search | Found direct earlier JPEG attribution, detection/attribution comparison, contrastive quality learning, and recent overlapping multi-task work |
| GPU | RTX 5070; reported memory 12,227 MiB; driver 610.88 |
| CPU | Ryzen 7 7700 identified through processor registry |
| RAM | Native Windows check: 16,229,781,504 usable bytes; 4,781,965,312 available bytes at inspection |
| Free storage | About 209 GiB on C: at inspection; full dataset footprint not measured |
| Python | Bundled Python 3.12.14; NumPy 2.3.5; Pillow 12.3.0 |
| ML and codecs | torch/torchvision absent in inspected Python; FFmpeg/FFprobe absent from PATH |
| Python syntax | All experiment Python files compiled successfully |
| JPEG pipeline | Passed synthetic test: 15 reference samples, 135 manifest rows after variants |
| Transforms | DCT orthonormality/energy and Haar energy checks passed |
| Descriptors | Finite outputs; dimensions RGB=768, FFT=256, DCT=128, wavelet=28, residual=263 |
| Provenance checks | Duplicate rows, cross-split original leakage, inconsistent clusters, changed variant labels rejected |
| Metrics checks | Perfect classification and tied-score AUROC cases passed; two frame predictions aggregated into each video |
| Overwrite protection | Existing JPEG output rejected |
| GPU benchmark | Attempted prerequisite check returned not_run because torch is absent; no speed/VRAM training measurement |
| Actual model training | Not run; requires research dependencies and authorized dataset |

Synthetic checks establish code-path and mathematical behavior. They do not validate face extraction, dataset integrity, training convergence, attribution accuracy, or the scientific hypotheses.

## Update after the dataset download

The C23 mirror is now present. All 7,000 metadata paths exist; 5,000 videos belong to the primary real/four-method study. Official split mapping yielded 500 complete source-pair groups without known input-ID conflicts. MP4 structural checks passed for all core videos; 15 sampled videos decoded completely with matching metadata counts/dimensions. A private preprocessing environment was created and locked. YuNet extraction produced 800 crops from 100 videos with zero logged failures; a training crop preview was inspected. JPEG/history generation produced 7,200 audited rows. Details and qualification of these checks are in `dataset_audit_c23.md`. At the time of that update, classifier training and scientific accuracy were unmeasured.

## Status reconciliation (23 October 2026)

| Historical row or statement (10 October) | Current evidence |
|---|---|
| "No real-data experiments were run" | Superseded. Main cohort assembled (39,898 crops, 4,999 videos); nine RGB runs, six reference probes, and Q40/Q80 known-quality probes completed on validation. See the module tracker and aggregate milestone. |
| Python 3.12.14, NumPy 2.3.5; torch absent | Historical preprocessing interpreter only. Experiments use the Python 3.11.9 DeepFake environment with torch 2.7.1+cu128 (`environment_deepfake.json`). |
| GPU benchmark `not_run` | Superseded by measured runs: main RGB runs took 49.1–56.0 min with about 500 MiB peak reserved PyTorch GPU memory; system RAM peaks remain unmeasured. |
| Actual model training: not run | Superseded. All classifier results to date are **validation** evidence; classifier **test** scoring has not occurred. |
| FFmpeg/FFprobe absent from PATH | Not rechecked. Video-codec work is an extension outside the frozen core. |
| Free storage about 209 GiB | Snapshot only; about 198 GiB was free on 23 October 2026. |

**Update, 24 October 2026.** The statement above that classifier test scoring has not occurred
is now superseded. Test scoring took place once, under register gates G2–G4: scorers were
verified on validation, the model-freeze manifest and Amendment 1 were recorded, then test
was scored and analysed. See [main_results_test.json](main_results_test.json). About 196 GiB
was free afterwards.

The synthetic-check rows remain valid as code-path checks; they are not scientific results.
