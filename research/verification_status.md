# Verification record

Date: 10 October 2026. No real-data experiments were run.

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

The C23 mirror is now present. All 7,000 metadata paths exist; 5,000 videos belong to the primary real/four-method study. Official split mapping yielded 500 complete source-pair groups without known input-ID conflicts. MP4 structural checks passed for all core videos; 15 sampled videos decoded completely with matching metadata counts/dimensions. A private preprocessing environment was created and locked. YuNet extraction produced 800 crops from 100 videos with zero logged failures; a training crop preview was inspected. JPEG/history generation produced 7,200 audited rows. Details and qualification of these checks are in `dataset_audit_c23.md`. Classifier training and scientific accuracy remain unmeasured.
