# Downloaded FF++ C23: audit and preparation results

**Updated after dataset arrival, 10 October 2026.** The downloaded folder was located without further input:

`C:/Users/hassa/Desktop/Sem7/MBR/experiments/FaceForensics/FaceForensics++_C23`

**Decision:** suitable to proceed with the C23 pilot based on the inventory, official source-pair mapping, MP4 structure checks, sampled decoding, and a real face-extraction smoke test. This is a study of additional compression from an already compressed reference. Byte-for-byte authenticity against the official release and full decoding of every file have not been established.

## Inventory

| Folder | Videos | Use |
|---|---:|---|
| original | 1,000 | Primary real examples |
| Deepfakes | 1,000 | Primary fake-source class |
| Face2Face | 1,000 | Primary fake-source class |
| FaceSwap | 1,000 | Primary fake-source class |
| NeuralTextures | 1,000 | Primary fake-source class |
| FaceShifter | 1,000 | Available optional unseen-manipulation extension |
| DeepFakeDetection | 1,000 | Separate additional collection; outside primary source pairing |

All 7,000 listed paths exist. The master metadata has 7,000 unique paths, no duplicates, and no missing files. Its source-folder real/fake labels agree for all 5,000 core videos. Original integer IDs and target/source filename pairs are preserved.

## Official split/provenance audit

The three small public official FF++ split files were downloaded and saved with source URLs and SHA256 hashes. The 1,000 underlying original IDs map to 500 complete two-original provenance groups. Every core manipulated filename's two IDs fall within the same official group and split.

| Split | Groups | Videos per source class | Total core videos |
|---|---:|---:|---:|
| Train | 360 | 720 | 3,600 |
| Validation | 70 | 140 | 700 |
| Test | 70 | 140 | 700 |
| Total | 500 | 1,000 | 5,000 |

No known original-ID split conflicts or incomplete groups were detected. This check establishes filename-based source lineage; it does not establish disjoint people or original YouTube uploads.

Artifacts: `experiments/data/splits/`, `experiments/data/audit_c23/audit.json`, and `videos_core.csv`.

## Pilot selection

Seed 42 selected 72 train, 14 validation, and 14 test complete provenance groups, retaining every core source class within each group.

| Split | Selected groups | Videos per class | Total videos |
|---|---:|---:|---:|
| Train | 72 | 144 | 720 |
| Validation | 14 | 28 | 140 |
| Test | 14 | 28 | 140 |
| Total | 100 | 200 | 1,000 |

Pilot manifest: `experiments/data/audit_c23/videos_pilot.csv`. This is video-level input for extraction, not yet the full pilot's face-crop manifest.

## Readability checks

All 5,000 core files passed a bounded MP4 container/header check for valid boxes and expected ftyp/moov/mdat sections. Structural validity alone does not prove that every compressed frame is intact.

Fifteen deterministic sampled videos, one per source/split combination, were decoded completely using PyAV. All passed; their decoded frame counts and dimensions matched the supplied metadata. The selected videos use H.264. The report is `experiments/data/audit_c23/decoded_sample.json`.

## Private preprocessing environment

Created `.venv` within this workspace and installed preprocessing dependencies there, leaving the global runtime unchanged. Resolved versions are saved in `experiments/environment-preprocessing.lock.txt`:

- Python 3.12.14
- NumPy 2.5.3
- Pillow 12.3.0
- PyAV 19.0.1
- OpenCV headless 4.14.0.94

The frozen YuNet 2023mar detector was obtained from the official OpenCV model repository; its URL/hash are saved in `experiments/models/yunet_provenance.json`. This is a face-preprocessing detector, not a trained research classifier. PyTorch/torchvision and the standalone FFmpeg tools are still pending; PyAV's bundled codec libraries already support the decoding used here.

## Real face-extraction and compression smoke test

A nested 8/1/1-group subset contains 100 videos, balanced across the five core labels within each split. Eight fixed frames per video produced **800 reference crops with zero logged extraction failures**, using fixed detector settings, square 1.3× boxes, and 224² Lanczos crops. Extraction took about 52.7 seconds on this selected subset; this is preprocessing timing, not training throughput.

| Split | Reference crops per class | Total reference crops |
|---|---:|---:|
| Train | 128 | 640 |
| Validation | 16 | 80 |
| Test | 16 | 80 |

Metadata records frame indices/PTS, crop boxes, confidences, face counts, versions, hashes, and original C23 history. A training-only crop preview was visually checked across all five labels. The PNGs retain inherited H.264 artifacts; saving them losslessly does not make the reference uncompressed.

The JPEG preparation script then produced five single-pass qualities and three recompression histories, plus the reference rows: **7,200 audited manifest rows** for the same 800 samples. Reference pairing and official split checks passed. These are preparation outputs, not accuracy measurements.

Artifacts:

- `experiments/data/c23_smoke/reference.csv`
- `experiments/data/c23_smoke/extraction_summary.json`
- `experiments/data/c23_smoke/crop_metadata.jsonl`
- `experiments/data/c23_smoke/train_crop_preview.png`
- `experiments/data/c23_smoke/jpeg/manifest.csv`
- `experiments/data/c23_smoke/jpeg/compression_metadata.json`

The smoke set has only ten provenance groups, including one each for validation and test. It is for engineering checks and cannot support a strong generalization or uncertainty claim.

FFT descriptor extraction also completed on all 7,200 smoke rows, producing a disk-backed 7,200×256 feature array with the matching manifest in `experiments/features/c23_smoke_fft`. This verifies real feature extraction; no classifier was fitted to those features.

## Temporal correspondence caveat

The mirror metadata shows frame-count differences between some manipulated files and the original identified by the first filename ID: 0/1,000 Deepfakes, 992/1,000 Face2Face, 496/1,000 FaceSwap, and 497/1,000 NeuralTextures. The recorded comparison is in `frame_count_comparison.json`. These differences do not establish file corruption or explain which pipeline trimmed/retimed a clip.

Do not assume that an equal frame number identifies equal visual content across an original video and every manipulation output. Our compression pairing is within each video's own reference and degraded counterpart, where indices/geometry are preserved. Cross-method temporal/content matching and frame-count/duration shortcut checks require separate verification before stronger content-controlled claims. The uniform eight-frame extraction is an initial frame-based baseline, not a temporally aligned comparison of all manipulation pipelines.

## Next gates

1. Complete the full 1,000-video pilot's crop extraction and source/failure coverage review.
2. Finish the statistical evaluator and stronger representation extraction.
3. Install/benchmark the GPU research stack and run a tiny training sanity check.
4. Launch the predeclared C23-reference JPEG baselines with validation-only model selection.

The initial C23-specific scope is now recorded in `module1_specification.md`. No detection/attribution classifier has been trained or evaluated during this dataset preparation.
