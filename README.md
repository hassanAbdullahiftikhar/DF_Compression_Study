# What Survives Compression?

Updated 2026-10-12. **Research is paused at the user's request.**

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
- Q80 known-quality adaptation: **pending**.

The validation cohort contains 699 videos, including 559 fake videos.
Classifier test evaluation remains pending.

## Main RGB results at JPEG Q40

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

## Reference-trained representation probes at Q40

Values are validation balanced accuracy in percent, with frozen detection
thresholds. Pooled RGB is a pixel descriptor, separate from the trained ResNet.

| Representation | Detection | Fake-only attribution |
|---|---:|---:|
| FFT | 57.48 | 33.11 |
| Pooled RGB | 60.40 | 45.82 |
| DCT | 50.36 | 25.54 |
| WAVELET | 55.16 | 29.73 |
| RESIDUAL | 57.80 | 37.79 |
| DINOv2 | 66.84 | 59.78 |

DINOv2 has the highest absolute Q40 performance among these probes.
The reference-trained DCT probe becomes nearly chance-level under JPEG.
FFT detection retention and wavelet attribution retention have unstable
reference denominators; their affected confidence intervals are unavailable.

The usable probe intervals do not establish a general ordering of detection
versus attribution degradation at Q40. Representation, classifier capacity
and pretraining differ across tracks.

## Known-quality adaptation control at Q40

New scaling and ridge heads were fitted on Q40-compressed training crops
and selected using Q40 validation. Values below are percentages.

| Metric | Reference-trained, evaluated at Q40 | Q40-trained, evaluated at Q40 |
|---|---:|---:|
| DCT: Detection AUROC | 52.55 | 62.09 |
| DCT: Attribution balanced accuracy | 25.54 | 39.36 |
| DINOv2: Detection AUROC | 71.40 | 74.58 |
| DINOv2: Attribution balanced accuracy | 59.78 | 65.85 |

DCT's partial recovery supports training/compression mismatch as a
contributor to its earlier failure. These results demonstrate usable label
information on this validation cohort; they do not establish a physical
forensic fingerprint. Threshold selection domains differ between the two
regimes, so AUROC and fake-only attribution are the clearest comparisons.

## Interpretation and limitations

- All reported classifier results are validation-selected and optimistic.
  Bootstrap intervals condition on selected models and thresholds and
  exclude selection uncertainty.
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

1. Finish Q80 known-quality adaptation if pending.
2. Finalize and run the selected background/border/content, task-sharing,
   source-supervision and preprocessing controls.
3. Run the complete-eight-frame sensitivity analysis.
4. Finalize the comparison register and any remaining mixed-quality probe baselines.
5. Evaluate frozen models on the test cohort and the held-out JPEG qualities
   Q95/Q60/Q20; these qualities have not selected the current models.
6. Complete paired test statistics, seed reporting, distortion and error analysis.
7. Run recompression-history experiments if retained in the final JPEG scope.
8. Refresh the literature/novelty assessment, generate final figures/tables,
   write the manuscript and revise its claims.

Whole-video H.264/H.265 and external datasets remain separate extensions.

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

- [Aggregate validation milestone](research/main_results_validation.json)
- [Current module tracker](research/module_tracker.md)
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
The aggregate milestone preserves metrics, confidence intervals,
configuration/resource records and artifact hashes.

Saved hashes describe execution artifact bytes. The snapshot also records
LF-normalized source hashes for comparison with Git's text normalization.
