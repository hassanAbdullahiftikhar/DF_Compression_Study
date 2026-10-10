# Experimental protocol and hardware feasibility

Prepared 10 October 2026. This is a proposed protocol, not measured experimental results. Numerical compression settings below replace equations/values missing from the supplied Markdown; they are recommendations, not recovered text.

## 1. Scope and operational definitions

The primary project is **closed-set FF++ manipulation-method attribution under compression**, alongside binary real/fake detection. Fake source labels are Deepfakes, Face2Face, FaceSwap, NeuralTextures. Real examples do not have a fake source label. Distinguish this from architecture identification, trained checkpoint attribution, open-set discovery, and fully synthetic image attribution.

The minimum defensible study has RGB, FFT, block-DCT, Haar-wavelet, residual, and frozen DINOv2-small representations; a controlled JPEG sweep; two tasks; quality adaptation controls; and paired uncertainty. H.264 and a held-out H.265 subset strengthen the study. Repeated compression and source-aware learning follow after the baseline works. GenImage and temporal/3D/restoration models are extensions.

RQ2 is a hypothesis, not an expected result: attribution might degrade faster, similarly, or more slowly for some sources. Do not define success as obtaining the predicted ordering.

## 2. Dataset access and lineage

Obtain authorized FF++ access through the [official repository](https://github.com/ondyari/FaceForensics), accept the dataset terms yourself, and use its released split JSON files. No access request or acceptance of terms was submitted during this preparation. Do not start by downloading all three qualities, every added dataset, and every source model. Obtain a balanced pilot first, preferring the least additionally compressed videos for controlled re-encoding.

Use official train/validation/test assignments, and map **both IDs** in a manipulated video filename to the same split. Record every underlying original ID. Build connected provenance groups for original/manipulated videos sharing an input. Keep all conditions, frame samples, and compression variants of a media item in its existing split. If any official split has a lineage collision, report it and resolve before experiments rather than silently randomizing frames. Dataset provenance grouping is not identity-disjoint splitting; repeated identities or YouTube lineage may require a stronger sensitivity analysis.

Suggested manifest fields:

| Field | Meaning |
|---|---|
| sample_id | Stable video-plus-frame identifier, common across compression variants |
| path | Absolute crop path |
| reference_path | Absolute lossless crop path before the controlled transform |
| split | train, val, or test |
| group_id | Connected lineage group for clustered uncertainty |
| video_id | Unique media item, including manipulation method |
| original_ids | Pipe-separated original input IDs, e.g. 001\|002 |
| source | real, Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| frame_index | Fixed decoded-frame index |
| condition | ref, jpeg_qXX, jpeg_q80_q80, etc. |
| history | Ordered transformations, encoder settings, and prior history if known |

Additional production fields should include dataset release, original file SHA256, timestamps/PTS, crop coordinates, detector/version, bounding box confidence, crop margin, color pipeline, dimensions, failure status, and codec metadata. Hash manifests and retain full run configurations. Empty original history means unknown, not pristine.

## 3. Preprocessing: control geometry before studying compression

Sample eight frames per video at fixed positions after validating duration/frame count. At 5,000 nominal videos this produces about 40,000 crops: 28,800 train, 5,600 validation, 5,600 test if all videos yield eight valid samples. Actual counts depend on coverage and failures.

For each reference frame, detect a face once with a fixed detector, choose a deterministic crop margin, and save its box. Crop and resize to 224×224 with a declared interpolation method and save PNG. Avoid face warping in the main experiment because it modifies fine texture. Apply JPEG to that fixed lossless crop. Verify equal detection-failure treatment and paired support across classes. Do not discard a degraded sample solely because its detector fails while keeping its reference; model that failure in a separate end-to-end arm.

For video tests, **transcode the complete video before extracting corresponding frame indices and applying the saved reference boxes**. Keep dimensions, frame rate, and timing consistent; check decoded frame counts/PTS. Redetecting faces at every quality would mix codec effects with crop jitter. A separate realistic arm may redetect and report coverage/failure as part of system performance.

FF++ originals have earlier acquisition compression. Use “reference” or “least additionally compressed” rather than “pristine.” Codec-native quantization statistics cannot be inferred directly from a crop's recomputed DCT coefficients.

## 4. Compression matrix

| Arm | Recommended conditions | Training allocation | Interpretation |
|---|---|---|---|
| JPEG main sweep | reference; Q=95,80,60,40,20 | mixed training uses reference and Q=80,40; val at those same conditions | Q=95,60,20 are withheld from training and model selection |
| JPEG history | 80→80, 80→40, 40→80 | test after single-quality training | Order includes decoding between passes; 40→80 cannot restore the lost signal |
| H.264 extension | libx264 CRF=18,23,32,40 | train/reference plus selected levels; evaluate withheld levels | Whole-video compression, fixed preset/pixel format/GOP |
| H.265 extension | libx265 CRF=18,23,32,40, or matched distortion operating points | no H.265 training or validation | Held-out codec evaluation |
| Video history | H.264 23→23; 23→40; H.264 23→H.265 matched condition | held-out history | Measure final bitrate/distortion and preserve intermediate metadata |
| Resizing sensitivity | 224 reference versus one declared downsample/upsample condition | separate arm | Do not merge resizing with the main severity axis |

JPEG quality is implementation-specific. Fix Pillow/libjpeg version, chroma subsampling (initially 4:2:0), optimize/progressive flags, and color mode. Include a small 4:4:4 control to separate chroma effects from quantization. Quality 100 is still a lossy encoding and is not the reference.

Video settings must include codec/version, preset, pixel format, rate-control mode, CRF, GOP/keyframe spacing, B-frames, dimensions, frame rate, and audio handling. Software x264/x265 provide an interpretable initial codec family. Hardware NVENC is another implementation domain, not interchangeable ground truth for software encoding. Do not encode individual JPEG frames and call the experiment H.264/H.265.

**Equal CRF across codecs does not mean equal distortion or bitrate.** Report bitrate/bits per pixel and PSNR/SSIM of aligned decoded frames alongside CRF. Compare codecs at matched distortion and, separately if practical, matched bitrate. Repeated mild encoding versus one severe pass needs a final-distortion match to isolate history. Treat real platform redistribution as an additional experiment requiring documented platform settings, not a property proven by local re-encoding. [FFmpeg codec documentation](https://ffmpeg.org/ffmpeg-codecs.html).

## 5. Representations and fairness

| Representation | Initial implementation | Necessary qualification |
|---|---|---|
| RGB | ResNet-18 with two heads; RGB moments/pooled descriptor for pipeline checks | Descriptor baseline is deliberately weak; do not present it as a strong RGB neural baseline |
| FFT | log magnitude of centered luminance FFT; fixed pooling | Magnitude discards phase and localization; normalization fit on train only |
| DCT | orthonormal 8×8 block DCT statistics, optional global DCT | Recomputed on decoded pixels, not native JPEG coefficient extraction; keep block grid fixed |
| Wavelet | one-level Haar subbands with statistics; later localized maps | Declare levels/filter; Haar results do not represent every wavelet |
| Residual | declared Laplacian/high-pass filter statistics | Starter Laplacian is **not SRM**; add a verified fixed SRM bank before making SRM claims |
| DINOv2 | frozen ViT-S/14 CLS embedding, batched offline extraction | Semantic/pretrained features may encode identity/content; include content controls |

Do two distinct comparisons. First, fit equal-protocol linear probes on frozen representations, with training-only standardization and validation-selected regularization. This measures accessibility to a simple probe, not total information. Use learning curves to check sample sufficiency. Second, use a common small learnable encoder/head budget for compatible representation maps, with declared channel adapters and initialization. Include pretrained RGB alongside pretrained DINO: otherwise representation and pretraining are confounded.

DINOv2 small is substantially preferable to giant variants here. Freeze its encoder and store features rather than repeated end-to-end training. At 384 float32 components, 40,000 embeddings take 61.44 MB decimal; six conditions take about 369 MB. Use the same sample mask for every representation. [Official DINOv2 repository](https://github.com/facebookresearch/dinov2).

## 6. Baselines and source-aware paired learning

Required training regimes:

1. Reference-only binary and source heads (unaligned multi-task).
2. Reference-only separate binary and fake-only attribution models, to check effects of shared capacity.
3. Mixed-quality augmentation using the same number of presentations and optimizer updates.
4. Two-view training with no pair loss (compute/data exposure control).
5. Binary-oriented pairing without source supervision, then a train-only source probe.
6. Source-supervised two-head pairing with a small validation sweep of pair weight.
7. An auxiliary compression-prediction head, as a lightweight comparator to compression invariance; hold encoder/update budget fixed.

If time permits, reproduce ADD or QAD on a small backbone and transparently adapt its source head. Published scores using different splits/backbones/frame counts must not be treated as directly comparable experimental results. Keep exact paper reproduction distinct from architecture-inspired controls.

For x and its degraded counterpart C_q(x), let z=E(x), z_q=E(C_q(x)), binary target y∈{0,1}, fake source s∈{0,1,2,3}. A transparent objective is:

\[
L=L_{det}+\lambda_{src}L_{src}+\lambda_{pair}L_{pair},
\]

\[
L_{det}=\tfrac12[CE(h_{det}(z),y)+CE(h_{det}(z_q),y)],
\]

\[
L_{src}=\tfrac12[CE(h_{src}(z),s)+CE(h_{src}(z_q),s)]\quad\text{computed only on fake examples},
\]

\[
L_{pair}=\mathbb{E}[1-\cos(Pz,Pz_q)].
\]

Use a projection P so a paired constraint need not force every encoder feature to be identical. Compare λ_pair=0,0.1,1.0 on validation, λ_src=1 initially. The starter trainer implements this modest **baseline**, not a novel algorithm or full ADD/QAD. Supervised contrastive pairing is optional and needs class-balanced batches; real samples must not acquire a fabricated generator label.

Uniformly sampling five classes makes four times as many fakes as reals. Counter this with balanced detection sampling/loss weights while keeping fake sources balanced. Report the chosen training prior; AUROC is prior-independent in its definition, but operating thresholds and precision are not. Use the true test distribution and balanced accuracy.

## 7. Metrics, paired inference, and interpretation

Aggregate frame probabilities per video before primary metrics. Binary: AUROC, EER, balanced accuracy, macro F1; report low-FPR sensitivity if test support permits. Select the operational threshold on validation, then freeze it across test compression levels. EER is a descriptive test ROC statistic rather than the deployed threshold. Four-way attribution: fake-only balanced accuracy, macro F1, top-1, per-source recall, confusion matrix. Also report end-to-end probability of correct fake-source assignment with detection gating. Restricting evaluation to detected fakes biases conditional accuracy, so show that separately with coverage.

Use a common chance-adjusted scale for the task-degradation comparison:

\[
S_t(q)=\frac{BA_t(q)-b_t}{BA_t(ref)-b_t},\quad b_{det}=0.5,\ b_{attr}=0.25.
\]

This equals 1 at reference, 0 at chance, and may be negative or above 1. Do not clip for analysis. It is undefined/unstable if the reference score is at or near chance; mark that representation as uninformative rather than manufacturing retention. Predeclare a minimum above-chance margin, e.g. 0.05, as an analysis rule. The four-way chance value does not apply to macro F1 or a five-way real-plus-source classifier. AUROC has its own 0.5 baseline; do not divide attribution accuracy by detection AUROC and call that a matched comparison.

Test Δ(q)=S_attr(q)−S_det(q) using a **paired lineage-group bootstrap**, resampling groups with all their videos/frames/conditions retained. Use 2,000 bootstrap draws as a starting point and report 95% intervals plus three training seeds for final selected runs. Account for normalization-denominator variation within the bootstrap. Predefine a few primary contrasts (e.g. Q=60,20, held-out H.265, history effect) and correct multiple tests or mark the rest exploratory. Nominal thousands of frames do not imply thousands of independent observations.

Two complementary tests avoid a common interpretation error:

- Train at reference and test degraded inputs: measures deployment sensitivity/domain shift.
- Retrain or probe at matched quality and test that quality: estimates how recoverable cues are with adaptation.

Poor transfer with good matched-quality performance means the representation/classifier can adapt. Poor performance under both regimes still does not prove information-theoretic irrecoverability. UMAP/silhouette/CKA are supplementary views, not proof of a physical fingerprint or causal compression invariance.

## 8. Shortcut and negative controls

- Apply the identical controlled compression distribution to every source and to real media. Track original histories; PNG conversion does not erase them.
- Train a background-only/crop-border probe and a codec-only metadata probe. High source accuracy suggests a confound requiring investigation.
- Within each quality, probe codec/quality from embeddings. A successful probe alone is not evidence of label leakage: investigate correlations with source.
- Compare fixed reference boxes with redetected boxes; measure face-detection failure and bounding-box drift by source/quality.
- Use label-shuffle controls at the lineage-group level, and train-only feature scaling/regularization selection.
- Use matched-duration/frame-count video sampling, balanced source coverage, and synchronized frame indices. Log missing frames rather than silently changing support.
- Test whether identity/content predicts source within held-out groups. Same-source positives from different identities can reduce content shortcuts but do not guarantee removal.

For GenImage, preserve generator and semantic/content labels, balance resolution and file histories, and avoid duplicate real images crossing splits. Its official design is detection-focused; it should become a separate generator-family attribution experiment rather than casually mixed with FF++. [GenImage paper](https://arxiv.org/abs/2306.08571).

## 9. Hardware feasibility and local observations

**Yes: the primary frame-based project fits this hardware with conservative batches and staged data preparation.** Full reproduction of every temporal/restoration baseline and storing the entire codec grid at once would be a poor scope choice.

Observed locally: NVIDIA GeForce RTX 5070; 12,227 MiB total GPU memory reported by nvidia-smi; driver 610.88. CPU registry: AMD Ryzen 7 7700 8-Core Processor. C: free space at initial inspection: 224,768,544,768 bytes (about 209.3 GiB), a snapshot subject to change. The initial Windows system-information query was unavailable, but a second native memory check succeeded: 16,229,781,504 bytes usable physical RAM (about 15.1 GiB, consistent with nominal 16 GB), with 4,781,965,312 bytes (about 4.5 GiB) available at that moment. RAM pressure is therefore a practical consideration even for streaming. FFmpeg was not on the inspected PATH. The inspected bundled Python has NumPy/Pillow but no torch, torchvision, SciPy, scikit-learn, OpenCV, or PyWavelets.

NVIDIA specifies the desktop RTX 5070 as 12 GB GDDR7 and compute capability 12.0. Keep system RAM and GPU VRAM separate. Ryzen 7 7700's 8-core processor is sufficient for controlled offline preprocessing, though x265 encoding may dominate elapsed time. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).

Blackwell needs a compatible PyTorch binary. Current releases should follow the [official install matrix](https://pytorch.org/get-started/previous-versions/); the [2.12 release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/) move newer GPUs to CUDA 13.0+ builds. The starter setup pins documented torch 2.7.1/torchvision 0.22.1 cu128 as a conservative reproducible baseline, rather than claiming it is latest. Use one internally compatible stack and run the GPU forward/backward test. The driver's reported CUDA capability is not the installed toolkit version. Old repository CUDA/PyTorch environments may not support this GPU. Global software was not changed during preparation.

### Starting resource settings — estimates, not benchmarks

| Workload | Starting microbatch | Local assessment |
|---|---:|---|
| Deterministic descriptors / linear probes | stream one crop; small feature arrays | Comfortable; CPU-dominated |
| ResNet-18, single-view, 224² | 16–32 | Should be practical; measure actual peak VRAM |
| ResNet-18, two quality views | 8–16 pairs | Practical candidate; two forward graphs increase memory |
| ResNet-50 / EfficientNet-B0/B1 | 8–16 | Likely practical; backbone and extra losses affect capacity |
| Frozen DINOv2-small | 8–16 | Practical candidate for cached features |
| ADD teacher+student / QAD AWP / DPL | 4–8 initially | Conditional; benchmark the complete training step |
| Temporal encoder / multi-branch restoration | 1–4 short clips initially | Optional, high implementation/runtime uncertainty |
| DINOv2 giant / full million-scale experiments | — | Outside the recommended local scope |

Use AMP, one training run at a time, DataLoader workers=0 initially on Windows (try 2 after measurement), modest prefetch, and disk-backed images/features. Effective batch through gradient accumulation does not reproduce BatchNorm or contrastive-batch behavior of a larger physical batch; disclose this when adapting published protocols. Do not interpret marketing AI TOPS or gaming frame generation as training throughput.

40,000 RGB crops at 224² occupy about 6.02 GB as an uncompressed uint8 array and 24.08 GB as float32. The train subset alone is about 17.34 GB float32. Both exceed a prudent RAM budget when materialized with the OS and model. Stream them. Per-input tensor storage is only a lower bound; activations, gradients, optimizer states, two-view graphs, pinned buffers, and workspace allocations dominate training memory. Disk PNG size is content-dependent and must be measured, not equated with tensor size.

Use pilot measured averages to forecast storage: bytes_per_crop × valid_crops × materialized_conditions plus original videos, transient codec outputs, and checkpoints. Do not assume the full raw FF++ download fits the 209 GiB currently free. Materialize JPEG test variants selectively; perform training augmentation on demand. Encode one codec condition at a time, retain necessary crops/metadata, and remove only explicitly designated reproducible intermediates after checking results. Do not automatically delete original data.

For runtime, benchmark 100–200 warmed-up real training steps and one representative full-video encode. Total training estimate = epochs × ceil(train_samples/batch) × measured step time; multiply by views/runs only as represented by that benchmark. Example: 28,800 samples/batch 32 × 20 epochs = 18,000 steps. At a hypothetical 0.1–0.5 seconds/step this is 0.5–2.5 hours of compute, excluding decoding/validation; these rates are illustrative, not measured RTX 5070 performance. A grid of 6 representations × 3 regimes × 3 seeds is 54 runs before ablations, so stage it rather than launching the entire grid.

## 10. Stage gates and practical schedule

| Stage | Work | Exit criterion |
|---|---|---|
| A: preparation | literature correction, dataset access, Python/codec/GPU audit | access obtained; lineage audit clean; compression tool smoke tests pass |
| B: pilot | about 100 provenance groups, all sources; preserve official split ratios | valid crops and timing; batch memory/time measured; no leaked groups |
| C: main JPEG study | six representations, reference→quality and matched/mixed controls | meaningful above-chance reference scores; paired curves/intervals |
| D: video extension | H.264 sweep; H.265 held out; distortion-matched comparisons | frame alignment and crop controls verified; matched test support |
| E: optional method | two-view/no-pair versus source-aware pair; λ sweep | held-out gains exceed uncertainty; clean attribution trade-off reported |
| F: final | three seeds on selected runs, ablations, limitations, reproducible release | conclusions match evidence; scripts/configs/manifests saved |

The proposal's sixteen weeks are reasonable if weeks 1–2 cover access/preparation, 3–4 the pilot, 5–8 the JPEG study, 9–11 codec extensions, 12–14 selected pairing experiments, and 15–16 analysis/writing. Dataset access delay is a real dependency. If access is delayed or encoding dominates, complete the JPEG study first and narrow the final claims to its measured scope.

## 11. Preparation completed versus pending

Completed: review of twelve references with access qualifications; novelty search including close predecessors; local GPU/CPU/disk audit; explicit protocol and configs; manifest/leakage checks; deterministic JPEG/recompression tools; descriptor extraction; optional ResNet two-head trainer and GPU benchmark entry points; synthetic smoke verification of the dependency-light preparation path.

Pending: dataset access/download, detector choice and face extraction, production provenance grouping, installed research environment/FFmpeg, GPU training benchmark, DINOv2 extraction, full learned transform-map models, paper-baseline reproduction, grouped confidence intervals on real predictions, and all scientific results. These are ongoing research stages, not findings that can be fabricated during setup.
