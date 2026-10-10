# Complete research roadmap: from preparation to the final paper

**Project:** What Survives Compression? Detection and manipulation-method attribution under controlled lossy compression.

**Prepared:** 10 October 2026. **Purpose:** an execution guide identifying every module's prerequisites, data, steps, outputs, and completion checks. This roadmap does not claim that planned experiments have been completed.

**Dataset amendment:** the user has now provided the Kaggle C23 mirror. The current core experiment measures additional compression from already compressed C23-derived references. The actual audit/preparation status is in `dataset_audit_c23.md`, the initial frozen scope is in `module1_specification.md`, and the tracker takes precedence over earlier prospective status descriptions below. A raw-reference subset remains an optional stronger extension.

## How to use this roadmap

Work through the core modules first. Dataset access and environment preparation can progress independently. Build the evaluation system before launching the main training grid. Add video codecs, recompression, and source-aware learning after the JPEG pilot works. Write the introduction, related work, and methods while experiments are running; write results, conclusions, and the abstract from actual evidence.

The main scope is **four-way manipulation-method attribution on known FF++ methods**, not identification of an individual trained generator or discovery of unknown generators. A separate GenImage extension would address generator-family attribution. A source label here describes a known pipeline and its dataset implementation; successful classification alone does not establish a physical fingerprint.

### Scope tiers

| Tier | Modules/experiments | What the resulting paper can claim |
|---|---|---|
| Core | Scope/literature, environment/data/splits/crops, JPEG, metrics, pilot, representations, main runs, statistics, reproducibility, writing | Controlled JPEG sensitivity and adaptation of detection versus closed-set manipulation attribution |
| Main extension | H.264/H.265 and repeated compression | Video-codec generalization and processing-history effects under the tested settings |
| Learning extension | Source-aware paired training and ablations | Whether source supervision and pairing improve robustness over matched controls |
| Optional external validation | Celeb-DF detection, or a separately designed GenImage attribution study | Additional evidence within each external dataset's compatible task/label scope |

If the video extension is not completed, narrow the title and conclusions to JPEG-compressed face crops. If paired learning does not improve performance, retain the empirical study and report the negative result. The full combined scope is a candidate contribution; no worldwide-first claim is established.

### Dataset and asset inventory

| Dataset/asset | Required? | Where used | What to obtain | Important boundary |
|---|---|---|---|---|
| FaceForensics++ | Yes | Most experimental modules | Authorized least-additionally-compressed original and four manipulation videos; official split JSON files | Main labels identify manipulation methods, not generator checkpoints |
| FF++ c23/c40 | Optional | Published-setting checks and an additional compression comparison | Selected released variants if storage permits | These do not replace a controlled sweep from common references |
| FaceShifter addition | Optional | Detection on an unseen manipulation | Authorized addition from the official source | It is unknown to a four-source head; do not force it into a known class |
| Celeb-DF v2 | Optional | External detection | Authorized data and official test list | Primarily a binary external evaluation here; no direct four-way FF++ attribution target |
| GenImage subset | Optional separate study | Generator-family attribution | Generator labels, paired real-data provenance where available, semantic labels, sizes, original histories | Requires its own confound controls and splitting design |
| ImageNet pretrained weights | Recommended comparison asset | RGB pretrained encoder | Official backbone weights and exact version/hash | A checkpoint is an asset, not a new dataset to download in full |
| DINOv2-small weights | Core representation asset | Frozen pretrained features | Official ViT-S/14 checkpoint and preprocessing specification | No need to train DINOv2 or download its pretraining data |
| Face detector checkpoint | Yes for video-to-face preprocessing | Fixed crops | A chosen detector's official weights and version | No face-detector training dataset is required if the detector is frozen |

Dataset access and definitions are documented by the [FF++ authors](https://github.com/ondyari/FaceForensics), [Celeb-DF authors](https://github.com/yuezunli/celeb-deepfakeforensics), and [GenImage authors](https://github.com/GenImage-Dataset/GenImage). Use the [official DINOv2 implementation and weights](https://github.com/facebookresearch/dinov2). Start with FF++; external datasets are not prerequisites for the pilot.

## Module 01 — Freeze the research question and analysis rules

**Purpose:** decide exactly which claims the experiment will support before observing its test results.

**Prepare beforehand:** read the revised literature review and experimental protocol; establish the available research period and compute budget. **Dataset needed:** none yet; know FF++'s label definitions. **Dependencies:** none.

**Steps**

1. Write RQ1: how do tested representations change under compression for detection and manipulation attribution?
2. Write RQ2: after accounting for different chance baselines, does attribution degrade faster than detection?
3. Write RQ3: does source-aware quality pairing improve held-out compression relative to augmentation and ordinary multi-task learning?
4. Define binary labels and four fake-source labels. Real media has no fake source label.
5. Specify the primary JPEG conditions, training conditions, held-out qualities, codec extension, and recompression histories.
6. Fix the primary unit as a video and uncertainty unit as a connected provenance group.
7. Record a small set of primary contrasts, validation rules, seed plan, and treatment of near-chance references. Label remaining comparisons exploratory.
8. Define what counts as a useful result without requiring the hypothesis to be true.
9. Version this document. Log later changes and their reason, including whether test results were already visible.

**Outputs:** a short analysis specification and a frozen initial experiment matrix. **Done when:** every proposed conclusion maps to a measured task, dataset, condition, and metric.

**Current status:** a recommended specification exists in `experimental_protocol.md` and `configs/pilot.json`; it still needs to be finalized against real pilot resources before main experiments.

## Module 02 — Consolidate the literature and comparison baselines

**Purpose:** make the final contribution defensible and select a realistic set of comparisons.

**Prepare beforehand:** Module 01's scope and the supplied proposal. **Dataset needed:** none. **Dependencies:** Module 01 for the final framing; reading can begin immediately.

**Steps**

1. Use the existing review of all twelve cited papers and the corrected bibliography.
2. Explicitly discuss Yu's JPEG attribution experiment, Frank's frequency/source identification study, and the 2025 detection-versus-attribution comparison.
3. Keep a paper matrix containing task, quality transform, data, model, metric, accessible code, and limitations.
4. Obtain complete text for the three papers whose review relied on abstracts/previews if their exact methods will be reproduced.
5. Audit repository availability, licenses, checkpoints, preprocessing, and environment requirements before committing to a reproduction.
6. Pick one complex compression-oriented reference method, such as ADD or QAD, if the core work leaves enough time.
7. Distinguish exact reproduction, small-backbone adaptation, and architecture-inspired comparison in notes and later tables.
8. Update the novelty search shortly before writing/submission so the claim reflects intervening papers.

**Outputs:** final related-work matrix, bibliography, chosen baselines, and a narrowly worded contribution statement. **Done when:** the study's difference from its closest predecessors is concrete and no priority claim relies on overlooking attribution literature.

**Current status:** substantial preparation completed. Complete-text and implementation audit remain necessary for any chosen complex reproduction.

## Module 03 — Establish the software environment and measured resource budget

**Purpose:** ensure that the planned workload can actually execute on the local machine.

**Prepare beforehand:** RTX 5070/driver information, disk space, a private project environment, the existing requirements and benchmark scripts. **Dataset needed:** synthetic tensors first; one representative real video/crop batch later. **Dependencies:** none for the synthetic check.

**Steps**

1. Create an isolated Python environment in the workspace; preserve global software.
2. Install mutually compatible torch/torchvision with Blackwell support, plus the research dependencies.
3. Install or locate FFmpeg/FFprobe with software x264/x265 support for the video extension.
4. Run `preflight.py` and record interpreter, package, driver, codec, CPU, RAM, and disk information.
5. Run `selftest.py` to verify the preparation path in that environment.
6. Run `benchmark_gpu.py` with eight pairs initially; perform an actual forward/backward step, not merely a CUDA availability check.
7. Measure peak allocated/reserved GPU memory and step time; test a smaller batch if needed.
8. Once a real pilot exists, measure JPEG generation, decoding, DataLoader throughput, and validation overhead as well.
9. Encode one representative video to measure time/storage for each codec. Extrapolate from measured video duration and source sizes.
10. Freeze practical batch size, worker count, preprocessing concurrency, and the order of experiments. Save a package lock/version record.

**Outputs:** environment record, benchmark results, installation instructions, and a measured run budget. **Done when:** a tiny real training job completes within GPU/RAM limits and video preprocessing is measured if that extension is included.

**Current status:** hardware audit and synthetic preparation checks completed. PyTorch/FFmpeg installation and measured GPU training are pending. Twelve GB GPU memory is distinct from the nominal sixteen GB system RAM; stream images from disk.

## Module 04 — Obtain and inventory the main dataset

**Purpose:** acquire enough authorized data without unnecessary downloads.

**Prepare beforehand:** storage estimate, official access process, a download/integrity log, and a planned folder structure. **Dataset needed:** FF++ originals plus Deepfakes, Face2Face, FaceSwap, NeuralTextures and official split files. **Dependencies:** Module 01; can proceed alongside Module 03.

**Steps**

1. Obtain FF++ access using the authors' documented process and follow the data terms.
2. Save the official train/validation/test JSON files and record the release/source version.
3. Begin with a stratified pilot of about 100 original-video provenance groups, preserving official split assignments. This is a suggested preparation budget, not a statistically powered sample requirement.
4. Include all four manipulation types and corresponding real references for selected groups; determine complete pairing from the actual dataset rather than assuming it.
5. Download least-additionally-compressed variants first for controlled transforms.
6. Verify files decode, record duration/dimensions/frame rate/codec, and hash originals.
7. Produce inventory counts by split, source, quality, and provenance coverage.
8. Record missing/corrupt media and resolve it before sampling crops.
9. Expand toward the full main split only after the pilot's storage/compute gate passes.

**Outputs:** authorized media inventory, integrity report, hashes, and official splits. **Done when:** every selected media file has an auditable source, label, and expected split.

**Current status:** no real dataset is present. Access/download is the first external dependency.

## Module 05 — Build lineage groups and leakage-safe splits

**Purpose:** prevent closely related videos and frames from appearing on both sides of evaluation.

**Prepare beforehand:** Module 04's inventory, official split JSON files, naming conventions, and any available original/identity metadata. **Dataset needed:** FF++ original/manipulation metadata; decoding media is not required for most lineage checks. **Dependencies:** Module 04.

**Steps**

1. Parse all original inputs used by each manipulation; preserve both filename IDs where applicable.
2. Build a graph connecting originals and manipulated outputs that share inputs.
3. Assign connected provenance groups and keep each group in a single split.
4. Verify consistency with the official splits. Investigate conflicts rather than quietly creating a frame-level random split.
5. Carry identical group/split assignments into every frame, crop, compression variant, and representation.
6. Check for duplicates, shared original IDs across splits, and conflicting labels.
7. If identity or common YouTube-source metadata are available, audit them separately: lineage-disjoint does not necessarily mean identity-disjoint.
8. Save a complete split audit and a manifest version/hash.

**Outputs:** provenance graph/group mapping and audited media manifest. **Done when:** no known original input or connected provenance group crosses splits, and grouping limitations are documented.

**Current status:** `common.py` checks several leakage/consistency cases. Automatic video inventory parsing and connected-group construction still require implementation against real metadata.

## Module 06 — Extract stable reference face crops

**Purpose:** produce consistent images whose geometry will remain fixed across degradation conditions.

**Prepare beforehand:** audited splits, one fixed face detector/version, crop margin, deterministic sampling rule, and interpolation/color choices. **Dataset needed:** pilot FF++ videos, then the expanded main set. **Dependencies:** Modules 03–05.

**Steps**

1. Validate frame count, duration, and timestamps for each video.
2. Choose eight fixed frame positions per video for the frame-level study; record decoded indices and timestamps.
3. Detect the intended face on reference frames with the frozen detector.
4. Resolve multi-face cases consistently; do not choose different targets by source or quality.
5. Store boxes, detector confidence, original dimensions, and crop margin.
6. Crop and resize once to 224×224; avoid warping/alignment in the main texture study unless explicitly made part of the protocol.
7. Save lossless reference PNG crops, linking them back to videos and original IDs.
8. Review an equal-sized visual sample from each source and split.
9. Record missing faces, corrupted frames, and failures; define a common paired evaluation support set.
10. Export `reference.csv` and run the manifest audit with official splits.

**Outputs:** reference crops, boxes/timing metadata, exclusion log, and crop manifest. **Done when:** labels and face regions are correct, preprocessing is identical across sources, and selected frames are reproducible.

**Current status:** manifest template exists; face extraction itself is pending. Eight separated frames do not support a claim of temporal trajectory modeling.

## Module 07 — Prepare controlled JPEG conditions

**Purpose:** vary compression while keeping sample identity and crop geometry constant.

**Prepare beforehand:** Module 06's reference PNGs and manifest; encoder version, subsampling, and quality schedule. **Dataset needed:** reference face crops, no additional dataset. **Dependencies:** Module 06.

**Steps**

1. Preserve reference PNGs; call them reference/least-additionally-compressed rather than pristine sensor data.
2. Generate JPEG Q=95,80,60,40,20 from the same reference crop.
3. Keep encoder implementation, 4:2:0 subsampling, color mode, and optimize/progressive options fixed.
4. Record transform order, hashes, byte counts, encoder/library version, and manifest associations.
5. Verify that decoded dimensions and sample IDs match references.
6. Materialize pilot and evaluation variants; use on-demand compression for training to conserve storage.
7. Reserve reference/Q80/Q40 for mixed-quality training and model selection in the initial protocol; keep Q95/Q60/Q20 held out.
8. Measure distortion/file size and create an illustrative comparison panel.
9. Check the same quality distribution is applied to real and every fake source.

**Outputs:** paired JPEG manifest, metadata, distortion summary, and qualitative examples. **Done when:** every tested JPEG maps to its exact reference, and training/validation have not accessed held-out conditions.

**Current status:** `prepare_jpeg.py` and on-demand trainer compression exist; JPEG/history checks passed on synthetic images. Real-data verification and distortion summaries are pending.

## Module 08 — Build and validate the evaluation system

**Purpose:** make every model produce comparable, correctly aggregated results.

**Prepare beforehand:** task labels, condition names, grouping rules, and the analysis specification. **Dataset needed:** synthetic prediction cases first; real validation predictions later. **Dependencies:** Module 01; finish before Module 11.

**Steps**

1. Standardize prediction output: sample/video/group/condition, true binary/source labels, fake probability, and four source probabilities.
2. Average frame probabilities per video and condition for primary reporting; retain frame-level scores as secondary data.
3. Implement binary AUROC, balanced accuracy, macro F1, EER, and a declared operating threshold rule.
4. Implement fake-only source top-1, balanced accuracy, macro F1, per-source recall, and confusion matrices.
5. Select the operating threshold on validation and freeze it across test qualities.
6. Report detection-gated correct attribution separately, with coverage; do not hide hard examples by evaluating only detected fakes.
7. Implement chance-adjusted retention using binary balanced accuracy with chance 0.5 and four-way source balanced accuracy with chance 0.25.
8. Mark retention undefined when the reference score is at/near chance; predeclare the margin and keep raw scores visible.
9. Implement paired group-level resampling that retains every video/frame/condition in a sampled provenance group.
10. Verify perfect, chance/tied-score, missing-class, imbalanced, duplicate, and aggregated prediction cases.
11. Define primary contrasts and treatment of multiple comparisons before inspecting final test results.

**Outputs:** evaluation scripts, prediction schema, metric verification report, and analysis templates. **Done when:** synthetic known-answer cases pass and the same sample support is used for all paired comparisons.

**Current status:** `metrics.py` implements several core metrics and video aggregation. EER, detailed confusion/coverage reporting, retention, and paired bootstrap still need implementation; current checks are not a complete final statistical pipeline.

## Module 09 — Run the small end-to-end pilot

**Purpose:** identify pipeline defects and measure costs before the large experiment grid.

**Prepare beforehand:** working environment, pilot media, audited crops/compression, and core evaluation. **Dataset needed:** the stratified FF++ pilot from Module 04. **Dependencies:** Modules 03–08.

**Steps**

1. Check that every pilot split has real examples and all four fake classes; revise group selection if class support is incomplete.
2. Overfit a tiny training subset with pairing disabled to check labels, gradients, preprocessing, and output heads.
3. Run a short reference-only RGB baseline with validation monitoring.
4. Evaluate on pilot reference and JPEG conditions using video aggregation.
5. Run one cheap FFT/DCT descriptor probe for an independent code path.
6. Run label-shuffle and duplicate/leakage checks; investigate suspiciously perfect validation results.
7. Measure peak VRAM, available RAM, seconds/step, JPEG/decoder time, crop size, and output storage.
8. Inspect a few failures by source and quality rather than relying only on an overall number.
9. Fix defects and repeat only affected checks/runs; label pilot results exploratory.
10. Freeze the main dataset selection and training budget before touching main test predictions.

**Outputs:** pilot results, issue log, measured budget, and a go/no-go decision for expansion. **Done when:** the pipeline runs reproducibly, the reference task shows useful learning, and capacity/data issues are understood. A pilot does not establish the final scientific conclusion.

**Current status:** not started on real media.

## Module 10 — Complete the representation panel

**Purpose:** compare which kinds of features retain usable information.

**Prepare beforehand:** successful pilot, train/validation/test manifests, defined representation settings, and cache budget. **Dataset needed:** the same FF++ crops/conditions used by every representation; official pretrained weights for RGB/DINOv2. **Dependencies:** Module 09.

**Steps**

1. Establish RGB pretrained ResNet-18 as a practical learned baseline.
2. Extract global log-magnitude FFT features; declare luminance conversion, centering, pooling, and retained phase/magnitude choices.
3. Extract orthonormal 8×8 block-DCT features; distinguish decoded-pixel DCT from native codec coefficients.
4. Extract declared Haar-wavelet subbands/statistics; fix decomposition levels and filter settings.
5. Extract the declared Laplacian residual baseline. Implement a verified SRM bank separately if making SRM claims.
6. Implement frozen DINOv2-small feature extraction, using official preprocessing and small batches.
7. Cache all features with sample ID order, dimensions, version, checkpoint hash, and transform configuration.
8. Fit scaling only on training features; choose probe regularization only on validation.
9. Compare common linear probes, while acknowledging that different fixed representations expose different information to a linear classifier.
10. If time permits, add learned map encoders with declared parameter/channel budgets and matched initialization/pretraining comparisons.
11. Check equal evaluation support and sanity-test descriptor magnitudes, frequency alignment, and feature dimensionality.

**Outputs:** documented six-family feature panel, extraction scripts/caches, and trained probes. **Done when:** each claimed representation has a real implementation with auditable preprocessing and a comparable evaluation protocol.

**Current status:** five deterministic descriptor families and probe/trainer starters exist. DINOv2 extraction, full SRM, and learned transform-map backbones are pending. Descriptors must not be presented as reproductions of F³-Net or other named papers.

## Module 11 — Run the main JPEG detection/attribution experiments

**Purpose:** obtain the principal evidence for RQ1 and the JPEG part of RQ2.

**Prepare beforehand:** Modules 01/08's frozen analysis rules, the representation panel, expanded data, and a run register. **Dataset needed:** audited FF++ reference/JPEG crops; no extra dataset. **Dependencies:** Modules 08–10.

**Steps**

1. Train reference-only baselines and test each fixed model across the JPEG sweep. This measures sensitivity to deployment degradation.
2. Train matched-quality baselines at selected JPEG conditions and test at those conditions. This measures adaptation; label it separately from held-out-quality evaluation.
3. Train mixed-quality baselines on reference/Q80/Q40 and test the full sweep, holding Q95/Q60/Q20 out of training and model selection.
4. Evaluate both tasks with identical representation/sample support and retain raw probabilities.
5. Compare a shared two-head RGB encoder with separate detection and attribution training for a selected backbone, to check shared-capacity effects.
6. Keep training updates, sample presentations, pretraining, and selection rules comparable; record unavoidable differences.
7. Run a small initial seed budget, then repeat preselected final comparisons with seeds 42,43,44. Do not select final comparisons because one seed looked favorable on the test set.
8. Save best validation checkpoint, run settings, environment/hash, validation selection, predictions, and timing.
9. Generate raw-score and retention curves with uncertainty from Module 16.
10. Examine source-specific changes; an average may conceal one fragile manipulation.

**Outputs:** principal JPEG results, predictions, checkpoints, curves, and per-source comparisons. **Done when:** reference→compression, matched-quality, and mixed/held-out protocols are clearly distinguished and the main results are reproducible.

**Current status:** not run. Starter tools cover portions of this grid; the full experiment runner/register still needs preparation.

## Module 12 — Add whole-video H.264/H.265 experiments

**Purpose:** test video compression and a held-out codec while preserving temporal codec behavior.

**Prepare beforehand:** software codecs, reference video timing/boxes, measured encoding budget, aligned comparison logic, and representation choices frozen from pilot/validation evidence. **Dataset needed:** original FF++ videos, not just cropped JPEG images. **Dependencies:** Module 09; usually execute after Module 11's pipeline is sound.

**Steps**

1. Select a provenance-balanced video subset before evaluating this extension's test results.
2. Transcode whole videos with H.264 at declared CRFs, initially 18,23,32,40.
3. Freeze preset, dimensions, frame rate handling, pixel format, GOP/B-frame settings, and encoder version.
4. Check decoded frame counts/PTS and extract corresponding frames using saved reference crop boxes.
5. Record bitrate, file size, and frame distortion against aligned references.
6. Train H.264-domain baselines using predeclared conditions and evaluate withheld H.264 severities.
7. Produce H.265 conditions without using them for training or model selection.
8. Choose H.265 operating points for distortion-matched comparisons using a separate calibration/validation pool, then fix that mapping for test; equal CRF is not equal quality.
9. Evaluate both tasks on the held-out H.265 test conditions and clearly state the training domain.
10. Compare fixed boxes against a separate redetection arm if measuring practical preprocessing failures.
11. Record complete encoding and extraction histories. Keep this distinct from zero-shot transfer of a JPEG-trained model to video codecs.

**Outputs:** whole-video codec manifests, aligned crops, bitrate/distortion measurements, cross-quality/cross-codec results. **Done when:** timing/crop alignment is verified and codec effects are not confused with quality or preprocessing changes.

**Current status:** a codec planning/execution script exists; real encoding, alignment, calibration, extraction, and training remain pending.

## Module 13 — Test repeated compression and processing order

**Purpose:** separate the effect of processing history from simply ending at worse quality.

**Prepare beforehand:** single-pass baselines, ordered transformation metadata, common references, and a final-distortion matching rule. **Dataset needed:** existing FF++ reference crops; whole videos for the video-history arm. **Dependencies:** Module 11, plus Module 12 for video history.

**Steps**

1. Produce JPEG 80→80, 80→40, and 40→80 with decoding between passes.
2. Keep transformations ordered; a later high-quality encoding cannot restore earlier discarded information.
3. Compare against single-pass Q80/Q40 and, where useful, distortion-matched single-pass controls.
4. Measure achieved distortion/size; do not assume repeated and single encodes with the same label have equal output quality.
5. Test already-trained single/mixed-quality models on unseen histories without history-specific tuning.
6. Optionally test H.264 23→23, 23→40 and H.264→H.265 histories using whole videos.
7. Analyze task changes, source confusion, and history-order effects with paired samples.
8. If later training includes histories, label that as adaptation and keep a new held-out-history evaluation.

**Outputs:** processing-history dataset manifest and single-versus-repeated comparison. **Done when:** any history claim is supported beyond a difference in final distortion alone.

**Current status:** JPEG history generation is implemented and synthetically tested; scientific comparisons and distortion matching are pending.

## Module 14 — Test source-aware compression-paired learning

**Purpose:** evaluate whether pairing helps preserve source-discriminative information beyond ordinary augmentation.

**Prepare beforehand:** baseline results, training-only paired views, source labels, validation search budget, and fair compute controls. **Dataset needed:** existing FF++ references plus on-demand selected compressed counterparts; no new dataset. **Dependencies:** Modules 09–11; include video pairing only after Module 12 is stable.

**Steps**

1. Start from the same small encoder and initialization as the RGB baseline.
2. Use binary and four-source heads; apply source loss only to fake examples.
3. Supervise both reference and compressed views, then add a small projection-space agreement loss.
4. Compare the two-view no-pair control with identical sample exposure and update count.
5. Compare source-aware pairing against reference-only training and mixed-quality augmentation.
6. Add a binary-oriented paired encoder plus a train-only source probe, to test the value of explicit source supervision.
7. Sweep pair strength on validation only, initially 0,0.1,1.0; keep held-out qualities/codecs untouched.
8. Examine clean attribution as well as compressed attribution, detection, calibration/coverage, and runtime.
9. If implementing supervised contrastive learning, balance fake classes and distinguish instance pairs from same-method positives across identities.
10. Add a compression-supervised auxiliary head as a declared comparison if feasible.
11. Reproduce or transparently adapt one literature method if selected in Module 02.
12. Repeat selected contrasts across final seeds and report gains, trade-offs, or lack of benefit.

**Outputs:** trained paired model, matched controls, ablation table, and compute/performance trade-offs. **Done when:** any claimed improvement survives fair data/compute comparisons and uncertainty analysis.

**Current status:** a simple paired two-head trainer exists and is syntax-checked. Its training, most ablations, and literature-method adaptations are pending. This is a modest baseline extension, not an established novel algorithm.

## Module 15 — Optional external validation or generator-family study

**Purpose:** assess the bounds of generalization beyond the primary benchmark.

**Prepare beforehand:** an explicit external task/label mapping, additional access/storage, frozen models, and separate preprocessing/split audits. **Dataset needed:** choose Celeb-DF v2 for detection, or a separately designed GenImage subset for generator-family attribution. **Dependencies:** Module 11; usually execute after stronger core evidence exists.

**Steps: external detection option**

1. Obtain Celeb-DF v2 and its official test list.
2. Apply the same declared crop/compression pipeline and record its pre-existing video histories.
3. Evaluate frozen FF++ detection models without external tuning; if adaptation is added, establish a separate external validation split and protocol.
4. Report binary performance and coverage by condition.
5. Do not treat its fakes as automatically matching one of the four exact FF++ source implementations.

**Steps: generator-family attribution option**

1. Select a manageable GenImage generator subset and define exactly which family/checkpoint labels are available.
2. Balance semantic content, resolution, sample count, and processing histories as far as possible; audit duplicate/reused real data.
3. Split by underlying content/provenance where available, including prompts or generation batches as appropriate.
4. Build a separate reference/JPEG attribution experiment with its own source head and chance baseline.
5. Compare the same declared representation families and report remaining content/history confounds.
6. Interpret this as independent generator-family evidence; do not merge its scores into FF++ method-attribution averages.

**Outputs:** external validation table or separate generator-family study. **Done when:** label/task compatibility and external preprocessing histories are transparent.

**Current status:** optional; no external data downloaded or analyzed. It is preferable to omit this module than rush a poorly controlled additional dataset.

## Module 16 — Statistical analysis, shortcut controls, and failure analysis

**Purpose:** determine which findings are reliable and what explains them.

**Prepare beforehand:** frozen main predictions, provenance metadata, primary contrasts, and metric/bootstrap implementations. **Dataset needed:** existing predictions and selected original/reference/degraded examples; no new dataset. **Dependencies:** Module 08 and the experimental modules actually included.

**Steps**

1. Recheck complete paired support for every compared model/condition and document missingness.
2. Produce video-level detection and source metrics separately, including raw-score tables.
3. Compute chance-adjusted retention with predeclared near-chance handling; values can exceed one or fall below zero.
4. Estimate paired differences between detection and attribution retention through provenance-group bootstrap, initially 2,000 draws.
5. Include uncertainty in the reference denominator inside bootstrap calculations.
6. Keep fixed-model sampling uncertainty distinct from seed-to-seed training variation; report both and explain any combined procedure.
7. Address multiple predeclared contrasts and clearly label exploratory analyses.
8. Compare sensitivity of a reference-trained model with matched-quality adaptation. Failure to transfer does not prove that source information was erased.
9. Train background/border and codec-metadata controls; investigate any source predictability from irrelevant cues.
10. Probe embedding quality/codec information, while recognizing that codec predictability alone does not prove label leakage.
11. Compare preprocessing failure/crop drift across sources and qualities and, if available, check identity/content shortcuts.
12. Plot per-source confusion matrices, raw and retained performance, distortion/bitrate curves, and selected error examples.
13. Use embedding visualizations as illustrative evidence, not proof of causal invariance or fingerprint existence.
14. Create an evidence-to-claim table stating which observations support each conclusion and which alternative explanations remain.

**Outputs:** final figures/tables, confidence intervals, controls, error analysis, limitations, and an evidence-backed contribution statement. **Done when:** each conclusion is supported by an appropriate comparison rather than a favored curve or one seed.

**Current status:** design exists; final inferential code, control runs, and real results are pending.

## Module 17 — Prepare the reproducibility package

**Purpose:** make the work checkable by a supervisor, examiner, or reviewer.

**Prepare beforehand:** completed runs, stable code/configuration, data terms, and the final experiment register. **Dataset needed:** manifests/provenance references and permitted examples; original datasets need not be redistributed. **Dependencies:** ongoing from Module 03; finalized after Module 16.

**Steps**

1. Organize scripts for download/inventory, grouping, crop extraction, transforms, feature extraction, training, prediction, and analysis.
2. Give every run a stable ID linking config, code version, seed, checkpoint, logs, predictions, timing, and outputs.
3. Save exact environment/library/encoder/checkpoint versions and hashes.
4. Include data-access instructions and split/group manifests without redistributing restricted media.
5. Provide one small permitted/synthetic smoke example and a clearly labeled scientific reproduction workflow.
6. Add commands that regenerate the paper's tables/figures from saved predictions.
7. Verify a small reproduction in a fresh environment or separate output directory.
8. Check figure labels, source ordering, conditions, thresholds, and sample counts against the manuscript.
9. Preserve original data and raw predictions; identify reproducible intermediates separately for storage management.

**Outputs:** documented code package, run register, environment lock, manifests, and figure/table reproduction instructions. **Done when:** an independent reader can follow a sample through preprocessing, prediction, and the reported metric.

**Current status:** starter code and documentation exist; a complete research release will require the missing modules and real runs.

## Module 18 — Write, revise, and finalize the paper

**Purpose:** present the actual contribution clearly and accurately.

**Prepare beforehand:** university/venue format, verified bibliography, methods log, frozen results, and limitations. **Dataset needed:** existing result artifacts and permitted illustrative examples; no new data collection is normally required. **Dependencies:** begin background/methods early; finish after Modules 16–17.

**Steps**

1. Create the manuscript outline and select the required template. Use an institution/venue's actual formatting and page rules once chosen.
2. Draft the introduction around compressed-media use and the difference between detection and source attribution.
3. Draft related work around compression detection, fingerprint/attribution prior art, quality learning, and dataset biases. Cite the closest predecessors explicitly.
4. State contributions as measured deliverables/results, avoiding unsupported first-ever or universal-fingerprint claims.
5. Write data/preprocessing: dataset access/version, splits, lineage, frame sampling, detector, crop geometry, failure handling, source definitions, and histories.
6. Write experimental methods: representation definitions, pretraining, heads/losses, regimes, compute matching, hyperparameter selection, held-out conditions, and seeds.
7. Write evaluation/statistics: aggregation, threshold selection, chance baselines, retention definition, bootstrap, sample support, and multiple-comparison treatment.
8. Present results in the order of the research questions, rather than the order in which experiments happened.
9. Explain what reference transfer and matched-quality adaptation show separately. Include per-source effects and controls.
10. Write a discussion of robustness/attribution trade-offs and the practical limits of the measured evidence.
11. State limitations: older FF++ methods, prior source compression, dataset/identity/content confounds, selected representations/probes, limited seeds or subsets, and closed-set attribution.
12. Include an ethics/data statement consistent with actual authorized data use; describe the defensive forensic purpose and permitted example sharing.
13. Write the conclusion from the strongest supported findings. Do not equate model failure with proven irrecoverability.
14. Write the abstract last with actual dataset/protocol, main numerical findings, uncertainty where useful, and the bounded contribution.
15. Finalize title/keywords to match completed scope: JPEG-only, video codecs, and generator families must reflect experiments actually performed.
16. Cross-check every numerical claim against generated tables; check bibliography metadata, captions, units, source labels, and train/test wording.
17. Prepare supplementary details: settings, derivations, additional curves, exclusions, per-seed results, and reproduction instructions.
18. Obtain supervisor feedback, answer methodological objections, make necessary corrections, and document any extra experiments as such.
19. Render/compile the final document, inspect figures/tables/references for layout, and archive the final source/export with its exact result package.

**Outputs:** final paper/thesis chapter, supplement, bibliography, and linked reproducibility package. **Done when:** the manuscript's scope, claims, numbers, and reproducibility instructions agree with the evidence.

**Current status:** reviewed related-work material and a corrected bibliography are ready; no scientific results or final manuscript exist yet.

## Suggested figures and tables for the paper

| Artifact | Contents | Produced by |
|---|---|---|
| Figure 1 | Data lineage → fixed crops → compression → features → two tasks → paired analysis | Methods/modules 05–08 |
| Figure 2 | Same reference crop at several qualities, with representative frequency/residual views | Modules 07/10 |
| Figure 3 | Raw detection and attribution performance versus JPEG severity | Modules 11/16 |
| Figure 4 | Chance-adjusted retention with paired uncertainty | Module 16 |
| Figure 5 | Source confusion under selected severe conditions | Module 16 |
| Figure 6, if included | H.264/H.265 performance against achieved distortion or bitrate | Modules 12/16 |
| Figure 7, if included | Single/repeated-history comparison at comparable final distortion | Modules 13/16 |
| Table 1 | Closest prior work and the exact difference of this study | Module 02 |
| Table 2 | Dataset, split/group/video/frame counts and exclusions | Modules 04–06 |
| Table 3 | Representation/model settings and training/evaluation regimes | Modules 10–11 |
| Table 4 | Main raw detection/attribution metrics and seed variability | Modules 11/16 |
| Table 5, if included | Pair/source/augmentation/compute ablations | Module 14 |
| Table 6 | Runtime, memory, storage, and reproducibility settings | Modules 03/17 |

Do not reserve a table for a hoped-for improvement before results exist. Negative or mixed findings can be presented with the same rigorous structure.

## Suggested sixteen-week schedule

This is a planning allocation rather than a guaranteed completion time. Dataset access and measured encoding/training costs may change it.

| Weeks | Main modules | Milestone |
|---|---|---|
| 1–2 | 01–04; begin 08 and manuscript background | Frozen question, environment, access/inventory |
| 3–4 | 05–09 | Audited paired pilot and measured compute/storage budget |
| 5–6 | 10–11, RGB/pretrained and cheap probes first | First trustworthy JPEG results |
| 7–8 | Complete 10–11; initial controls from 16 | Full core representation comparison |
| 9–10 | 12, if included | Aligned H.264 and held-out H.265 evidence |
| 11 | 13, if included | Recompression/history controls |
| 12–13 | 14 | Source-aware pairing and fair ablations |
| 14 | Selected final seeds; 15 only if core is complete | Final experimental evidence frozen |
| 15 | 16–17 | Final statistics/figures and reproduction package |
| 16 | 18 | Revised, checked paper and final archive |

Keep Module 18's background/methods drafts and Module 17's logs active throughout. If time is tight, omit Module 15, reduce complex paper reproductions, and narrow optional extensions before sacrificing leakage checks or statistical validity.

## Immediate next actions

1. Finalize Module 01's core scope and populate the run register using `experiment_matrix.md`.
2. Complete the isolated environment and GPU forward/backward check from Module 03.
3. Obtain FF++ access and official split files for Module 04.
4. Implement video inventory/group mapping and fixed reference crop extraction for Modules 05–06.
5. Extend the evaluation scripts in Module 08 while access/preprocessing is underway.
6. Run the small real-data pilot in Module 09 before expanding the grid.

The current review, protocol, and synthetic starter checks are useful preparation; they do not substitute for these first real-data milestones.
