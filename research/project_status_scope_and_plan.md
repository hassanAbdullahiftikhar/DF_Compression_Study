# Project status, proposal scope, and future plan

**Status snapshot:** 28 October 2026, updated after the registered FF++ test evaluation (gates G1–G4, 24 October), the post-freeze extensions X1–X7 (Amendments 2–5, 25 October), and the separately registered DFDM + Celeb-DF study with the external test E10 (28 October). Exact metrics are in the project README, `main_results_test.json`, `main_results_extensions.json`, `main_results_dfdm.json` and `main_results_validation.json`.

## Research question in plain language

The project asks whether compression changes evidence used for two forensic tasks:

1. **Detection:** decide whether a face video is real or manipulated.
2. **Manipulation-method attribution:** for a manipulated video, identify which of four known FaceForensics++ methods produced it: Deepfakes, Face2Face, FaceSwap, or NeuralTextures.

The study tests whether broad clues can remain useful for detecting a fake while finer clues that distinguish manipulation methods are more affected by compression.

## What the original proposal planned

The proposal, *What Survives Compression? A Study of Deepfake Detection and Manipulation/Generator Fingerprint Persistence Under Lossy Media Compression*, outlined a broad experimental program:

- Use FaceForensics++ as the primary dataset, with Celeb-DF as a possible external test and GenImage as a possible generator-attribution extension.
- Compare RGB, Fourier/FFT, DCT, wavelet, noise-residual, and pretrained visual representations.
- Evaluate several JPEG qualities, H.264 at several CRF levels, H.265 as a held-out codec, and repeated recompression.
- Compare clean-to-compressed, matched-quality, mixed-quality, cross-quality, cross-codec, and recompression-history settings.
- Review and potentially reproduce compression-aware methods such as ADD, QAD, DPL, F3-Net, and Perelli et al.; the proposal also discusses restoration, motion, and temporal approaches.
- Explore lightweight compression-paired learning intended to improve robustness across quality changes while retaining attribution information.
- Report detection and attribution metrics, source-level uncertainty intervals, error analysis, and a final reproducible paper.

These were proposed objectives, not a statement that all have been implemented.

## Scope now being studied

The executable study has been narrowed to **FaceForensics++ C23-derived face crops with additional JPEG compression**. The supplied reference videos are already H.264 C23-compressed. The reference condition is therefore not pristine media; current JPEG results describe adding JPEG compression on top of that reference.

Attribution is **closed-set manipulation-method classification** among four known FF++ methods. It is not open-set attribution, identification of an individual generator checkpoint, or proof of a unique physical fingerprint.

The current core compares reference, JPEG augmentation, and compression-paired RGB training, alongside six lightweight reference-trained representation probes. Completed compression analysis and adaptation work is concentrated on reference/Q80/Q40 validation conditions. Whole-video H.264/H.265, repeated compression, external datasets, and reproductions of all proposed published methods are outside the current core; they remain possible extensions.

## Work documented as completed

- Literature review and qualified novelty assessment prepared. The evidence supports a focused comparative study; broad “first-ever” claims are unsupported.
- The FF++ C23 core inventory was audited: 5,000 planned videos across real and four manipulation sources.
- Face crops were extracted and the main cohort assembled: **39,898 valid crops from 4,999 videos**, organized into 500 provenance groups. Failed detections were logged without replacement; one zero-crop video was excluded.
- Nine RGB training runs completed: reference training, JPEG augmentation, and paired training, each with seeds 42, 43, and 44, for 20 epochs per run.
- Six representation probes completed: pooled RGB, FFT, DCT, Haar wavelet, Laplacian residual, and frozen DINOv2-small.
- Validation analyses, frozen operating thresholds, compression measurements, label-shuffle diagnostics, and 2,000 provenance-group bootstrap draws were recorded.
- Six Q40 and six Q80 known-quality adaptation probes completed (validation only; Q80 run on 23 October 2026). At Q80, matched-quality heads partly restored the reference-trained DCT probe, which had fallen close to chance; other representations changed by about two percentage points or less.
- The RTX 5070, Ryzen 7 7700, and nominal 16 GB RAM system completed the documented runs. Each RGB run took about 49–56 minutes, with about 500 MiB peak reserved PyTorch GPU memory. System RAM peaks were not measured.

Main RGB Q40 validation summary, mean ± sample standard deviation across three seeds:

| Training approach | Detection balanced accuracy | Fake-only attribution balanced accuracy |
|---|---:|---:|
| Reference training | 76.72 ± 5.73% | 91.54 ± 2.50% |
| JPEG augmentation | 92.28 ± 1.34% | 97.37 ± 0.10% |
| Paired training, weight 1 | 92.64 ± 0.45% | 97.49 ± 0.31% |

These are validation results, not final test results or confidence intervals. JPEG augmentation improved Q40 validation scores over reference training. The paired method's extra gain over augmentation is small, and the existing paired-versus-augmentation intervals include zero. DINOv2 had the highest absolute Q40 performance among the reference-trained probes. Those probes vary in representation and classifier capacity, so they do not establish a universal ranking.

### Registered held-out test evaluation (24 October 2026)

Steps, in register order:
- **G1:** the registered controls C1–C3 were completed on validation.
- **G2:** test scorers reproduced the stored validation predictions of all 49 registered models.
- **G3:** a model-freeze manifest and register Amendment 1 were recorded.
- **G4:** each model was scored once on the 700 test videos at reference and Q95/Q80/Q60/Q40/Q20, and the registered analysis ran once.

Registered readings (details and intervals in the README and `main_results_test.json`):

- **Primary contrast, reference-trained RGB (P1-RGB):** at Q20, chance-adjusted attribution retention exceeds detection retention in all three seeds. Δ(Q20) is 0.29, 0.35 and 0.49, and every interval excludes zero. This is the reverse of the proposal's expectation that method cues would be more fragile. Detection is measured at the frozen validation threshold. Exploratory AUROC shows that much of the detection loss is a score shift past that threshold, so the gap is smaller without a fixed operating point.
- **Primary contrast, reference-trained probes (P1-PROBE):** no probe contrast survives Holm correction. Several probes are close to chance even on reference test crops, and FFT's contrast is undefined.
- **RQ3 (S2):** JPEG augmentation improves strongly over reference training at Q20. Paired training shows **no reliable benefit over augmentation**.
- **Controls:**
  - Matched-quality heads recover part of the reference-trained losses, most clearly for DCT (S3).
  - Separate single-task models strengthen the P1-RGB direction (S4).
  - Removing source supervision lowers both readout attribution and reference detection (S5).
  - Face-masked probes are not material.
  - The **metadata/geometry probe is material for attribution** (35.4%, 95% interval 33.0–37.9%), so attribution findings carry a dataset-level caveat (S6).
  - Complete-eight supports change no interval's exclusion of zero (S7).

### Post-freeze extensions (25 October 2026; Amendments 2–3)

These were designed after the registered results were seen and are labelled as extensions. Each was
pre-registered in Amendment 2, validation-selected, frozen in Amendment 3 and scored once. Details are in
the README and `main_results_extensions.json`.

- **Operating point (X1):** the primary direction survives per-quality detection thresholds and a
  like-for-like AUROC comparison. At Q20, attribution AUROC stays about 94–96% while detection AUROC falls
  to about 73–80%.
  - Correction: an earlier exploratory remark compared detection AUROC retention with attribution BA
    retention. Those metrics are not comparable, so that remark overstated how much the gap shrinks.
- **Frozen-encoder Q20 readout (X2):** a linear readout on the frozen reference-trained encoder recovers
  15–27 points of detection BA and 19–23 points of attribution BA at Q20. The information is still linearly
  available in the encoder.
- **Seed-pooled intervals (X3):** F1 Δ(Q20) is 0.38 [0.28, 0.48]. Paired minus augmentation stays null.
- **Video codecs (X5):** H.265 repeats the JPEG pattern. **H.264 at CRF 40 does not:** attribution falls as
  far as detection there, and the direction is not established. JPEG augmentation does not improve H.264
  detection.
- **EfficientNet-B4 (X6):** replicates the primary direction and the augmentation gains.
- **Metadata cue (X4):** no sign that crop models rely on it at the reference condition.
- **CLIP ViT-L/14 probes (X7, Amendments 4–5):**
  - CLIP is the strongest frozen probe at reference: 80.9% detection BA and 83.4% attribution BA.
  - At its frozen operating point the sign reverses. Argmax attribution collapses at Q20 (42.1%), giving
    Δ(Q20) = −0.22 [−0.35, −0.09].
  - Threshold-free, attribution is still retained slightly better: Δ_AUC(Q20) = 0.07 [0.01, 0.14].
  - Matched-quality heads recover 10–16 points of attribution.
- **Across representations,** the threshold-free contrast is:
  - positive for the trained CNNs, DINOv2 and CLIP;
  - near zero for the pooled-RGB, wavelet and residual descriptors;
  - negative for FFT and DCT.

  **The defensible headline is therefore representation- and codec-specific, not universal.**

### Second dataset: DFDM + Celeb-DF, and external test E10 (28 October 2026)

- **Register:** `dfdm_register.md` v1.0 was frozen before any crop extraction. Amendment D-1 handles shared
  real sources; Amendment D-2 is the model freeze.
- **Data:** DFDM provides five related autoencoder face-swap generators with a lossless reference. Real
  videos come from Celeb-DF v2 (official copy) and are the exact source clips, re-encoded with DFDM's x264
  settings. Face boxes are shared, so backgrounds are pixel-identical.
- **Split:** identity-disjoint.
- **Models:** ResNet-18 and EfficientNet-B4 (F1 and F2 recipes, 3 seeds each), DINOv2 and CLIP probes, and
  two controls. All 18 models reproduced their validation predictions exactly before one test pass.
- **Fine-grained attribution collapses first:**
  - Five-way attribution (chance 20%) falls from about 47–54% at ref to chance by JPEG Q60.
  - The registered contrast is "direction not established" for ResNet-18 and "attribution retained less
    than detection" for EfficientNet-B4.
  - Threshold-free Δ_AUC(Q20) is negative in all six reference-trained models.
- **Detection is confounded by compression history:** the C1-DCT control is material, at 84.0%
  [80.6, 87.9] detection BA from block-DCT statistics. The face-masked control is at chance.
- **External E10** (FF++ register Amendment 6): frozen FF++ detectors reach about 63–72% AUROC on the
  518-video Celeb-DF v2 test set. JPEG augmentation improves transfer.
- **Synthesis:** compression retention of method evidence depends on attribution granularity. Coarse FF++
  method differences survive; fine-grained distinctions collapse first, both DFDM's generators and FF++'s
  Face2Face vs. NeuralTextures.
- **Disclosed process deviation:** two copies of one EfficientNet run overlapped for under five minutes.
  Both were stopped before any epoch, and the run was restarted alone.

## What changed and why

The original proposal combined still-image and video compression, two codecs, processing histories, external datasets, reproduction of several published architectures, and a new paired-learning method. The implemented scope prioritizes a controlled, reproducible study that fits the available dataset and hardware, with limited storage and manageable training time.

| Original proposal | Current core scope |
|---|---|
| Pristine media and multiple compression paths | Existing H.264 C23 reference videos, then additional JPEG compression |
| JPEG, H.264, H.265, and repeated recompression | JPEG is the completed core; video codecs and recompression remain extensions |
| Possible Celeb-DF and GenImage studies | FF++ C23 only in the documented core |
| Broad manipulation/generator fingerprint language | Four-way closed-set attribution to FF++ manipulation methods |
| Reproduce multiple state-of-the-art architectures | Controlled RGB training variants and lightweight representation probes; cited methods mainly provide context and future comparison options |
| Develop a new compression-paired learning strategy | Evaluate one lightweight paired-training candidate against JPEG augmentation |

This is a deliberate refinement, not completion of every item in the initial proposal. The strongest defensible contribution is a bounded empirical comparison of detection and known-method attribution across representations and additional JPEG compression, with validation evidence about compression-aware training.

## Future plan

1. **Finish declared validation controls:** the complete-eight-frame validation sensitivity is done; its support definitions are frozen for test. The [comparison register](comparison_register.md) v1.0 was frozen on 23 October 2026, before any test scoring. It keeps three controls: C1 task sharing (separate binary and fake-only RGB models), C2 source supervision (binary-only paired model with a train-only source readout), and C3 background/border, metadata/geometry and content analyses. These must be trained and frozen on validation (register gates G1–G3) before test scoring. **Gate G1 was completed on 24 October 2026.** The nine control RGB runs, the C2 readout and the C3a–C3c analyses are recorded on validation in the project README and the aggregate milestone. Gates G2 (scorer verification) and G3 (model-freeze amendment) were completed on 24 October 2026.
2. **Evaluate frozen models:** completed 24 October 2026 (gates G2–G4). The held-out test cohort was scored once at reference and Q95/Q80/Q60/Q40/Q20. These results must not be used to select models, thresholds or settings, and the pass may be rerun only to fix a documented bug.
3. **Statistical and error analysis:** the registered analysis is complete: per-seed results, provenance-group intervals, support sensitivity, confusion matrices, coverage and distortion. Any further error analysis is exploratory and must be labelled as such.
4. **Processing-history experiments:** outside the frozen core (register section 10). If they are added later, a register amendment must come before their test scoring, and they must compare repeated compression with a single-pass condition at matched final distortion.
5. **Write and revise the paper:** refresh the literature review, verify each citation, create final figures and tables, explain the revised scope, and report uncertain or negative findings accurately.

Whole-video H.264/H.265, Celeb-DF, and GenImage are optional extensions. Add them only if they answer a necessary research question without delaying the held-out evaluation and controls.

## Interpretation limits

- The reference is already C23-compressed; describe the condition as **additional JPEG compression of C23-derived crops**.
- Validation metrics are selection-stage evidence. Test metrics come from one registered pass; their intervals condition on the selected models and thresholds and omit selection and training-seed uncertainty.
- The P1-RGB reading concerns detection at a fixed validation operating point.
- Four-way attribution identifies one of four known FF++ manipulation methods, not an unknown source or specific generator checkpoint.
- The cohort is one benchmark mirror. Provenance-group separation does not prove identity-level or YouTube-video independence, and the mirror's official-byte authenticity has not been established.
- A lower classifier score does not prove forensic information was physically or irreversibly erased.
- Paired training showed no reliable gain over JPEG augmentation on test (registered S2 reading).
- A video-level metadata/geometry cue predicts the manipulation method above chance on test, which is a dataset-level caveat for attribution.

## Related project records

- [Project README and results](../README.md)
- [Module tracker](module_tracker.md)
- [Experiment register](experiment_matrix.md)
- [Main cohort inclusion specification](main_cohort_specification.md)
- [Literature and novelty assessment](literature_review.md)
- [Aggregate validation results](main_results_validation.json)
- [Registered test results](main_results_test.json) and [model-freeze manifest](model_freeze_manifest.json)
