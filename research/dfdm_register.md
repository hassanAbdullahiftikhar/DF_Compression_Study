# Second-dataset register: DFDM + Celeb-DF (D-study)

**Version 1.0 — frozen 27 October 2026, before any DFDM crop extraction, fitting or scoring.** Its hash is in
`research/dfdm_register_freeze.json`. Sections 1–10 are never edited; changes are appended as dated
amendments in section 11.
Rules follow `comparison_register.md` (the FF++ register) unless stated otherwise.

## 1. Purpose

The FF++ study found that method-attribution cues in trained CNNs survive JPEG and H.265 better than
real/fake cues. The finding is conditional on representation and codec. This study tests it on independent
data where attribution is **finer-grained** and the reference is **truly lossless**:

- **D-RQ1 (primary):** On DFDM, does chance-adjusted attribution retention under additional JPEG
  compression differ from detection retention?
- **D-RQ2:** Does JPEG augmentation help at held-out qualities and at DFDM's native H.264 levels?
- **D-RQ3:** How much of the real/fake separation could come from compression history rather than
  manipulation? This is a control; see section 7.

The questions are descriptive and comparative. No first-ever claims are made, and lower scores are not
read as physical erasure.

## 2. Data, audited 25–27 October 2026

- **DFDM** (Jia, Li & Lyu 2022), from the authors' repository: 430 clips × 5 autoencoder face-swap
  generators (Faceswap baseline, Lightweight, IAE, Dfaker, DFL-H128) × 3 H.264 levels (lossless `qp=0`,
  CRF 10, CRF 23). All 4:4:4. Per-file SHA-256 values are in `experiments/data/audit_dfdm/videos.csv`.
- **Celeb-DF v2,** the official copy from the authors: 590 Celeb-real, 300 YouTube-real, 5,639
  Celeb-synthesis, plus the 518-entry official test list.
- **Real class:** for each DFDM clip `idT_idS_NNNN`, the source video is `Celeb-real/idT_NNNN.mp4`. All 430
  exist, all resolutions match, and frame counts match for 428. In the other two, DFDM ends a few frames
  early.
  - Alignment check: frame *k* of each fake equals frame *k* of its real source. The best temporal offset
    was 0 in every probe, with a background difference of about 0.3/255.
  - Real/fake pairs are therefore content-matched outside the face.
- **Split, by identity:** connected components of the identity graph (target–source edges). The largest
  component goes to training, the middle to test, the smallest to validation. No identity appears in two
  splits.

  | Split | Identities | Clips | Videos per level (real + 5 fakes per clip) |
  |---|---:|---:|---:|
  | Train | 32 | 265 | 1,590 |
  | Validation | 10 | 70 | 420 |
  | Test | 16 | 95 | 570 |

## 3. Reference and compression conditions

- **Fakes:** DFDM files as distributed.
- **Reals:** Celeb-real (MPEG-4 Part 2, 4:2:0) is decoded and re-encoded in memory with DFDM's own x264
  option string, read from the DFDM files (`cabac=0 ref=16 deblock=0:0:0 analyse=0:0 me=dia subme=0 …
  bframes=0 keyint=250 scenecut=0 aq=0`). Re-encoding uses `qp=0` for the reference and CRF 10 / CRF 23 to
  match DFDM's levels, in 4:4:4.
  - So each real and each fake shares its *final* encode.
  - Disclosed difference: the reals carry an earlier MPEG-4 encode, and the fakes' backgrounds also derive
    from it. The x264 build is core 164/165 versus DFDM's core 155.
- **Conditions:**

  | Condition | Definition | Role |
  |---|---|---|
  | ref | Lossless (`qp=0`) videos | Selection and reference |
  | jpeg_q80, jpeg_q40 | Pillow JPEG 4:2:0 on the ref crop | Selection |
  | jpeg_q95, jpeg_q60, jpeg_q20 | Same | **Held out**; Q20 is primary |
  | h264_crf10, h264_crf23 | DFDM's own encodes; reals re-encoded to match | **Held out**; native-codec secondary |

## 4. Crops (no generator-specific geometry)

- **Frames:** indices `rint(linspace(0, n−1, 8))`, where *n* is the DFDM frame count. These indices are
  valid in the real video too.
- **Detection:** YuNet 2023mar with the FF++ settings (score threshold 0.9; largest face, confidence as
  tie-break) runs on the **real** frame, decoded from the original Celeb-real file.
- **Box reuse:** the same square box (margin 1.3) is applied to the real video and all five fakes at
  every level; resize is 224 px Lanczos.
- **Failures:** if YuNet finds no face on the real frame, that frame is dropped for all six videos and
  logged, never replaced.
- **Why:** box geometry cannot differ by generator, which removes the shortcut flagged by the FF++
  metadata probe C3b.

## 5. Models (validation-only selection)

- **D-F1:** ResNet-18 (ImageNet initialization) with two heads: detection (2) and generator (5, fake-only).
  Reference-only training with two identical views and pair weight 0.
- **D-F2:** as D-F1, but the second view is a random Q80 or Q40 JPEG (the augmentation baseline).
- **Shared settings:** seeds 42/43/44; 20 epochs; AdamW 3e-4, weight decay 1e-4; 8 pairs per batch; AMP;
  inverse-count class weights (detection 1 real : 5 fake).
- **Checkpoint:** maximum mean validation (detection BA + attribution BA)/2 over ref/Q80/Q40, at 0.5.
- **Threshold:** maximizes mean validation video detection BA over ref/Q80/Q40. Ties go to the value
  closest to 0.5, then the smaller.
- **D-EB4-F1 / D-EB4-F2:** the same two recipes with an EfficientNet-B4 backbone (torchvision IMAGENET1K_V1,
  1,792-d features, 224 px input), as in FF++ extension X6. They use the identical training, selection and
  threshold rules. Seeds 42/43/44.
- **Probes:** DINOv2-small and CLIP ViT-L/14 with class-weighted ridge heads, using the FF++ F4 procedure.
  Each has a reference-trained head and a Q40 known-quality head (F6 procedure).
- **Verification:** before test, every scorer must reproduce its stored validation predictions.

## 6. Metrics and statistics

- **Balanced accuracy:** chance 0.50 for detection, 0.20 for five-way fake-only attribution.
- **Retention:** S(q) = (BA(q) − b)/(BA(ref) − b), with b = 0.5 or 0.2. Undefined if BA(ref) < b + 0.05.
  Δ(q) = S_attr − S_det.
- **Bootstrap:** 2,000 draws, seed 4242, 95% percentile intervals, paired across models.
  - **Unit:** ordered identity pair (T, S), with all its clips, reals and fakes. Test has 19 groups and
    validation 14.
  - This is coarse, and is disclosed as such.

## 7. Contrasts

- **D-P1 (primary):** Δ(Q20) for D-F1. Direction is claimed only if all three seed intervals exclude zero
  with the same sign. The same rule is applied to D-EB4-F1 as a co-reported replication.
- **Secondary (intervals reported, no stand-alone significance claims):**
  - D-S1: Δ_AUC(Q20). Detection AUROC versus mean one-vs-rest generator AUROC on fakes; chance 0.5 for
    both.
  - D-S2: Δ at h264_crf23 and h264_crf10 for D-F1.
  - D-S3: D-F2 − D-F1 and D-EB4-F2 − D-EB4-F1 detection and attribution BA at Q20, Q60, Q95, CRF 23
    and ref, per seed.
  - D-S4: probe Δ(Q20) and Δ_AUC(Q20), plus Q40-head minus reference-head attribution BA at Q40.
  - D-S5: generator confusion at ref and Q20, which tests whether fine-grained attribution degrades
    first.
- **D-C1, compression-history control:**
  - A detection-only ridge probe on the block-DCT descriptor (FF++ F4 DCT features), trained and
    evaluated at ref.
  - **Material** if its test detection BA lower bound exceeds 0.60. A material flag adds a
    compression-history caveat to all detection results.
- **D-C2, face-masked probe:** the FF++ C3a mask applied to the DINOv2 probe. Content-matched
  backgrounds predict chance; anything above chance flags background leakage.
- **Comparison with FF++:** descriptive only. Retention is compared across datasets, with no pooled
  test.

## 8. Gates

1. **D-G1:** crops extracted; D-F1, D-F2, probes and controls fitted and frozen on validation.
2. **D-G2:** scorers verified against the stored validation predictions.
3. **D-G3:** model-freeze manifest recorded and cited by a dated amendment.
4. **D-G4:** one test pass and one analysis. A rerun is allowed only for a documented bug.

## 9. Linked FF++ amendment (to be appended to `comparison_register.md` before its scoring)

- **E10, external detection test:** the frozen FF++ models (F1–F3, EB4-F1, EB4-F2) score the 518 official
  Celeb-DF v2 test videos.
  - Crops follow the FF++ protocol: YuNet per video, 8 frames.
  - Conditions: ref and JPEG Q95/Q80/Q60/Q40/Q20.
  - Reported: detection AUROC, and BA at the frozen FF++ thresholds.
  - Descriptive. Celeb-DF has a single manipulation method, so there is no attribution.
  - 81 of these test videos are the real sources of DFDM clips. That is irrelevant to E10, because the FF++
    models never saw Celeb-DF, but it is disclosed.

## 10. Known limitations, disclosed in advance

- The reals carry an earlier MPEG-4 encode.
- The x264 build differs from DFDM's.
- The DFDM generators are older autoencoder face-swappers.
- There are only 16 test identities and 19 bootstrap groups.
- Attribution is closed-set among five generators.
- Two clips are truncated by DFDM.

## 11. Amendments

*None yet. Append entries as "Amendment D-N — date — before/after test scoring — change — reason".*

### Amendment D-1 — 27 October 2026 — before crop extraction — shared real sources

- **Finding:** 15 Celeb-real videos are each the source of two DFDM clips (the same target video swapped
  with two different source identities), so the real class has 415 distinct videos, not 430. All 30 of
  these clips are in the training split, and paired clips have equal frame counts.
- **Change:** each real video is included once. Its frame indices and face boxes come from its clip
  (equal frame counts make the choice immaterial), and the same boxes apply to the fakes of both clips.
- **Corrected counts:** distinct real videos are 250 (train), 70 (validation) and 95 (test). Fake videos
  are 5 × clips. Validation and test are unaffected.
- **Reason:** the section 2 table assumed one real video per clip.

### Amendment D-2 — 28 October 2026 — before test scoring — model freeze (gate D-G3)

- **Change:** this amendment adopts `research/dfdm_model_freeze_manifest.json`
  (SHA-256 `4d240c900e2bcd034420e8b186baa70c40f7c9d015c02ba7bad6fa713a335de4`). The manifest lists 18
  models with their artifact hashes and thresholds:
  - 12 RGB models: D-F1, D-F2, D-EB4-F1 and D-EB4-F2, three seeds each;
  - four probes: P-DINO-REF, P-DINO-Q40, P-CLIP-REF and P-CLIP-Q40;
  - two controls: C1-DCT and C2-MASK.

  It also hashes the frozen DFDM code.
- **Gate D-G2:** every model and probe reproduced its stored validation predictions exactly (maximum
  difference 0).
- **Observations disclosed before test (validation only; nothing reselected):**
  - **Very low F1-recipe thresholds.** Under the registered rule, the reference-only models (D-F1,
    D-EB4-F1) have thresholds very close to 0. Under JPEG, fake scores collapse while staying ranked above
    real scores.
  - **Detection breaks under JPEG for F1-recipe models.** Their detection is near-perfect at ref but much
    weaker under JPEG. The JPEG-augmented models stay near-perfect.
  - **Compression-history control.** On validation, C1-DCT (block-DCT statistics) detects at 86.9% BA,
    which suggests detection is strongly confounded by compression history.
  - **Background control.** C2-MASK is at chance, so backgrounds carry no signal.
  - **Frozen-feature attribution.** The DINOv2 and CLIP probes are near chance for attribution even at ref.
- **Reason:** gate D-G3 requires this freeze before the single test pass (D-G4).
