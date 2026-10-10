# Literature review, citation audit, and novelty assessment

**Project:** What Survives Compression? A Study of Deepfake Detection and Manipulation/Generator Fingerprint Persistence Under Lossy Media Compression.

**Review date:** 10 October 2026. **Input:** the supplied `Mgr.md`. **Verdict:** feasible as a carefully bounded undergraduate research project; the broad novelty rationale is incomplete and must be revised. The strongest contribution is a controlled empirical study, with an optional learning extension after the evidence is established.

## 1. Review method and strength of evidence

I searched the exact titles of all twelve references, checked author/publisher or conference records, and read accessible primary paper text. I then searched outside the proposal for generator fingerprints, manipulation attribution, compression robustness, detection-versus-attribution comparisons, and JPEG dataset biases. This is a targeted critical literature review, not an exhaustive systematic review of every database. No Scopus/Web of Science census, citation-count analysis, or guarantee of worldwide priority is claimed.

Paper summaries distinguish reported findings from implications for this project. Full text was accessible for [1–4], [6–7], [10–12] and several missing prior-art papers. BZNet [5] was checked through its authors' university publication record and abstract; its complete method and tables were not independently inspected. For [8–9], the publisher abstract/highlights and indexed article preview were accessible, but complete experimental details were not. Those three summaries therefore support conceptual comparison, not an exact reproduction recipe. Some CVF PDF fetches failed; author arXiv full text was used where available.

The proposal's rendered Markdown has missing equations and compression values. These cannot be recovered reliably from the supplied file. The companion protocol supplies explicit recommended values and equations and labels them as new design choices.

## 2. Citation audit

| Ref. | Verified identity and publication | Correction or qualification |
|---|---|---|
| [1] | Andreas Rössler et al.; FaceForensics++; ICCV 2019 | Title/year are valid. Clean up corrupted author accents. |
| [2] | Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, Jing Shao; F³-Net; ECCV 2020 | The transform is DCT; a generic FFT input is not a reproduction of F³-Net. |
| [3] | Juan Hu, Xin Liao, Wei Wang, Zheng Qin; TCSVT 32(3), 1089–1102, March 2022 | 2021 online/early-access date explains the year discrepancy; 2022 issue citation is valid. DOI: 10.1109/TCSVT.2021.3074259. |
| [4] | **Binh M. Le**, Simon S. Woo; ADD; AAAI 2022 | Replace “T.-N. Le.” arXiv:2112.03553. |
| [5] | Sangyup Lee, **Jaeju An**, Simon S. Woo; BZNet; WWW 2022, 3500–3510 | Replace “S. An.” DOI: 10.1145/3485447.3512245. |
| [6] | **Xin Liao**, Yumei Wang, Tianyi Wang, Juan Hu, Xiaoshuai Wu; FAMM; TCSVT 2023 | Replace “Y. Liao.” DOI: 10.1109/TCSVT.2023.3278310. |
| [7] | **Binh M. Le**, Simon S. Woo; QAD; ICCV 2023 | Replace “T.-N. Le.” arXiv:2309.05911. |
| [8] | Ying Li, Shan Bian, Chuntao Wang, Kemal Polat, Adi Alhudhaif, Fayadh Alenezi; ESWA 2023 | The paper exists; expand the incomplete citation using the publisher's export. |
| [9] | **Yizhe Zhu**, Chunhui Zhang, Jialin Gao, Xin Sun, Zihan Rui, Xi Zhou; Neurocomputing 565, 126872, 2024 | Replace “Z. Zhu.” DOI: 10.1016/j.neucom.2023.126872. The DOI's 2023 does not invalidate its 2024 issue date. |
| [10] | **Dongliang Zhang**, Yunfei Li, Jiaran Zhou, Yuezun Li; DPL; ACCV 2024 | Replace “Z. Zhang.” arXiv:2410.07633. |
| [11] | **Zongmei Chen**, Xin Liao, Xiaoshuai Wu, Yanxiang Chen; APSIPA ASC 2024 | Replace “Y. Chen.” arXiv:2404.18149. |
| [12] | **Gianpaolo Perelli**, Marco Micheletto, Sara Concas, Giovanni Puglisi, Gian Luca Marcialis; ESWA 317, 131761, 2026 | Replace “A. Perelli.” Genuine 2026 article, available online 20 February 2026. DOI: 10.1016/j.eswa.2026.131761. |

Primary links and corrected names are supplied in the summaries below. These errors are bibliographic rather than evidence that the twelve papers are fabricated.

## 3. Critical review of the twelve cited papers

### [1] Rössler et al. — FaceForensics++ (2019)

FF++ establishes a controlled facial manipulation benchmark with 1,000 original sequences and four manipulation methods: Deepfakes, Face2Face, FaceSwap, and NeuralTextures. Its raw, c23, and c40 versions make compression comparisons possible; the standard split is 720/140/140. This directly supports the primary experiment. The official repository documents access conditions, additional FaceShifter data, and dataset terms. [Paper](https://arxiv.org/html/1901.08971), [official repository](https://github.com/ondyari/FaceForensics).

**Critical implication:** four-way classification demonstrates manipulation-pipeline attribution within FF++, not identification of a particular trained generator checkpoint. YouTube-derived originals also have prior processing history: “raw” means the dataset's least additionally compressed reference, not sensor-raw imagery. Source pairing is useful for holding content approximately constant, but it requires careful lineage grouping. A result on these older methods does not establish robustness for current diffusion or talking-head generators. Use FF++ as the controlled main study and reserve modern data for a separate extension.

### [2] Qian et al. — Thinking in Frequency / F³-Net (2020)

F³-Net combines frequency-aware decomposed components with local frequency statistics through a two-stream architecture. It uses DCT and evaluates face forgery detection across FF++ compression qualities. The paper reports stronger low-quality performance than its tested competing detectors. [Full text](https://arxiv.org/html/2007.09355).

**Critical implication:** frequency cues can be useful even after compression, so “high frequencies are removed, therefore frequency models must fail first” is not a defensible assumption. A codec may suppress generation traces while producing other discriminative patterns. The proposed generic FFT/DCT branches should be called representation baselines; they do not reproduce F³-Net's learned decomposition and local statistics. Include a global spectrum and a localized transform so that frequency-domain performance is not conflated with loss of spatial localization. Its binary evaluation does not establish manipulation-source persistence. If reproducing it, maintain its architecture in a separate baseline track and adapt an attribution head transparently.

### [3] Hu et al. — Frame-temporality two-stream network (2022)

This work combines frame-level and temporality-level evidence for compressed deepfake video detection. Its central argument is that spatial evidence and temporal relationships complement each other when video compression weakens local cues. The authors' accessible issue PDF confirms the March 2022 publication and exact authors. [Author-hosted full text](https://msf-hnu.github.io/resource/IEEE%20TCSVT-Detecting%20compressed%20Deepfake%20video%20in%20social%20networks%20using%20frame-temporality%20two-stream%20convolutional%20network.pdf).

**Critical implication:** the proposal's eight uniformly sampled frames are adequate for a frame classifier but do not faithfully test short-term motion or temporal consistency. A reproduction needs contiguous clips and the original temporal sampling protocol. Keep this paper as a temporal robustness comparator or an optional second-stage baseline. Its contribution is binary compressed-video detection, not a common-representation comparison of binary and source labels. A still-image JPEG study should not claim to test this mechanism. Also distinguish social-platform uploads from locally applied codecs; those are different experimental environments.

### [4] Le and Woo — ADD (2022)

ADD transfers high-quality teacher information to a low-quality student using frequency attention and multi-view distillation, with the latter informed by sliced Wasserstein distance. The paper tests multiple manipulation subsets and backbones. Its NeuralTextures c40 ablation raises reported accuracy from 60.27% for the baseline to 68.53% for the combined distillation method; this is a specific binary setting, not an attribution score. [Full text](https://arxiv.org/html/2112.03553), [author-linked code](https://github.com/Leminhbinh0209/ADD).

**Critical implication:** clean/compressed pairing and cross-quality representation transfer already exist. The proposed pair loss is therefore not independently novel. Distillation improves decision-relevant representations; it does not prove that the original erased pixels or physical fingerprint have been recovered. A fair project comparison should ask whether binary-oriented distillation also retains source discriminability. Freeze the teacher, measure memory, and prefer a small backbone initially. A locally adapted ADD variant needs its own reported protocol rather than comparison against unrelated published leaderboard numbers.

### [5] Lee, An, and Woo — BZNet (2022)

The authors' university record describes a multi-scale branch zooming network using unsupervised super-resolution and training only on highly compressed low-quality inputs, without readily available high-quality training data. It reports evaluation on FF++ low-quality, GAN-generated data, and an internet-collected deepfake set. The record reports average CNN detection improvement of 4.21% and a separate in-the-wild improvement of 4.13%; these are the abstract's reported figures, not independently reproduced results. [University publication record](https://pure.skku.edu/en/publications/bznet-unsupervised-multi-scale-branch-zooming-network-for-detecti/).

**Critical implication:** its availability assumption differs from ADD, so they are not interchangeable baselines. Restoration may help classify a fake while adding learned textures unrelated to its true source. Treat fingerprint fidelity as a testable concern, not an established failure of BZNet. Full-text and implementation audit remain necessary before assigning exact memory or reproduction settings. On the stated hardware, multi-branch restoration is a lower-priority experiment than the controlled study.

### [6] Liao et al. — FAMM (2023)

FAMM constructs geometric facial muscle-motion features from landmarks and evaluates compressed-video detection. The full text uses consecutive frames, face detection/alignment, and geometric measurements across sensory and facial regions, with temporal classifiers including GRU/SVM. This provides a complementary route to texture-based evidence. [Author-hosted full text](https://msf-hnu.github.io/resource/IEEE%20TCSVT-FAMM%20Facial%20muscle%20motions%20for%20detecting%20compressed%20Deepfake%20videos%20over%20social%20networks.pdf).

**Critical implication:** robust landmark trajectories could support detection without uniquely distinguishing the manipulation pipeline. That is a hypothesis for this project, not a result supplied by FAMM. Landmark failure under strong compression can itself alter classification; log extraction success separately by source and condition. A temporal ablation must sample contiguous sequences, not eight unrelated frames. Facial alignment introduces resampling and should not silently differ across compression levels. This is a useful optional contrast to residual/frequency models, but including it makes the project substantially broader.

### [7] Le and Woo — QAD (2023)

QAD learns from corresponding quality variants using one model. Its intra-model collaborative learning increases dependence between intermediate representations through HSIC, combined with adversarial weight perturbation. It evaluates video qualities and random JPEG degradation, using ResNet-50/EfficientNet-B1 configurations. [Full text](https://arxiv.org/html/2309.05911).

**Critical implication:** this is a close predecessor to compression-paired representation learning. Its dependence objective is more nuanced than forcing every representation to be identical; describing all prior work as aggressively eliminating quality-sensitive information would misrepresent it. The proposed extension must show whether an explicit source objective improves attribution beyond quality pairing, mixed-quality augmentation, and a common multi-task backbone. HSIC on batch representations and adversarial perturbation add implementation and training costs. Use an ordinary paired cosine loss as a transparent small baseline, then reproduce QAD only if time allows. A reduced QAD-inspired implementation is not an exact QAD reproduction.

### [8] Li et al. — Spatial Restored Detection Framework (2023)

The publisher describes restoration and enhancement of weakened spatial texture features in low-quality social-network-style deepfake video. This places the work in the restoration family alongside BZNet, rather than source attribution. [Publisher record/preview](https://www.sciencedirect.com/science/article/pii/S095741742301148X).

**Critical implication:** restoration changes the evidence presented to the classifier. Improvement in binary classification does not demonstrate correspondence with the uncompressed source signature. A useful extension would compare attribution before and after restoration using exactly the same examples, while tracking clean-domain performance and restoration artifacts. However, the accessible abstract/preview is insufficient to verify the proposal's exact dataset details, training budget, and reproduction feasibility. Do not populate a comparison table with unsupported implementation settings or numerical claims. Keep this as literature context unless complete text and a working implementation become available. This limitation concerns review access, not the existence or quality of the paper.

### [9] Zhu et al. — Contrastive spatiotemporal distillation (2024)

The publisher's abstract identifies a two-stage spatiotemporal encoder, spatial-frequency distillation, and temporal contrastive transfer from high-quality teacher cues to heavily compressed video. The publication is Neurocomputing 565, article 126872. [Publisher abstract and highlights](https://www.sciencedirect.com/science/article/pii/S0925231223009955).

**Critical implication:** pairing, contrastive learning, frequency cues, and quality transfer are already combined in the compression literature. A source-aware extension must demonstrate a specific benefit rather than claim novelty from this ingredient list. Temporal encoding plus teacher/student paths is likely more expensive than a frame-level ResNet, but exact memory cannot be inferred from the abstract alone. Review the complete method before reproduction. This paper supports the rationale for an optional temporal comparator; it does not validate the claim that manipulation attribution degrades faster or establish that all fingerprints disappear under a given codec. No exact dataset-performance table was independently verified here.

### [10] Zhang et al. — DPL (2024)

DPL uses two progressive branches guided by visual quality and forgery identifiability. A CLIP-based quality indicator and feature selection inform sequential processing with shared weights. The paper's implementation uses ConvNeXt-T, 224×224 inputs, Adam at 5e-5, batch 32, and ten epochs split between two stages, with random JPEG augmentation. [Full text](https://arxiv.org/html/2410.07633).

**Critical implication:** quality-aware dynamic processing is established; the project should compare against simple augmentation before adding complexity. These reported settings are a reproduction reference, not evidence that batch 32 fits a 12 GB RTX 5070. Measure the combined quality estimator, encoder, and backward pass. DPL's cross-quality and cross-dataset objectives overlap RQ3's broad motivation, but its central task remains detection. An attribution adaptation must disclose changed outputs and losses. A verified official repository was not located in this review; do not use unrelated code links appearing in aggregated search snippets.

### [11] Chen et al. — 3D spatiotemporal trajectories (2024)

This work builds facial motion features from a 3D model, integrating 2D/3D information and separating expression from head movement. It analyzes phase-space trajectories and checks landmark distributions before and after compression. It therefore supplies evidence that useful non-texture cues can be relatively stable. [Full text](https://arxiv.org/html/2404.18149).

**Critical implication:** this is a different representation family and a different preprocessing workload from RGB/frequency crops. Stability of landmark estimates does not imply stability of source identity information. Contiguous sequences, tracking, reconstruction failures, and pose need to be measured if included. A small number of unrelated frames cannot reproduce trajectory analysis. The paper is relevant to the explanation of detection survival, but it should not become a mandatory baseline for the initial six-representation study. Its exact generator-source attribution performance is not established by the cited detection experiments.

### [12] Perelli et al. — Scalable network strategies (2026)

The paper combines video recompression augmentation with Multi-Head Networks predicting detection and compression level; a Multi-Branch Network aggregates such models. It evaluates FF++ and Celeb-DF and reports that lightweight augmented networks can improve compressed detection. Its full text describes offline whole-video re-encoding before face preprocessing. [Open-access author repository PDF](https://iris.unica.it/retrieve/afa7e27d-f046-4915-b1a3-bf8a01a073ee/1-s2.0-S0957417426006743-main.pdf), [publisher](https://www.sciencedirect.com/science/article/pii/S0957417426006743).

**Critical implication:** multiple heads and compression-aware supervision are established. A detection-plus-source head is a sensible baseline, but this structural change alone is weak novelty. Its preprocessing approach reinforces the need to preserve video codec temporal context. A lightweight detection/compression auxiliary head is practical to adapt on the stated GPU; the full multi-branch model should be benchmarked separately. Its reported detection improvements do not resolve whether source-discriminative evidence is retained. Match sample count, encoder, and augmentation when testing an auxiliary compression head against source-aware learning.

## 4. Missing prior art that materially changes the novelty assessment

### Yu, Davis, and Fritz — Attributing Fake Images to GANs (ICCV 2019)

This is the most direct correction to a broad claim that source fingerprints under compression are unexplored. It learns GAN fingerprints for real/fake classification and model attribution, including fine-grained differences between training instances. Section 4.4 explicitly attacks attribution using JPEG quality drawn from 10–75, noise, blur, cropping, relighting, and combinations. The clean-trained attribution classifier deteriorates under perturbations; finetuning recovers some robustness. [Full text, especially §4.4](https://arxiv.org/html/1811.08180).

**What remains different:** the proposed study would use controlled facial manipulation pipelines, a broader common representation panel, whole-video codecs, and paired task survival estimates. JPEG-sensitive attribution and robustness finetuning themselves are already prior art. Architectural or checkpoint attribution in this paper should not be collapsed into FF++ manipulation-method labels.

### Marra et al. — Do GANs leave artificial fingerprints? (2019)

Marra, Gragnaniello, Verdoliva, and Poggi study source-specific fingerprints in generated imagery, analogizing them to camera fingerprints and demonstrating source identification across GANs. This is foundational passive attribution prior art. [Author preprint](https://arxiv.org/abs/1812.11842).

**What remains different:** the current proposal's potential contribution is the degradation protocol and task comparison, not introducing source attribution or the existence of generator signatures. Its residual baseline should explicitly acknowledge this tradition. Residual or spectral patterns need controls for content and image processing before they can be described as generator fingerprints.

### Frank et al. — Leveraging Frequency Analysis for Deep Fake Image Recognition (ICML 2020)

Frank et al. use DCT for detection and source identification, investigate architecture/upsampling-related artifacts, and test perturbations including JPEG quality sampled from 10–75. They also evaluate augmentation with perturbed training data. [Full text, §4.3–4.5](https://arxiv.org/html/2003.08685), [official code](https://github.com/RUB-SysSec/GANDCTAnalysis).

**What remains different:** a controlled comparison across six representation families, two tasks, video codecs, and recompression history could extend this work. Merely comparing RGB versus DCT attribution under JPEG or adding compression augmentation does not. The GAN experiments should not be assumed to apply unchanged to modern diffusion models or face manipulation pipelines.

### Baxavanakis, Schinas, and Papadopoulos — Do DeepFake Attribution Models Generalize? (MAD 2025)

This close predecessor compares binary and multi-class manipulation models, evaluates shared manipulation types across datasets, and tests Triplet, NT-Xent, and supervised contrastive learning. FF++ c23 is used in its main setup; its principal axis is dataset shift rather than a controlled codec/severity sweep. It reports stronger binary generalization and model-dependent contrastive gains. [Full text](https://arxiv.org/html/2505.21520), [publisher](https://doi.org/10.1145/3733567.3735572).

**What remains different:** separate compression from dataset shift, keep each media example paired across conditions, compare chance-adjusted task degradation, and control crop geometry and acquisition history. The general proposition that attribution is harder than detection and may benefit from contrastive learning is insufficient novelty. This paper should be explicitly discussed in the revised introduction and related work.

### Open-world attribution and recent overlapping work

The authors' repository for **Contrastive Pseudo Learning for Open-World DeepFake Attribution** (ICCV 2023) and **Rethinking Open-World Deepfake Attribution with Multi-perspective Sensory Learning** (IJCV 2025) demonstrates established work on unknown sources. The present closed-set FF++ experiment does not solve open-world attribution. [Official repository](https://github.com/TencentYoutuResearch/OpenWorld-DeepFakeAttribution).

**What if Retrieval Could Work Before Decoding? The case of JPEG AI Latents for Deepfake Source Attribution** (DFF 2025) works with compressed learned latents and retrieval. JPEG AI is distinct from ordinary JPEG/H.264, but it further invalidates a sweeping claim that compression and attribution have not been connected. [Paper](https://openreview.net/pdf/db403d5e7459dda89faf478eb4ebd7d1dc91a28c.pdf).

A 2026 primary article, **A Unified Multi-Task Vision Transformer for AI-Generated Image Detection and Generator Attribution**, also describes a shared detection/attribution model and robustness assessment with JPEG/resizing. Its evidence should receive less weight than the established conference studies until independently reproduced, but it is another reason not to claim novelty from two heads or a robustness test alone. [Publisher paper](https://www.ijraset.com/best-journal/a-unified-multitask-vision-transformer-for-ai-generated-image-detection-and-generator-attribution).

### Dataset shortcuts: Fake or JPEG? (2024), and bias-aware detection (2025)

**Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets** identifies JPEG and image-size biases, including GenImage, and explains why applying random JPEG does not necessarily equalize pre-existing compression histories. [Full text](https://arxiv.org/html/2403.17608). This is central to validity: a codec-sensitive feature could predict source acquisition or file history instead of generation method.

**A Bias-Free Training Paradigm for More General AI-generated Image Detection** explicitly evaluates format/content controls and a modified GenImage subset with reduced format mismatch. [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf). A GenImage extension needs generator-balanced semantics, sizes, and histories, rather than assuming its labels automatically isolate generator signatures. Applying a common PNG output does not erase prior JPEG artifacts.

## 5. Synthesis: what the literature supports

The reviewed work supplies four established strategies: spatial/frequency cues, high-to-low-quality transfer, quality-conditioned or quality-generalized learning, and temporal/geometric evidence. Their design goals differ. An ADD teacher assumes high-quality supervision; BZNet addresses missing high-quality training inputs; QAD uses corresponding qualities in a shared model; DPL uses progressive quality-guided processing; Perelli et al. explicitly supervise compression. A meaningful experimental comparison needs to specify these information and compute differences.

The proposal correctly recognizes that detectability and source discriminability are different. However, the twelve selected papers disproportionately cover detection, creating an apparent gap that partly disappears when attribution literature is included. The missing papers establish source signatures, compression-sensitive attribution, direct detection/attribution comparison, and contrastive attribution training. A literature review based only on the proposal's twelve references would overstate originality.

There is nevertheless a worthwhile empirical question: **under a controlled degradation intervention, which accessible representation retains each kind of discriminative signal, and how much does source supervision alter that trade-off?** This statement is narrower, falsifiable, and hardware-compatible. It does not presuppose a frequency ordering, that attribution fails first, or that invariance necessarily damages attribution.

“Fingerprint persistence” also needs operational discipline. A trained classifier succeeds on distributions, not on a physical trace measured directly. Strong evidence would combine probe accuracy, controlled representation ablations, content/background checks, processing-history balance, and transferable behavior. Failure of one model is evidence of that model's limitations, not proof that no recoverable information exists. Calling the surviving information “source-discriminative cues” is safer until stronger fingerprint evidence is available.

## 6. Novelty verdict by claim

| Proposed claim | Assessment | Evidence or required change |
|---|---|---|
| Compression hurts deepfake detection | Established | Already central to much of the cited work. |
| Source fingerprints and attribution exist | Established | Marra; Yu; Frank. |
| JPEG affects attribution | Established | Yu §4.4; Frank §4.5. |
| Compare binary and attribution models | Established | Baxavanakis et al. 2025. |
| Paired quality representations / contrastive alignment | Established ingredients | ADD, QAD, CSTD, contrastive attribution work. |
| Two-head detection and source prediction | Weak novelty alone | Standard multi-task design; overlapping 2026 article. |
| Chance-normalized retention index | Analysis choice, not a standalone contribution | A normalized skill score; validate interpretability and denominators. |
| Unified six-representation, paired task-survival study spanning JPEG, H.264/H.265, and recompression | **Plausibly distinct combination** | No exact match identified in this targeted search; controlled design and meaningful findings still required. |
| Source-aware pairing improves held-out compression while preserving clean attribution | **Potential empirical contribution** | Must beat matched augmentation/multi-task baselines with uncertainty and compute controls. |
| First such study / inherently novel algorithm | **Not supported** | Avoid priority language unless an expanded systematic search establishes it. |

## 7. Recommended revised research claim

> We evaluate the relative retention of binary detection and closed-set manipulation-method attribution under controlled compression histories. Using a common data lineage, fixed crop geometry, multiple representation families, and paired video-level uncertainty estimates, we distinguish degradation sensitivity from adaptation to known quality. We then test whether source-supervised quality pairing improves attribution under held-out compression compared with augmentation and ordinary multi-task learning.

This supports an empirical/benchmark contribution with an optional modest method extension. It is a credible BS research direction; publication strength will depend on methodological care, the size and reliability of the observed effects, and relevance beyond FF++.

## 8. Reading and implementation priorities

Read Yu §4.4, Frank §4.3–4.5, Baxavanakis et al., and Fake or JPEG? before revising the gap. Then read FF++ splitting/preprocessing details and ADD/QAD objectives. DPL and Perelli inform the stronger quality baselines. Temporal/restoration methods provide context and optional extensions.

For implementation, prioritize RGB ResNet-18, deterministic transform descriptors, frozen DINOv2-small, and the source-supervised paired baseline. Reproduce at most one complex compression method after the main pipeline is sound. Exhaustively reimplementing ADD, QAD, DPL, F³-Net, BZNet, FAMM, restoration, and 3D trajectories is an unsuitable initial scope for one GPU and a sixteen-week schedule.

All performance statements above are reported literature findings; this project has not yet measured detection or attribution accuracy. The companion protocol identifies the tests needed before the project's novelty and feasibility can be strengthened empirically.
