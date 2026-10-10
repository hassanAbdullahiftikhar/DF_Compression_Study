# What Survives Compression? — research preparation

Prepared 10 October 2026 from `C:/Users/hassa/Downloads/Mgr.md` and independently checked primary literature. The proposal is source material; its claims and suggested methods are evaluated rather than treated as instructions.

Read these first:

1. [Literature review and novelty audit](literature_review.md): all 12 references, missing prior art, corrected citations, and a qualified novelty verdict.
2. [Experimental protocol and hardware assessment](experimental_protocol.md): the proposed experiment, controls, metrics, budget, and stage gates.
3. [Starter experiment instructions](../experiments/README.md): runnable preparation tools and exactly what has and has not been tested.
4. [Complete module roadmap](research_roadmap.md): preparation, datasets, detailed steps and completion checks through the final paper.
5. [Module tracker](module_tracker.md) and [experiment register](experiment_matrix.md): current status and planned run families.
6. [Initial Module 01 specification](module1_specification.md): the current C23-based scope and planned primary comparisons, recorded before classifier test results.

**Dataset now available:** the Kaggle FF++ C23 mirror is in `experiments/FaceForensics/FaceForensics++_C23`. The primary inventory and official split audit are in `experiments/data/audit_c23`. This changes the initial experiment to additional compression from C23-derived reference crops. The earlier raw-reference design remains a possible stronger extension.

Read the [dataset audit and preparation results](dataset_audit_c23.md) for the actual inventory, official splits, sampled decoding, 800 real smoke crops, JPEG variants, and remaining gates.

**Decision:** proceed with a controlled compression study of detection versus manipulation-method attribution. Broad claims of first studying attribution under compression, first comparing detection with attribution, or inventing compression-paired learning are unsupported. A defensible contribution may be the controlled combination of representation comparisons, paired survival analysis, held-out codecs, recompression histories, and shortcut controls. This is a candidate contribution, not proof of uniqueness.

The RTX 5070 was observed locally: 12,227 MiB reported memory, driver 610.88. The processor registry identifies a Ryzen 7 7700. A second Windows memory check reports 16,229,781,504 bytes usable physical RAM (consistent with nominal 16 GB), with roughly 4.5 GiB available at that moment. No deepfake dataset or trained research model was present in this initially empty workspace. Training speed and accuracy are unmeasured.
