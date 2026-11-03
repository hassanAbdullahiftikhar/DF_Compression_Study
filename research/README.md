# Research documents and current evidence

Updated 2026-10-28. Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); DFDM + Celeb-DF second-dataset study, external test E10 and the I-JEPA probe extension X8 (Amendments 7-8) completed on 28 October 2026; manuscript work remains.

The proposal in Mgr.md was evaluated as source material. The literature
review covers its references and qualifies the project's novelty claims.

## Current evidence

The main cohort contains 39,898 crops from 4,999 videos.
Nine RGB runs across three seeds, six reference-trained probes,
validation compression sweeps, frozen operating thresholds and
70-group bootstrap analyses are complete.
Six Q40 adaptation probes are complete; Q80 adaptation: complete, validation only.

Registered held-out test results (gates G2-G4, 24 October 2026) are in
[main_results_test.json](main_results_test.json). They come from one scoring pass
after the [model-freeze manifest](model_freeze_manifest.json) and register Amendment 1.
Validation evidence remains in the validation milestone. The manuscript is unfinished.

Read these first:

1. [Project overview and results](../README.md)
2. [Registered test results](main_results_test.json) and [aggregate validation milestone](main_results_validation.json)
3. [Module tracker](module_tracker.md)
4. [Frozen comparison register](comparison_register.md) and [experiment register](experiment_matrix.md)
5. [Main cohort specification](main_cohort_specification.md)
6. [Initial analysis specification](module1_specification.md)
7. [Literature and novelty assessment](literature_review.md)
8. [Experimental protocol](experimental_protocol.md)
9. [Full module roadmap](research_roadmap.md)
10. [Script workflow](../experiments/README.md)

The original preparation documents retain their historical dates and
design rationale. Current execution status is maintained in the tracker,
register and aggregate milestone.

Broad claims of first studying attribution under compression or inventing
paired compression learning remain unsupported. Final conclusions depend
on the registered held-out results and controls, including null results.

Next: Prepare final figures and tables from research/main_results_test.json, refresh the literature review and write the manuscript, reporting every registered outcome.
