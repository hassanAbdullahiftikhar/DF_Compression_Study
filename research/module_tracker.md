# Module progress tracker

Updated 2026-10-28. Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); DFDM + Celeb-DF second-dataset study, external test E10 and the I-JEPA probe extension X8 (Amendments 7-8) completed on 28 October 2026; manuscript work remains.
Status describes completed evidence rather than planned experiments.

| ID | Module | Current evidence | Work remaining |
|---|---|---|---|
| 01 | Scope and rules | Inclusion policy, validation thresholds, comparison register v1.0 (frozen 2026-10-23) and Amendment 1 (G3 model freeze, 2026-10-24) recorded | Report the amendment in the paper |
| 02 | Literature | Initial review and qualified novelty assessment prepared | Final refresh and reproduction audits |
| 03 | Environment | Nine measured RGB runs and frozen DINOv2 completed | RAM peaks and extension budgets |
| 04 | Inventory | 5,000 core videos inventoried | Mirror authenticity limitations in paper |
| 05 | Splits | 500 provenance groups; known input-ID separation audited | Identity/YouTube limitations and final support checks |
| 06 | Crops | 39,898 crops; 102 failures; 4,999 usable videos; sampled QA; complete-eight validation sensitivity | Report test support sensitivity (S7) |
| 07 | JPEG | Ref/Q80/Q40 validation; test at ref/Q95/Q80/Q60/Q40/Q20 with crop PSNR/SSIM | Final figures |
| 08 | Evaluation | Frozen thresholds, aggregation, retention, gating and bootstrap; registered test analysis completed | Figures and tables |
| 09 | Pilot | Pilot training, probes and preprocessing checks completed | Historical evidence retained |
| 10 | Representations | Six reference probes and validation compression analyses completed | Report P1-PROBE (no Holm-supported contrast) and S3 |
| 11 | Main JPEG | Nine RGB runs; Q40 and Q80 adaptation probes; held-out test at six conditions completed | Paper reporting |
| 12 | Video codecs | Post-freeze extension X5: H.264 CRF 18/23/32/40 and H.265 CRF 23/32 test re-encoding scored with frozen models (2026-10-25) | Paper reporting as a labelled extension |
| 13 | Recompression | Generator and engineering checks available | Scientific history controls if retained |
| 14 | Paired learning | Weight-1 candidate at three seeds; C1 and C2 controls; registered test comparisons S2, S4 and S5 completed | Paper reporting (no reliable paired benefit over augmentation) |
| 15 | External validation | DFDM + Celeb-DF study (separate register, 28 October 2026) and E10 external Celeb-DF test completed | Paper reporting |
| 16 | Statistics/controls | Validation bootstrap and controls; gates G1-G4 completed; registered test inference recorded | Paper reporting of every registered outcome |
| 17 | Reproducibility | Scripts, configs, environment, model-freeze manifest and validation/test aggregates prepared | Figure/table regeneration |
| 18 | Paper | Literature and protocol material available | Manuscript, figures, numerical checks and revisions |

Validation outcomes are selection-stage evidence. Registered test outcomes come from one
scoring pass (see main_results_test.json). Bootstrap intervals exclude selection
uncertainty; seed SD is descriptive.

Next: Prepare final figures and tables from research/main_results_test.json, refresh the literature review and write the manuscript, reporting every registered outcome.

See [the test aggregate](main_results_test.json) and [the validation milestone](main_results_validation.json)
for exact metrics, intervals, resource records and artifact hashes.
