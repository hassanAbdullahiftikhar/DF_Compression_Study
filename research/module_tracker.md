# Module progress tracker

Updated 11 October 2026 from user-supplied execution results.
Status describes completed evidence, not planned experiments.

| ID | Module | Current evidence | Work remaining |
|---|---|---|---|
| 01 | Scope and analysis rules | Initial C23 scope recorded; validation threshold rule frozen | Final dataset size and main comparison register |
| 02 | Literature and baselines | Review and corrected bibliography prepared | Selected reproduction audits and final novelty refresh |
| 03 | Environment and resources | Python 3.11/CUDA stack working; synthetic and real RGB training passed | Real peak-resource records and codec budget |
| 04 | Dataset inventory | C23 mirror inventoried; 5,000 core videos; 1,000-video pilot assembled | Final expansion decision; authenticity limitations retained |
| 05 | Lineage and splits | Official provenance groups audited; no known input-ID split conflicts | Identity/YouTube-level limitations and final support checks |
| 06 | Reference crops | 8,000 crops from 1,000 videos; zero extraction failures | Visual crop QA and final documentation |
| 07 | JPEG conditions | Reference/Q80/Q40 used in RGB validation and augmentation | Distortion measurements and full controlled sweep |
| 08 | Evaluation | EER, confusion, gating, retention and paired bootstrap checked; thresholds recorded | Generalized final test analysis and repeated-seed reporting |
| 09 | End-to-end pilot | RGB training and independent descriptor probes completed | Remaining crop, error and resource closeout checks |
| 10 | Representations | Five reference descriptor probes and shuffled controls completed | Frozen DINOv2 features and compression comparisons |
| 11 | Main JPEG study | Seed-42 pilot reference/augmentation RGB runs completed | Matched-quality, held-out evaluation and final seeds |
| 12 | Video codecs | Starter implementation only | Encoding, alignment, calibration and experiments |
| 13 | Recompression | Transform generator available | History experiments and distortion controls |
| 14 | Paired learning | Weights 0, 0.1 and 1 evaluated on seed-42 validation | Final ablations and repeated-seed held-out comparisons |
| 15 | External validation | Optional; unstarted | Separate data/task preparation if retained |
| 16 | Statistics and controls | Conditional 14-group validation bootstrap; descriptor label-shuffle diagnostics | Test inference, seed variability, shortcut controls and error analysis |
| 17 | Reproducibility | Source scripts, environment snapshot and aggregate pilot records prepared | Final run register and figure/table regeneration |
| 18 | Paper | Literature and protocol material available | Manuscript, final figures, revisions and numerical checks |

Pilot results are exploratory validation evidence. Threshold tuning and checkpoint
selection use validation data; current intervals exclude selection uncertainty
and training-seed variability. Main test results are not yet reported.

See pilot_results_seed42.json for recorded pilot evidence and
environment_deepfake.json for the current package snapshot.
Original datasets, feature caches, checkpoints and raw predictions remain local.
