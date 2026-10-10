# Module progress tracker

Prepared 10 October 2026. Status describes this workspace's current evidence, not planned completion.

| ID | Module | Prerequisites | Current state | Next completion check |
|---|---|---|---|---|
| 01 | Research scope and analysis rules | None | Initial C23 specification recorded | Finalize measured run budget and log any subsequent design changes |
| 02 | Literature/baseline selection | 01 | Review/bibliography prepared | Complete access/code audit for the method selected for reproduction |
| 03 | Environment/resource benchmark | None; real pilot later | Private preprocessing environment ready | Install GPU research stack; measure actual training step |
| 04 | Dataset access/inventory | 01 | C23 mirror inventoried; 5,000 core videos | Full media decoding if needed; raw-reference extension optional |
| 05 | Lineage/splits | 04 | Official 500 source-pair groups mapped without conflicts | Document remaining identity/YouTube-level limitations |
| 06 | Reference crops | 03–05 | 800 crops from 100-video smoke set, zero failures | Complete full 1,000-video pilot and coverage checks |
| 07 | JPEG conditions | 06 | 7,200 real smoke manifest rows passed audit | Expand after full-pilot extraction; add distortion summaries |
| 08 | Metrics/statistical harness | 01 | Partial implementation | Complete EER, retention, coverage and group bootstrap; verify edge cases |
| 09 | End-to-end pilot | 03–08 | Real preprocessing smoke passed; classifier unrun | Learning sanity, source support and measured training-resource gate |
| 10 | Representation panel | 09 | Five cheap descriptors; ML starters prepared | DINOv2 plus all claimed representations/probes verified on real data |
| 11 | Main JPEG experiments | 08–10 | Unrun | Frozen reference/matched/mixed comparisons and predictions |
| 12 | Video codecs | 09; 11 recommended | Encoder starter only | Full-video alignment, calibration and held-out-codec results |
| 13 | Recompression | 11; 12 for video | JPEG generator tested synthetically | History comparison with final-distortion controls |
| 14 | Source-aware pairing | 09–11 | Baseline trainer syntax-checked | Fair controls, validation sweep, actual seeds and held-out results |
| 15 | External study | 11 | Optional, unstarted | Compatible external task/labels and independently audited data |
| 16 | Statistics/controls/errors | 08; completed run families | Design prepared | Paired uncertainty, source failures, shortcut checks and bounded claims |
| 17 | Reproducibility package | Continuous; finalized after 16 | Starter documentation | Traceable runs and regenerated paper figures/tables |
| 18 | Paper | Background can begin now; results after 16–17 | Review material only | Complete checked manuscript matching actual evidence |

Read `research_roadmap.md` for each module's data, detailed steps, deliverables, and exit criterion. Use `experiment_matrix.md` to register runs before execution. Update this tracker only when the corresponding evidence exists.
