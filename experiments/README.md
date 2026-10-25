# Experiment workflow

Updated 2026-10-25. Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); DFDM second-dataset study in preparation; manuscript work remains.
Current evidence and results are summarized in [the project README](../README.md)
and [the validation milestone](../research/main_results_validation.json).

## Current environment

Use the existing Python 3.11 DeepFake environment from the repository root:

```powershell
& .\DeepFake\Scripts\python.exe experiments/preflight.py
```

This is an environment check for reproduction, not part of the documentation
export. The current stack is recorded in
[environment_deepfake.json](../research/environment_deepfake.json).
The earlier Python 3.12 lock describes historical preprocessing.

## Completed workflow

| Stage | Scripts |
|---|---|
| Inventory and official splits | audit_ffpp.py, fetch_ffpp_splits.py, common.py |
| Fixed face crops | extract_reference_faces.py |
| Reuse pilot crops and expand the cohort | prepare_main_expansion.py, assemble_main.py |
| Three-seed measured RGB training | train_measured.py, train_paired.py, model.py |
| RGB threshold freezing and validation analysis | freeze_main_thresholds.py, analyze_main_validation.py |
| Deterministic and frozen DINOv2 features | features.py, extract_dinov2_features.py |
| Reference ridge probes and label shuffles | probe_validation.py |
| In-memory probe JPEG validation | evaluate_probe_compression.py |
| Probe bootstrap | analyze_probe_validation.py |
| Known-quality adaptation | fit_known_quality_probes.py |
| Complete-eight-frame sensitivity | eight_frame_sensitivity.py |
| Comparison-register freeze | freeze_register.py |
| C1/C2 control training and thresholds | train_control.py, freeze_control_thresholds.py |
| C2 frozen-encoder source readout | fit_source_readout.py |
| C3 border, metadata and content controls | fit_border_probes.py, fit_metadata_probe.py, content_decomposition.py |
| Registered model list | registered_models.py |
| Test scorers, verified on validation first (G2, G4) | score_rgb.py, score_probes.py |
| Model-freeze manifest and scoring guard (G3) | freeze_models.py |
| Registered test analysis and its validation dry run (G4) | analyze_test.py |
| Documentation/results export | export_validation_milestone.py |

All nine main RGB runs use 20 epochs, eight pairs per batch and seeds
42/43/44. The original checkpoint-selection rule is retained.
Pair weight 1 was selected from the exploratory pilot.

Reference probes select ridge penalties using reference validation.
Known-quality probes use matched-quality validation. Scalers are fitted
on the appropriate training features only. These are distinct tracks.

## Resume point

The registered test pass is complete and must not be rerun except to fix a documented
bug (register section 9). It was produced, in order, by:

```powershell
& .\DeepFake\Scripts\python.exe experiments/score_rgb.py --split val      # G2
& .\DeepFake\Scripts\python.exe experiments/score_probes.py --split val   # G2
& .\DeepFake\Scripts\python.exe experiments/analyze_test.py --split val   # dry run
& .\DeepFake\Scripts\python.exe experiments/freeze_models.py              # G3, then Amendment 1
& .\DeepFake\Scripts\python.exe experiments/score_rgb.py --split test     # G4
& .\DeepFake\Scripts\python.exe experiments/score_probes.py --split test  # G4
& .\DeepFake\Scripts\python.exe experiments/analyze_test.py --split test  # G4
```

The test scorers refuse to run without the manifest, the amendment and unchanged
hashes, and they refuse to overwrite outputs.

Next: Prepare final figures and tables from research/main_results_test.json, refresh the literature review and write the manuscript, reporting every registered outcome.

predict_rgb.py and probe_features.py are earlier pilot scorers and are not part of
the registered pass.

## Storage and reproduction

JPEGs are generated in memory. Main manifests reuse existing crops and
copy no images. Keep the underlying crop folders referenced by the manifests.

Known-quality fitting uses temporary numeric buffers in RAM and saves
fitted heads, predictions and reports. Output tools refuse to overwrite
existing destinations.

Raw media, feature caches, model parameters and predictions remain local.
Aggregate evidence is exported to research/main_results_validation.json and
research/main_results_test.json. Test predictions use about 270 MB under
experiments/runs/c23_test_rgb and c23_test_probes.
Fresh reproduction requires dataset access and the recorded preprocessing
and pretrained assets.

Frame failures are logged and unreplaced. The mixed-decoder compatibility
check covered 80 crops in one training group; global equivalence is unverified.
