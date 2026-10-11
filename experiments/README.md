# Experiment workflow

Updated 2026-10-12. Research is paused.
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
| Documentation/results export | export_validation_milestone.py |

All nine main RGB runs use 20 epochs, eight pairs per batch and seeds
42/43/44. The original checkpoint-selection rule is retained.
Pair weight 1 was selected from the exploratory pilot.

Reference probes select ridge penalties using reference validation.
Known-quality probes use matched-quality validation. Scalers are fitted
on the appropriate training features only. These are distinct tracks.

## Resume point

Q80 known-quality adaptation is pending.
If pending, resume with:

```powershell
& .\DeepFake\Scripts\python.exe experiments/fit_known_quality_probes.py --quality 80
```

Then prepare the remaining controls and final comparison register.
Reference/Q80/Q40 are selection conditions; Q95/Q60/Q20 remain held out
for the relevant generalization track.

predict_rgb.py produces test predictions. The earlier probe_features.py
also scores test data; current validation uses probe_validation.py.

## Storage and reproduction

JPEGs are generated in memory. Main manifests reuse existing crops and
copy no images. Keep the underlying crop folders referenced by the manifests.

Known-quality fitting uses temporary numeric buffers in RAM and saves
fitted heads, predictions and reports. Output tools refuse to overwrite
existing destinations.

Raw media, feature caches, model parameters and predictions remain local.
Aggregate evidence is exported to research/main_results_validation.json.
Fresh reproduction requires dataset access and the recorded preprocessing
and pretrained assets.

Frame failures are logged and unreplaced. The mixed-decoder compatibility
check covered 80 crops in one training group; global equivalence is unverified.
