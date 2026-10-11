# Main cohort inclusion specification

Recorded after reference preprocessing and before main classifier test scoring.

Use the 5,000-video FF++ C23 core inventory and official provenance splits.
Retain every valid crop from the eight originally scheduled frame positions.
Include videos with at least one valid crop; exclude zero-crop videos.
Average probabilities over available frames, giving each evaluated video equal weight.
Use the same reference sample IDs across all compared models and compression conditions.
Log failures and coverage by source/split. Keep the detector threshold, margin,
sampling positions and crop settings unchanged; do not replace failed frames.

The assembled cohort contains 39,898 crops from 4,999 videos.
Every missing scheduled crop is reconciled with a no-face failure entry.
A complete-eight-frame sensitivity analysis remains planned.

The pilot used validation for model and threshold selection; its results remain exploratory.
Retrain and select final models using the expanded training/validation cohort.
The earlier checkpoint-selection and threshold-selection rules remain in force.
Freeze each model's validation-selected operating threshold across test conditions.

Reference images derive from already H.264 C23-compressed videos.
Mixed decoder versions and the limited 80-crop compatibility comparison remain documented.
Known input-ID/provenance separation does not establish identity/YouTube independence.
See experiments/data/c23_main/assembly.json for counts, coverage and source hashes.
