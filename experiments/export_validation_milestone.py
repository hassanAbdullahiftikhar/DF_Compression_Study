"""Export existing validation evidence and refresh project documentation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-12"
REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")
MODES = ("reference", "augmented", "paired_w1")
SEEDS = (42, 43, 44)
artifact_hashes = {}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(relative):
    path = ROOT / relative
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    artifact_hashes[relative] = sha256(path)
    return data


def write(relative, text):
    path = ROOT / relative
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.strip() + "\n")


def percentage(value):
    return f"{100 * value:.2f}"


assembly = read("experiments/data/c23_main/assembly.json")
environment = read("research/environment_deepfake.json")
thresholds = read("experiments/results/main_validation_thresholds_seeds42_44.json")
rgb_analysis = read("experiments/results/main_validation_analysis_seeds42_44.json")
probe_compression = read(
    "experiments/results/c23_main_probe_compression_validation/report.json"
)
probe_analysis = read("experiments/results/main_probe_validation_analysis.json")
matched40 = read("experiments/runs/c23_main_matched_q40_probes_validation/report.json")
dino = read("experiments/features/c23_main_dinov2_trainval/feature_metadata.json")

q80_path = "experiments/runs/c23_main_matched_q80_probes_validation/report.json"
matched80 = read(q80_path) if (ROOT / q80_path).exists() else None

assert assembly["audit"]["rows"] == 39898
assert assembly["audit"]["videos"] == 4999
assert thresholds["test_scores_used"] is False
assert probe_compression["test_scores_used"] is False
assert matched40["test_scores_used"] is False
assert matched40["training_condition"] == "jpeg_q40"
assert rgb_analysis["bootstrap"]["groups"] == 70
assert probe_analysis["bootstrap"]["groups"] == 70
assert rgb_analysis["bootstrap"]["draws"] == 2000
assert probe_analysis["bootstrap"]["draws"] == 2000
assert thresholds["manifest_sha256"] == assembly["reference_manifest_sha256"]
assert probe_compression["manifest_sha256"] == assembly["train_val_manifest_sha256"]
assert matched40["manifest_sha256"] == assembly["train_val_manifest_sha256"]

if matched80:
    assert matched80["test_scores_used"] is False
    assert matched80["training_condition"] == "jpeg_q80"
    assert matched80["manifest_sha256"] == assembly["train_val_manifest_sha256"]

rgb_runs = {}
for seed in SEEDS:
    for mode in MODES:
        name = f"c23_main_{mode}_seed{seed}"
        prefix = f"experiments/runs/{name}"
        history = read(f"{prefix}/history.json")
        metadata = read(f"{prefix}/run_metadata.json")
        resources = read(f"{prefix}/resource_usage.json")
        registered = thresholds["runs"][name]
        assert len(history) == 20
        assert metadata["config"]["seed"] == seed
        assert resources["status"] == "completed_measured_main_training"
        assert resources["manifest_sha256"] == assembly["reference_manifest_sha256"]
        assert sha256(ROOT / prefix / "best.pt") == registered["checkpoint_sha256"]
        selected = [
            epoch for epoch in history
            if epoch["epoch"] == registered["selected_epoch"]
        ]
        assert len(selected) == 1
        rgb_runs[name] = {
            "metadata": metadata,
            "resources": resources,
            "selected_epoch": selected[0]["epoch"],
            "selected_validation_at_threshold_0_5": selected[0]["validation"],
            "frozen_threshold": registered["threshold"],
            "checkpoint_sha256": registered["checkpoint_sha256"],
        }

reference_selections = {}
for rep in REPS:
    prefix = f"experiments/runs/c23_main_{rep}_probe_validation"
    reference_selections[rep] = read(f"{prefix}/selection.json")
    registered = probe_compression["representations"][rep]
    assert sha256(ROOT / prefix / "probe_parameters.npz") == registered["parameters_sha256"]
    assert sha256(ROOT / prefix / "selection.json") == registered["selection_sha256"]
    assert reference_selections[rep]["train_frames"] == 28735
    assert reference_selections[rep]["validation_frames"] == 5582
    for report, quality in ((matched40, 40), (matched80, 80)):
        if report:
            path = ROOT / (
                f"experiments/runs/c23_main_matched_q{quality}_probes_validation/"
                f"{rep}/probe_parameters.npz"
            )
            assert sha256(path) == report["representations"][rep]["parameters_sha256"]

source_hashes = {}
paths = sorted((ROOT / "experiments").glob("*.py"))
paths += sorted((ROOT / "experiments/configs").glob("*.json"))
for path in paths:
    raw = path.read_bytes()
    relative = path.relative_to(ROOT).as_posix()
    source_hashes[relative] = {
        "working_tree_sha256": hashlib.sha256(raw).hexdigest(),
        "lf_normalized_sha256": hashlib.sha256(
            raw.replace(b"\r\n", b"\n")
        ).hexdigest(),
    }

snapshot = {
    "schema_version": 1,
    "as_of": DATE,
    "project_status": "paused",
    "evidence_scope": "Selected validation results; final classifier test evaluation pending",
    "cohort": assembly,
    "environment": environment,
    "rgb": {
        "runs": rgb_runs,
        "thresholds": thresholds,
        "analysis": rgb_analysis,
    },
    "reference_probes": {
        "selections": reference_selections,
        "compression": probe_compression,
        "analysis": probe_analysis,
    },
    "known_quality_probes": {
        "jpeg_q40": matched40,
        "jpeg_q80": matched80,
    },
    "dinov2_extraction": dino,
    "source_snapshot_at_export": source_hashes,
    "input_artifact_sha256": artifact_hashes,
    "qualification": (
        "Contains aggregate metrics and provenance, not raw predictions, "
        "images, feature arrays or model parameters. Bootstrap intervals "
        "condition on selected models and thresholds. Seed SD is descriptive. "
        "Source LF hashes account for repository text normalization."
    ),
}
snapshot_text = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False)
assert len(snapshot_text.encode("utf-8")) < 2 * 1024 * 1024
write("research/main_results_validation.json", snapshot_text)

rgb_rows = []
labels = {
    "reference": "Reference training",
    "augmented": "JPEG augmentation",
    "paired_w1": "Paired training, weight 1",
}
for mode in MODES:
    scores = rgb_analysis["descriptive_seed_variation"][mode]["jpeg_q40"]
    cells = []
    for metric in ("detection_ba", "attribution_ba"):
        value = scores[metric]
        cells.append(
            f'{percentage(value["mean"])} ± {percentage(value["sample_sd"])}'
        )
    rgb_rows.append(f"| {labels[mode]} | {cells[0]} | {cells[1]} |")

probe_rows = []
for rep in REPS:
    scores = probe_compression["representations"][rep]["metrics"]["jpeg_q40"]
    label = {"rgb": "Pooled RGB", "dinov2": "DINOv2"}.get(rep, rep.upper())
    probe_rows.append(
        f'| {label} | {percentage(scores["detection_ba"])} | '
        f'{percentage(scores["attribution_ba"])} |'
    )

adaptation_rows = []
for rep, label in (("dct", "DCT"), ("dinov2", "DINOv2")):
    result = matched40["representations"][rep]
    before = result["reference_trained_validation"]
    after = result["threshold_validation"]
    for metric, description in (
        ("detection_auc", "Detection AUROC"),
        ("attribution_ba", "Attribution balanced accuracy"),
    ):
        adaptation_rows.append(
            f"| {label}: {description} | {percentage(before[metric])} | "
            f"{percentage(after[metric])} |"
        )

minutes = [run["resources"]["elapsed_minutes"] for run in rgb_runs.values()]
q80_status = "Complete, validation only" if matched80 else "Pending"
next_step = (
    "Prepare the remaining shortcut/task-sharing controls and final evaluation register."
    if matched80 else
    "Resume with the Q80 known-quality probe control, then prepare the remaining controls."
)

root_readme = f"""
# What Survives Compression?

Updated {DATE}. **Research is paused at the user's request.**

A controlled study of deepfake detection and four-way, closed-set
manipulation-method attribution under additional JPEG compression.
The reference crops come from an already H.264 C23-compressed
FaceForensics++ mirror.

## Completed work

- Literature review and qualified novelty assessment prepared.
- Main core inventory: 5,000 videos across real, Deepfakes, Face2Face,
  FaceSwap and NeuralTextures.
- Main cohort: 39,898 valid crops from 4,999 videos, in 500 provenance groups.
- All nine main RGB runs completed: reference, JPEG augmentation and paired
  training at seeds 42, 43 and 44, with 20 epochs per run.
- Selected RGB checkpoints and validation operating thresholds recorded.
- Six reference-trained probes completed: pooled RGB, FFT, block DCT,
  Haar wavelet, Laplacian residual and frozen DINOv2-small.
- Probe validation evaluated at reference/Q80/Q40, with frozen thresholds.
- Group bootstrap: 2,000 draws across 70 validation provenance groups.
- Six Q40 known-quality adaptation probes and shuffled-label diagnostics completed.
- Q80 known-quality adaptation: **{q80_status.lower()}**.

The validation cohort contains 699 videos, including 559 fake videos.
Classifier test evaluation remains pending.

## Main RGB results at JPEG Q40

Mean ± sample standard deviation across seeds 42/43/44.
Both columns are balanced accuracy in percent; SD is in percentage points.
These are descriptive seed summaries, not confidence intervals.

| Training approach | Detection | Fake-only attribution |
|---|---:|---:|
{chr(10).join(rgb_rows)}

JPEG augmentation improves Q40 validation performance in all three seeds.
The paired approach's incremental advantage is small and mixed.
All per-seed Q40 paired-versus-augmentation confidence intervals include
zero for detection and attribution.

Checkpoint selection used mean task balanced accuracy across validation
reference/Q80/Q40 at threshold 0.5. Each selected model's detection threshold
was subsequently chosen on those validation conditions and frozen.

## Reference-trained representation probes at Q40

Values are validation balanced accuracy in percent, with frozen detection
thresholds. Pooled RGB is a pixel descriptor, separate from the trained ResNet.

| Representation | Detection | Fake-only attribution |
|---|---:|---:|
{chr(10).join(probe_rows)}

DINOv2 has the highest absolute Q40 performance among these probes.
The reference-trained DCT probe becomes nearly chance-level under JPEG.
FFT detection retention and wavelet attribution retention have unstable
reference denominators; their affected confidence intervals are unavailable.

The usable probe intervals do not establish a general ordering of detection
versus attribution degradation at Q40. Representation, classifier capacity
and pretraining differ across tracks.

## Known-quality adaptation control at Q40

New scaling and ridge heads were fitted on Q40-compressed training crops
and selected using Q40 validation. Values below are percentages.

| Metric | Reference-trained, evaluated at Q40 | Q40-trained, evaluated at Q40 |
|---|---:|---:|
{chr(10).join(adaptation_rows)}

DCT's partial recovery supports training/compression mismatch as a
contributor to its earlier failure. These results demonstrate usable label
information on this validation cohort; they do not establish a physical
forensic fingerprint. Threshold selection domains differ between the two
regimes, so AUROC and fake-only attribution are the clearest comparisons.

## Interpretation and limitations

- All reported classifier results are validation-selected and optimistic.
  Bootstrap intervals condition on selected models and thresholds and
  exclude selection uncertainty.
- Attribution concerns four known manipulation methods on fake videos.
  It does not identify individual generator checkpoints or unknown methods.
- JPEG is additional compression of C23-derived crops.
- Official splits separate known original IDs/provenance groups.
  Identity/YouTube-level independence and official-byte authenticity
  of the mirror are not established.
- Failures were logged without replacement. One zero-crop validation video,
  Deepfakes/638_640, was excluded; partial videos remain included.
- Classifier failure does not establish irreversible loss of identifying information.
- Broad first-ever novelty claims are unsupported. The candidate contribution
  is a bounded comparative empirical study.

## Remaining work

1. Finish Q80 known-quality adaptation if pending.
2. Finalize and run the selected background/border/content, task-sharing,
   source-supervision and preprocessing controls.
3. Run the complete-eight-frame sensitivity analysis.
4. Finalize the comparison register and any remaining mixed-quality probe baselines.
5. Evaluate frozen models on the test cohort and the held-out JPEG qualities
   Q95/Q60/Q20; these qualities have not selected the current models.
6. Complete paired test statistics, seed reporting, distortion and error analysis.
7. Run recompression-history experiments if retained in the final JPEG scope.
8. Refresh the literature/novelty assessment, generate final figures/tables,
   write the manuscript and revise its claims.

Whole-video H.264/H.265 and external datasets remain separate extensions.

## Measured environment and feasibility

Current environment: Python 3.11.9, torch 2.7.1+cu128,
torchvision 0.22.1+cu128, CUDA runtime 12.8, NumPy 2.4.6,
Pillow 12.3.0, PyAV 18.1.0 and OpenCV-headless 4.14.0.94.

Implemented JPEG experiments completed on an RTX 5070, Ryzen 7 7700
and nominal 16 GB RAM system. Main RGB runs took approximately
{min(minutes):.1f}–{max(minutes):.1f} minutes each, with 500 MiB peak reserved
PyTorch GPU memory. DINOv2 reference extraction took {dino["seconds"]:.1f}
seconds for 34,317 crops, excluding model loading, with 166 MiB peak reserved
GPU memory. System/process RAM peaks were not measured.

The earlier Python 3.12 preprocessing lock is historical.
The current environment is recorded in
[environment_deepfake.json](research/environment_deepfake.json).

## Documentation and evidence

- [Aggregate validation milestone](research/main_results_validation.json)
- [Current module tracker](research/module_tracker.md)
- [Experiment register](research/experiment_matrix.md)
- [Experiment scripts and workflow](experiments/README.md)
- [Main cohort inclusion specification](research/main_cohort_specification.md)
- [Initial scope and analysis rules](research/module1_specification.md)
- [Literature review](research/literature_review.md)
- [Protocol](research/experimental_protocol.md)
- [Full roadmap](research/research_roadmap.md)
- [Bibliography](research/references.bib)

Dataset access: [official FaceForensics++](https://github.com/ondyari/FaceForensics)
and the [C23 mirror used here](https://www.kaggle.com/datasets/xdxd003/ff-c23).

Dataset media, crops, raw predictions, feature caches, weights, virtual
environments and local results directories remain excluded from Git.
The aggregate milestone preserves metrics, confidence intervals,
configuration/resource records and artifact hashes.

Saved hashes describe execution artifact bytes. The snapshot also records
LF-normalized source hashes for comparison with Git's text normalization.
"""
write("README.md", root_readme)

write("research/README.md", f"""
# Research documents and current evidence

Updated {DATE}. Research is paused after the Q40 known-quality control.

The proposal in Mgr.md was evaluated as source material. The literature
review covers its references and qualifies the project's novelty claims.

## Current evidence

The main cohort contains 39,898 crops from 4,999 videos.
Nine RGB runs across three seeds, six reference-trained probes,
validation compression sweeps, frozen operating thresholds and
70-group bootstrap analyses are complete.
Six Q40 adaptation probes are complete; Q80 adaptation is {q80_status.lower()}.

All classifier results currently reported are validation evidence.
Final test classification, the held-out quality sweep, selected controls
and the manuscript remain unfinished.

Read these first:

1. [Project overview and results](../README.md)
2. [Aggregate validation milestone](main_results_validation.json)
3. [Module tracker](module_tracker.md)
4. [Experiment register](experiment_matrix.md)
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
on held-out evaluation and controls.

Next when resumed: {next_step}
""")

write("experiments/README.md", rf"""
# Experiment workflow

Updated {DATE}. Research is paused.
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

Q80 known-quality adaptation is {q80_status.lower()}.
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
""")

tracker_rows = [
    ("01", "Scope and rules", "Initial rules, main inclusion policy and validation thresholds recorded", "Final comparison/control register"),
    ("02", "Literature", "Initial review and qualified novelty assessment prepared", "Final refresh and reproduction audits"),
    ("03", "Environment", "Nine measured RGB runs and frozen DINOv2 completed", "RAM peaks and extension budgets"),
    ("04", "Inventory", "5,000 core videos inventoried", "Mirror authenticity limitations in paper"),
    ("05", "Splits", "500 provenance groups; known input-ID separation audited", "Identity/YouTube limitations and final support checks"),
    ("06", "Crops", "39,898 crops; 102 failures; 4,999 usable videos; sampled QA", "Complete-eight-frame sensitivity"),
    ("07", "JPEG", "Ref/Q80/Q40 validation and distortion measurements completed", "Held-out qualities and final distortion analysis"),
    ("08", "Evaluation", "Frozen thresholds, aggregation, retention, gating and bootstrap implemented", "Final test analysis"),
    ("09", "Pilot", "Pilot training, probes and preprocessing checks completed", "Historical evidence retained"),
    ("10", "Representations", "Six reference probes and validation compression analyses completed", "Test comparisons and retained mixed-quality baselines"),
    ("11", "Main JPEG", f"Nine RGB runs; six Q40 probes; Q80 adaptation {q80_status.lower()}", "Remaining controls/regimes and held-out testing"),
    ("12", "Video codecs", "Starter design/code only", "Separate optional extension"),
    ("13", "Recompression", "Generator and engineering checks available", "Scientific history controls if retained"),
    ("14", "Paired learning", "Weight-1 candidate evaluated at three seeds on validation", "Held-out and task/source-supervision controls"),
    ("15", "External validation", "Unstarted", "Optional external studies"),
    ("16", "Statistics/controls", "70-group bootstrap, seed summaries and shuffle diagnostics completed", "Shortcut/content controls and test inference"),
    ("17", "Reproducibility", "Scripts, configs, environment and aggregate milestone prepared", "Final register and figure/table regeneration"),
    ("18", "Paper", "Literature and protocol material available", "Manuscript, figures, numerical checks and revisions"),
]
tracker_table = "\n".join(
    f"| {number} | {module} | {evidence} | {remaining} |"
    for number, module, evidence, remaining in tracker_rows
)
write("research/module_tracker.md", f"""
# Module progress tracker

Updated {DATE}. Research is paused at the user's request.
Status describes completed evidence rather than planned experiments.

| ID | Module | Current evidence | Work remaining |
|---|---|---|---|
{tracker_table}

All classifier outcomes reported here are selected validation results.
Bootstrap intervals exclude selection uncertainty; seed SD is descriptive.
Test preprocessing has occurred; final classifier test evaluation is pending.

Next when resumed: {next_step}

See [the aggregate milestone](main_results_validation.json) for exact
metrics, intervals, resource records and artifact hashes.
""")

matrix_path = ROOT / "research/experiment_matrix.md"
old_matrix = matrix_path.read_text(encoding="utf-8-sig")
marker = "\n## Planned families of experiments"
assert old_matrix.count(marker) == 1

matrix_header = f"""
# Experiment register and planned comparisons

Initial design prepared 10 October 2026; execution status updated {DATE}.
Research is paused. The family table below retains the initial plan.

## Current execution status

| Family | Current evidence | Remaining |
|---|---|---|
| E00: Pipeline pilot | Completed | Historical evidence retained |
| E01: Reference sensitivity | Six probes and RGB validation at ref/Q80/Q40 completed | Test and held-out qualities |
| E02: Known-quality adaptation | Six Q40 probes completed; Q80 {q80_status.lower()} | Final matched-quality test comparison |
| E03: Mixed-quality generalization | RGB augmentation trained at three seeds | Held-out testing; retained probe regimes |
| E04: Task-sharing | Separate RGB encoder control unstarted | Selected control |
| E05: Video coding | Starter design/code | Separate extension |
| E06: Processing history | Generator and engineering checks | Scientific experiments if retained |
| E07: Paired learning | Main weight-1 candidate at three seeds | Held-out comparisons |
| E08: Source supervision | Binary-only RGB control unstarted | Selected control |
| E09: Explanatory controls | Label shuffles and sampled crop QA completed | Background/border/content and sensitivity checks |
| E10: External detection | Unstarted | Optional |
| E11: Generator-family study | Unstarted | Optional separate study |

Current results are validation evidence. Models, thresholds, penalties and
regimes are recorded in [main_results_validation.json](main_results_validation.json).
Final classifier test evaluation remains pending.
"""
write(
    "research/experiment_matrix.md",
    matrix_header.strip() + "\n" + marker + old_matrix.split(marker, 1)[1],
)

print("Updated README.md, research/README.md, experiments/README.md")
print("Updated research/module_tracker.md and research/experiment_matrix.md")
print(
    "Exported research/main_results_validation.json:",
    (ROOT / "research/main_results_validation.json").stat().st_size,
    "bytes",
)
print("No experiments were run. Research remains paused.")
