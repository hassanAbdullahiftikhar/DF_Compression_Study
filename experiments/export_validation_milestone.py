"""Export existing validation evidence and refresh project documentation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-25"
REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")
MODES = ("reference", "augmented", "paired_w1")
SEEDS = (42, 43, 44)
LABELS = {"fft": "FFT", "rgb": "Pooled RGB", "dct": "DCT", "wavelet": "Wavelet",
          "residual": "Residual", "dinov2": "DINOv2"}
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
eight = read("experiments/results/eight_frame_sensitivity_validation.json")
register_freeze = read("research/comparison_register_freeze.json")
control_thresholds = read("experiments/results/control_validation_thresholds.json")
readout = read("experiments/runs/c23_control_source_readout_validation/report.json")
border = read("experiments/runs/c23_control_border_probes_validation/report.json")
metadata_probe = read("experiments/runs/c23_control_metadata_probe_validation/report.json")
decomposition = read("experiments/results/content_decomposition_validation.json")
for record in (control_thresholds, readout, border, metadata_probe, decomposition):
    assert record["test_scores_used"] is False
assert len(control_thresholds["runs"]) == 9
assert eight["test_scores_used"] is False
assert register_freeze["test_scoring_status"] == "not_started"
assert not register_freeze["test_prediction_scan"]["files_with_test_videos"]

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
    "project_status": "active",
    "evidence_scope": ("Selected validation results. Registered held-out test results are exported "
                       "separately to research/main_results_test.json."),
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
    "complete_eight_frame_sensitivity": {
        key: eight[key] for key in (
            "status", "definition", "support", "bootstrap",
            "max_abs_ba_shift_vs_full", "contrast_status_changes",
            "script_sha256", "qualification",
        )
    },
    "comparison_register": register_freeze,
    "registered_controls_validation": {
        "c1_c2_rgb_runs": control_thresholds,
        "c2_source_readout": readout,
        "c3a_border_probes": border,
        "c3b_metadata_probe": metadata_probe,
        "c3c_content_decomposition": decomposition,
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
    label = LABELS[rep]
    probe_rows.append(
        f'| {label} | {percentage(scores["detection_ba"])} | '
        f'{percentage(scores["attribution_ba"])} |'
    )

matched = [(80, matched80), (40, matched40)] if matched80 else [(40, matched40)]
adaptation_header = "| Metric | " + " | ".join(
    f"Reference-trained at Q{q} | Q{q}-trained at Q{q}" for q, _ in matched
) + " |"
adaptation_rule = "|---|" + "---:|---:|" * len(matched)
adaptation_rows = []
for rep, label in (("dct", "DCT"), ("dinov2", "DINOv2")):
    for metric, description in (
        ("detection_auc", "Detection AUROC"),
        ("attribution_ba", "Attribution balanced accuracy"),
    ):
        cells = []
        for _, report in matched:
            result = report["representations"][rep]
            cells.append(percentage(result["reference_trained_validation"][metric]))
            cells.append(percentage(result["threshold_validation"][metric]))
        adaptation_rows.append(f"| {label}: {description} | " + " | ".join(cells) + " |")

minutes = [run["resources"]["elapsed_minutes"] for run in rgb_runs.values()]
q80_status = "Complete, validation only" if matched80 else "Pending"
next_step = (
    "Prepare final figures and tables from research/main_results_test.json, refresh the "
    "literature review and write the manuscript, reporting every registered outcome."
)
changes = {
    support: sum(c["support"] == support for c in eight["contrast_status_changes"])
    for support in ("complete_video", "complete_group")
}
status_line = (
    "Registered held-out test evaluation completed on 24 October 2026 (register gates G1-G4); "
    "post-freeze extensions X1-X7 completed on 25 October 2026 (Amendments 2-5); "
    "DFDM second-dataset study in preparation; manuscript work remains."
)
adaptation_title = "Q80 and Q40" if matched80 else "Q40"
adaptation_note = (
    "\nAt Q80, the reference-trained DCT probe falls close to chance, and\n"
    "Q80-matched heads restore much of its reference-condition score. For the\n"
    "other four representations, Q80 adaptation changes AUROC and fake-only\n"
    "attribution by at most about two percentage points. These differences\n"
    "have no bootstrap intervals and are descriptive.\n"
    if matched80 else ""
)
remaining = [
    "Generate final figures and tables from the registered test aggregate\n"
    "   (research/main_results_test.json); report every registered outcome,\n"
    "   including null and negative ones, with validation and test kept separate.",
    "Any further error analysis (for example the Q20 detection score shift) is\n"
    "   exploratory and must be labelled as such; it cannot change registered readings.",
    "Second dataset (Tier 3): DFDM is downloaded and inventoried (6,450 videos, identity-disjoint\n"
    "   split by identity components). Celeb-DF real videos are needed for its detection arm; a new\n"
    "   register must be written before any DFDM fitting.",
    "Recompression history (E06) and other extensions are outside the frozen core;\n"
    "   each needs a register amendment written before its own test scoring.",
    "Refresh the literature/novelty assessment, write the manuscript and revise\n"
    "   its claims against the test evidence.",
]
remaining_list = "\n".join(
    f"{number}. {item}" for number, item in enumerate(remaining, 1)
)


def seed_mean(getter):
    return float(sum(getter(seed) for seed in SEEDS) / len(SEEDS))


points = rgb_analysis["point_metrics"]
controls = control_thresholds["runs"]
control_rows = []
for label, left, right in (
    ("Detection BA: shared F1 / separate C1-DET",
     lambda s, c: points[f"c23_main_reference_seed{s}"][c]["detection_ba"],
     lambda s, c: controls[f"c23_control_c1_det_seed{s}"]["metrics"][c]["detection_ba"]),
    ("Attribution BA: shared F1 / separate C1-ATT",
     lambda s, c: points[f"c23_main_reference_seed{s}"][c]["attribution_ba"],
     lambda s, c: controls[f"c23_control_c1_att_seed{s}"]["metrics"][c]["attribution_ba"]),
    ("Detection BA: F3 / binary-only paired C2-BIN",
     lambda s, c: points[f"c23_main_paired_w1_seed{s}"][c]["detection_ba"],
     lambda s, c: controls[f"c23_control_c2_bin_seed{s}"]["metrics"][c]["detection_ba"]),
    ("Readout R attribution BA: F3-R / C2-R",
     lambda s, c: readout["encoders"][f"f3_seed{s}"]["readout_attribution_ba"][c],
     lambda s, c: readout["encoders"][f"c2_bin_seed{s}"]["readout_attribution_ba"][c]),
):
    cells = [
        f"{percentage(seed_mean(lambda s: left(s, c)))} / {percentage(seed_mean(lambda s: right(s, c)))}"
        for c in ("ref", "jpeg_q40")
    ]
    control_rows.append(f"| {label} | {cells[0]} | {cells[1]} |")

border_rows = []
for rep in REPS:
    masked = border["representations"][rep]["metrics"]["ref"]
    full = probe_compression["representations"][rep]["metrics"]["ref"]
    shuffled = max(s["attribution_ba"] for s in border["representations"][rep]["shuffled_labels"])
    label = LABELS[rep]
    border_rows.append(
        f"| {label} | {percentage(full['attribution_ba'])} | {percentage(masked['attribution_ba'])} | "
        f"{percentage(shuffled)} | {percentage(masked['detection_ba'])} |"
    )
meta = metadata_probe["metrics"]
meta_shuffled = max(s["attribution_ba"] for s in metadata_probe["shuffled_labels"])
border_rows.append(
    f"| Metadata/geometry (C3b, video level) | — | {percentage(meta['attribution_ba'])} | "
    f"{percentage(meta_shuffled)} | {percentage(meta['detection_ba'])} |"
)

parts = decomposition["models"]
decomposition_rows = []
for label, getter in (
    ("Reference-trained RGB (seed mean)",
     lambda c, k: seed_mean(lambda s: parts[f"rgb_reference_seed{s}"][c][k])),
    ("JPEG-augmented RGB (seed mean)",
     lambda c, k: seed_mean(lambda s: parts[f"rgb_augmented_seed{s}"][c][k])),
    ("Reference-trained DINOv2 probe", lambda c, k: parts["reference_probe_dinov2"][c][k]),
):
    cells = []
    for c in ("ref", "jpeg_q40"):
        cells += [percentage(getter(c, k)) for k in ("family_ba", "within_swap_ba", "within_reenactment_ba")]
    decomposition_rows.append(f"| {label} | " + " | ".join(cells) + " |")
quadruplets = decomposition["design_check"]

# Registered held-out test evidence (gates G2-G4), exported to its own aggregate.
test_hashes = {}


def read_test(relative):
    path = ROOT / relative
    test_hashes[relative] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


freeze_manifest = read_test("research/model_freeze_manifest.json")
manifest_sha = test_hashes["research/model_freeze_manifest.json"]
test_analysis = read_test("experiments/results/test_analysis.json")
test_reports = {kind: read_test(f"experiments/runs/c23_test_{kind}/report.json") for kind in ("rgb", "probes")}
verification = {kind: read_test(f"experiments/results/scorer_verification_{kind}_validation.json")
                for kind in ("rgb", "probes")}
verification_attempt1 = read_test("experiments/results/scorer_verification_probes_validation_attempt1.json")
dry_run = read_test("experiments/results/test_analysis_dryrun_validation.json")
assert test_analysis["status"] == "registered_test_analysis"
assert test_analysis["provenance"]["model_freeze_manifest_sha256"] == manifest_sha
assert all(r["model_freeze_manifest_sha256"] == manifest_sha and r["all_passed"] for r in test_reports.values())
assert all(r["all_passed"] for r in verification.values()) and not verification_attempt1["all_passed"]
assert dry_run["validation_cross_check"]["max_abs_difference"] < 1e-10
register_text = (ROOT / "research/comparison_register.md").read_text(encoding="utf-8")
assert manifest_sha in register_text and "Amendment 1" in register_text

test_snapshot = {
    "schema_version": 1,
    "as_of": DATE,
    "evidence_scope": "Registered held-out test results (register gates G2-G4); scored once after the G3 model freeze",
    "register": {
        "file": "research/comparison_register.md",
        "v1_sha256": freeze_manifest["register_v1_sha256"],
        "current_sha256": sha256(ROOT / "research/comparison_register.md"),
        "amendments": ["Amendment 1 — 24 October 2026 — before test scoring — model freeze (gate G3)"],
    },
    "model_freeze_manifest_sha256": manifest_sha,
    "g2_scorer_verification": {
        kind: {"all_passed": report["all_passed"], "tolerance": report["tolerance"],
               "max_abs_probability_difference": max(m["max_abs_probability_difference"]
                                                     for m in report["models"].values()),
               "threshold_boundary_videos": [dict(b, model=k) for k, m in report["models"].items()
                                             for b in m.get("threshold_boundary_videos", [])],
               "script_sha256": report["script_sha256"]}
        for kind, report in verification.items()
    },
    "g2_probe_attempt1": {"all_passed": False,
                          "failed_models": [k for k, m in verification_attempt1["models"].items() if not m["passed"]],
                          "note": verification["probes"]["verification_note"]},
    "analysis_dry_run_on_validation": dry_run["validation_cross_check"],
    "test_scoring": {
        kind: {key: report[key] for key in ("status", "conditions", "manifest_audit", "manifest_sha256",
                                             "batch", "script_sha256", "elapsed_seconds", "peak_reserved_bytes")}
        | {"test_predictions_sha256": {k: m["test_predictions_sha256"] for k, m in report["models"].items()}}
        for kind, report in test_reports.items()
    },
    "analysis": test_analysis,
    "input_artifact_sha256": test_hashes,
    "qualification": (
        "Aggregate metrics and provenance only; raw test predictions remain local. Intervals condition on "
        "validation-selected models and thresholds and omit selection and training-seed uncertainty. "
        "Additional JPEG compression of C23-derived crops; closed-set attribution among four FF++ methods. "
        "A lower score does not show that information was physically or irreversibly erased."
    ),
}
test_text = json.dumps(test_snapshot, indent=2, ensure_ascii=False, allow_nan=False)
assert len(test_text.encode("utf-8")) < 3 * 1024 * 1024
write("research/main_results_test.json", test_text)

TEST_COLUMNS = ("ref", "jpeg_q95", "jpeg_q80", "jpeg_q60", "jpeg_q40", "jpeg_q20")
COLUMN_LABELS = "| Ref | Q95 | Q80 | Q60 | Q40 | Q20 |"
seed_summary = test_analysis["exploratory"]["seed_summary"]
intervals = test_analysis["model_intervals"]
primary = test_analysis["primary_contrast"]
secondary = test_analysis["secondary"]


def ci(entry, scale=100, digits=1):
    if entry is None or entry["point"] is None:
        return "undefined"
    point = f"{scale * entry['point']:.{digits}f}"
    if entry["ci95"] is None:
        return f"{point} (no interval)"
    low, high = entry["ci95"]
    return f"{point} [{scale * low:.{digits}f}, {scale * high:.{digits}f}]"


def mean_sd(family, condition, metric):
    value = seed_summary[family][condition][metric]
    return f"{percentage(value['mean'])} ± {percentage(value['sample_sd'])}"


curve_rows = []
for metric, metric_label in (("detection_ba", "Detection"), ("attribution_ba", "Attribution")):
    for family, label in (("F1", "Reference training"), ("F2", "JPEG augmentation"), ("F3", "Paired, weight 1")):
        cells = " | ".join(mean_sd(family, c, metric) for c in TEST_COLUMNS)
        curve_rows.append(f"| {metric_label}: {label} | {cells} |")

p1_rows = []
for seed in SEEDS:
    name = f"F1_s{seed}"
    q20 = intervals[name]["jpeg_q20"]
    p1_rows.append(f"| {seed} | {ci(q20['detection_retention'], 1, 2)} | {ci(q20['attribution_retention'], 1, 2)} | "
                   f"{ci(primary['P1_RGB']['per_seed'][name], 1, 2)} |")
probe_p1_rows = []
for rep in REPS:
    entry = primary["P1_PROBE"]["per_probe"][rep]
    ref = intervals[f"F4_{rep}"]["ref"]
    label = LABELS[rep]
    probe_p1_rows.append(
        f"| {label} | {percentage(ref['detection_ba']['point'])} | {percentage(ref['attribution_ba']['point'])} | "
        f"{ci(entry, 1, 2)} | {entry['bootstrap_p']:.3f} | {entry['holm']['holm_bound']:.4f} | "
        f"{'yes' if entry['holm']['rejected'] else 'no'} |")
points_test = test_analysis["exploratory"]["point_metrics"]
coverage = [points_test[f"F1_s{s}"]["jpeg_q20"]["fake_detection_coverage"] for s in SEEDS]
auroc = {c: seed_mean(lambda s: points_test[f"F1_s{s}"][c]["detection_auc"]) for c in ("ref", "jpeg_q20")}
s1_rows = [f"| {seed} | " + " | ".join(ci(secondary["S1_delta_other_qualities"][c][f"F1_s{seed}"], 1, 2)
                                       for c in ("jpeg_q95", "jpeg_q60")) + " |" for seed in SEEDS]
s2 = secondary["S2_training"]["per_seed"]
s2_rows = [
    f"| {seed} | {ci(s2[f'seed{seed}']['F3_minus_F2']['jpeg_q20']['attribution_ba'])} | "
    f"{ci(s2[f'seed{seed}']['F3_minus_F2']['ref']['attribution_ba'])} | "
    f"{ci(s2[f'seed{seed}']['F3_minus_F2']['jpeg_q20']['detection_ba'])} | "
    f"{ci(s2[f'seed{seed}']['F2_minus_F1']['jpeg_q20']['attribution_ba'])} | "
    f"{ci(s2[f'seed{seed}']['F2_minus_F1']['jpeg_q20']['detection_ba'])} |"
    for seed in SEEDS
]
s3 = secondary["S3_known_quality_adaptation"]
s3_rows = []
for rep in REPS:
    label = LABELS[rep]
    q80, q40 = s3[f"F5_minus_F4_{rep}_jpeg_q80"], s3[f"F6_minus_F4_{rep}_jpeg_q40"]
    s3_rows.append(f"| {label} | {ci(q80['attribution_ba'])} | {100 * q80['auroc_point_difference']:+.1f} | "
                   f"{ci(q40['attribution_ba'])} | {100 * q40['auroc_point_difference']:+.1f} |")
s4 = secondary["S4_task_sharing"]
s4_rows = [
    f"| {seed} | {ci(s4[f'seed{seed}']['C1-DET_minus_F1_detection_ba']['ref'])} | "
    f"{ci(s4[f'seed{seed}']['C1-DET_minus_F1_detection_ba']['jpeg_q20'])} | "
    f"{ci(s4[f'seed{seed}']['C1-ATT_minus_F1_attribution_ba']['ref'])} | "
    f"{ci(s4[f'seed{seed}']['C1-ATT_minus_F1_attribution_ba']['jpeg_q20'])} | "
    f"{ci(s4[f'seed{seed}']['separate_minus_shared_delta'], 1, 2)} |"
    for seed in SEEDS
]
s5 = secondary["S5_source_supervision"]
s5_rows = [
    f"| {seed} | {ci(s5[f'seed{seed}']['C2-R_minus_F3-R_attribution_ba']['ref'])} | "
    f"{ci(s5[f'seed{seed}']['C2-R_minus_F3-R_attribution_ba']['jpeg_q20'])} | "
    f"{ci(s5[f'seed{seed}']['C2-BIN_minus_F3_detection_ba']['ref'])} | "
    f"{ci(s5[f'seed{seed}']['C2-BIN_minus_F3_detection_ba']['jpeg_q20'])} |"
    for seed in SEEDS
]
s6 = secondary["S6_shortcut_and_content"]
shortcut_rows = []
for rep in REPS:
    label = LABELS[rep]
    entry = s6["C3a_border"][rep]
    flags = [k.split("_")[0] for k, v in entry["material_flag_ref"].items() if v]
    shortcut_rows.append(f"| C3a face-masked {label} (ref) | {ci(entry['ref']['attribution_ba'])} | "
                         f"{ci(entry['ref']['detection_ba'])} | {', '.join(flags) or 'none'} |")
meta_flags = [k.split("_")[0] for k, v in s6["C3b_metadata"]["material_flag"].items() if v]
shortcut_rows.append(f"| C3b metadata/geometry (video level) | {ci(s6['C3b_metadata']['attribution_ba'])} | "
                     f"{ci(s6['C3b_metadata']['detection_ba'])} | {', '.join(meta_flags) or 'none'} |")
c3c_rows = []
for family, label in (("F1", "Reference-trained RGB"), ("F2", "JPEG-augmented RGB"), ("C1-ATT", "Fake-only RGB (C1-ATT)")):
    cells = []
    for c in ("ref", "jpeg_q60", "jpeg_q20"):
        cells += [percentage(seed_mean(lambda s: s6["C3c"][f"{family}_s{s}"][c][k]["point"]))
                  for k in ("family_ba", "within_swap_ba", "within_reenactment_ba")]
    c3c_rows.append(f"| {label} (seed mean) | " + " | ".join(cells) + " |")
s7 = secondary["S7_support_sensitivity"]["status_changes"]
s7_exclusion = sum(((c["full"]["ci95"] is not None and (c["full"]["ci95"][0] > 0 or c["full"]["ci95"][1] < 0))
                    != (c["subset"]["ci95"] is not None and (c["subset"]["ci95"][0] > 0 or c["subset"]["ci95"][1] < 0)))
                   for c in s7)
distortion_rows = [
    f"| Q{c[6:]} | {v['mean_jpeg_bytes']:.0f} | {v['mean_rgb_psnr_db']:.2f} | {v['mean_luma_ssim']:.4f} |"
    for c, v in test_analysis["exploratory"]["distortion"].items()
]
test_support = test_analysis["support"]
holm_supported = [rep for rep, e in primary["P1_PROBE"]["per_probe"].items() if e["holm"]["rejected"]]
undefined_probes = [rep for rep, e in primary["P1_PROBE"]["per_probe"].items() if e["point"] is None]
unavailable_probes = [rep for rep, e in primary["P1_PROBE"]["per_probe"].items()
                      if e["point"] is not None and not e["interval_available"]]


def lower(entry):
    return entry["ci95"][0] if entry["ci95"] else None


def upper(entry):
    return entry["ci95"][1] if entry["ci95"] else None


# Prose below states these patterns; check them against the test record.
assert all(lower(s2[f"seed{s}"]["F2_minus_F1"]["jpeg_q20"][m]) > 0 for s in SEEDS for m in ("detection_ba", "attribution_ba"))
assert all(lower(s4[f"seed{s}"]["C1-ATT_minus_F1_attribution_ba"]["jpeg_q20"]) > 0 for s in SEEDS)
assert all(lower(s4[f"seed{s}"]["separate_minus_shared_delta"]) > 0 for s in SEEDS)
assert all(upper(s5[f"seed{s}"]["C2-R_minus_F3-R_attribution_ba"][c]) < 0 for s in SEEDS for c in ("ref", "jpeg_q20"))
assert all(upper(s5[f"seed{s}"]["C2-BIN_minus_F3_detection_ba"]["ref"]) < 0 for s in SEEDS)
c2_ref = [-100 * s5[f"seed{s}"]["C2-R_minus_F3-R_attribution_ba"]["ref"]["point"] for s in SEEDS]
c2_q20 = [-100 * s5[f"seed{s}"]["C2-R_minus_F3-R_attribution_ba"]["jpeg_q20"]["point"] for s in SEEDS]
c2_det = [-100 * s5[f"seed{s}"]["C2-BIN_minus_F3_detection_ba"]["ref"]["point"] for s in SEEDS]
c2_det_q20 = [s for s in SEEDS if upper(s5[f"seed{s}"]["C2-BIN_minus_F3_detection_ba"]["jpeg_q20"]) < 0]
c1_det_q20 = {s: s4[f"seed{s}"]["C1-DET_minus_F1_detection_ba"]["jpeg_q20"] for s in SEEDS}
c1_det_worse = [s for s, e in c1_det_q20.items() if upper(e) < 0]
c1_det_better = [s for s, e in c1_det_q20.items() if lower(e) > 0]
dct_gain = [100 * s3[f"{a}_minus_F4_dct_{c}"]["attribution_ba"]["point"] for a, c in (("F5", "jpeg_q80"), ("F6", "jpeg_q40"))]
q40_gains = {rep: s3[f"F6_minus_F4_{rep}_jpeg_q40"]["attribution_ba"] for rep in REPS if rep != "dct"}
q40_recovered = [rep for rep, e in q40_gains.items() if lower(e) > 0]
auroc_retention = (auroc["jpeg_q20"] - 0.5) / (auroc["ref"] - 0.5)
attr_retention = seed_mean(lambda s: intervals[f"F1_s{s}"]["jpeg_q20"]["attribution_retention"]["point"])
det_retention = seed_mean(lambda s: intervals[f"F1_s{s}"]["jpeg_q20"]["detection_retention"]["point"])
names = {"rgb": "pooled RGB", "dinov2": "DINOv2", "fft": "FFT", "dct": "DCT", "wavelet": "wavelet", "residual": "residual"}


def join(items):
    items = [str(i) for i in items]
    return " and ".join([", ".join(items[:-1]), items[-1]]) if len(items) > 1 else (items[0] if items else "none")


def seeds_text(seeds):
    return ("seed " if len(seeds) == 1 else "seeds ") + join(seeds) if seeds else "no seed"

# Post-freeze extensions (Amendments 2-3), exported separately and labelled as such.
ext_hashes = {}


def read_ext(relative):
    path = ROOT / relative
    ext_hashes[relative] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


ext_manifest_sha = sha256(ROOT / "research/extension_freeze_manifest.json")
ext = read_ext("experiments/results/ext_analysis_test.json")
ext_reports = {name: read_ext(f"experiments/runs/{name}/report.json")
               for name in ("c23_ext_test_jpeg", "c23_ext_test_codec")}
codec_report = read_ext("experiments/data/c23_codec_test/report.json")
ext_verification = read_ext("experiments/results/ext_scorer_verification_validation.json")
assert ext["provenance"]["extension_freeze_manifest_sha256"] == ext_manifest_sha
assert ext["bootstrap"]["registered_F1_intervals_reproduced_max_abs_difference"] < 1e-10
assert ext_verification["all_passed"] and ext_manifest_sha in register_text and "Amendment 3" in register_text
assert codec_report["reference_crops_reproduced"] == 5581 and codec_report["frame_count_preserved"]
clip_sha = sha256(ROOT / "research/clip_freeze_manifest.json")
clip = read_ext("experiments/results/ext_clip_analysis_test.json")
clip_verification = read_ext("experiments/results/ext_clip_verification_validation.json")
assert clip["provenance"]["clip_freeze_manifest_sha256"] == clip_sha and clip_sha in register_text
assert clip_verification["all_passed"] and clip["registered_F4_dinov2_reproduced_max_abs_difference"] < 1e-10
ext_snapshot = {
    "schema_version": 1, "as_of": "2026-10-25",
    "evidence_scope": ("Post-freeze extensions X1-X7 (register Amendments 2-5), designed after the registered "
                       "test results were seen; validation-selected; scored once. Not registered results."),
    "extension_freeze_manifest_sha256": ext_manifest_sha,
    "verification": {"all_passed": ext_verification["all_passed"],
                     "max_abs_probability_difference": max(m["max_abs_probability_difference"]
                                                          for m in ext_verification["models"].values())},
    "codec_extraction": {k: codec_report[k] for k in ("settings", "encoder_options", "pyav",
                                                      "reference_crops_reproduced", "frame_count_preserved",
                                                      "crop_rule", "elapsed_hours")},
    "test_scoring": {name: {"elapsed_seconds": r["elapsed_seconds"],
                            "test_predictions_sha256": {k: m["test_predictions_sha256"] for k, m in r["models"].items()}}
                     for name, r in ext_reports.items()},
    "analysis": ext,
    "X7_clip": {"clip_freeze_manifest_sha256": clip_sha, "verification": clip_verification, "analysis": clip},
    "input_artifact_sha256": ext_hashes,
    "disclosed_deviation": ("Two copies of the EfficientNet-B4 F2 seed-43 run overlapped for under five minutes "
                            "on 24 October; both were stopped before any epoch and the run was restarted alone."),
}
write("research/main_results_extensions.json",
      json.dumps(ext_snapshot, indent=2, ensure_ascii=False, allow_nan=False))

x1a, x1b_ext, x2_ext = ext["X1a_threshold_free"], ext["X1b_per_quality_calibration"]["models"], ext["X2_q20_readout"]
op_rows = []
for s in SEEDS:
    a = x1a[f"F1_s{s}"]["jpeg_q20"]
    op_rows.append(
        f"| {s} | {ci(intervals[f'F1_s{s}']['jpeg_q20']['detection_ba'])} | "
        f"{ci(x1b_ext[f'F1-CAL_s{s}']['jpeg_q20']['detection_ba'])} | {ci(a['detection_auc'])} | {ci(a['attribution_auc'])} | "
        f"{ci(primary['P1_RGB']['per_seed'][f'F1_s{s}'], 1, 2)} | "
        f"{ci(x1b_ext[f'F1-CAL_s{s}']['jpeg_q20']['attribution_minus_detection_retention'], 1, 2)} | "
        f"{ci(a['delta_auc'], 1, 2)} |")
eb4_auc_rows = [f"| EB4-F1 seed {s} | {ci(x1a[f'EB4-F1_s{s}']['jpeg_q20']['detection_auc'])} | "
                f"{ci(x1a[f'EB4-F1_s{s}']['jpeg_q20']['attribution_auc'])} | {ci(x1a[f'EB4-F1_s{s}']['jpeg_q20']['delta_auc'], 1, 2)} |"
                for s in SEEDS]
x2_rows = [f"| {s} | {ci(intervals[f'F1_s{s}']['jpeg_q20']['detection_ba'])} | {ci(x2_ext[f'seed{s}']['readout']['jpeg_q20']['detection_ba'])} | "
           f"{ci(intervals[f'F1_s{s}']['jpeg_q20']['attribution_ba'])} | {ci(x2_ext[f'seed{s}']['readout']['jpeg_q20']['attribution_ba'])} |"
           for s in SEEDS]
x3 = ext["X3_seed_pooled"]
x3_rows = [f"| {label} | {ci(x3[key], *scale)} |" for label, key, scale in (
    ("F1 Δ(Q20)", "F1_delta_q20", (1, 2)), ("EfficientNet-B4 F1 Δ(Q20)", "EB4F1_delta_q20", (1, 2)),
    ("F3 − F2 attribution BA, Q20 (points)", "F3_minus_F2_attribution_ba_q20", (100, 1)),
    ("F3 − F2 detection BA, Q20 (points)", "F3_minus_F2_detection_ba_q20", (100, 1)),
    ("F2 − F1 attribution BA, Q20 (points)", "F2_minus_F1_attribution_ba_q20", (100, 1)),
    ("F2 − F1 detection BA, Q20 (points)", "F2_minus_F1_detection_ba_q20", (100, 1)),
    ("EB4 F2 − F1 attribution BA, Q20 (points)", "EB4F2_minus_EB4F1_attribution_ba_q20", (100, 1)),
    ("EB4 F2 − F1 detection BA, Q20 (points)", "EB4F2_minus_EB4F1_detection_ba_q20", (100, 1)))]
CODEC_COLUMNS = ("h264_crf18", "h264_crf23", "h264_crf32", "h264_crf40", "h265_crf23", "h265_crf32")
codec_models = ext["X5_codec"]["models"]


def codec_mean(family, condition, metric):
    return percentage(seed_mean(lambda s: codec_models[f"{family}_s{s}"][condition][metric]["point"]))


codec_rows = [f"| {metric_label}: {label} | " + " | ".join(codec_mean(family, c, metric) for c in CODEC_COLUMNS) + " |"
              for metric, metric_label in (("detection_ba", "Detection"), ("attribution_ba", "Attribution"))
              for family, label in (("F1", "Reference training"), ("F2", "JPEG augmentation"), ("F3", "Paired"),
                                    ("EB4-F1", "EfficientNet-B4, reference"), ("EB4-F2", "EfficientNet-B4, augmentation"))]
dist = ext["X5_codec"]["distortion"]
codec_rows.append("| Mean crop PSNR vs. reference (dB) | " + " | ".join(f"{dist[c]['mean_crop_psnr_db']:.1f}" for c in CODEC_COLUMNS) + " |")
codec_rows.append("| Mean bitrate (kbps) | " + " | ".join(f"{dist[c]['mean_kbps']:.0f}" for c in CODEC_COLUMNS) + " |")
codec_delta_rows = [f"| {s} | " + " | ".join(ci(ext["X5_codec"]["F1_delta"][c][f"F1_s{s}"], 1, 2) for c in CODEC_COLUMNS) + " |"
                    for s in SEEDS]
eb4_curve = ext["X6_efficientnet_b4"]["curves"]
eb4_rows = [f"| {metric_label}: EfficientNet-B4, {label} | " + " | ".join(
    percentage(seed_mean(lambda s: eb4_curve[f"{family}_s{s}"][c][metric]["point"])) for c in TEST_COLUMNS) + " |"
    for metric, metric_label in (("detection_ba", "Detection"), ("attribution_ba", "Attribution"))
    for family, label in (("EB4-F1", "reference"), ("EB4-F2", "augmentation"))]
eb4_p1 = ext["X6_efficientnet_b4"]["P1_rule_at_q20"]
# Checks behind the extension prose.
assert all(lower(x1a[f"F1_s{s}"]["jpeg_q20"]["delta_auc"]) > 0 for s in SEEDS)
assert all(lower(x1a[f"EB4-F1_s{s}"]["jpeg_q20"]["delta_auc"]) > 0 for s in SEEDS)
assert ext["X1b_per_quality_calibration"]["F1_q20_reading_descriptive"] == "attribution retained more"
assert all(lower(x2_ext[f"seed{s}"]["readout_minus_native_q20"][m]) > 0 for s in SEEDS for m in ("detection_ba", "attribution_ba"))
assert eb4_p1["reading_descriptive"] == "attribution retained more"
crf40 = [ext["X5_codec"]["F1_delta"]["h264_crf40"][f"F1_s{s}"] for s in SEEDS]
h265 = [ext["X5_codec"]["F1_delta"][c][f"F1_s{s}"] for c in ("h265_crf23", "h265_crf32") for s in SEEDS]
assert len({np.sign(e["point"]) for e in crf40}) > 1 and all(lower(e) > 0 for e in h265)
assert all(lower(x3[k]) > 0 for k in ("F1_delta_q20", "EB4F1_delta_q20")) and lower(x3["F3_minus_F2_attribution_ba_q20"]) < 0
x4 = ext["X4_metadata_cue"]
clip_curve = clip["curves"]
clip_rows = [f"| {metric_label}: {label} | " + " | ".join(percentage(clip_curve[name][c][metric]["point"]) for c in TEST_COLUMNS) + " |"
             for metric, metric_label in (("detection_ba", "Detection BA"), ("attribution_ba", "Attribution BA"))
             for name, label in (("X7-REF", "CLIP ViT-L/14 probe"), ("F4_dinov2", "DINOv2 probe (F4)"))]
clip_rows.append("| Detection AUROC: CLIP probe | " + " | ".join(percentage(clip["auroc_points"]["X7-REF"][c]) for c in TEST_COLUMNS) + " |")
clip_delta, clip_auc = clip["X7_REF_delta_q20"], clip["X7_REF_delta_auc"]["jpeg_q20"]
clip_adapt = clip["adaptation"]
assert upper(clip_delta) < 0 and lower(clip_auc["delta_auc"]) > 0
assert all(lower(v["attribution_ba"]) > 0 for v in clip_adapt.values())
assert all(lower(v) <= 0 <= upper(v) for c, v in clip["codec_delta"].items() if c.startswith("h264"))
assert upper(x1a["F4_fft"]["jpeg_q20"]["delta_auc"]) < 0 and upper(x1a["F4_dct"]["jpeg_q20"]["delta_auc"]) < 0
assert lower(x1a["F4_dinov2"]["jpeg_q20"]["delta_auc"]) > 0
assert all(lower(x1a[f"F4_{r}"]["jpeg_q20"]["delta_auc"]) <= 0 <= upper(x1a[f"F4_{r}"]["jpeg_q20"]["delta_auc"])
           for r in ("rgb", "wavelet", "residual"))
assert lower(clip["X7_REF_minus_F4_dinov2"]["ref"]["attribution_ba"]) > 0 and upper(clip["X7_REF_minus_F4_dinov2"]["jpeg_q20"]["attribution_ba"]) < 0
x4_ref_zero = all(lower(x4[n]["ref"]["accuracy_meta_correct_minus_wrong"]) <= 0 <= upper(x4[n]["ref"]["accuracy_meta_correct_minus_wrong"])
                  for n in x4 if n.startswith(("F1_", "F2_", "F3_", "C1-ATT", "EB4")))

root_readme = f"""
# What Survives Compression?

Updated {DATE}. **{status_line}**

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
- Complete-eight-frame sensitivity on validation completed; its definition is frozen for test.
- Registered controls completed on validation (gate G1):
  - nine control RGB runs (C1-DET, C1-ATT, C2-BIN at three seeds);
  - the C2 source readout;
  - face-masked border probes, a metadata/geometry probe and a content-matched decomposition.
- [Comparison register](research/comparison_register.md) v{register_freeze["register_version"]}
  frozen {register_freeze["frozen_on"]}, before any classifier test scoring.
- Gates G2–G4 completed on 24 October 2026:
  - G2: all 49 registered models' scorers reproduced their stored validation predictions;
  - G3: model-freeze manifest and register Amendment 1 recorded;
  - G4: one test-scoring pass at six conditions and one registered analysis.

The validation cohort contains 699 videos (559 fake); the test cohort contains
{test_support["full"]} videos in 70 provenance groups.

## Held-out test results (registered analysis, gate G4)

Each registered model was scored once on the official test split at reference and JPEG
Q95/Q80/Q60/Q40/Q20, after the gates above. Q95, Q60 and Q20 played no part in any
selection. Intervals are 95% provenance-group bootstrap intervals (2,000 paired draws,
seed 4242). They condition on the validation-selected models and thresholds and omit
selection and training-seed uncertainty. Full results, including every registered
outcome, are in [main_results_test.json](research/main_results_test.json).

### RGB quality curves

Test balanced accuracy in percent, mean ± sample SD over seeds 42/43/44 (descriptive).
Detection uses each model's frozen validation threshold; attribution is fake-only.

| Model {COLUMN_LABELS}
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(curve_rows)}

### Primary contrast (RQ2): Δ(Q20) = S_attr − S_det for reference-trained models

S is chance-adjusted retention relative to the test reference condition
(1 = no loss, 0 = chance).

**P1-RGB (F1).** Registered reading: **{primary["P1_RGB"]["reading"]}**.
A direction is claimed only when all three seed intervals exclude zero with the same sign.

| Seed | S_det(Q20) | S_attr(Q20) | Δ(Q20) |
|---|---:|---:|---:|
{chr(10).join(p1_rows)}

**P1-PROBE (F4).** Two-sided bootstrap p, with Holm correction over the fixed six probes.

| Probe | Ref detection BA | Ref attribution BA | Δ(Q20) | p | Holm bound | Supported |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(probe_p1_rows)}

{"No probe contrast survives Holm correction." if not holm_supported else "Holm-supported: " + ", ".join(names[r] for r in holm_supported) + "."}
Δ is undefined for {join(names[r] for r in undefined_probes)}: a test reference BA is below chance + 5 points.
Intervals are unavailable for {join(names[r] for r in unavailable_probes)}, because more than 5% of
bootstrap draws had an undefined retention. Several probes are close to chance
even at the reference condition, so they carry little information about retention.

**Exploratory context (not part of the registered contrast).** Detection BA is measured at a
fixed validation operating point, but attribution is a threshold-free argmax. At Q20 the F1 models
flag {join(f"{100 * c:.0f}%" for c in coverage)} of fake videos (seeds 42, 43 and 44). Their mean
detection AUROC falls only from {percentage(auroc["ref"])}% to {percentage(auroc["jpeg_q20"])}%.
Chance-adjusted detection AUROC retention ({auroc_retention:.2f}) is well above the mean fixed-threshold
detection retention ({det_retention:.2f}). Much of the detection loss is therefore a shift of scores past
the validation threshold, not a loss of ranking. The P1-RGB reading concerns detection at the frozen
operating point. Extension X1a below compares the two tasks like for like, both by AUROC. It keeps a
large gap in the same direction. (The AUROC retention above is not directly comparable with the
attribution BA retention of {attr_retention:.2f}.)

**S1: Δ at the other held-out qualities (F1).**

| Seed | Δ(Q95) | Δ(Q60) |
|---|---:|---:|
{chr(10).join(s1_rows)}

### Compression-aware training (RQ3, S2)

Registered reading: **{secondary["S2_training"]["reading"]}**. Paired-minus-augmentation (F3 − F2)
attribution at Q20 does not exclude zero positively in any seed. JPEG augmentation (F2) improves
on reference training (F1) at Q20 in every seed, for both detection and attribution.
Differences are in percentage points of balanced accuracy.

| Seed | F3 − F2 attribution, Q20 | F3 − F2 attribution, ref | F3 − F2 detection, Q20 | F2 − F1 attribution, Q20 | F2 − F1 detection, Q20 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(s2_rows)}

### Known-quality adaptation (S3)

Matched-quality heads minus reference-trained heads, evaluated at the matched quality.
Attribution BA differences have intervals; AUROC differences are point estimates.

| Probe | Attribution, F5 − F4 at Q80 | AUROC Δ at Q80 | Attribution, F6 − F4 at Q40 | AUROC Δ at Q40 |
|---|---:|---:|---:|---:|
{chr(10).join(s3_rows)}

Matched heads recover {dct_gain[0]:.1f} (Q80) and {dct_gain[1]:.1f} (Q40) points of DCT attribution.
At Q40, attribution also improves, with intervals excluding zero, for
{join(names[r] for r in q40_recovered)}. A reference-trained loss therefore partly reflects the
mismatch between training and test compression. It does not show that the information is absent.

### Task sharing (S4) and source supervision (S5)

S4 compares separate single-task models (C1) with the shared F1 model. The last column is the
separate system's Δ(Q20) minus F1's.

| Seed | C1-DET − F1 detection, ref | C1-DET − F1 detection, Q20 | C1-ATT − F1 attribution, ref | C1-ATT − F1 attribution, Q20 | Separate − shared Δ(Q20) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(s4_rows)}

At Q20 the separate fake-only attribution model is better than the shared model in every seed. The
separate detector is worse in {seeds_text(c1_det_worse)} and better in {seeds_text(c1_det_better)}. The separate system's Δ(Q20) is larger than F1's in
every seed, so the P1-RGB direction does not depend on the two tasks sharing capacity.

S5 compares paired training with and without source supervision (C2).

| Seed | C2-R − F3-R attribution, ref | C2-R − F3-R attribution, Q20 | C2-BIN − F3 detection, ref | C2-BIN − F3 detection, Q20 |
|---|---:|---:|---:|---:|
{chr(10).join(s5_rows)}

Without the source loss, the same fixed linear readout recovers {min(c2_ref):.1f}–{max(c2_ref):.1f}
points less attribution at the reference condition and {min(c2_q20):.1f}–{max(c2_q20):.1f} points
less at Q20. Detection is also lower at the reference condition ({min(c2_det):.1f}–{max(c2_det):.1f} points).
At Q20 detection is lower, with an interval excluding zero, only in {seeds_text(c2_det_q20)}.

### Shortcut and content controls (S6, S7)

Test balanced accuracy in percent. A result is flagged "material" when the 95% lower bound is
above 30% for attribution (chance 25%) or above 55% for detection (chance 50%).

| Control | Attribution BA | Detection BA | Material flags |
|---|---:|---:|---|
{chr(10).join(shortcut_rows)}

None of the face-masked probes is flagged. The video-level metadata/geometry probe is flagged for
**{", ".join(meta_flags) or "nothing"}**: file size, frame count and face-box geometry predict
the manipulation method above chance. Under the register, full-crop attribution findings carry
this dataset-level caveat. It does not show that the crop models use that cue.

C3c content-matched decomposition (chance 50%): family BA (identity swap vs. reenactment) and the
two within-family BAs, at ref, Q60 and Q20.

| Model | Family | DF vs FS | F2F vs NT | Family Q60 | DF vs FS Q60 | F2F vs NT Q60 | Family Q20 | DF vs FS Q20 | F2F vs NT Q20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(c3c_rows)}

S7 (complete-eight supports) flagged {len(s7)} status changes. {"None" if s7_exclusion == 0 else s7_exclusion} of them
change{"s" if s7_exclusion == 0 else ""} whether an interval excludes zero; {"all" if s7_exclusion == 0 else "the rest"} are sign changes of points near zero. The support set
is {test_support["complete_video"]} complete videos and {test_support["complete_groups"]} complete groups.

Exploratory test-crop distortion, averaged over the {test_analysis["exploratory"]["distortion"]["jpeg_q20"]["crops"]:,} test crops:

| JPEG | Mean bytes | RGB PSNR (dB) | Luma SSIM |
|---|---:|---:|---:|
{chr(10).join(distortion_rows)}

## Post-freeze extensions (register Amendments 2–3)

These analyses were designed **after** the registered test results were seen. Each was pre-registered
in Amendment 2 before it ran, selected on validation only, frozen in Amendment 3 (all nine new models
reproduced their validation predictions exactly) and scored once. They are not registered results and
do not change the readings above. Full results:
[main_results_extensions.json](research/main_results_extensions.json).

### Operating point (X1, X2, X3)

F1 at Q20. Detection BA is shown at the frozen threshold and at a per-quality threshold chosen on
validation at Q20 (X1b). The AUROC columns score both tasks without a threshold; attribution AUROC is
the mean one-vs-rest AUROC over the four methods on fake videos (X1a). The last three columns are Δ
under the registered metric, with per-quality thresholds, and with AUROC for both tasks.

| Seed | Detection BA, frozen | Detection BA, Q20 threshold | Detection AUROC | Attribution AUROC | Δ (registered) | Δ (Q20 threshold) | Δ (AUROC) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(op_rows)}

The direction survives both checks. Recalibrating the detection threshold recovers part of the
detection loss. Comparing AUROC with AUROC still shows attribution retained far better: at Q20,
attribution AUROC stays about 94–96% while detection AUROC falls to about 73–80%.

| EfficientNet-B4, Q20 | Detection AUROC | Attribution AUROC | Δ (AUROC) |
|---|---:|---:|---:|
{chr(10).join(eb4_auc_rows)}

**X2, frozen-encoder Q20 readout.** A linear readout refitted at Q20 on the frozen reference-trained
F1 encoder recovers most of the loss:

| Seed | Native detection BA | Readout detection BA | Native attribution BA | Readout attribution BA |
|---|---:|---:|---:|---:|
{chr(10).join(x2_rows)}

The Q20 information is still linearly available in the encoder; the reference-trained heads no longer
read it. This is evidence against irreversible loss for this model, not proof of physical presence.

**X3, intervals pooled over seeds** (training seeds and groups both resampled):

| Contrast | Pooled estimate [95% interval] |
|---|---:|
{chr(10).join(x3_rows)}

### Whole-video H.264/H.265 re-encoding (X5)

Each test video was re-encoded in memory at six settings. The scheduled frames were re-cropped with
the stored boxes; every reference crop was first reproduced pixel for pixel. Frozen models and
thresholds were used. Values are test BA in percent, seed means.

| Model | H.264 CRF 18 | H.264 CRF 23 | H.264 CRF 32 | H.264 CRF 40 | H.265 CRF 23 | H.265 CRF 32 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(codec_rows)}

F1 Δ (attribution minus detection retention) per seed:

| Seed | H.264 CRF 18 | H.264 CRF 23 | H.264 CRF 32 | H.264 CRF 40 | H.265 CRF 23 | H.265 CRF 32 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(codec_delta_rows)}

- **H.265:** the JPEG pattern repeats (Δ > 0 in every seed).
- **H.264 at CRF 40, the harshest setting:** attribution falls as far as detection, and Δ changes sign
  across seeds. The direction is not established there.
- **Augmentation does not transfer to H.264 detection:** under H.264 CRF 32/40, JPEG augmentation (F2) has
  lower seed-mean detection BA than reference training (at CRF 40, lower with intervals excluding zero in two
  of three seeds), although it still helps attribution. EfficientNet-B4 augmentation shows no detection gain there.
- Equal CRF across codecs is not equal distortion; compare by PSNR and bitrate.

### EfficientNet-B4 replication (X6)

| Model {COLUMN_LABELS}
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(eb4_rows)}

Under the P1-RGB rule, applied descriptively, the reading for EfficientNet-B4 is
**{eb4_p1["reading_descriptive"]}** (Δ(Q20) {", ".join(ci(e, 1, 2) for e in eb4_p1["per_seed"].values())}).
JPEG augmentation again gives large Q20 gains, so the main findings are not specific to ResNet-18.

### Metadata cue (X4, exploratory)

{"At the reference condition, no crop model's attribution accuracy differs between videos where the metadata probe is right and where it is wrong (every interval includes zero)" if x4_ref_zero else "Some crop models' attribution accuracy differs by metadata-probe correctness at the reference condition"}.
Re-weighting test videos to balance the metadata cue changes attribution BA by under one point. This
does not suggest the crop models rely on the cue at the reference condition. At Q20 the intervals are wide.

### CLIP ViT-L/14 probes (X7, Amendments 4–5)

Frozen OpenAI CLIP ViT-L/14 image embeddings with ridge heads, using the F4/F5/F6 procedures. CLIP ran
deterministically from its TorchScript checkpoint with no added packages, and re-extracting validation
reproduced every stored probability exactly. Test balanced accuracy and AUROC in percent:

| Probe {COLUMN_LABELS}
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(clip_rows)}

- **Strongest frozen probe at reference.** CLIP exceeds DINOv2 by
  {ci(clip["X7_REF_minus_F4_dinov2"]["ref"]["attribution_ba"])} points of attribution BA.
- **Opposite sign under JPEG at the operating point.** CLIP's argmax attribution collapses at Q20, giving
  Δ(Q20) = {ci(clip_delta, 1, 2)} (bootstrap p < 0.001; descriptive, outside the Holm family). That is the
  reverse of the RGB models.
- **Positive again without a threshold.** Attribution AUROC ({ci(clip_auc["attribution_auc"])}) still exceeds
  detection AUROC ({ci(clip_auc["detection_auc"])}), so Δ_AUC(Q20) = {ci(clip_auc["delta_auc"], 1, 2)}.
  The four methods remain separable; the reference-trained head's decisions drift.
- **Calibration again at Q95.** CLIP detection BA at Q95 is lower than at Q80 while its AUROC barely
  changes, so the frozen threshold misfires at a quality it never saw.
- **Matched heads recover attribution:** {ci(clip_adapt["X7-Q80_minus_X7-REF_jpeg_q80"]["attribution_ba"])}
  points at Q80 and {ci(clip_adapt["X7-Q40_minus_X7-REF_jpeg_q40"]["attribution_ba"])} at Q40.
- **Codecs:** Δ is near zero at every H.264 setting and positive at H.265 CRF 32
  ({ci(clip["codec_delta"]["h265_crf32"], 1, 2)}).

The sign of a fixed-operating-point detection/attribution contrast therefore depends on the representation.
The threshold-free contrast Δ_AUC(Q20) also depends on it:

- **Attribution retained better** for both trained CNNs, DINOv2 ({ci(x1a["F4_dinov2"]["jpeg_q20"]["delta_auc"], 1, 2)})
  and CLIP.
- **Near zero** for the pooled-RGB, wavelet and residual descriptors.
- **Attribution retained worse** for the FFT ({ci(x1a["F4_fft"]["jpeg_q20"]["delta_auc"], 1, 2)}) and DCT
  ({ci(x1a["F4_dct"]["jpeg_q20"]["delta_auc"], 1, 2)}) descriptors. Their method cues appear tied to
  compression-sensitive frequency statistics, consistent with DCT's recovery under matched heads.

No single direction holds across representations.

## Validation: main RGB results at JPEG Q40

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

## Validation: reference-trained representation probes at Q40

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

## Validation: known-quality adaptation control at {adaptation_title}

At each quality, new scaling and ridge heads were fitted on training crops
compressed at that quality and selected using validation at the same quality.
Values below are percentages.

{adaptation_header}
{adaptation_rule}
{chr(10).join(adaptation_rows)}
{adaptation_note}
DCT's partial recovery supports training/compression mismatch as a
contributor to its earlier failure. These results demonstrate usable label
information on this validation cohort; they do not establish a physical
forensic fingerprint. Threshold selection domains differ between the two
regimes, so AUROC and fake-only attribution are the clearest comparisons.

## Complete-eight-frame sensitivity (validation)

Partial videos were re-examined with frozen thresholds and selections.
On `complete_video` support ({eight["support"]["complete_video"]["videos"]} of
{eight["support"]["full"]["videos"]} videos), the largest change in any model's
detection or attribution balanced accuracy is
{percentage(eight["max_abs_ba_shift_vs_full"]["complete_video"])} points.
On `complete_group` support ({eight["support"]["complete_group"]["groups"]} of 70 groups),
it is {percentage(eight["max_abs_ba_shift_vs_full"]["complete_group"])} points.
RGB paired contrasts changing status: {changes["complete_video"]} on complete-video support and
{changes["complete_group"]} on complete-group support. Every changed contrast is near zero, and its
interval includes or borders zero on both supports.
The Q40 conclusions above are unchanged.

## Registered controls on validation (gate G1)

These controls are registered in the [comparison register](research/comparison_register.md).
Their test comparisons are in the held-out test section above. Values here are validation percentages
at frozen thresholds: the seed mean over 42/43/44 for RGB models. They are descriptive,
with no intervals.

| Validation quantity | Reference | Q40 |
|---|---:|---:|
{chr(10).join(control_rows)}

- **C1 (task sharing):** a separate detector is weaker than the shared two-head model,
  especially at Q40. A separate fake-only attribution model is at least as good.
- **C2 (source supervision):** removing the source loss from paired training lowers
  detection. It also lowers the source information that a fixed linear readout can recover.

**C3a and C3b (border and metadata shortcut checks).** Attribution BA has chance 25%,
and the shuffle column is the best of five shuffled-label refits.
Detection BA has chance 50%. All values are at the reference condition.

| Probe | Full crop attribution | Face-masked attribution | Best shuffle | Face-masked detection |
|---|---:|---:|---:|---:|
{chr(10).join(border_rows)}

With the face masked, the probes fall close to chance. Video metadata and
face-box geometry carry a modest cue. The registered "material" flags were
judged on test intervals (see above).

**C3c (content-matched decomposition).** Every fake pair ID has all four methods
({quadruplets["val"]["pairs_with_all_four_usable_videos"]} of
{quadruplets["val"]["fake_pair_ids"]} validation quadruplets fully usable).
Below are family BA (identity swap vs. reenactment) and the two within-family BAs
(chance 50%), first at the reference condition and then at Q40.

| Model | Family | DF vs. FS | F2F vs. NT | Family Q40 | DF vs. FS Q40 | F2F vs. NT Q40 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(decomposition_rows)}

For reference-trained RGB, Deepfakes vs. FaceSwap stays near perfect at Q40.
Its losses come from family separation and from Face2Face vs. NeuralTextures.

## Interpretation and limitations

- Validation results are selection-stage evidence and optimistic. Test results come from one
  registered pass. Their intervals condition on the selected models and thresholds and
  exclude selection and training-seed uncertainty.
- The registered P1-RGB reading is that, at Q20, attribution is retained more than detection.
  This is the reverse of the proposal's expectation that method cues would be more fragile. It
  describes detection at a fixed validation operating point. In the post-freeze extensions, the same
  direction holds for the trained CNNs with per-quality thresholds, with AUROC for both tasks, and for
  EfficientNet-B4. It does **not** generalize everywhere:
  - under H.264 re-encoding at CRF 40, attribution falls as far as detection;
  - the frozen CLIP probe reverses the sign at its operating point;
  - the FFT and DCT descriptors retain attribution worse even without a threshold.

  The defensible claim is representation- and codec-specific.
- No reference-trained probe contrast survives Holm correction, and several probes are close to
  chance even on reference test crops. The probe track does not establish a representation ranking.
- JPEG augmentation is the clear compression-aware gain. Paired training shows no reliable
  benefit over augmentation on test.
- The metadata/geometry probe is material for attribution on test. A dataset-level cue exists, and
  attribution claims carry that caveat.
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

{remaining_list}

Whole-video H.264/H.265 was added as post-freeze extension X5; external datasets are in preparation.

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

- [Registered test results](research/main_results_test.json)
- [Model-freeze manifest](research/model_freeze_manifest.json)
- [Post-freeze extension results](research/main_results_extensions.json), with the
  [extension](research/extension_freeze_manifest.json) and [CLIP](research/clip_freeze_manifest.json) freeze manifests
- [Aggregate validation milestone](research/main_results_validation.json)
- [Current module tracker](research/module_tracker.md)
- [Frozen comparison register](research/comparison_register.md)
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
The aggregate records preserve metrics, confidence intervals,
configuration/resource records and artifact hashes.

Saved hashes describe execution artifact bytes. The snapshot also records
LF-normalized source hashes for comparison with Git's text normalization.
"""
write("README.md", root_readme)

write("research/README.md", f"""
# Research documents and current evidence

Updated {DATE}. {status_line}

The proposal in Mgr.md was evaluated as source material. The literature
review covers its references and qualifies the project's novelty claims.

## Current evidence

The main cohort contains 39,898 crops from 4,999 videos.
Nine RGB runs across three seeds, six reference-trained probes,
validation compression sweeps, frozen operating thresholds and
70-group bootstrap analyses are complete.
Six Q40 adaptation probes are complete; Q80 adaptation: {q80_status.lower()}.

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

Next: {next_step}
""")

write("experiments/README.md", rf"""
# Experiment workflow

Updated {DATE}. {status_line}
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

Next: {next_step}

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
""")

tracker_rows = [
    ("01", "Scope and rules", "Inclusion policy, validation thresholds, comparison register v1.0 (frozen 2026-10-23) and Amendment 1 (G3 model freeze, 2026-10-24) recorded", "Report the amendment in the paper"),
    ("02", "Literature", "Initial review and qualified novelty assessment prepared", "Final refresh and reproduction audits"),
    ("03", "Environment", "Nine measured RGB runs and frozen DINOv2 completed", "RAM peaks and extension budgets"),
    ("04", "Inventory", "5,000 core videos inventoried", "Mirror authenticity limitations in paper"),
    ("05", "Splits", "500 provenance groups; known input-ID separation audited", "Identity/YouTube limitations and final support checks"),
    ("06", "Crops", "39,898 crops; 102 failures; 4,999 usable videos; sampled QA; complete-eight validation sensitivity", "Report test support sensitivity (S7)"),
    ("07", "JPEG", "Ref/Q80/Q40 validation; test at ref/Q95/Q80/Q60/Q40/Q20 with crop PSNR/SSIM", "Final figures"),
    ("08", "Evaluation", "Frozen thresholds, aggregation, retention, gating and bootstrap; registered test analysis completed", "Figures and tables"),
    ("09", "Pilot", "Pilot training, probes and preprocessing checks completed", "Historical evidence retained"),
    ("10", "Representations", "Six reference probes and validation compression analyses completed", "Report P1-PROBE (no Holm-supported contrast) and S3"),
    ("11", "Main JPEG", "Nine RGB runs; Q40 and Q80 adaptation probes; held-out test at six conditions completed", "Paper reporting"),
    ("12", "Video codecs", "Post-freeze extension X5: H.264 CRF 18/23/32/40 and H.265 CRF 23/32 test re-encoding scored with frozen models (2026-10-25)", "Paper reporting as a labelled extension"),
    ("13", "Recompression", "Generator and engineering checks available", "Scientific history controls if retained"),
    ("14", "Paired learning", "Weight-1 candidate at three seeds; C1 and C2 controls; registered test comparisons S2, S4 and S5 completed", "Paper reporting (no reliable paired benefit over augmentation)"),
    ("15", "External validation", "Unstarted", "Optional external studies"),
    ("16", "Statistics/controls", "Validation bootstrap and controls; gates G1-G4 completed; registered test inference recorded", "Paper reporting of every registered outcome"),
    ("17", "Reproducibility", "Scripts, configs, environment, model-freeze manifest and validation/test aggregates prepared", "Figure/table regeneration"),
    ("18", "Paper", "Literature and protocol material available", "Manuscript, figures, numerical checks and revisions"),
]
tracker_table = "\n".join(
    f"| {number} | {module} | {evidence} | {remaining} |"
    for number, module, evidence, remaining in tracker_rows
)
write("research/module_tracker.md", f"""
# Module progress tracker

Updated {DATE}. {status_line}
Status describes completed evidence rather than planned experiments.

| ID | Module | Current evidence | Work remaining |
|---|---|---|---|
{tracker_table}

Validation outcomes are selection-stage evidence. Registered test outcomes come from one
scoring pass (see main_results_test.json). Bootstrap intervals exclude selection
uncertainty; seed SD is descriptive.

Next: {next_step}

See [the test aggregate](main_results_test.json) and [the validation milestone](main_results_validation.json)
for exact metrics, intervals, resource records and artifact hashes.
""")

matrix_path = ROOT / "research/experiment_matrix.md"
old_matrix = matrix_path.read_text(encoding="utf-8-sig")
marker = "\n## Planned families of experiments"
assert old_matrix.count(marker) == 1

matrix_header = f"""
# Experiment register and planned comparisons

Initial design prepared 10 October 2026; execution status updated {DATE}.
{status_line} The family table below retains the initial plan.

## Current execution status

| Family | Current evidence | Remaining |
|---|---|---|
| E00: Pipeline pilot | Completed | Historical evidence retained |
| E01: Reference sensitivity | Six probes and RGB at ref/Q80/Q40 (validation); test at six conditions completed | Paper reporting |
| E02: Known-quality adaptation | Q40 and Q80 probes; registered test comparison S3 completed | Paper reporting |
| E03: Mixed-quality generalization | RGB augmentation at three seeds; tested at held-out Q95/Q60/Q20 | Paper reporting |
| E04: Task-sharing | C1 separate models at three seeds; registered test comparison S4 completed | Paper reporting |
| E05: Video coding | Post-freeze extension X5 on test (Amendments 2-3): H.264 CRF 40 breaks the JPEG direction; H.265 repeats it | Paper reporting as a labelled extension |
| E06: Processing history | Generator and engineering checks | Scientific experiments if retained |
| E07: Paired learning | Weight-1 candidate at three seeds; test S2: no reliable benefit over augmentation | Paper reporting |
| E08: Source supervision | C2 binary-only paired model and readout; registered test comparison S5 completed | Paper reporting |
| E09: Explanatory controls | Validation controls; test S6/S7 completed (C3b flagged material for attribution) | Paper reporting |
| E10: External detection | Unstarted | Optional |
| E11: Generator-family study | Unstarted | Optional separate study |

Models, thresholds, penalties and regimes are recorded in
[main_results_validation.json](main_results_validation.json) and frozen in
[model_freeze_manifest.json](model_freeze_manifest.json). Registered test results are in
[main_results_test.json](main_results_test.json).
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
print("Exported research/main_results_test.json:", (ROOT / "research/main_results_test.json").stat().st_size, "bytes")
print("This export runs no experiments.")
