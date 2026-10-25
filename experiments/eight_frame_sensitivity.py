"""Complete-eight-frame sensitivity on frozen validation predictions.

Support definitions are frozen for the matching test analysis:
- complete_video: videos with all eight scheduled reference crops valid.
- complete_group: provenance groups whose ten core videos all qualify.
Thresholds, checkpoints and probe heads are the frozen full-cohort
validation selections; nothing is reselected on a subset.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from metrics import aggregate, auc
from research_metrics import retention

CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
SEEDS = (42, 43, 44)
MODES = ("reference", "augmented", "paired_w1")
REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")
SUPPORTS = ("full", "complete_video", "complete_group")
SCHEDULED_FRAMES = 8
GROUP_VIDEOS = 10
PAIRS = {
    f"{name}_seed{seed}": (f"c23_main_{left}_seed{seed}", f"c23_main_{right}_seed{seed}")
    for seed in SEEDS
    for name, left, right in (
        ("augmentation_minus_reference", "augmented", "reference"),
        ("pair_w1_minus_augmentation", "paired_w1", "augmented"),
    )
}
OUTPUT = Path("experiments/results/eight_frame_sensitivity_validation.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_support(manifest_rows, split):
    """Return (complete_video ids, complete_group ids) from crop coverage only."""
    counts = Counter(r["video_id"] for r in manifest_rows if r["split"] == split)
    groups = defaultdict(set)
    for row in manifest_rows:
        if row["split"] == split:
            groups[row["group_id"]].add(row["video_id"])
    videos = {v for v, n in counts.items() if n == SCHEDULED_FRAMES}
    assert max(counts.values()) == SCHEDULED_FRAMES
    complete_groups = {
        g for g, members in groups.items()
        if len(members) == GROUP_VIDEOS and members <= videos
    }
    return videos, complete_groups


def measures(det_correct, src_correct, labels, sources, indices, conditions):
    det = det_correct[:, indices]
    src = src_correct[:, indices]
    y, s = labels[indices], sources[indices]
    det_ba = (det[:, y == 0].mean(axis=1) + det[:, y == 1].mean(axis=1)) / 2
    src_ba = np.mean([src[:, s == k].mean(axis=1) for k in range(4)], axis=0)
    result = {}
    for i, condition in enumerate(conditions):
        values = {"detection_ba": float(det_ba[i]), "attribution_ba": float(src_ba[i])}
        if conditions[0] == "ref":
            det_r = retention(values["detection_ba"], float(det_ba[0]), 0.5)
            src_r = retention(values["attribution_ba"], float(src_ba[0]), 0.25)
            values.update({
                "detection_retention": det_r,
                "attribution_retention": src_r,
                "attribution_minus_detection_retention": (
                    src_r - det_r if src_r is not None and det_r is not None else None
                ),
            })
        result[condition] = values
    return result


def difference(left, right):
    return None if left is None or right is None else float(left - right)


def interval(values, point):
    valid = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    usable = point is not None and len(valid) > 0 and len(valid) >= int(np.ceil(0.95 * len(values)))
    return {
        "point": point,
        "ci95": np.quantile(valid, [0.025, 0.975], method="linear").tolist() if usable else None,
        "valid_draws": len(valid),
        "undefined_draws": len(values) - len(valid),
        "interval_available": usable,
    }


def excludes_zero(entry):
    return entry["ci95"] is not None and (entry["ci95"][0] > 0 or entry["ci95"][1] < 0)


def load_models():
    """Frozen validation predictions and thresholds, with hash checks."""
    models = {}
    thresholds_path = Path("experiments/results/main_validation_thresholds_seeds42_44.json")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert thresholds["test_scores_used"] is False
    for seed in SEEDS:
        for mode in MODES:
            name = f"c23_main_{mode}_seed{seed}"
            registered = thresholds["runs"][name]
            path = Path("experiments/runs") / name / "validation_predictions.csv"
            models[name] = (path, registered["validation_predictions_sha256"],
                            registered["threshold"], CONDITIONS, "rgb_resnet18")

    probes_path = Path("experiments/results/c23_main_probe_compression_validation/report.json")
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    assert probes["test_scores_used"] is False
    for rep in REPS:
        registered = probes["representations"][rep]
        models[f"reference_probe_{rep}"] = (
            probes_path.parent / rep / "validation_predictions.csv",
            registered["validation_predictions_sha256"], registered["threshold"],
            CONDITIONS, "reference_trained_probe")

    for quality in (80, 40):
        report_path = Path(f"experiments/runs/c23_main_matched_q{quality}_probes_validation/report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["test_scores_used"] is False
        for rep in REPS:
            registered = report["representations"][rep]
            models[f"matched_q{quality}_probe_{rep}"] = (
                report_path.parent / rep / "validation_predictions.csv",
                registered["validation_predictions_sha256"], registered["threshold"],
                (f"jpeg_q{quality}",), "known_quality_probe")
    inputs = [thresholds_path, probes_path] + [
        Path(f"experiments/runs/c23_main_matched_q{q}_probes_validation/report.json")
        for q in (80, 40)
    ]
    return models, {p.as_posix(): sha256(p) for p in inputs}


def selftest():
    labels = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    sources = np.array([-1, 0, 1, 2, 3] * 2)
    det = np.ones((3, 10), dtype=bool)
    src = np.ones((3, 10), dtype=bool)
    det[2, labels == 1] = False
    src[2] = sources == 0
    full = measures(det, src, labels, sources, np.arange(10), CONDITIONS)
    half = measures(det, src, labels, sources, np.arange(5), CONDITIONS)
    assert full == half
    assert full["jpeg_q40"]["detection_retention"] == 0
    assert full["jpeg_q40"]["attribution_retention"] == 0
    single = measures(det[:1], src[:1], labels, sources, np.arange(10), ("jpeg_q40",))
    assert set(single["jpeg_q40"]) == {"detection_ba", "attribution_ba"}

    rows = []
    for group, video, frames in (("g1", "a", 8), ("g1", "b", 7), ("g2", "c", 8)):
        rows += [{"split": "val", "group_id": group, "video_id": video}] * frames
    videos, _ = complete_support(rows, "val")
    assert videos == {"a", "c"}
    assert excludes_zero({"ci95": [0.1, 0.2]}) and not excludes_zero({"ci95": [-0.1, 0.2]})
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main(draws, seed):
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    assembly = json.loads(Path("experiments/data/c23_main/assembly.json").read_text(encoding="utf-8-sig"))
    manifest = Path("experiments/data/c23_main/reference.csv")
    assert sha256(manifest) == assembly["reference_manifest_sha256"]
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    complete_videos, complete_groups = complete_support(manifest_rows, "val")

    models, input_hashes = load_models()
    canonical, arrays, aucs, kinds, thresholds = None, {}, {}, {}, {}
    for name, (path, expected, threshold, conditions, kind) in models.items():
        if sha256(path) != expected:
            raise ValueError(f"Registered file changed: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            videos = aggregate(list(csv.DictReader(handle)))
        by_condition = {c: {r["video_id"]: r for r in videos if r["condition"] == c} for c in conditions}
        assert {r["condition"] for r in videos} == set(conditions)
        video_ids = sorted(by_condition[conditions[0]])
        metadata = [
            (v, by_condition[conditions[0]][v]["group_id"],
             by_condition[conditions[0]][v]["label"], by_condition[conditions[0]][v]["source_index"])
            for v in video_ids
        ]
        if canonical is None:
            canonical = metadata
        elif metadata != canonical:
            raise ValueError(f"Video support or provenance differs: {name}")
        labels = np.array([m[2] for m in metadata])
        sources = np.array([m[3] for m in metadata])
        det, src, aucs[name] = [], [], {}
        for condition in conditions:
            if set(by_condition[condition]) != set(video_ids):
                raise ValueError("Conditions must share video support")
            ordered = [by_condition[condition][v] for v in video_ids]
            scores = np.array([r["p_fake"] for r in ordered])
            probabilities = np.array([[r[f"p_source_{i}"] for i in range(4)] for r in ordered])
            det.append((scores >= threshold) == labels)
            src.append(probabilities.argmax(axis=1) == sources)
            aucs[name][condition] = scores, labels
        arrays[name] = (np.array(det), np.array(src), conditions)
        kinds[name], thresholds[name] = kind, threshold

    assert len(canonical) == 699
    video_ids = [m[0] for m in canonical]
    groups_of = np.array([m[1] for m in canonical])
    masks = {
        "full": np.ones(len(canonical), dtype=bool),
        "complete_video": np.array([v in complete_videos for v in video_ids]),
        "complete_group": np.array([g in complete_groups for g in groups_of]),
    }
    group_ids = sorted(set(groups_of))
    members = [np.flatnonzero(groups_of == g) for g in group_ids]
    assert len(members) == 70

    def evaluate_all(indices):
        out = {}
        for support in SUPPORTS:
            chosen = indices[masks[support][indices]]
            out[support] = {
                name: measures(det, src, labels, sources, chosen, conditions)
                for name, (det, src, conditions) in arrays.items()
            }
        return out

    points = evaluate_all(np.arange(len(canonical)))

    # Cross-check full-cohort points against the recorded analyses.
    recorded_rgb = json.loads(Path("experiments/results/main_validation_analysis_seeds42_44.json").read_text(encoding="utf-8"))
    recorded_probe = json.loads(Path("experiments/results/main_probe_validation_analysis.json").read_text(encoding="utf-8"))
    max_point_error = 0.0
    for name, (_, _, conditions) in arrays.items():
        if kinds[name] == "rgb_resnet18":
            recorded = recorded_rgb["point_metrics"][name]
        elif kinds[name] == "reference_trained_probe":
            recorded = recorded_probe["point_metrics"][name.removeprefix("reference_probe_")]
        else:
            report = models[name][0].parents[1] / "report.json"
            rep = name.rsplit("_", 1)[1]
            recorded = {conditions[0]: json.loads(report.read_text(encoding="utf-8"))["representations"][rep]["threshold_validation"]}
        for condition in conditions:
            for metric in ("detection_ba", "attribution_ba"):
                max_point_error = max(max_point_error, abs(points["full"][name][condition][metric] - recorded[condition][metric]))
    assert max_point_error < 1e-12, max_point_error

    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for draw in range(draws):
        indices = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        current = evaluate_all(indices)
        for name, (_, _, conditions) in arrays.items():
            for condition in conditions:
                for metric, value in current["full"][name][condition].items():
                    for support in SUPPORTS:
                        sub = current[support][name][condition][metric]
                        samples[(name, condition, metric, support)].append(sub)
                        if support != "full":
                            samples[(name, condition, metric, f"{support}_minus_full")].append(difference(sub, value))
        for pair, (left, right) in PAIRS.items():
            for condition in CONDITIONS:
                for metric in current["full"][left][condition]:
                    for support in SUPPORTS:
                        samples[(pair, condition, metric, support)].append(difference(
                            current[support][left][condition][metric], current[support][right][condition][metric]))
        if (draw + 1) % 500 == 0:
            print(json.dumps({"bootstrap_draws_completed": draw + 1}), flush=True)

    model_results = {}
    for name, (_, _, conditions) in arrays.items():
        model_results[name] = {"kind": kinds[name], "frozen_threshold": thresholds[name]}
        for condition in conditions:
            entry = {}
            for metric, value in points["full"][name][condition].items():
                entry[metric] = {support: interval(samples[(name, condition, metric, support)], points[support][name][condition][metric]) for support in SUPPORTS}
                for support in SUPPORTS[1:]:
                    entry[metric][f"{support}_minus_full"] = interval(
                        samples[(name, condition, metric, f"{support}_minus_full")],
                        difference(points[support][name][condition][metric], value))
            scores, labels = aucs[name][condition]
            entry["detection_auc_point"] = {s: auc(labels[masks[s]], scores[masks[s]]) for s in SUPPORTS}
            model_results[name][condition] = entry

    contrast_results, status_changes = {}, []
    for pair, (left, right) in PAIRS.items():
        contrast_results[pair] = {}
        for condition in CONDITIONS:
            contrast_results[pair][condition] = {}
            for metric in points["full"][left][condition]:
                contrast_results[pair][condition][metric] = {
                    support: interval(samples[(pair, condition, metric, support)], difference(
                        points[support][left][condition][metric], points[support][right][condition][metric]))
                    for support in SUPPORTS
                }
                full = contrast_results[pair][condition][metric]["full"]
                recorded = recorded_rgb["paired_differences"][pair][condition][metric]
                assert full["ci95"] == recorded["ci95"] or (full["ci95"] is None and recorded["ci95"] is None)
                for support in SUPPORTS[1:]:
                    sub = contrast_results[pair][condition][metric][support]
                    changed_sign = (full["point"] is not None and sub["point"] is not None
                                    and np.sign(full["point"]) != np.sign(sub["point"]))
                    if excludes_zero(full) != excludes_zero(sub) or changed_sign:
                        status_changes.append({"contrast": pair, "condition": condition, "metric": metric,
                                               "support": support, "full": full, "subset": sub})

    ba_shift = {
        support: max(
            abs(entry[metric][f"{support}_minus_full"]["point"])
            for result in model_results.values()
            for condition, entry in result.items() if isinstance(entry, dict)
            for metric in ("detection_ba", "attribution_ba")
        )
        for support in SUPPORTS[1:]
    }

    excluded = sorted(set(video_ids) - complete_videos)
    record = {
        "status": "validation_sensitivity_analysis",
        "test_scores_used": False,
        "definition": {
            "complete_video": "Video with all eight scheduled reference crops valid in reference.csv",
            "complete_group": "Provenance group whose ten core videos are all complete_video",
            "primary_support": "complete_video",
            "thresholds": "Frozen full-cohort validation thresholds; no reselection on subsets",
            "models": "All nine RGB runs, six reference-trained probes, twelve known-quality probes",
            "bootstrap": "Provenance-group resampling over the full validation groups; subset metrics use drawn indices within each support, preserving multiplicity",
            "flags": "Contrast status change = sign change or change in whether the 95% interval excludes zero",
            "test_application": "Apply the identical support rules and frozen thresholds to test predictions; reselection is not permitted",
        },
        "support": {
            "full": {"videos": int(masks["full"].sum()), "groups": 70},
            "complete_video": {"videos": int(masks["complete_video"].sum()), "groups": len(set(groups_of[masks["complete_video"]])), "excluded_videos": excluded},
            "complete_group": {"videos": int(masks["complete_group"].sum()), "groups": len(complete_groups),
                               "excluded_groups": sorted(set(group_ids) - complete_groups)},
            "zero_crop_exclusion_already_applied": [e["video_id"] for e in assembly["excluded_videos"]],
        },
        "bootstrap": {"draws": draws, "seed": seed, "unit": "provenance_group", "groups": 70,
                      "interval": "95% percentile, linear quantiles", "minimum_valid_draw_fraction": 0.95,
                      "full_cohort_rgb_contrast_intervals_reproduced": True},
        "full_cohort_point_check_max_abs_error": max_point_error,
        "max_abs_ba_shift_vs_full": ba_shift,
        "contrast_status_changes": status_changes,
        "models": model_results,
        "contrasts": contrast_results,
        "manifest_sha256": sha256(manifest),
        "input_record_sha256": input_hashes,
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n)) for n in ("metrics.py", "research_metrics.py")},
        "numpy": np.__version__,
        "qualification": (
            "Validation-only sensitivity of selected models and frozen thresholds; "
            "intervals exclude selection uncertainty and seed variability. "
            "Subsets are nested in the full cohort, so differences are not independent comparisons."
        ),
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({k: record[k] for k in ("support", "max_abs_ba_shift_vs_full", "full_cohort_point_check_max_abs_error")}, indent=2))
    print(json.dumps({"contrast_status_changes": len(status_changes)}))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        main(args.draws, args.seed)
