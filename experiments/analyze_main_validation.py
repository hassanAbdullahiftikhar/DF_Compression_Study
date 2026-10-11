"""Descriptive validation analysis with paired provenance-group bootstrap."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from common import SOURCES
from metrics import aggregate
from research_metrics import retention, summarize

CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
RUNS = ('c23_main_reference_seed42', 'c23_main_augmented_seed42', 'c23_main_paired_w1_seed42', 'c23_main_reference_seed43', 'c23_main_augmented_seed43', 'c23_main_paired_w1_seed43', 'c23_main_reference_seed44', 'c23_main_augmented_seed44', 'c23_main_paired_w1_seed44')
MEASURES = (
    "detection_ba",
    "attribution_ba",
    "detection_retention",
    "attribution_retention",
    "attribution_minus_detection_retention",
)
PAIRS = {'augmentation_minus_reference_seed42': ('c23_main_augmented_seed42', 'c23_main_reference_seed42'), 'pair_w1_minus_augmentation_seed42': ('c23_main_paired_w1_seed42', 'c23_main_augmented_seed42'), 'augmentation_minus_reference_seed43': ('c23_main_augmented_seed43', 'c23_main_reference_seed43'), 'pair_w1_minus_augmentation_seed43': ('c23_main_paired_w1_seed43', 'c23_main_augmented_seed43'), 'augmentation_minus_reference_seed44': ('c23_main_augmented_seed44', 'c23_main_reference_seed44'), 'pair_w1_minus_augmentation_seed44': ('c23_main_paired_w1_seed44', 'c23_main_augmented_seed44')}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_indices(members, draws):
    # Repeated groups remain repeated observations.
    return np.concatenate([members[int(i)] for i in draws])


def estimates(det_correct, src_correct, labels, sources, indices):
    selected_det = det_correct[:, indices]
    selected_src = src_correct[:, indices]
    selected_labels = labels[indices]
    selected_sources = sources[indices]

    det_ba = (
        selected_det[:, selected_labels == 0].mean(axis=1)
        + selected_det[:, selected_labels == 1].mean(axis=1)
    ) / 2
    src_ba = np.mean([
        selected_src[:, selected_sources == source].mean(axis=1)
        for source in range(4)
    ], axis=0)

    result = {}
    for i, condition in enumerate(CONDITIONS):
        det_r = retention(float(det_ba[i]), float(det_ba[0]), 0.5)
        src_r = retention(float(src_ba[i]), float(src_ba[0]), 0.25)
        result[condition] = {
            "detection_ba": float(det_ba[i]),
            "attribution_ba": float(src_ba[i]),
            "detection_retention": det_r,
            "attribution_retention": src_r,
            "attribution_minus_detection_retention": (
                src_r - det_r if src_r is not None and det_r is not None
                else None
            ),
        }
    return result


def difference(left, right):
    return None if left is None or right is None else float(left - right)


def interval(values, point):
    valid = np.array([
        value for value in values
        if value is not None and np.isfinite(value)
    ], dtype=float)
    required = int(np.ceil(0.95 * len(values)))
    usable = point is not None and len(valid) > 0 and len(valid) >= required

    return {
        "point": point,
        "ci95": (
            np.quantile(valid, [0.025, 0.975], method="linear").tolist()
            if usable else None
        ),
        "valid_draws": len(valid),
        "undefined_draws": len(values) - len(valid),
        "interval_available": usable,
    }


def selftest():
    members = [np.array([0, 1]), np.array([2, 3])]
    assert sample_indices(members, [1, 1]).tolist() == [2, 3, 2, 3]

    labels = np.array([0, 1, 1, 1, 1] * 2)
    sources = np.array([-1, 0, 1, 2, 3] * 2)
    det = np.ones((3, 10), dtype=bool)
    src = np.ones((3, 10), dtype=bool)
    det[2, labels == 1] = False
    src[2] = sources == 0

    groups = [np.arange(5), np.arange(5, 10)]
    indices = sample_indices(groups, [0, 0])
    result = estimates(det, src, labels, sources, indices)
    assert result["ref"]["detection_ba"] == 1
    assert result["jpeg_q40"]["detection_ba"] == 0.5
    assert result["jpeg_q40"]["attribution_ba"] == 0.25
    assert result["jpeg_q40"]["detection_retention"] == 0
    assert result["jpeg_q40"]["attribution_retention"] == 0

    changed = det.copy()
    changed[0, :5] = labels[:5] == 0
    weak = estimates(changed, src, labels, sources, indices)
    strong = estimates(
        changed, src, labels, sources, sample_indices(groups, [1, 1])
    )
    assert weak["jpeg_q40"]["detection_retention"] is None
    assert strong["jpeg_q40"]["detection_retention"] == 0

    rng = np.random.default_rng(4242)
    for _ in range(20):
        indices = sample_indices(groups, rng.integers(0, 2, size=2))
        left = estimates(det, src, labels, sources, indices)
        right = estimates(det.copy(), src.copy(), labels, sources, indices)
        for condition in CONDITIONS:
            for metric in MEASURES:
                assert difference(
                    left[condition][metric], right[condition][metric]
                ) == 0

    assert interval([0.0] * 20, 0.0)["ci95"] == [0.0, 0.0]
    assert interval([None] * 20, None)["ci95"] is None

    print(json.dumps({
        "status": "passed",
        "synthetic_only": True,
        "checks": [
            "Repeated group multiplicity preserved",
            "Known-answer BA and retention",
            "Reference denominator recomputed per draw",
            "Identical models have zero paired differences",
            "Constant and undefined interval handling"
        ]
    }, indent=2))


def main(draws, seed):
    if draws < 100:
        raise ValueError("Use at least 100 draws")

    threshold_path = Path(
        "experiments/results/main_validation_thresholds_seeds42_44.json"
    )
    output = Path("experiments/results/main_validation_analysis_seeds42_44.json")
    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    frozen = json.loads(threshold_path.read_text(encoding="utf-8"))
    manifest = Path("experiments/data/c23_main/reference.csv")
    if sha256(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Manifest changed after threshold selection")

    points, arrays, canonical = {}, {}, None

    for name in RUNS:
        run = Path("experiments/runs") / name
        prediction_path = run / "validation_predictions.csv"
        registered = frozen["runs"][name]

        for path, expected in (
            (prediction_path, registered["validation_predictions_sha256"]),
            (run / "best.pt", registered["checkpoint_sha256"]),
        ):
            if sha256(path) != expected:
                raise ValueError(f"Registered file changed: {path}")

        with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
            videos = aggregate(list(csv.DictReader(handle)))

        threshold = registered["threshold"]
        points[name] = summarize(videos, threshold)
        if set(points[name]) != set(CONDITIONS):
            raise ValueError("Unexpected validation conditions")

        by_condition = {
            condition: {
                row["video_id"]: row for row in videos
                if row["condition"] == condition
            }
            for condition in CONDITIONS
        }
        video_ids = sorted(by_condition["ref"])
        metadata = [
            (
                video_id,
                by_condition["ref"][video_id]["group_id"],
                by_condition["ref"][video_id]["label"],
                by_condition["ref"][video_id]["source_index"],
            )
            for video_id in video_ids
        ]

        if canonical is None:
            canonical = metadata
        elif metadata != canonical:
            raise ValueError("Model video support or provenance differs")

        labels = np.array([row[2] for row in metadata])
        sources = np.array([row[3] for row in metadata])
        det_correct, src_correct = [], []

        for condition in CONDITIONS:
            ordered = [by_condition[condition][v] for v in video_ids]
            scores = np.array([row["p_fake"] for row in ordered])
            probabilities = np.array([
                [row[f"p_source_{i}"] for i in range(4)]
                for row in ordered
            ])
            det_correct.append((scores >= threshold) == labels)
            src_correct.append(probabilities.argmax(axis=1) == sources)

        arrays[name] = (
            np.array(det_correct, dtype=bool),
            np.array(src_correct, dtype=bool),
        )

        direct = estimates(*arrays[name], labels, sources, np.arange(len(labels)))
        for condition in CONDITIONS:
            for metric in MEASURES:
                expected = points[name][condition][metric]
                actual = direct[condition][metric]
                if expected is None or actual is None:
                    assert expected is None and actual is None
                else:
                    assert np.isclose(expected, actual)

    groups = {}
    for index, row in enumerate(canonical):
        groups.setdefault(row[1], []).append(index)
    members = [
        np.array(groups[group], dtype=int) for group in sorted(groups)
    ]
    for indices in members:
        if set(sources[indices].tolist()) != {-1, 0, 1, 2, 3}:
            raise ValueError("Each validation group must cover real and all sources")

    assert len(canonical) == 699
    assert len(members) == 70

    def empty_store():
        return {
            condition: {metric: [] for metric in MEASURES}
            for condition in CONDITIONS
        }

    samples = {name: empty_store() for name in RUNS}
    paired_samples = {name: empty_store() for name in PAIRS}
    rng = np.random.default_rng(seed)

    for draw in range(draws):
        indices = sample_indices(
            members, rng.integers(0, len(members), size=len(members))
        )
        current = {
            name: estimates(*arrays[name], labels, sources, indices)
            for name in RUNS
        }

        for condition in CONDITIONS:
            for metric in MEASURES:
                for name in RUNS:
                    samples[name][condition][metric].append(
                        current[name][condition][metric]
                    )
                for comparison, (left, right) in PAIRS.items():
                    paired_samples[comparison][condition][metric].append(
                        difference(
                            current[left][condition][metric],
                            current[right][condition][metric],
                        )
                    )
        if (draw + 1) % 500 == 0:
            print(json.dumps({"bootstrap_draws_completed": draw + 1}), flush=True)

    intervals = {
        name: {
            condition: {
                metric: interval(
                    samples[name][condition][metric],
                    points[name][condition][metric],
                )
                for metric in MEASURES
            }
            for condition in CONDITIONS
        }
        for name in RUNS
    }
    paired_intervals = {
        comparison: {
            condition: {
                metric: interval(
                    paired_samples[comparison][condition][metric],
                    difference(
                        points[left][condition][metric],
                        points[right][condition][metric],
                    ),
                )
                for metric in MEASURES
            }
            for condition in CONDITIONS
        }
        for comparison, (left, right) in PAIRS.items()
    }

    seed_summary = {}
    for mode in ("reference", "augmented", "paired_w1"):
        seed_summary[mode] = {}
        for condition in CONDITIONS:
            seed_summary[mode][condition] = {}
            for metric in MEASURES:
                values = {
                    str(seed): points[f"c23_main_{mode}_seed{seed}"][condition][metric]
                    for seed in (42, 43, 44)
                }
                valid = all(
                    value is not None and np.isfinite(value)
                    for value in values.values()
                )
                seed_summary[mode][condition][metric] = {
                    "values_by_seed": values,
                    "mean": float(np.mean(list(values.values()))) if valid else None,
                    "sample_sd": float(np.std(list(values.values()), ddof=1)) if valid else None,
                }

    record = {
        "descriptive_seed_variation": seed_summary,
        "seed_variation_note": (
            "Mean and sample SD across three seeds are descriptive; "
            "they are not confidence intervals."
        ),
        "status": "exploratory_validation_analysis",
        "qualification": (
            "Conditional on selected checkpoints and thresholds; excludes "
            "selection uncertainty and training-seed variability"
        ),
        "bootstrap": {
            "unit": "provenance_group",
            "groups": len(members),
            "videos_per_condition": len(canonical),
            "draws": draws,
            "seed": seed,
            "interval": "95% percentile, linear quantiles",
            "minimum_valid_draw_fraction": 0.95,
            "near_chance_reference_margin": 0.05,
            "paired_across_models_and_conditions": True,
        },
        "source_order": list(SOURCES[1:]),
        "threshold_record_sha256": sha256(threshold_path),
        "manifest_sha256": sha256(manifest),
        "script_sha256": sha256(Path(__file__)),
        "research_metrics_sha256": sha256(
            Path(__file__).with_name("research_metrics.py")
        ),
        "numpy": np.__version__,
        "point_metrics": points,
        "intervals": intervals,
        "paired_differences": paired_intervals,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)

    compact = {
        "validation_groups": len(members),
        "q40_descriptive_seed_variation": {
            mode: seed_summary[mode]["jpeg_q40"]
            for mode in seed_summary
        },
        "q40": {
            name: {
                "eer": points[name]["jpeg_q40"]["detection_eer"],
                "gated_source_accuracy": points[name]["jpeg_q40"][
                    "detection_gated_source_accuracy"
                ],
                **intervals[name]["jpeg_q40"],
            }
            for name in RUNS
        },
        "q40_paired_differences": {
            name: {
                metric: paired_intervals[name]["jpeg_q40"][metric]
                for metric in ("detection_ba", "attribution_ba")
            }
            for name in PAIRS
        },
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    print("Saved:", output)


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
