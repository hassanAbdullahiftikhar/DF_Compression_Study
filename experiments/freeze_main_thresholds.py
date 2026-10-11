import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from common import SOURCES, read_manifest
from metrics import aggregate, balanced_accuracy, evaluate

CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
RUNS = ('c23_main_reference_seed42', 'c23_main_augmented_seed42', 'c23_main_paired_w1_seed42', 'c23_main_reference_seed43', 'c23_main_augmented_seed43', 'c23_main_paired_w1_seed43', 'c23_main_reference_seed44', 'c23_main_augmented_seed44', 'c23_main_paired_w1_seed44')

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = Path("experiments/data/c23_main/reference.csv")
expected = {
    r["sample_id"]: r
    for r in read_manifest(manifest)
    if r["split"] == "val"
}
assert len(expected) == 5582

output = Path("experiments/results/main_validation_thresholds_seeds42_44.json")
if output.exists():
    raise FileExistsError(f"Already exists: {output}")

reports = {}

for name in RUNS:
    run = Path("experiments/runs") / name
    seed = int(name.rsplit("seed", 1)[1])
    mode = name[len("c23_main_"):].rsplit("_seed", 1)[0]
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    resources = json.loads((run / "resource_usage.json").read_text(encoding="utf-8"))
    base_config = json.loads(
        Path("experiments/configs/pilot.json").read_text(encoding="utf-8")
    )
    base_config["seed"] = seed
    assert metadata["config"] == base_config, "Training configuration differs"
    arguments = metadata["arguments"]
    assert arguments["reference_only"] == (mode == "reference")
    assert arguments["pair_weight"] == (1.0 if mode == "paired_w1" else 0.0)
    assert arguments["pretrained"] is True
    assert Path(arguments["manifest"]).resolve() == manifest.resolve()
    assert resources["status"] == "completed_measured_main_training"
    assert resources["manifest_sha256"] == sha256(manifest)
    assert resources["trainer_sha256"] == sha256(Path("experiments/train_paired.py"))
    predictions_path = run / "validation_predictions.csv"

    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {r["condition"] for r in rows} == set(CONDITIONS)
    assert len(rows) == len(expected) * len(CONDITIONS)
    assert len({(r["sample_id"], r["condition"]) for r in rows}) == len(rows)

    for condition in CONDITIONS:
        assert {
            r["sample_id"] for r in rows if r["condition"] == condition
        } == set(expected)

    for row in rows:
        reference = expected[row["sample_id"]]
        for field in ("video_id", "group_id"):
            assert row[field] == reference[field]

        source_index = SOURCES.index(reference["source"]) - 1
        assert int(row["source_index"]) == source_index
        assert int(row["label"]) == int(source_index >= 0)

        probabilities = np.array([
            float(row["p_fake"]),
            *[float(row[f"p_source_{i}"]) for i in range(4)]
        ])
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.isclose(probabilities[1:].sum(), 1, atol=1e-5)

    # Unique, exact sample coverage above verifies every available frame,
    # including videos with fewer than eight successful crops.

    videos = aggregate(rows)
    data = {}
    for condition in CONDITIONS:
        subset = [r for r in videos if r["condition"] == condition]
        assert len(subset) == 699
        labels = np.array([r["label"] for r in subset])
        scores = np.array([r["p_fake"] for r in subset])
        assert (labels == 0).sum() == 140
        assert (labels == 1).sum() == 559
        data[condition] = (labels, scores)

    all_scores = np.concatenate([scores for _, scores in data.values()])
    candidates = np.unique(np.concatenate([
        all_scores,
        [0.5, np.nextafter(all_scores.max(), np.inf)]
    ]))

    best_score, best_threshold = -1.0, None
    for threshold in candidates:
        score = float(np.mean([
            balanced_accuracy(labels, (scores >= threshold).astype(int), 2)
            for labels, scores in data.values()
        ]))
        better = score > best_score + 1e-12
        tied = abs(score - best_score) <= 1e-12

        if better or (
            tied and (
                best_threshold is None or
                (abs(threshold - 0.5), threshold) <
                (abs(best_threshold - 0.5), best_threshold)
            )
        ):
            best_score = score
            best_threshold = float(threshold)

    original = evaluate(rows, threshold=0.5)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert len(history) == base_config["training"]["epochs"]
    selected = max(history, key=lambda epoch: float(np.mean([
        (scores["detection_ba"] + scores["attribution_ba"]) / 2
        for scores in epoch["validation"].values()
    ])))
    for condition in CONDITIONS:
        for metric in ("detection_auc", "detection_ba", "attribution_ba"):
            assert np.isclose(
                original[condition][metric],
                selected["validation"][condition][metric],
                rtol=0, atol=1e-10
            ), "Saved predictions disagree with selected epoch"
    tuned = evaluate(rows, threshold=best_threshold)

    reports[name] = {
        "seed": seed,
        "selected_epoch": selected["epoch"],
        "validation_frames_per_condition": len(expected),
        "validation_videos_per_condition": 699,
        "threshold": best_threshold,
        "checkpoint_sha256": sha256(run / "best.pt"),
        "validation_predictions_sha256": sha256(predictions_path),
        "validation_groups": len({r["group_id"] for r in videos}),
        "mean_validation_detection_ba": best_score,
        "detection_ba": {
            condition: {
                "threshold_0_5": original[condition]["detection_ba"],
                "selected_threshold": tuned[condition]["detection_ba"]
            }
            for condition in CONDITIONS
        }
    }

record = {
    "rule": "Maximize mean video detection BA across validation ref/Q80/Q40",
    "ties": "Closest to 0.5, then smaller threshold; tolerance 1e-12",
    "prediction_rule": "Fake when p_fake >= threshold",
    "checkpoint_selection": "Original selection at threshold 0.5 retained",
    "test_scores_used": False,
    "qualification": "Threshold-tuned validation scores are optimistic",
    "manifest_sha256": sha256(manifest),
    "script_sha256": sha256(Path(__file__)),
    "numpy": np.__version__,
    "runs": reports
}

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, allow_nan=False)

print(json.dumps(reports, indent=2, allow_nan=False))
print("Saved:", output)
