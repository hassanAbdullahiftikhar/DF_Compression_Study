"""Validation-selected logistic probes on cached descriptors; needs scikit-learn."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import SOURCES, audit, read_manifest
from metrics import evaluate


def predictions(rows, indices, detection, attribution, features):
    p_fake = detection.predict_proba(features)[:, list(detection.classes_).index(1)]
    source_probabilities = attribution.predict_proba(features)
    result = []
    for offset, index in enumerate(indices):
        row = rows[index]
        source = SOURCES.index(row["source"]) - 1
        result.append(dict(sample_id=row["sample_id"], video_id=row["video_id"],
            group_id=row["group_id"], condition=row["condition"], label=int(source >= 0),
            source_index=source, p_fake=float(p_fake[offset]),
            **{f"p_source_{k}": float(source_probabilities[offset, list(attribution.classes_).index(k)])
               for k in range(4)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_directory")
    parser.add_argument("output")
    parser.add_argument("--official-splits", required=True)
    parser.add_argument("--train-conditions", nargs="+", default=["ref"])
    parser.add_argument("--val-conditions", nargs="+", default=None)
    args = parser.parse_args()
    directory, output = Path(args.feature_directory), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    rows = read_manifest(directory / "rows.csv")
    audit(rows, official_dir=args.official_splits)
    matrix = np.load(directory / "features.npy", mmap_mode="r")
    if len(matrix) != len(rows):
        raise ValueError("Feature/manifest row mismatch")
    val_conditions = args.val_conditions or args.train_conditions
    for split, conditions in (("train", args.train_conditions), ("val", val_conditions)):
        for condition in conditions:
            support = {r["source"] for r in rows if r["split"] == split and r["condition"] == condition}
            if support != set(SOURCES):
                raise ValueError(f"Missing source support in {split}/{condition}")
    train_indices = [i for i, r in enumerate(rows) if r["split"] == "train" and r["condition"] in args.train_conditions]
    val_indices = [i for i, r in enumerate(rows) if r["split"] == "val" and r["condition"] in val_conditions]
    test_indices = [i for i, r in enumerate(rows) if r["split"] == "test"]
    if not test_indices:
        raise ValueError("No test examples")
    scaler = StandardScaler().fit(matrix[train_indices])
    train = scaler.transform(matrix[train_indices])
    val = scaler.transform(matrix[val_indices])
    labels = np.array([SOURCES.index(rows[i]["source"]) - 1 for i in train_indices])
    fake_mask = labels >= 0
    best_score, best_models, search = -float("inf"), None, []
    for strength in (0.01, 0.1, 1.0, 10.0):
        detection = LogisticRegression(C=strength, class_weight="balanced", max_iter=1000, random_state=42)
        attribution = LogisticRegression(C=strength, class_weight="balanced", max_iter=1000, random_state=42)
        detection.fit(train, fake_mask.astype(int))
        attribution.fit(train[fake_mask], labels[fake_mask])
        scores = evaluate(predictions(rows, val_indices, detection, attribution, val))
        score = float(np.mean([(r["detection_ba"] + r["attribution_ba"]) / 2 for r in scores.values()]))
        search.append({"C": strength, "score": score, "validation": scores})
        if score > best_score:
            best_score, best_models = score, (detection, attribution, strength)
    if best_models is None:
        raise ValueError("No valid validation score")
    detection, attribution, strength = best_models
    # Selected model is fixed before test data are transformed or scored.
    test = scaler.transform(matrix[test_indices])
    test_rows = predictions(rows, test_indices, detection, attribution, test)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_rows[0].keys())
        writer.writeheader()
        writer.writerows(test_rows)
    (output / "selection.json").write_text(json.dumps({"arguments": vars(args), "C": strength,
        "search": search, "threshold": 0.5, "note": "Select/freeze a validation operating threshold for final reporting."}, indent=2))
    # Save numeric parameters without pickle, allowing independent reconstruction.
    np.savez(output / "probe_parameters.npz", scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             det_coef=detection.coef_, det_intercept=detection.intercept_, det_classes=detection.classes_,
             src_coef=attribution.coef_, src_intercept=attribution.intercept_, src_classes=attribution.classes_)
    print(json.dumps({"selected_C": strength, "test": evaluate(test_rows)}, indent=2))
