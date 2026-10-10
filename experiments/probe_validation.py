"""Training-only scaling and validation-selected ridge probes; no test scoring."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from common import SOURCES, audit, read_manifest
from metrics import evaluate

ALPHAS = (0.001, 0.01, 0.1, 1.0)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_system(features, labels, classes):
    labels = np.asarray(labels, dtype=int)
    if set(labels.tolist()) != set(range(classes)):
        raise ValueError("Missing training class")
    counts = np.bincount(labels, minlength=classes)
    weights = len(labels) / (classes * counts[labels])
    targets = np.eye(classes)[labels]

    gram = features.T @ (features * weights[:, None]) / len(labels)
    rhs = features.T @ (targets * weights[:, None]) / len(labels)
    return gram, rhs


def solve(system, alpha):
    gram, rhs = system
    regularized = gram.copy()
    # Last column is the intercept; do not penalize it.
    indices = np.arange(len(gram) - 1)
    regularized[indices, indices] += alpha
    return np.linalg.solve(regularized, rhs)


def softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def shuffle_video_labels(video_ids, labels, seed):
    mapping = {}
    for video, label in zip(video_ids, labels):
        previous = mapping.setdefault(video, int(label))
        if previous != int(label):
            raise ValueError("Training video has conflicting labels")

    ordered = sorted(mapping)
    original = np.array([mapping[video] for video in ordered])
    shuffled = np.random.default_rng(seed).permutation(original)
    assigned = dict(zip(ordered, shuffled.tolist()))
    return np.array([assigned[video] for video in video_ids], dtype=int)


def prediction_rows(rows, detection, attribution):
    result = []
    for index, row in enumerate(rows):
        source = SOURCES.index(row["source"]) - 1
        result.append({
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "group_id": row["group_id"],
            "condition": row["condition"],
            "label": int(source >= 0),
            "source_index": source,
            "p_fake": float(detection[index, 1]),
            **{
                f"p_source_{k}": float(attribution[index, k])
                for k in range(4)
            },
        })
    return result


def fit_probe(train, val, source_targets, val_rows):
    fake = source_targets >= 0
    det_system = weighted_system(train, fake.astype(int), 2)
    src_system = weighted_system(train[fake], source_targets[fake], 4)

    best = None
    search = []
    for alpha in ALPHAS:
        det_coef = solve(det_system, alpha)
        src_coef = solve(src_system, alpha)
        predictions = prediction_rows(
            val_rows,
            softmax(val @ det_coef),
            softmax(val @ src_coef),
        )
        scores = evaluate(predictions, threshold=0.5)["ref"]
        selection = (scores["detection_ba"] + scores["attribution_ba"]) / 2
        entry = {
            "alpha": alpha,
            "selection_score": float(selection),
            "validation": scores,
        }
        search.append(entry)

        # ALPHAS is ascending: exact ties retain smaller alpha.
        if best is None or selection > best["selection_score"] + 1e-12:
            best = {
                **entry,
                "det_coef": det_coef,
                "src_coef": src_coef,
                "predictions": predictions,
            }

    report = {
        key: best[key]
        for key in ("alpha", "selection_score", "validation")
    }
    report["search"] = search
    return best, report


def selftest():
    features = np.array([
        [-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]
    ])
    labels = np.array([0, 0, 1, 1])
    coefficients = solve(weighted_system(features, labels, 2), 0.1)
    assert np.array_equal((features @ coefficients).argmax(axis=1), labels)

    features = np.column_stack([np.eye(4), np.ones(4)])
    labels = np.arange(4)
    coefficients = solve(weighted_system(features, labels, 4), 0.1)
    assert np.array_equal((features @ coefficients).argmax(axis=1), labels)

    videos = np.repeat(["a", "b", "c", "d", "e"], 2)
    labels = np.repeat([-1, 0, 1, 2, 3], 2)
    shuffled = shuffle_video_labels(videos, labels, 101)
    assert np.array_equal(np.sort(shuffled), np.sort(labels))
    assert np.array_equal(shuffled[::2], shuffled[1::2])
    assert np.array_equal(
        shuffled, shuffle_video_labels(videos, labels, 101)
    )
    assert np.allclose(softmax(np.array([[1000.0, 999.0]])).sum(axis=1), 1)

    print(json.dumps({
        "status": "passed",
        "synthetic_only": True,
        "checks": [
            "Known-answer binary and four-class ridge predictions",
            "Video-level label shuffle preserves counts and frame labels",
            "Deterministic shuffling",
            "Stable softmax"
        ]
    }, indent=2))


def main(directory, output, official_splits):
    started = time.perf_counter()
    directory, output = Path(directory), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")

    rows = read_manifest(directory / "rows.csv")
    audit(rows, official_dir=official_splits)
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("This pilot probe requires training/validation only")
    if any(row["condition"] != "ref" for row in rows):
        raise ValueError("This pilot check requires reference features only")

    matrix = np.load(directory / "features.npy", mmap_mode="r")
    if matrix.ndim != 2 or len(matrix) != len(rows):
        raise ValueError("Feature/manifest shape mismatch")

    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    val_indices = [i for i, row in enumerate(rows) if row["split"] == "val"]
    val_rows = [rows[i] for i in val_indices]

    train_raw = np.asarray(matrix[train_indices], dtype=np.float64)
    val_raw = np.asarray(matrix[val_indices], dtype=np.float64)
    if not np.isfinite(train_raw).all() or not np.isfinite(val_raw).all():
        raise ValueError("Nonfinite features")

    mean = train_raw.mean(axis=0)
    scale = train_raw.std(axis=0)
    scale[scale < 1e-12] = 1.0

    train = np.column_stack([
        (train_raw - mean) / scale, np.ones(len(train_raw))
    ])
    val = np.column_stack([
        (val_raw - mean) / scale, np.ones(len(val_raw))
    ])

    targets = np.array([
        SOURCES.index(rows[i]["source"]) - 1 for i in train_indices
    ])
    video_ids = [rows[i]["video_id"] for i in train_indices]

    best, real_report = fit_probe(train, val, targets, val_rows)
    shuffled_reports = []
    for seed in range(101, 106):
        shuffled_targets = shuffle_video_labels(video_ids, targets, seed)
        _, report = fit_probe(train, val, shuffled_targets, val_rows)
        report["shuffle_seed"] = seed
        report["training_targets_sha256"] = hashlib.sha256(
            shuffled_targets.astype("<i8").tobytes()
        ).hexdigest()
        shuffled_reports.append(report)
        print(json.dumps({"shuffle_completed": seed}), flush=True)

    metadata = json.loads(
        (directory / "feature_metadata.json").read_text(encoding="utf-8")
    )
    record = {
        "status": "validation_only_descriptor_probe",
        "representation": metadata["representation"],
        "classifier": "Separate class-weighted ridge detection/source heads",
        "objective": "Mean class-weighted squared loss plus alpha*L2",
        "scaling": "Mean and population standard deviation fitted on training only",
        "intercept": "Unpenalized",
        "selection_rule": "Mean detection/source validation BA at threshold 0.5",
        "tie_rule": "Smaller alpha",
        "probability_mapping": "Softmax of ridge scores; calibration unverified",
        "shuffle_rule": "Permute five-way labels across training videos",
        "qualification": "Diagnostic controls; validation tuning is optimistic",
        "source_order": list(SOURCES[1:]),
        "train_frames": len(train_indices),
        "validation_frames": len(val_indices),
        "dimensions": matrix.shape[1],
        "feature_sha256": sha256(directory / "features.npy"),
        "rows_sha256": sha256(directory / "rows.csv"),
        "script_sha256": sha256(Path(__file__)),
        "numpy": np.__version__,
        "real_labels": real_report,
        "shuffled_labels": shuffled_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=best["predictions"][0].keys())
        writer.writeheader()
        writer.writerows(best["predictions"])

    np.savez_compressed(
        output / "probe_parameters.npz",
        scaler_mean=mean,
        scaler_scale=scale,
        det_coef=best["det_coef"],
        src_coef=best["src_coef"],
    )
    with (output / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)

    def compact(report):
        return {
            "alpha": report["alpha"],
            "selection_score": report["selection_score"],
            **{
                metric: report["validation"][metric]
                for metric in (
                    "detection_auc", "detection_ba", "attribution_ba"
                )
            },
        }

    print(json.dumps({
        "status": record["status"],
        "real_labels": compact(real_report),
        "shuffled_labels": [
            {"seed": report["shuffle_seed"], **compact(report)}
            for report in shuffled_reports
        ],
        "elapsed_seconds": record["elapsed_seconds"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_directory", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--official-splits")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        if not all((args.feature_directory, args.output, args.official_splits)):
            parser.error("Feature directory, output and official splits required")
        main(args.feature_directory, args.output, args.official_splits)
