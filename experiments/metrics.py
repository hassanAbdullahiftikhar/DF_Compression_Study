"""Video aggregation and standard metrics; bootstrap inference remains a research stage."""
from collections import defaultdict
import numpy as np


def balanced_accuracy(labels, predictions, classes):
    recalls = []
    for label in range(classes):
        mask = labels == label
        if not mask.any():
            return float("nan")
        recalls.append((predictions[mask] == label).mean())
    return float(np.mean(recalls))


def macro_f1(labels, predictions, classes):
    values = []
    for label in range(classes):
        tp = np.sum((labels == label) & (predictions == label))
        fp = np.sum((labels != label) & (predictions == label))
        fn = np.sum((labels == label) & (predictions != label))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def auc(labels, scores):
    positive, negative = scores[labels == 1], np.sort(scores[labels == 0])
    if not len(positive) or not len(negative):
        return float("nan")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.mean((lower + upper) / (2 * len(negative))))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["video_id"], row["condition"])].append(row)
    videos = []
    for (video, condition), frames in groups.items():
        if len({(r["label"], r["source_index"], r["group_id"]) for r in frames}) != 1:
            raise ValueError("Video predictions disagree on provenance/labels")
        videos.append(dict(video_id=video, condition=condition, group_id=frames[0]["group_id"],
            label=int(frames[0]["label"]), source_index=int(frames[0]["source_index"]),
            p_fake=float(np.mean([float(r["p_fake"]) for r in frames])),
            **{f"p_source_{i}": float(np.mean([float(r[f"p_source_{i}"]) for r in frames]))
               for i in range(4)}))
    return videos


def evaluate(rows, threshold=0.5):
    videos = aggregate(rows)
    results = {}
    for condition in sorted({r["condition"] for r in videos}):
        subset = [r for r in videos if r["condition"] == condition]
        y = np.array([r["label"] for r in subset])
        scores = np.array([r["p_fake"] for r in subset])
        prediction = (scores >= threshold).astype(int)
        fakes = [r for r in subset if r["label"] == 1]
        source_y = np.array([r["source_index"] for r in fakes], dtype=int)
        source_pred = (np.argmax([[r[f"p_source_{i}"] for i in range(4)] for r in fakes], axis=1)
                       if fakes else np.array([], dtype=int))
        results[condition] = {"videos": len(subset), "fake_videos": len(fakes),
            "detection_auc": auc(y, scores), "detection_ba": balanced_accuracy(y, prediction, 2),
            "detection_macro_f1": macro_f1(y, prediction, 2),
            "attribution_ba": balanced_accuracy(source_y, source_pred, 4),
            "attribution_top1": float(np.mean(source_y == source_pred)) if fakes else float("nan"),
            "attribution_macro_f1": macro_f1(source_y, source_pred, 4)}
    return results


if __name__ == "__main__":
    import argparse
    import csv
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed threshold selected on validation before test")
    args = parser.parse_args()
    with open(args.predictions, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(json.dumps(evaluate(rows, args.threshold), indent=2))
