"""Research metrics on video-averaged predictions."""
import argparse
import json

import numpy as np
from metrics import evaluate


def eer(labels, scores):
    """EER from linear interpolation between empirical ROC points."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if (
        labels.ndim != 1 or scores.shape != labels.shape
        or not len(labels) or not np.isfinite(scores).all()
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError("EER requires finite scores and both binary classes")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Move all observations with tied scores together.
    ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0),
        len(labels) - 1
    ]
    cumulative_tp = np.cumsum(sorted_labels)
    tp = np.r_[0, cumulative_tp[ends]]
    fp = np.r_[0, (np.arange(1, len(labels) + 1) - cumulative_tp)[ends]]

    fpr = fp / np.sum(labels == 0)
    fnr = 1 - tp / np.sum(labels == 1)
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])

    if abs(difference[right]) < 1e-12:
        return float((fpr[right] + fnr[right]) / 2)

    left = right - 1
    fraction = (
        -difference[left] / (difference[right] - difference[left])
    )
    crossing_fpr = fpr[left] + fraction * (fpr[right] - fpr[left])
    crossing_fnr = fnr[left] + fraction * (fnr[right] - fnr[left])
    return float((crossing_fpr + crossing_fnr) / 2)


def retention(score, reference, chance, minimum_margin=0.05):
    """Unclipped chance-adjusted BA retention; None for weak references."""
    if score is None or reference is None:
        return None
    if not np.isfinite([score, reference]).all():
        return None
    if reference - chance < minimum_margin:
        return None
    return float((score - chance) / (reference - chance))


def summarize(videos, threshold):
    """Input must contain exactly one row per video and condition."""
    if not videos or not np.isfinite(threshold):
        raise ValueError("Nonempty predictions and finite threshold required")

    keys = [(r["video_id"], r["condition"]) for r in videos]
    if len(set(keys)) != len(keys):
        raise ValueError("Aggregate frame probabilities before summarizing")

    conditions = sorted({r["condition"] for r in videos})
    if "ref" not in conditions:
        raise ValueError("Reference predictions required for retention")

    reference_ids = {
        r["video_id"] for r in videos if r["condition"] == "ref"
    }
    provenance = {}
    for row in videos:
        identity = (
            int(row["label"]), int(row["source_index"]), row["group_id"]
        )
        previous = provenance.setdefault(row["video_id"], identity)
        if previous != identity:
            raise ValueError("Provenance differs across conditions")

    results = evaluate(videos, threshold=threshold)

    for condition in conditions:
        subset = [r for r in videos if r["condition"] == condition]
        if {r["video_id"] for r in subset} != reference_ids:
            raise ValueError("Conditions must have identical video support")

        labels = np.array([int(r["label"]) for r in subset])
        source_labels = np.array([int(r["source_index"]) for r in subset])
        scores = np.array([float(r["p_fake"]) for r in subset])
        probabilities = np.array([
            [float(r[f"p_source_{i}"]) for i in range(4)]
            for r in subset
        ])

        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Both binary classes required")
        if not (
            np.isfinite(scores).all()
            and np.isfinite(probabilities).all()
            and ((scores >= 0) & (scores <= 1)).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all()
            and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Invalid probabilities")

        fake = labels == 1
        if not (source_labels[~fake] == -1).all():
            raise ValueError("Real videos must have source_index -1")
        if set(source_labels[fake].tolist()) != {0, 1, 2, 3}:
            raise ValueError("All four fake sources required")

        detected = scores >= threshold
        source_predictions = probabilities[fake].argmax(axis=1)
        source_correct = source_predictions == source_labels[fake]
        fake_detected = detected[fake]

        detection_confusion = np.bincount(
            2 * labels + detected.astype(int), minlength=4
        ).reshape(2, 2)
        source_confusion = np.bincount(
            4 * source_labels[fake] + source_predictions, minlength=16
        ).reshape(4, 4)

        results[condition].update({
            "detection_eer": eer(labels, scores),
            "detection_confusion": detection_confusion.tolist(),
            "source_confusion": source_confusion.tolist(),
            "source_recall": (
                np.diag(source_confusion) / source_confusion.sum(axis=1)
            ).tolist(),
            "fake_detection_coverage": float(fake_detected.mean()),
            "detection_gated_source_accuracy": float(
                (fake_detected & source_correct).mean()
            ),
            "source_accuracy_among_detected_fakes": (
                float(source_correct[fake_detected].mean())
                if fake_detected.any() else None
            ),
        })

    reference = results["ref"]
    for values in results.values():
        detection_retention = retention(
            values["detection_ba"], reference["detection_ba"], 0.5
        )
        attribution_retention = retention(
            values["attribution_ba"], reference["attribution_ba"], 0.25
        )
        values.update({
            "detection_retention": detection_retention,
            "attribution_retention": attribution_retention,
            "attribution_minus_detection_retention": (
                attribution_retention - detection_retention
                if attribution_retention is not None
                and detection_retention is not None else None
            ),
        })

    return results


def selftest():
    assert np.isclose(eer([0, 1], [0.1, 0.9]), 0)
    assert np.isclose(eer([0, 1], [0.5, 0.5]), 0.5)
    assert np.isclose(eer([0, 1], [0.9, 0.1]), 1)
    assert np.isclose(
        eer([0, 0, 0, 1, 1], [0.8, 0.7, 0.6, 0.9, 0.75]), 1 / 3
    )
    assert np.isclose(
        eer([1, 0, 1, 0], [0.7, 0.7, 0.2, 0.2]), 0.5
    )
    try:
        eer([1, 1], [0.2, 0.8])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing binary class was accepted")

    assert np.isclose(retention(0.9, 0.9, 0.5), 1)
    assert np.isclose(retention(0.5, 0.9, 0.5), 0)
    assert np.isclose(retention(0.4, 0.9, 0.5), -0.25)
    assert np.isclose(retention(1.0, 0.9, 0.5), 1.25)
    assert retention(0.8, 0.54, 0.5) is None

    videos = []
    for condition in ("ref", "jpeg_q40"):
        for index in range(5):
            fake = index > 0
            source = index - 1
            predicted_source = source if condition == "ref" and fake else 0
            probabilities = [0.0] * 4
            probabilities[predicted_source] = 1.0
            videos.append({
                "video_id": f"synthetic_{index}",
                "condition": condition,
                "group_id": "synthetic_group",
                "label": int(fake),
                "source_index": source,
                "p_fake": 0.9 if fake and condition == "ref" else 0.1,
                **{
                    f"p_source_{i}": probabilities[i] for i in range(4)
                },
            })

    result = summarize(videos, threshold=0.5)
    assert result["ref"]["detection_ba"] == 1
    assert result["ref"]["attribution_ba"] == 1
    assert result["ref"]["detection_gated_source_accuracy"] == 1
    assert result["jpeg_q40"]["detection_retention"] == 0
    assert result["jpeg_q40"]["attribution_retention"] == 0
    assert result["jpeg_q40"]["fake_detection_coverage"] == 0
    assert result["jpeg_q40"]["detection_gated_source_accuracy"] == 0
    assert result["jpeg_q40"]["source_accuracy_among_detected_fakes"] is None

    try:
        summarize(videos + [videos[0]], threshold=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate video prediction was accepted")

    print(json.dumps({
        "status": "passed",
        "synthetic_only": True,
        "checks": [
            "EER perfect, tied, reversed and interpolated ROC cases",
            "Missing binary class rejection",
            "Chance-adjusted retention and weak-reference handling",
            "Unclipped negative and above-one retention",
            "Known-answer confusion, coverage and gated attribution",
            "Common video support and duplicate rejection"
        ],
        "confusion_convention": "Rows=true class; columns=predicted class"
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("Use --selftest, or import summarize for analysis")
    selftest()
