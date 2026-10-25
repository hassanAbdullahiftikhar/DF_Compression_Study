"""Verify registered RGB control runs and freeze their validation thresholds (F1 rule)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_probe_compression import choose_threshold
from metrics import aggregate, evaluate

CONTROLS = ("c1_det", "c1_att", "c2_bin")
SEEDS = (42, 43, 44)
CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
OUTPUT = Path("experiments/results/control_validation_thresholds.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = Path("experiments/data/c23_main/reference.csv")
    trainer = Path("experiments/train_control.py")
    runs = {}
    for control in CONTROLS:
        for seed in SEEDS:
            name = f"c23_control_{control}_seed{seed}"
            run = Path("experiments/runs") / name
            metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
            assert metadata["control"] == control and metadata["seed"] == seed
            assert metadata["test_scores_used"] is False
            assert metadata["manifest_sha256"] == sha256(manifest)
            assert metadata["script_sha256"] == sha256(trainer)
            assert metadata["checkpoint_sha256"] == sha256(run / "best.pt")
            predictions_path = run / "validation_predictions.csv"
            assert metadata["validation_predictions_sha256"] == sha256(predictions_path)
            history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            assert len(history) == metadata["config"]["training"]["epochs"]
            selected = max(history, key=lambda epoch: epoch["selection_score"])
            assert selected["epoch"] == metadata["selected_epoch"]
            with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert {r["condition"] for r in rows} == set(CONDITIONS)
            assert len(rows) == 5582 * len(CONDITIONS)
            saved = evaluate(rows, threshold=0.5)
            for condition in CONDITIONS:
                for metric in ("detection_auc", "detection_ba", "attribution_ba"):
                    assert np.isclose(saved[condition][metric], selected["validation"][condition][metric],
                                      rtol=0, atol=1e-10), "Saved predictions disagree with selected epoch"
            entry = {
                "control": control,
                "seed": seed,
                "selected_epoch": selected["epoch"],
                "selection_score": selected["selection_score"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "validation_predictions_sha256": metadata["validation_predictions_sha256"],
                "elapsed_minutes": metadata["elapsed_minutes"],
                "peak_reserved_bytes": metadata["peak_reserved_bytes"],
                "optimizer_updates": metadata["optimizer_updates"],
            }
            if metadata["detection_loss_weight"]:
                threshold, score = choose_threshold(aggregate(rows))
                tuned = evaluate(rows, threshold=threshold)
                entry.update({"threshold": threshold, "mean_validation_detection_ba": score,
                              "metrics": {c: {k: tuned[c][k] for k in ("detection_auc", "detection_ba")}
                                          for c in CONDITIONS}})
            if metadata["source_loss_weight"]:
                entry["metrics"] = {c: {k: saved[c][k] for k in ("attribution_ba", "attribution_macro_f1")}
                                    for c in CONDITIONS}
            runs[name] = entry
    record = {
        "status": "validation_frozen_registered_controls",
        "register": "research/comparison_register.md v1.0 section 5",
        "rule": "Maximize mean video detection BA across validation ref/Q80/Q40 (detection-trained controls only)",
        "ties": "Closest to 0.5, then smaller threshold; tolerance 1e-12",
        "prediction_rule": "Fake when p_fake >= threshold",
        "untrained_heads": "C1-DET and C2-BIN source outputs and C1-ATT detection outputs are not interpreted",
        "test_scores_used": False,
        "manifest_sha256": sha256(manifest),
        "trainer_sha256": sha256(trainer),
        "script_sha256": sha256(Path(__file__)),
        "runs": runs,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps(runs, indent=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
