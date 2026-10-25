"""Extension X1b (register Amendment 2): per-quality detection thresholds chosen on validation.

Scores validation at Q95/Q60/Q20 with the frozen detection-trained RGB
checkpoints (F1-F3, C1-DET, C2-BIN) using train_paired.predict at batch 8,
then picks, per model and quality, the threshold maximizing validation video
detection BA at that quality (section 4 tie rule). Validation only.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from common import audit, read_manifest
from fit_known_quality_probes import threshold_for
from metrics import evaluate
from registered_models import rgb_models
from score_rgb import BATCH, MANIFEST, load_model, sha256
from train_paired import predict

QUALITIES = (95, 60, 20)
OUTPUT = Path("experiments/runs/c23_ext_x1b_quality_thresholds_validation")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = read_manifest(MANIFEST)
    checked = audit(rows, official_dir="experiments/data/splits")
    val = [r for r in rows if r["split"] == "val"]
    assert len(val) == 5582
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    OUTPUT.mkdir(parents=True)
    started, models = time.perf_counter(), {}
    for entry in rgb_models():
        if "detection" not in entry["tasks"]:
            continue
        model = load_model(entry["run"], entry["checkpoint_sha256"], device)
        # predict() always includes ref; keep only the held-out qualities.
        predictions = [r for r in predict(model, val, list(QUALITIES), 2, BATCH, device) if r["condition"] != "ref"]
        del model
        torch.cuda.empty_cache()
        destination = OUTPUT / entry["id"]
        destination.mkdir()
        with (destination / "validation_heldout_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        report = {"frozen_threshold": entry["threshold"], "thresholds": {}, "validation": {},
                  "predictions_sha256": sha256(destination / "validation_heldout_predictions.csv")}
        for quality in QUALITIES:
            condition = f"jpeg_q{quality}"
            subset = [r for r in predictions if r["condition"] == condition]
            threshold = threshold_for(subset, condition)
            report["thresholds"][condition] = threshold
            report["validation"][condition] = {
                "calibrated": {k: evaluate(subset, threshold)[condition][k] for k in ("detection_auc", "detection_ba")},
                "frozen": {"detection_ba": evaluate(subset, entry["threshold"])[condition]["detection_ba"]}}
        models[entry["id"]] = report
        print(json.dumps({entry["id"]: report["thresholds"]}), flush=True)
    record = {
        "status": "validation_only_post_freeze_extension",
        "extension": "X1b per-quality calibration, research/comparison_register.md Amendment 2",
        "test_scores_used": False,
        "qualities": [f"jpeg_q{q}" for q in QUALITIES],
        "rule": "Maximize validation video detection BA at each quality; closest to 0.5, then smaller (1e-12)",
        "manifest_audit": checked,
        "manifest_sha256": sha256(MANIFEST),
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
        "qualification": "Validation at Q95/Q60/Q20 was never used before; these thresholds are a post-freeze "
                         "extension motivated by the test result and are reported as exploratory.",
        "models": models,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", OUTPUT / "report.json")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
