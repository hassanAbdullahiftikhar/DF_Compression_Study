"""Extension freeze (register Amendment 3): thresholds and hashes for X1b, X2 and X6 before test scoring.

--thresholds : verify the EfficientNet-B4 runs and freeze their thresholds
               with the F1 rule (validation only).
--manifest   : write research/extension_freeze_manifest.json after the
               extension scorers pass validation verification. Amendment 3
               must then cite its hash. require_extension_freeze() is the
               guard used by every extension test scorer and analysis.
"""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from evaluate_probe_compression import choose_threshold
from metrics import aggregate, evaluate

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "research/comparison_register.md"
MANIFEST_OUT = ROOT / "research/extension_freeze_manifest.json"
THRESHOLDS = ROOT / "experiments/results/effb4_validation_thresholds.json"
X1B = ROOT / "experiments/runs/c23_ext_x1b_quality_thresholds_validation/report.json"
X2 = ROOT / "experiments/runs/c23_ext_x2_q20_readout_validation/report.json"
VERIFY = ROOT / "experiments/results/ext_scorer_verification_validation.json"
CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
SEEDS = (42, 43, 44)
SCRIPTS = ("train_effnet.py", "freeze_ext_models.py", "calibrate_quality_thresholds.py", "fit_q20_readout.py",
           "codec_sweep.py", "score_ext.py", "analyze_ext.py", "analyze_test.py", "registered_models.py",
           "train_paired.py", "model.py", "metrics.py", "research_metrics.py", "prepare_jpeg.py", "common.py",
           "fit_source_readout.py", "fit_known_quality_probes.py", "evaluate_probe_compression.py", "features.py",
           "probe_validation.py", "fit_border_probes.py", "score_rgb.py", "freeze_models.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effb4_runs():
    for recipe in ("f1", "f2"):
        for seed in SEEDS:
            yield f"EB4-{recipe.upper()}_s{seed}", ROOT / f"experiments/runs/c23_ext_effb4_{recipe}_seed{seed}"


def freeze_thresholds():
    if THRESHOLDS.exists():
        raise FileExistsError(THRESHOLDS)
    trainer = ROOT / "experiments/train_effnet.py"
    runs = {}
    for model_id, run in effb4_runs():
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["test_scores_used"] is False and metadata["script_sha256"] == sha256(trainer)
        assert metadata["checkpoint_sha256"] == sha256(run / "best.pt")
        assert metadata["validation_predictions_sha256"] == sha256(run / "validation_predictions.csv")
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        selected = max(history, key=lambda e: e["selection_score"])
        assert len(history) == 20 and selected["epoch"] == metadata["selected_epoch"]
        with (run / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        saved = evaluate(rows, 0.5)
        for c in CONDITIONS:
            for metric in ("detection_ba", "attribution_ba"):
                assert abs(saved[c][metric] - selected["validation"][c][metric]) < 1e-10
        threshold, score = choose_threshold(aggregate(rows))
        tuned = evaluate(rows, threshold)
        runs[model_id] = {"run": run.relative_to(ROOT).as_posix(), "selected_epoch": selected["epoch"],
                          "selection_score": selected["selection_score"], "threshold": threshold,
                          "mean_validation_detection_ba": score,
                          "checkpoint_sha256": metadata["checkpoint_sha256"],
                          "validation_predictions_sha256": metadata["validation_predictions_sha256"],
                          "elapsed_minutes": metadata["elapsed_minutes"],
                          "metrics": {c: {k: tuned[c][k] for k in ("detection_auc", "detection_ba", "attribution_ba")}
                                      for c in CONDITIONS}}
    record = {"status": "validation_frozen_post_freeze_extension", "extension": "X6 EfficientNet-B4",
              "rule": "Maximize mean video detection BA across validation ref/Q80/Q40; closest to 0.5, then smaller",
              "test_scores_used": False, "trainer_sha256": sha256(trainer), "script_sha256": sha256(Path(__file__)),
              "runs": runs}
    with THRESHOLDS.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({k: (v["selected_epoch"], round(v["selection_score"], 4), round(v["threshold"], 4))
                      for k, v in runs.items()}, indent=2))


def extension_models():
    """Frozen extension models: EfficientNet-B4 runs and the X2 readouts."""
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    x2 = json.loads(X2.read_text(encoding="utf-8"))
    models = {}
    for model_id, run in effb4_runs():
        entry = thresholds["runs"][model_id]
        models[model_id] = {"kind": "effb4", "run": entry["run"], "threshold": entry["threshold"],
                            "artifacts_sha256": {f"{entry['run']}/best.pt": entry["checkpoint_sha256"]},
                            "validation_predictions_sha256": entry["validation_predictions_sha256"]}
    for seed in SEEDS:
        entry = x2["models"][f"F1_s{seed}"]
        folder = f"experiments/runs/c23_ext_x2_q20_readout_validation/F1_s{seed}"
        models[f"X2-F1_s{seed}"] = {"kind": "q20_readout", "encoder": f"F1_s{seed}", "threshold": entry["threshold"],
                                    "artifacts_sha256": {f"{folder}/readout_parameters.npz": entry["parameters_sha256"]},
                                    "encoder_checkpoint_sha256": entry["encoder_checkpoint_sha256"],
                                    "validation_predictions_sha256": entry["validation_predictions_sha256"]}
    return models


def require_extension_freeze(script):
    """Return the extension manifest hash if Amendment 3 cites it and nothing listed changed."""
    if not MANIFEST_OUT.exists():
        raise RuntimeError("Extension freeze manifest missing; extension test scoring is not allowed")
    manifest = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
    digest = sha256(MANIFEST_OUT)
    text = REGISTER.read_text(encoding="utf-8")
    if "Amendment 3" not in text or digest not in text:
        raise RuntimeError("Register Amendment 3 citing the extension manifest is missing")
    assert f"experiments/{Path(script).name}" in manifest["scripts_sha256"]
    for path, expected in manifest["scripts_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Frozen code changed: {path}"
    for model_id, entry in manifest["models"].items():
        for path, expected in entry["artifacts_sha256"].items():
            assert sha256(ROOT / path) == expected, f"Artifact changed: {path}"
    assert sha256(X1B) == manifest["x1b_report_sha256"]
    return digest


def write_manifest():
    if MANIFEST_OUT.exists():
        raise FileExistsError(MANIFEST_OUT)
    verification = json.loads(VERIFY.read_text(encoding="utf-8"))
    assert verification["all_passed"] is True
    x1b = json.loads(X1B.read_text(encoding="utf-8"))
    models = extension_models()
    for entry in models.values():
        for path, expected in entry["artifacts_sha256"].items():
            assert sha256(ROOT / path) == expected
    record = {
        "status": "extension_freeze_manifest_before_extension_test_scoring",
        "register": "research/comparison_register.md Amendments 2 and 3",
        "frozen_on": date.today().isoformat(),
        "models": models,
        "x1b_thresholds": {k: v["thresholds"] for k, v in x1b["models"].items()},
        "x1b_report_sha256": sha256(X1B),
        "x2_report_sha256": sha256(X2),
        "effb4_thresholds_sha256": sha256(THRESHOLDS),
        "verification_sha256": sha256(VERIFY),
        "scripts_sha256": {f"experiments/{n}": sha256(ROOT / "experiments" / n) for n in SCRIPTS},
        "qualification": "Post-freeze extensions motivated in part by the registered test results; "
                         "selected on validation only; each is scored once on test.",
    }
    with MANIFEST_OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": MANIFEST_OUT.relative_to(ROOT).as_posix(), "sha256": sha256(MANIFEST_OUT),
                      "models": len(models)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    if args.thresholds:
        freeze_thresholds()
    elif args.manifest:
        write_manifest()
    else:
        parser.error("--thresholds or --manifest required")
