"""Gate G3: model-freeze manifest written before any test scoring.

Lists every registered model's checkpoint/head hashes and frozen threshold,
the verified scorer and analysis code, and the G2 verification reports.
The register then receives a dated amendment citing this manifest's hash.
Test scorers call require_model_freeze() and refuse to run unless the
manifest, the amendment and every listed hash still match.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import PIL
from PIL import Image

from freeze_register import REGISTER, test_prediction_scan
from registered_models import TEST_CONDITIONS, all_models, records

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/model_freeze_manifest.json"
REGISTER_FREEZE = ROOT / "research/comparison_register_freeze.json"
SCRIPTS = (
    "score_rgb.py", "score_probes.py", "analyze_test.py", "registered_models.py", "freeze_models.py",
    "train_paired.py", "train_control.py", "fit_source_readout.py", "model.py", "prepare_jpeg.py", "metrics.py",
    "research_metrics.py", "probe_validation.py", "evaluate_probe_compression.py", "fit_known_quality_probes.py",
    "fit_border_probes.py", "fit_metadata_probe.py", "features.py", "common.py", "eight_frame_sensitivity.py",
    "freeze_control_thresholds.py", "content_decomposition.py", "freeze_register.py",
)
VERIFICATION = (
    "experiments/results/scorer_verification_rgb_validation.json",
    "experiments/results/scorer_verification_probes_validation.json",
    "experiments/results/test_analysis_dryrun_validation.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts(model):
    files = {}
    if model["kind"] in ("rgb", "readout"):
        files[(model["run"] / "best.pt").as_posix()] = model["checkpoint_sha256"]
    if model["kind"] in ("readout", "probe", "metadata"):
        files[Path(model["parameters"]).as_posix()] = model["parameters_sha256"]
    return files


def model_entries():
    entries = {}
    for model in all_models():
        files = artifacts(model)
        for path, expected in files.items():
            assert sha256(ROOT / path) == expected, f"Artifact changed: {path}"
        entries[model["id"]] = {
            "family": model["family"], "kind": model["kind"], "tasks": list(model["tasks"]),
            "threshold": model["threshold"], "artifacts_sha256": files,
            "validation_predictions": Path(model["validation_predictions"]).as_posix(),
            "validation_predictions_sha256": model["validation_predictions_sha256"],
            "frozen_record": model["record"],
            **({"encoder": model["encoder"]} if "encoder" in model else {}),
        }
    return entries


def require_model_freeze(script):
    """Return the manifest hash if G3 is complete and nothing listed has changed."""
    if not OUTPUT.exists():
        raise RuntimeError("G3 model-freeze manifest missing; test scoring is not allowed")
    manifest = json.loads(OUTPUT.read_text(encoding="utf-8"))
    digest = sha256(OUTPUT)
    register = REGISTER.read_bytes()
    original = register[:manifest["register_v1_bytes"]]
    assert hashlib.sha256(original).hexdigest() == manifest["register_v1_sha256"], "Register sections 1-10 changed"
    amendment = register[manifest["register_v1_bytes"]:].decode("utf-8")
    if "Amendment 1" not in amendment or digest not in amendment:
        raise RuntimeError("Register amendment citing the model-freeze manifest is missing")
    name = f"experiments/{Path(script).name}"
    assert name in manifest["scripts_sha256"], f"{name} is not a frozen scorer"
    for path, expected in manifest["scripts_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Frozen code changed: {path}"
    current = model_entries()
    assert set(current) == set(manifest["models"])
    for model_id, entry in manifest["models"].items():
        assert current[model_id]["artifacts_sha256"] == entry["artifacts_sha256"], model_id
        assert current[model_id]["threshold"] == entry["threshold"], model_id
    return digest


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    freeze = json.loads(REGISTER_FREEZE.read_text(encoding="utf-8"))
    register = REGISTER.read_bytes()
    assert hashlib.sha256(register).hexdigest() == freeze["register_sha256"]["working_tree_sha256"], \
        "Register changed after the v1.0 freeze"
    scan = test_prediction_scan()
    if scan["files_with_test_videos"]:
        raise RuntimeError(f"Test predictions already exist: {scan}")
    verification = {}
    for path in VERIFICATION:
        report = json.loads((ROOT / path).read_text(encoding="utf-8"))
        if "all_passed" in report:
            assert report["all_passed"] is True, path
        else:
            assert report["validation_cross_check"]["max_abs_difference"] < 1e-10, path
        verification[path] = sha256(ROOT / path)
    preflight = json.loads((ROOT / "experiments/results/dinov2_preflight.json").read_text(encoding="utf-8"))
    models = model_entries()
    record = {
        "status": "model_freeze_manifest_before_test_scoring",
        "gate": "G3, research/comparison_register.md v1.0 section 9",
        "frozen_on": date.today().isoformat(),
        "register_v1_sha256": hashlib.sha256(register).hexdigest(),
        "register_v1_bytes": len(register),
        "test_scoring_status": "not_started",
        "test_prediction_scan": scan,
        "test_conditions": list(TEST_CONDITIONS),
        "jpeg": {"encoder": "Pillow", "pillow": PIL.__version__, "jpeg_library": Image.core.jpeglib_version,
                 "subsampling": "4:2:0", "optimize": False, "progressive": False},
        "prediction_rule": "Fake when p_fake >= frozen threshold; attribution = argmax of four source probabilities",
        "dinov2": {"source_commit": preflight["source_commit"], "checkpoint_sha256": preflight["checkpoint_sha256"]},
        "model_count": len(models),
        "models": models,
        "frozen_records_sha256": {p.as_posix(): sha256(ROOT / p) for p in records()},
        "scripts_sha256": {f"experiments/{n}": sha256(ROOT / "experiments" / n) for n in SCRIPTS},
        "g2_verification_sha256": verification,
        "qualification": "Thresholds, heads and checkpoints were selected on validation only. Nothing listed "
                         "here may change after test scoring starts (register section 9).",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": OUTPUT.relative_to(ROOT).as_posix(), "sha256": sha256(OUTPUT),
                      "models": len(models), **scan}, indent=2))


if __name__ == "__main__":
    main()
