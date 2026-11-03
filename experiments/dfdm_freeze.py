"""DFDM gate D-G3: model-freeze manifest and the guard used by every DFDM test scorer."""
import json
from datetime import date
from pathlib import Path

from dfdm_lib import ROOT, sha256

MANIFEST = ROOT / "research/dfdm_model_freeze_manifest.json"
REGISTER = ROOT / "research/dfdm_register.md"
REGISTER_FREEZE = ROOT / "research/dfdm_register_freeze.json"
THRESHOLDS = ROOT / "experiments/results/dfdm_validation_thresholds.json"
PROBES = ROOT / "experiments/runs/dfdm_probes_validation/report.json"
VERIFY = (ROOT / "experiments/results/dfdm_rgb_verification_validation.json",
          ROOT / "experiments/results/dfdm_probes_verification_validation.json")
SCRIPTS = ("dfdm_lib.py", "dfdm_crops.py", "dfdm_train.py", "dfdm_probes.py", "dfdm_score.py", "dfdm_freeze.py",
           "dfdm_analyze.py", "audit_dfdm.py", "freeze_dfdm_register.py", "model.py", "metrics.py", "prepare_jpeg.py",
           "probe_validation.py", "evaluate_probe_compression.py", "fit_border_probes.py", "clip_probes.py",
           "analyze_ext.py", "analyze_test.py", "research_metrics.py", "codec_sweep.py", "extract_reference_faces.py")


def write_manifest():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    for path in VERIFY:
        assert json.loads(path.read_text(encoding="utf-8"))["all_passed"] is True, path
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    models = {}
    for model_id, entry in thresholds["runs"].items():
        assert sha256(ROOT / entry["run"] / "best.pt") == entry["checkpoint_sha256"]
        models[model_id] = {"kind": "rgb", "threshold": entry["threshold"],
                            "artifacts_sha256": {f"{entry['run']}/best.pt": entry["checkpoint_sha256"]}}
    for probe, entry in probes["probes"].items():
        path = f"experiments/runs/dfdm_probes_validation/{probe}/probe_parameters.npz"
        assert sha256(ROOT / path) == entry["parameters_sha256"]
        models[probe] = {"kind": "probe", "threshold": entry["threshold"], "artifacts_sha256": {path: entry["parameters_sha256"]}}
    record = {"status": "dfdm_model_freeze_before_test_scoring", "gate": "D-G3", "frozen_on": date.today().isoformat(),
              "register_v1_sha256": json.loads(REGISTER_FREEZE.read_text(encoding="utf-8"))["register_sha256"]["working_tree_sha256"],
              "models": models, "thresholds_sha256": sha256(THRESHOLDS), "probes_report_sha256": sha256(PROBES),
              "verification_sha256": {p.name: sha256(p) for p in VERIFY},
              "crops_report_sha256": sha256(ROOT / "experiments/data/dfdm_crops/report.json"),
              "scripts_sha256": {f"experiments/{n}": sha256(ROOT / "experiments" / n) for n in SCRIPTS}}
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": MANIFEST.relative_to(ROOT).as_posix(), "sha256": sha256(MANIFEST), "models": len(models)}))


def require_dfdm_freeze(script):
    if not MANIFEST.exists():
        raise RuntimeError("DFDM model-freeze manifest missing; test scoring not allowed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = sha256(MANIFEST)
    text = REGISTER.read_text(encoding="utf-8")
    if "Amendment D-2" not in text or digest not in text:
        raise RuntimeError("DFDM register amendment citing the model-freeze manifest is missing")
    assert f"experiments/{Path(script).name}" in manifest["scripts_sha256"]
    for path, expected in manifest["scripts_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Frozen code changed: {path}"
    for entry in manifest["models"].values():
        for path, expected in entry["artifacts_sha256"].items():
            assert sha256(ROOT / path) == expected, f"Artifact changed: {path}"
    return digest


if __name__ == "__main__":
    write_manifest()
