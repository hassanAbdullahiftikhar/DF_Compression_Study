"""Registered scorer for RGB models (F1-F3, C1, C2-BIN) and readout R (gates G2 and G4).

--split val  : G2 check. Rescore validation ref/Q80/Q40 from the frozen
               checkpoints and compare with the stored predictions. Writes
               only a verification report.
--split test : G4. Score test ref + Q95/Q80/Q60/Q40/Q20 once. Requires the
               G3 model-freeze manifest and refuses to overwrite.

RGB predictions use train_paired.predict at batch 8, exactly as the stored
validation predictions were made (cuDNN TF32 algorithm choice depends on
batch size). Readout R reuses fit_source_readout.extract at batch 8.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import SOURCES, audit, read_manifest
from fit_source_readout import extract
from metrics import evaluate
from model import TwoHeadResNet
from probe_validation import prediction_rows, softmax
from registered_models import TEST_QUALITIES, readout_models, rgb_models
from train_paired import predict

MANIFEST = Path("experiments/data/c23_main/reference.csv")
BATCH = 8
VAL_QUALITIES = (80, 40)
TOLERANCE = 1e-6
VERIFY_OUTPUT = Path("experiments/results/scorer_verification_rgb_validation.json")
TEST_OUTPUT = Path("experiments/runs/c23_test_rgb")
FIELDS = ("sample_id", "video_id", "group_id", "condition", "label", "source_index", "p_fake",
          *[f"p_source_{k}" for k in range(4)])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_model(run, expected_sha256, device):
    path = run / "best.pt"
    assert sha256(path) == expected_sha256, f"Checkpoint changed: {path}"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(saved["source_order"]) == SOURCES[1:]
    model = TwoHeadResNet(pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def readout_predictions(model, rows, qualities, parameters, device):
    """Readout R source probabilities with the encoder's native p_fake."""
    result = []
    for quality in (0, *qualities):
        condition = "ref" if quality == 0 else f"jpeg_q{quality}"
        features, p_fake, _ = extract(model, rows, quality, device)
        design = np.column_stack([(features.astype(np.float64) - parameters["scaler_mean"])
                                  / parameters["scaler_scale"], np.ones(len(features))])
        source = softmax(design @ parameters["src_coef"])
        result += prediction_rows([dict(r, condition=condition) for r in rows],
                                  np.column_stack([1 - p_fake, p_fake]), source)
    return result


def compare(stored, rescored):
    """Max absolute probability difference after checking identical keys and labels."""
    reference = {(r["sample_id"], r["condition"]): r for r in stored}
    assert len(reference) == len(stored) == len(rescored)
    worst = 0.0
    for row in rescored:
        old = reference[(row["sample_id"], row["condition"])]
        for field in ("video_id", "group_id"):
            assert old[field] == row[field]
        for field in ("label", "source_index"):
            assert int(old[field]) == int(row[field])
        for field in FIELDS[6:]:
            worst = max(worst, abs(float(old[field]) - float(row[field])))
    return worst


def metric_check(stored, rescored, threshold):
    left, right = evaluate(stored, threshold), evaluate(rescored, threshold)
    return max(abs(left[c][m] - right[c][m]) for c in left for m in left[c]
               if isinstance(left[c][m], float) and np.isfinite(left[c][m]))


def selftest():
    rows = [{"sample_id": "a:0", "video_id": "a", "group_id": "g", "condition": "ref", "label": 1,
             "source_index": 2, "p_fake": 0.7, "p_source_0": 0.1, "p_source_1": 0.2, "p_source_2": 0.6,
             "p_source_3": 0.1}]
    shifted = [dict(rows[0], p_source_2=0.6 + 2e-7, p_source_3=0.1 - 2e-7)]
    assert abs(compare(rows, shifted) - 2e-7) < 1e-12
    try:
        compare(rows, [dict(rows[0], label=0)])
    except AssertionError:
        pass
    else:
        raise AssertionError("Label mismatch accepted")
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main(split):
    assembly = json.loads(Path("experiments/data/c23_main/assembly.json").read_text(encoding="utf-8-sig"))
    assert sha256(MANIFEST) == assembly["reference_manifest_sha256"]
    rows = read_manifest(MANIFEST)
    checked = audit(rows, official_dir="experiments/data/splits")
    subset = [r for r in rows if r["split"] == split]
    if split == "val":
        if VERIFY_OUTPUT.exists():
            raise FileExistsError(VERIFY_OUTPUT)
        assert len(subset) == 5582
        qualities, output = VAL_QUALITIES, None
    else:
        from freeze_models import require_model_freeze
        freeze = require_model_freeze(Path(__file__))
        if TEST_OUTPUT.exists():
            raise FileExistsError(TEST_OUTPUT)
        assert len(subset) == 5581 and len({r["video_id"] for r in subset}) == 700
        qualities, output = TEST_QUALITIES, TEST_OUTPUT
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    if output:
        output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started, reports = time.perf_counter(), {}

    jobs = [(m, None) for m in rgb_models()] + [(m, m["parameters"]) for m in readout_models()]
    for model_entry, parameter_path in jobs:
        tick = time.perf_counter()
        model = load_model(model_entry["run"], model_entry["checkpoint_sha256"], device)
        if parameter_path is None:
            predictions = predict(model, subset, list(qualities), 2, BATCH, device)
        else:
            assert sha256(parameter_path) == model_entry["parameters_sha256"]
            with np.load(parameter_path, allow_pickle=False) as saved:
                parameters = {k: saved[k].copy() for k in saved.files}
            predictions = readout_predictions(model, subset, qualities, parameters, device)
        del model
        torch.cuda.empty_cache()
        report = {"rows": len(predictions), "seconds": time.perf_counter() - tick}
        if split == "val":
            stored_path = model_entry["validation_predictions"]
            assert sha256(stored_path) == model_entry["validation_predictions_sha256"]
            stored = read_csv(stored_path)
            report["max_abs_probability_difference"] = compare(stored, predictions)
            report["max_abs_metric_difference"] = metric_check(stored, predictions,
                                                               model_entry["threshold"] or 0.5)
            report["passed"] = report["max_abs_probability_difference"] <= TOLERANCE
        else:
            destination = output / model_entry["id"]
            destination.mkdir()
            write_csv(destination / "test_predictions.csv", predictions)
            report["test_predictions_sha256"] = sha256(destination / "test_predictions.csv")
        reports[model_entry["id"]] = report
        print(json.dumps({model_entry["id"]: report}), flush=True)

    record = {
        "status": "scorer_verification_validation" if split == "val" else "registered_test_scores",
        "register": "research/comparison_register.md v1.0 section 9",
        "split": split,
        "conditions": ["ref", *[f"jpeg_q{q}" for q in qualities]],
        "manifest_audit": checked,
        "manifest_sha256": sha256(MANIFEST),
        "batch": BATCH,
        "tolerance": TOLERANCE if split == "val" else None,
        "all_passed": all(r.get("passed", True) for r in reports.values()),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n)) for n in (
            "train_paired.py", "fit_source_readout.py", "model.py", "prepare_jpeg.py", "metrics.py",
            "probe_validation.py", "registered_models.py", "common.py")},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "models": reports,
    }
    if split == "test":
        record["model_freeze_manifest_sha256"] = freeze
    destination = VERIFY_OUTPUT if split == "val" else output / "report.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"all_passed": record["all_passed"], "saved": destination.as_posix()}))
    if not record["all_passed"]:
        raise SystemExit("Scorer verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.split:
        main(args.split)
    else:
        parser.error("--split or --selftest required")
