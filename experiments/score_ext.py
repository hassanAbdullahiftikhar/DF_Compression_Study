"""Extension scorer for X2, X5 and X6 (register Amendments 2 and 3).

--split val  : verification. EfficientNet-B4 runs must reproduce their stored
               validation predictions (train_paired.predict, batch 8), and the
               X2 readouts their stored validation Q20 predictions.
--split test : score EfficientNet-B4 and X2 once at test ref + Q95/Q80/Q60/Q40/Q20.
--codec      : score the frozen registered RGB models, the six F4 probes and
               EfficientNet-B4 once on the X5 codec crops.
Test modes require the extension freeze (Amendment 3) and refuse to overwrite.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch
from PIL import Image

from codec_sweep import CONDITIONS as CODEC_CONDITIONS, OUTPUT as CODEC_DIR
from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import extract_batch, probabilities
from fit_border_probes import load_dinov2
from fit_source_readout import extract
from freeze_ext_models import X2, effb4_runs, extension_models, sha256
from metrics import evaluate
from probe_validation import prediction_rows
from registered_models import TEST_QUALITIES, probe_models, rgb_models
from score_rgb import BATCH, FIELDS, compare, load_model, read_csv, write_csv
from train_effnet import EffNetTwoHead
from train_paired import predict

MANIFEST = Path("experiments/data/c23_main/reference.csv")
VERIFY_OUTPUT = Path("experiments/results/ext_scorer_verification_validation.json")
TEST_OUTPUT = Path("experiments/runs/c23_ext_test_jpeg")
CODEC_OUTPUT = Path("experiments/runs/c23_ext_test_codec")
TOLERANCE = 1e-6


def load_effb4(run, expected, device):
    path = Path(run) / "best.pt"
    assert sha256(path) == expected, f"Checkpoint changed: {path}"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(saved["source_order"]) == SOURCES[1:] and saved["backbone"] == "efficientnet_b4"
    model = EffNetTwoHead(pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def readout_rows(model, rows, quality, parameters, device):
    condition = "ref" if quality == 0 else f"jpeg_q{quality}"
    features, _, _ = extract(model, rows, quality, device)
    det, src = probabilities(features, parameters)
    return prediction_rows([dict(r, condition=condition) for r in rows], det, src)


def load_npz(path, expected):
    assert sha256(path) == expected, f"Parameters changed: {path}"
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def frozen_entries():
    models = extension_models()
    registered = {m["id"]: m for m in rgb_models()}
    return models, registered


def verify(test_rows_unused=None):
    if VERIFY_OUTPUT.exists():
        raise FileExistsError(VERIFY_OUTPUT)
    rows = read_manifest(MANIFEST)
    audit(rows, official_dir="experiments/data/splits")
    val = [r for r in rows if r["split"] == "val"]
    models, registered = frozen_entries()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    reports = {}
    for model_id, entry in models.items():
        if entry["kind"] == "effb4":
            (path, expected), = entry["artifacts_sha256"].items()
            model = load_effb4(Path(path).parent, expected, device)
            rescored = predict(model, val, [80, 40], 2, BATCH, device)
            stored = read_csv(Path(entry["run"]) / "validation_predictions.csv")
        else:
            encoder = registered[entry["encoder"]]
            assert encoder["checkpoint_sha256"] == entry["encoder_checkpoint_sha256"]
            model = load_model(encoder["run"], encoder["checkpoint_sha256"], device)
            (path, expected), = entry["artifacts_sha256"].items()
            rescored = readout_rows(model, val, 20, load_npz(path, expected), device)
            stored = read_csv(Path(path).parent / "validation_predictions.csv")
        del model
        torch.cuda.empty_cache()
        difference = compare(stored, rescored)
        reports[model_id] = {"rows": len(rescored), "max_abs_probability_difference": difference,
                             "passed": difference <= TOLERANCE}
        print(json.dumps({model_id: reports[model_id]}), flush=True)
    record = {"status": "extension_scorer_verification_validation", "tolerance": TOLERANCE,
              "all_passed": all(r["passed"] for r in reports.values()),
              "script_sha256": sha256(Path(__file__)), "models": reports}
    with VERIFY_OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"all_passed": record["all_passed"]}))
    if not record["all_passed"]:
        raise SystemExit("Extension scorer verification failed")


def save(output, model_id, predictions):
    destination = output / model_id
    destination.mkdir()
    write_csv(destination / "test_predictions.csv", predictions)
    return sha256(destination / "test_predictions.csv")


def score_test():
    from freeze_ext_models import require_extension_freeze
    freeze = require_extension_freeze(Path(__file__))
    if TEST_OUTPUT.exists():
        raise FileExistsError(TEST_OUTPUT)
    rows = read_manifest(MANIFEST)
    audit(rows, official_dir="experiments/data/splits")
    test = [r for r in rows if r["split"] == "test"]
    assert len(test) == 5581
    models, registered = frozen_entries()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    TEST_OUTPUT.mkdir(parents=True)
    reports, started = {}, time.perf_counter()
    for model_id, entry in models.items():
        (path, expected), = entry["artifacts_sha256"].items()
        if entry["kind"] == "effb4":
            model = load_effb4(Path(path).parent, expected, device)
            predictions = predict(model, test, list(TEST_QUALITIES), 2, BATCH, device)
        else:
            encoder = registered[entry["encoder"]]
            model = load_model(encoder["run"], encoder["checkpoint_sha256"], device)
            parameters = load_npz(path, expected)
            predictions = [r for q in (0, *TEST_QUALITIES) for r in readout_rows(model, test, q, parameters, device)]
        del model
        torch.cuda.empty_cache()
        reports[model_id] = {"rows": len(predictions), "test_predictions_sha256": save(TEST_OUTPUT, model_id, predictions)}
        print(json.dumps({model_id: reports[model_id]}), flush=True)
    finish(TEST_OUTPUT, "extension_test_scores_jpeg", freeze, reports, started)


def codec_rows():
    report = json.loads((CODEC_DIR / "report.json").read_text(encoding="utf-8"))
    assert sha256(CODEC_DIR / "crops.csv") == report["crops_sha256"]
    rows = {r["sample_id"]: r for r in read_manifest(MANIFEST) if r["split"] == "test"}
    paths = {}
    for record in read_csv(CODEC_DIR / "crops.csv"):
        paths[(record["sample_id"], record["condition"])] = record["path"]
    ordered = [rows[k] for k in rows]
    return {c: [dict(r, reference_path=paths[(r["sample_id"], c)], path=paths[(r["sample_id"], c)])
                for r in ordered] for c in CODEC_CONDITIONS}, report


def score_codec():
    from freeze_ext_models import require_extension_freeze
    freeze = require_extension_freeze(Path(__file__))
    if CODEC_OUTPUT.exists():
        raise FileExistsError(CODEC_OUTPUT)
    by_condition, codec_report = codec_rows()
    models, _ = frozen_entries()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    CODEC_OUTPUT.mkdir(parents=True)
    reports, started = {}, time.perf_counter()
    jobs = [(m["id"], lambda m=m: load_model(m["run"], m["checkpoint_sha256"], device))
            for m in rgb_models()]
    jobs += [(model_id, lambda e=e: load_effb4(Path(next(iter(e["artifacts_sha256"]))).parent,
                                               next(iter(e["artifacts_sha256"].values())), device))
             for model_id, e in models.items() if e["kind"] == "effb4"]
    for model_id, loader in jobs:
        model = loader()
        predictions = []
        for condition, rows in by_condition.items():
            # Quality 0 feeds the stored codec crop unchanged; relabel the condition.
            predictions += [dict(r, condition=condition) for r in predict(model, rows, [], 2, BATCH, device)]
        del model
        torch.cuda.empty_cache()
        reports[model_id] = {"rows": len(predictions), "test_predictions_sha256": save(CODEC_OUTPUT, model_id, predictions)}
        print(json.dumps({model_id: reports[model_id]}), flush=True)

    probes = [m for m in probe_models() if m["family"] == "F4"]
    parameters = {m["id"]: load_npz(m["parameters"], m["parameters_sha256"]) for m in probes}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    dino, _ = load_dinov2()
    outputs = {m["id"]: [] for m in probes}
    for condition, rows in by_condition.items():
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            images = []
            for row in batch:
                with Image.open(row["reference_path"]) as handle:
                    images.append(handle.convert("RGB"))
            features = extract_batch(images, dino)
            variants = [dict(r, condition=condition) for r in batch]
            for m in probes:
                det, src = probabilities(features[m["rep"]], parameters[m["id"]])
                outputs[m["id"]] += prediction_rows(variants, det, src)
        print(json.dumps({"probes_condition_done": condition}), flush=True)
    for m in probes:
        reports[m["id"]] = {"rows": len(outputs[m["id"]]),
                            "test_predictions_sha256": save(CODEC_OUTPUT, m["id"], outputs[m["id"]])}
    finish(CODEC_OUTPUT, "extension_test_scores_codec", freeze, reports, started,
           {"codec_report_sha256": sha256(CODEC_DIR / "report.json"), "conditions": list(CODEC_CONDITIONS)})


def finish(output, status, freeze, reports, started, extra=None):
    record = {"status": status, "register": "research/comparison_register.md Amendments 2 and 3",
              "extension_freeze_manifest_sha256": freeze, "manifest_sha256": sha256(MANIFEST), "batch": BATCH,
              "script_sha256": sha256(Path(__file__)), "torch": torch.__version__,
              "elapsed_seconds": time.perf_counter() - started, "models": reports, **(extra or {})}
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--codec", action="store_true")
    args = parser.parse_args()
    if args.codec:
        score_codec()
    elif args.split == "val":
        verify()
    elif args.split == "test":
        score_test()
    else:
        parser.error("--split val|test or --codec required")
