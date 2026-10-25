"""Extension X2 (register Amendment 2): frozen F1 encoder, ridge heads refitted at Q20.

Features are the 512-d pooled outputs of each frozen reference-trained F1
encoder for training and validation crops compressed at Q20. Detection and
source ridge heads use the known-quality probe procedure (fit_known_quality_probes.fit):
training-only standardization, penalty by mean validation det/src BA at Q20,
threshold maximizing validation video detection BA at Q20. Validation only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import write_csv
from fit_known_quality_probes import compact, fit, threshold_for
from fit_source_readout import extract
from metrics import evaluate
from registered_models import rgb_models
from score_rgb import load_model, sha256

QUALITY, CONDITION = 20, "jpeg_q20"
MANIFEST = Path("experiments/data/c23_main/train_val_reference.csv")
OUTPUT = Path("experiments/runs/c23_ext_x2_q20_readout_validation")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = read_manifest(MANIFEST)
    checked = audit(rows, official_dir="experiments/data/splits")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    assert len(train) == 28735 and len(val) == 5582
    targets = np.array([SOURCES.index(r["source"]) - 1 for r in train])
    val_rows = [dict(r, condition=CONDITION) for r in val]
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    OUTPUT.mkdir(parents=True)
    started, reports = time.perf_counter(), {}
    for entry in [m for m in rgb_models() if m["family"] == "F1"]:
        model = load_model(entry["run"], entry["checkpoint_sha256"], device)
        train_raw, _, _ = extract(model, train, QUALITY, device)
        val_raw, native_fake, native_source = extract(model, val, QUALITY, device)
        del model
        torch.cuda.empty_cache()
        train_raw, val_raw = train_raw.astype(np.float64), val_raw.astype(np.float64)
        mean, scale = train_raw.mean(axis=0), train_raw.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design = lambda x: np.column_stack([(x - mean) / scale, np.ones(len(x))])
        best, search = fit(design(train_raw), design(val_raw), targets, val_rows, CONDITION)
        threshold = threshold_for(best["predictions"], CONDITION)
        destination = OUTPUT / entry["id"]
        destination.mkdir()
        write_csv(destination / "validation_predictions.csv", best["predictions"])
        np.savez_compressed(destination / "readout_parameters.npz", scaler_mean=mean, scaler_scale=scale,
                            det_coef=best["det_coef"], src_coef=best["src_coef"])
        native_rows = [dict(r, p_fake=float(native_fake[i]), **{f"p_source_{k}": float(native_source[i, k])
                                                                 for k in range(4)}) for i, r in enumerate(val_rows)]
        for r in native_rows:
            r["label"] = int(r["source"] != "real")
            r["source_index"] = SOURCES.index(r["source"]) - 1
        reports[entry["id"]] = {
            "encoder_checkpoint_sha256": entry["checkpoint_sha256"],
            "real_labels": compact(best),
            "alpha_search": search,
            "threshold": threshold,
            "readout_validation": evaluate(best["predictions"], threshold)[CONDITION],
            "native_validation_at_frozen_threshold": evaluate(native_rows, entry["threshold"])[CONDITION],
            "parameters_sha256": sha256(destination / "readout_parameters.npz"),
            "validation_predictions_sha256": sha256(destination / "validation_predictions.csv"),
        }
        print(json.dumps({entry["id"]: {k: reports[entry["id"]][k] for k in ("real_labels", "threshold")}}), flush=True)
    record = {
        "status": "validation_only_post_freeze_extension",
        "extension": "X2 frozen-encoder Q20 readout, research/comparison_register.md Amendment 2",
        "test_scores_used": False,
        "manifest_audit": checked,
        "manifest_sha256": sha256(MANIFEST),
        "features": "512-d pooled frozen F1 encoder output, eval mode, batch 8, Pillow JPEG Q20 4:2:0",
        "procedure": "fit_known_quality_probes.fit and threshold_for at jpeg_q20",
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
        "qualification": "Validation-selected and optimistic. A linear readout measures linearly available "
                         "information in the frozen encoder, not physical presence or absence.",
        "models": reports,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", OUTPUT / "report.json")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
