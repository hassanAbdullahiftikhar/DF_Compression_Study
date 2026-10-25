"""Registered control C3a: face-masked border probes on train/validation only.

Each crop's stored YuNet face box is mapped into the 224x224 crop, expanded
by 10% of its width/height on each side, clipped and filled with gray 128.
JPEG is applied before the mask. The six reference-probe representations
are refitted with the reference-probe (F4) procedure.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
from PIL import Image, ImageDraw

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import (
    BATCH, CONDITIONS, DIMS, REPS, choose_threshold, extract_batch, sha256, write_csv,
)
from metrics import aggregate, evaluate
from prepare_jpeg import jpeg_sequence
from probe_validation import fit_probe, prediction_rows, shuffle_video_labels, softmax

MASK_EXPANSION = 0.10
MASK_GRAY = 128
METADATA_DIRS = ("c23_smoke", "c23_remaining", "c23_main_remaining")
OUTPUT = Path("experiments/runs/c23_control_border_probes_validation")


def load_crop_metadata():
    metadata = {}
    for name in METADATA_DIRS:
        path = Path("experiments/data") / name / "crop_metadata.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["sample_id"] in metadata:
                    raise ValueError(f"Duplicate crop metadata: {record['sample_id']}")
                metadata[record["sample_id"]] = record
    return metadata


def mask_box(record, size=224):
    """Expanded face box in crop pixel coordinates (left, top, right, bottom)."""
    left, top, right, bottom = record["crop_box"]
    side = right - left
    assert side == bottom - top and side > 0
    scale = size / side
    x, y, width, height = record["face_xywh"]
    x0 = (x - MASK_EXPANSION * width - left) * scale
    y0 = (y - MASK_EXPANSION * height - top) * scale
    x1 = (x + (1 + MASK_EXPANSION) * width - left) * scale
    y1 = (y + (1 + MASK_EXPANSION) * height - top) * scale
    # The 1e-9 tolerance keeps float error from widening an exact edge by a pixel.
    box = [int(np.floor(max(0.0, x0) + 1e-9)), int(np.floor(max(0.0, y0) + 1e-9)),
           int(np.ceil(min(float(size), x1) - 1e-9)), int(np.ceil(min(float(size), y1) - 1e-9))]
    assert box[0] < box[2] and box[1] < box[3]
    return box


def apply_mask(image, box):
    masked = image.copy()
    # PIL rectangles include the far edge, so subtract one pixel.
    ImageDraw.Draw(masked).rectangle(
        [box[0], box[1], box[2] - 1, box[3] - 1], fill=(MASK_GRAY,) * 3
    )
    return masked


def masked_image(row, record, quality):
    with Image.open(row["reference_path"]) as handle:
        image = handle.convert("RGB")
    assert image.size == (224, 224)
    if quality:
        image = jpeg_sequence(image, [quality], subsampling=2)[1]
    return apply_mask(image, mask_box(record))


def load_dinov2():
    import torch
    preflight = json.loads(Path("experiments/results/dinov2_preflight.json").read_text(encoding="utf-8"))
    checkpoint = Path(preflight["checkpoint_path"])
    assert sha256(checkpoint) == preflight["checkpoint_sha256"]
    repository = Path(torch.hub.get_dir()) / f"facebookresearch_dinov2_{preflight['source_commit']}"
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    model = torch.hub.load(str(repository), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device="cuda", dtype=torch.float32), preflight


def selftest():
    record = {"crop_box": [100, 50, 200, 150], "face_xywh": [120.0, 70.0, 50.0, 60.0]}
    box = mask_box(record)
    # Face 120..170 x 70..130, expanded by 5/6 px, then scaled by 2.24.
    assert box == [33, 31, 168, 193], box
    image = Image.new("RGB", (224, 224), (10, 20, 30))
    masked = np.asarray(apply_mask(image, box))
    assert (masked[box[1]:box[3], box[0]:box[2]] == MASK_GRAY).all()
    assert (masked[:box[1]] == [10, 20, 30]).all() and (masked[:, box[2]:] == [10, 20, 30]).all()
    clipped = mask_box({"crop_box": [0, 0, 100, 100], "face_xywh": [-5.0, 60.0, 50.0, 45.0]})
    assert clipped[0] == 0 and clipped[3] == 224
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    import torch
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = Path("experiments/data/c23_main/train_val_reference.csv")
    rows = read_manifest(manifest)
    checked = audit(rows, official_dir="experiments/data/splits")
    assert len(rows) == 34317 and all(r["condition"] == "ref" for r in rows)
    metadata = load_crop_metadata()
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    assert len(train) == 28735 and len(val) == 5582

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, preflight = load_dinov2()

    jobs = [("train", "ref", 0)] + [("val", c, q) for c, q in zip(CONDITIONS, (0, 80, 40))]
    matrices = {}
    mask_area = []
    for split, condition, quality in jobs:
        subset = train if split == "train" else val
        matrix = {rep: np.empty((len(subset), DIMS[rep]), dtype=np.float32) for rep in REPS}
        for start in range(0, len(subset), BATCH):
            batch = subset[start:start + BATCH]
            images = [masked_image(r, metadata[r["sample_id"]], quality) for r in batch]
            if split == "train":
                for r in batch:
                    x0, y0, x1, y1 = mask_box(metadata[r["sample_id"]])
                    mask_area.append((x1 - x0) * (y1 - y0) / 224 ** 2)
            features = extract_batch(images, model)
            for rep in REPS:
                matrix[rep][start:start + len(batch)] = features[rep]
            if (start + len(batch)) % 4000 < BATCH or start + len(batch) == len(subset):
                print(json.dumps({"split": split, "condition": condition,
                                  "crops_completed": start + len(batch), "crops_total": len(subset)}), flush=True)
        matrices[(split, condition)] = matrix
    torch.cuda.synchronize()
    peak_reserved = torch.cuda.max_memory_reserved()
    del model
    torch.cuda.empty_cache()

    targets = np.array([SOURCES.index(r["source"]) - 1 for r in train])
    video_ids = [r["video_id"] for r in train]
    val_rows = {c: [dict(r, condition=c) for r in val] for c in CONDITIONS}
    OUTPUT.mkdir(parents=True)
    reports = {}
    for rep in REPS:
        raw = matrices[("train", "ref")][rep].astype(np.float64)
        assert np.isfinite(raw).all()
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale < 1e-12] = 1.0

        def design(values):
            return np.column_stack([(values.astype(np.float64) - mean) / scale, np.ones(len(values))])

        train_x = design(raw)
        val_x = {c: design(matrices[("val", c)][rep]) for c in CONDITIONS}
        best, report = fit_probe(train_x, val_x["ref"], targets, val_rows["ref"])
        predictions = []
        for condition in CONDITIONS:
            predictions += prediction_rows(
                val_rows[condition],
                softmax(val_x[condition] @ best["det_coef"]),
                softmax(val_x[condition] @ best["src_coef"]),
            )
        threshold, threshold_score = choose_threshold(aggregate(predictions))
        shuffled = []
        for seed in range(101, 106):
            _, control = fit_probe(train_x, val_x["ref"], shuffle_video_labels(video_ids, targets, seed), val_rows["ref"])
            shuffled.append({"seed": seed, "alpha": control["alpha"], **{
                k: control["validation"][k] for k in ("detection_auc", "detection_ba", "attribution_ba")}})
        destination = OUTPUT / rep
        destination.mkdir()
        write_csv(destination / "validation_predictions.csv", predictions)
        np.savez_compressed(destination / "probe_parameters.npz", scaler_mean=mean, scaler_scale=scale,
                            det_coef=best["det_coef"], src_coef=best["src_coef"])
        reports[rep] = {
            "alpha": best["alpha"],
            "alpha_search": report["search"],
            "threshold": threshold,
            "mean_validation_detection_ba_at_threshold": threshold_score,
            "metrics": evaluate(predictions, threshold=threshold),
            "metrics_threshold_0_5": evaluate(predictions, threshold=0.5),
            "shuffled_labels": shuffled,
            "parameters_sha256": sha256(destination / "probe_parameters.npz"),
            "validation_predictions_sha256": sha256(destination / "validation_predictions.csv"),
        }
        print(json.dumps({"representation": rep, "alpha": best["alpha"], "threshold": threshold,
                          "ref": {k: reports[rep]["metrics"]["ref"][k] for k in ("detection_auc", "detection_ba", "attribution_ba")}},
                         allow_nan=False), flush=True)
        del train_x, val_x

    record = {
        "status": "validation_only_registered_control",
        "control": "C3a face-masked border probes",
        "register": "research/comparison_register.md v1.0 section 5",
        "test_scores_used": False,
        "manifest_audit": checked,
        "manifest_sha256": sha256(manifest),
        "mask": {"source": "crop_metadata.jsonl face_xywh mapped through crop_box",
                 "expansion_per_side": MASK_EXPANSION, "fill": MASK_GRAY,
                 "order": "JPEG first, then mask",
                 "train_mask_area_fraction": {"mean": float(np.mean(mask_area)), "min": float(np.min(mask_area)),
                                              "max": float(np.max(mask_area))}},
        "crop_metadata_sha256": {n: sha256(Path("experiments/data") / n / "crop_metadata.jsonl") for n in METADATA_DIRS},
        "procedure": "F4: training-only scaling; alpha by reference-validation mean det/src BA at 0.5; "
                     "threshold maximizes mean video detection BA over validation ref/Q80/Q40",
        "source_order": list(SOURCES[1:]),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n)) for n in (
            "evaluate_probe_compression.py", "probe_validation.py", "features.py", "prepare_jpeg.py", "metrics.py")},
        "dinov2_source_commit": preflight["source_commit"],
        "dinov2_checkpoint_sha256": preflight["checkpoint_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": peak_reserved,
        "qualification": "Validation-selected and optimistic. Manipulation traces can extend beyond the "
                         "detector box, so border performance is a caveat, not proof of a shortcut.",
        "representations": reports,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", OUTPUT / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
