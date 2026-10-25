"""Registered control C3b: video-level metadata/geometry probe, train/validation only."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import sha256, write_csv
from fit_border_probes import METADATA_DIRS, load_crop_metadata
from fit_known_quality_probes import compact, fit, threshold_for
from metrics import evaluate
from probe_validation import shuffle_video_labels

CONDITION = "metadata"
OUTPUT = Path("experiments/runs/c23_control_metadata_probe_validation")
INVENTORY = Path("experiments/data/audit_c23/videos_core.csv")
CROP_FEATURES = ("face_width", "face_height", "face_centre_x", "face_centre_y",
                 "confidence", "faces_detected", "crop_side")
FEATURES = ("log_file_size", "frame_count", "width", "height", "log_bytes_per_pixel_frame",
            *[f"{name}_{stat}" for name in CROP_FEATURES for stat in ("mean", "sd")],
            "valid_crops")


def crop_values(record):
    width, height = record["original_dimensions"]
    x, y, w, h = record["face_xywh"]
    left, top, right, bottom = record["crop_box"]
    return [w / width, h / height, (x + w / 2) / width, (y + h / 2) / height,
            record["confidence"], record["faces_detected"], (right - left) / min(width, height)]


def video_features(video, crops):
    size, frames = float(video["file_size_bytes"]), float(video["frame_count_metadata"])
    width, height = float(video["width_metadata"]), float(video["height_metadata"])
    values = np.array(crops, dtype=np.float64)
    return [np.log(size), frames, width, height, np.log(size / (frames * width * height)),
            *[v for i in range(values.shape[1]) for v in (values[:, i].mean(), values[:, i].std())],
            float(len(crops))]


def selftest():
    record = {"original_dimensions": [200, 100], "face_xywh": [50.0, 20.0, 40.0, 50.0],
              "crop_box": [40, 10, 105, 75], "confidence": 0.9, "faces_detected": 1}
    assert np.allclose(crop_values(record), [0.2, 0.5, 0.35, 0.45, 0.9, 1, 0.65])
    video = {"file_size_bytes": "1000", "frame_count_metadata": "10", "width_metadata": "200", "height_metadata": "100"}
    values = video_features(video, [crop_values(record)] * 2)
    assert len(values) == len(FEATURES) and values[-1] == 2 and values[6] == 0
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = Path("experiments/data/c23_main/train_val_reference.csv")
    rows = read_manifest(manifest)
    checked = audit(rows, official_dir="experiments/data/splits")
    metadata = load_crop_metadata()
    with INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
        inventory = {r["video_id"]: r for r in csv.DictReader(handle)}

    crops, info = defaultdict(list), {}
    for row in rows:
        record = metadata[row["sample_id"]]
        assert record["split"] == row["split"] and record["source"] == row["source"]
        crops[row["video_id"]].append(crop_values(record))
        info[row["video_id"]] = row
    video_rows = {split: [] for split in ("train", "val")}
    matrices = {split: [] for split in ("train", "val")}
    for video_id in sorted(info):
        row, video = info[video_id], inventory[video_id]
        assert video["split"] == row["split"] and video["source"] == row["source"]
        video_rows[row["split"]].append({
            "sample_id": video_id, "video_id": video_id, "group_id": row["group_id"],
            "condition": CONDITION, "source": row["source"],
        })
        matrices[row["split"]].append(video_features(video, crops[video_id]))
    train_raw = np.array(matrices["train"])
    val_raw = np.array(matrices["val"])
    assert train_raw.shape == (3600, len(FEATURES)) and val_raw.shape == (699, len(FEATURES))
    assert np.isfinite(train_raw).all() and np.isfinite(val_raw).all()

    mean, scale = train_raw.mean(axis=0), train_raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = np.column_stack([(train_raw - mean) / scale, np.ones(len(train_raw))])
    val = np.column_stack([(val_raw - mean) / scale, np.ones(len(val_raw))])
    targets = np.array([SOURCES.index(r["source"]) - 1 for r in video_rows["train"]])
    video_ids = [r["video_id"] for r in video_rows["train"]]

    best, search = fit(train, val, targets, video_rows["val"], CONDITION)
    threshold = threshold_for(best["predictions"], CONDITION)
    shuffled = []
    for seed in range(101, 106):
        control, _ = fit(train, val, shuffle_video_labels(video_ids, targets, seed), video_rows["val"], CONDITION)
        shuffled.append({"seed": seed, **compact(control)})

    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "validation_predictions.csv", best["predictions"])
    np.savez_compressed(OUTPUT / "probe_parameters.npz", scaler_mean=mean, scaler_scale=scale,
                        det_coef=best["det_coef"], src_coef=best["src_coef"])
    # Standardized source-head weights show which features carry the signal.
    weights = {source: dict(zip(FEATURES, best["src_coef"][:-1, k].round(4).tolist()))
               for k, source in enumerate(SOURCES[1:])}
    record = {
        "status": "validation_only_registered_control",
        "control": "C3b metadata/geometry probe",
        "register": "research/comparison_register.md v1.0 section 5",
        "test_scores_used": False,
        "unit": "video; condition-independent metadata labelled 'metadata'",
        "features": list(FEATURES),
        "manifest_audit": checked,
        "manifest_sha256": sha256(manifest),
        "inventory_sha256": sha256(INVENTORY),
        "crop_metadata_sha256": {n: sha256(Path("experiments/data") / n / "crop_metadata.jsonl") for n in METADATA_DIRS},
        "train_videos": len(train_raw),
        "validation_videos": len(val_raw),
        "real_labels": compact(best),
        "alpha_search": search,
        "threshold": threshold,
        "metrics": evaluate(best["predictions"], threshold=threshold)[CONDITION],
        "shuffled_labels": shuffled,
        "standardized_source_weights": weights,
        "parameters_sha256": sha256(OUTPUT / "probe_parameters.npz"),
        "validation_predictions_sha256": sha256(OUTPUT / "validation_predictions.csv"),
        "script_sha256": sha256(Path(__file__)),
        "qualification": "Validation-selected and optimistic. Predictable metadata indicates a dataset-level "
                         "cue that crop models could only use indirectly; it is a caveat, not proof of use.",
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({k: record[k] for k in ("real_labels", "threshold", "metrics", "shuffled_labels")}, indent=2))
    print("Saved:", OUTPUT / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
