"""Validation-only compression evaluation of six already-fitted reference probes."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

from common import SOURCES, audit, read_manifest
from features import descriptor
from metrics import aggregate, balanced_accuracy, evaluate
from prepare_jpeg import jpeg_sequence
from probe_validation import prediction_rows, softmax
from research_metrics import summarize

REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")
DIMS = dict(fft=256, rgb=768, dct=128, wavelet=28, residual=263, dinov2=384)
CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
BATCH = 8


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probabilities(values, parameters):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    features = np.column_stack([
        (values - parameters["scaler_mean"]) / parameters["scaler_scale"],
        np.ones(len(values)),
    ])
    detection = softmax(features @ parameters["det_coef"])
    attribution = softmax(features @ parameters["src_coef"])
    assert np.isfinite(detection).all() and np.isfinite(attribution).all()
    return detection, attribution


def extract_batch(images, model):
    result = {
        rep: np.stack([descriptor(image, rep) for image in images])
        for rep in REPS if rep != "dinov2"
    }
    tensors = torch.stack([
        normalize(
            to_tensor(image),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        for image in images
    ])
    with torch.inference_mode():
        result["dinov2"] = model(tensors.to("cuda")).cpu().numpy()
    assert result["dinov2"].shape == (len(images), 384)
    return result


def choose_threshold(videos):
    data = []
    for condition in CONDITIONS:
        subset = [row for row in videos if row["condition"] == condition]
        assert len(subset) == 699
        labels = np.array([row["label"] for row in subset])
        scores = np.array([row["p_fake"] for row in subset])
        data.append((labels, scores))

    scores = np.concatenate([values for _, values in data])
    candidates = np.unique(np.r_[
        scores, 0.5, np.nextafter(scores.max(), np.inf)
    ])
    best_score, best_threshold = -1.0, None
    for threshold in candidates:
        score = float(np.mean([
            balanced_accuracy(labels, (values >= threshold).astype(int), 2)
            for labels, values in data
        ]))
        better = score > best_score + 1e-12
        tied = abs(score - best_score) <= 1e-12
        if better or (
            tied and (
                best_threshold is None or
                (abs(threshold - 0.5), threshold) <
                (abs(best_threshold - 0.5), best_threshold)
            )
        ):
            best_score, best_threshold = score, float(threshold)
    return best_threshold, best_score


def main():
    output = Path("experiments/results/c23_main_probe_compression_validation")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    manifest = Path("experiments/data/c23_main/train_val_reference.csv")
    rows = read_manifest(manifest)
    checked = audit(rows, official_dir="experiments/data/splits")
    assert len(rows) == 34317
    assert {row["split"] for row in rows} == {"train", "val"}
    assert all(row["condition"] == "ref" for row in rows)
    val = [row for row in rows if row["split"] == "val"]
    assert len(val) == 5582
    assert len({row["video_id"] for row in val}) == 699
    assert len({row["group_id"] for row in val}) == 70
    expected = {row["sample_id"]: row for row in val}
    positions = {row["sample_id"]: i for i, row in enumerate(rows)}
    manifest_hash = sha256(manifest)

    parameters, predictions, registers, cached_smoke = {}, {}, {}, {}
    smoke_rows = val[:BATCH]
    smoke_positions = [positions[row["sample_id"]] for row in smoke_rows]

    for rep in REPS:
        cache = Path(f"experiments/features/c23_main_{rep}_trainval")
        run = Path(f"experiments/runs/c23_main_{rep}_probe_validation")
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        assert selection["representation"] == rep
        assert selection["source_order"] == list(SOURCES[1:])
        assert selection["train_frames"] == 28735
        assert selection["validation_frames"] == len(val)
        assert sha256(cache / "rows.csv") == selection["rows_sha256"] == manifest_hash
        assert sha256(cache / "features.npy") == selection["feature_sha256"]

        matrix = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
        assert matrix.shape == (len(rows), DIMS[rep])
        cached_smoke[rep] = np.array(matrix[smoke_positions])
        del matrix

        parameter_path = run / "probe_parameters.npz"
        with np.load(parameter_path, allow_pickle=False) as saved:
            parameters[rep] = {name: saved[name].copy() for name in saved.files}
        p = parameters[rep]
        assert p["scaler_mean"].shape == p["scaler_scale"].shape == (DIMS[rep],)
        assert p["det_coef"].shape == (DIMS[rep] + 1, 2)
        assert p["src_coef"].shape == (DIMS[rep] + 1, 4)
        assert all(np.isfinite(value).all() for value in p.values())
        assert (p["scaler_scale"] > 0).all()

        reference_path = run / "validation_predictions.csv"
        reference = read_csv(reference_path)
        assert len(reference) == len(val)
        assert len({row["sample_id"] for row in reference}) == len(val)
        assert {row["sample_id"] for row in reference} == set(expected)
        for row in reference:
            truth = expected[row["sample_id"]]
            assert row["condition"] == "ref"
            assert row["video_id"] == truth["video_id"]
            assert row["group_id"] == truth["group_id"]
            source = SOURCES.index(truth["source"]) - 1
            assert int(row["source_index"]) == source
            assert int(row["label"]) == int(source >= 0)

        values = np.array([
            [float(row["p_fake"]), *[
                float(row[f"p_source_{i}"]) for i in range(4)
            ]]
            for row in reference
        ])
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        assert np.allclose(values[:, 1:].sum(axis=1), 1, atol=1e-5, rtol=0)

        original = evaluate(reference, threshold=0.5)["ref"]
        for metric in ("detection_auc", "detection_ba", "attribution_ba"):
            assert np.isclose(
                original[metric],
                selection["real_labels"]["validation"][metric],
                atol=1e-10, rtol=0,
            )
        predictions[rep] = reference
        registers[rep] = {
            "alpha": selection["real_labels"]["alpha"],
            "parameters_sha256": sha256(parameter_path),
            "selection_sha256": sha256(run / "selection.json"),
            "reference_predictions_sha256": sha256(reference_path),
            "features_sha256": selection["feature_sha256"],
            "rows_sha256": selection["rows_sha256"],
        }

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    preflight = json.loads(Path(
        "experiments/results/dinov2_preflight.json"
    ).read_text(encoding="utf-8"))
    checkpoint = Path(preflight["checkpoint_path"])
    assert sha256(checkpoint) == preflight["checkpoint_sha256"]
    dino_metadata = json.loads(Path(
        "experiments/features/c23_main_dinov2_trainval/feature_metadata.json"
    ).read_text(encoding="utf-8"))
    assert dino_metadata["source_commit"] == preflight["source_commit"]
    assert dino_metadata["checkpoint_sha256"] == preflight["checkpoint_sha256"]
    repository = Path(torch.hub.get_dir()) / (
        f"facebookresearch_dinov2_{preflight['source_commit']}"
    )
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    model = torch.hub.load(
        str(repository), "dinov2_vits14", source="local", pretrained=False
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    model = model.to(device="cuda", dtype=torch.float32)

    images = []
    for row in smoke_rows:
        with Image.open(row["reference_path"]) as handle:
            images.append(handle.convert("RGB"))
    smoke_features = extract_batch(images, model)
    for rep in REPS:
        tolerance = 1e-4 if rep == "dinov2" else 1e-7
        assert np.allclose(
            smoke_features[rep], cached_smoke[rep],
            atol=tolerance, rtol=1e-5,
        ), f"Cached feature mismatch: {rep}"
        det, src = probabilities(smoke_features[rep], parameters[rep])
        stored = {row["sample_id"]: row for row in predictions[rep]}
        expected_probabilities = np.array([
            [float(stored[row["sample_id"]]["p_fake"]), *[
                float(stored[row["sample_id"]][f"p_source_{i}"])
                for i in range(4)
            ]]
            for row in smoke_rows
        ])
        assert np.allclose(
            np.column_stack([det[:, 1], src]),
            expected_probabilities, atol=1e-5, rtol=0,
        ), f"Cached prediction mismatch: {rep}"
    print(json.dumps({
        "status": "passed_cached_reference_equivalence",
        "samples": len(smoke_rows), "representations": list(REPS),
    }), flush=True)

    output.mkdir(parents=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    distortions = []

    for quality in (80, 40):
        condition = f"jpeg_q{quality}"
        for start in range(0, len(val), BATCH):
            batch_rows = val[start:start + BATCH]
            images = []
            for row in batch_rows:
                with Image.open(row["reference_path"]) as handle:
                    clean = handle.convert("RGB")
                assert clean.size == (224, 224)
                payload, degraded = jpeg_sequence(clean, [quality], subsampling=2)
                difference = (
                    np.asarray(clean, dtype=np.float64) -
                    np.asarray(degraded, dtype=np.float64)
                )
                mse = float(np.square(difference).mean())
                distortions.append({
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "condition": condition,
                    "jpeg_bytes": len(payload),
                    "rgb_psnr_db": (
                        float(10 * np.log10(255.0 ** 2 / mse))
                        if mse > 0 else None
                    ),
                })
                images.append(degraded)

            batch_features = extract_batch(images, model)
            variants = [dict(row, condition=condition) for row in batch_rows]
            for rep in REPS:
                det, src = probabilities(batch_features[rep], parameters[rep])
                predictions[rep].extend(prediction_rows(variants, det, src))

            completed = start + len(batch_rows)
            if completed % 800 == 0 or completed == len(val):
                print(json.dumps({
                    "condition": condition, "crops_completed": completed,
                    "crops_total": len(val),
                }), flush=True)

    torch.cuda.synchronize()
    reports = {}
    for rep in REPS:
        frame_rows = predictions[rep]
        assert len(frame_rows) == len(val) * len(CONDITIONS)
        assert len({
            (row["sample_id"], row["condition"]) for row in frame_rows
        }) == len(frame_rows)
        for condition in CONDITIONS:
            assert {
                row["sample_id"] for row in frame_rows
                if row["condition"] == condition
            } == set(expected)

        destination = output / rep
        destination.mkdir()
        prediction_path = destination / "validation_predictions.csv"
        write_csv(prediction_path, frame_rows)
        videos = aggregate(frame_rows)
        threshold, score = choose_threshold(videos)
        reports[rep] = {
            **registers[rep],
            "threshold": threshold,
            "mean_validation_detection_ba": score,
            "validation_predictions_sha256": sha256(prediction_path),
            "metrics": summarize(videos, threshold),
        }

    write_csv(output / "distortion.csv", distortions)
    record = {
        "status": "validation_only_frozen_reference_probe_compression",
        "test_scores_used": False,
        "manifest_audit": checked,
        "manifest_sha256": manifest_hash,
        "validation_frames_per_condition": len(val),
        "validation_videos_per_condition": 699,
        "validation_groups": 70,
        "conditions": list(CONDITIONS),
        "probe_selection": "Existing reference-selected alpha and fitted heads retained",
        "threshold_rule": "Maximize mean video detection BA over validation ref/Q80/Q40",
        "threshold_ties": "Closest to 0.5, then smaller threshold; tolerance 1e-12",
        "threshold_candidates": "Observed video scores, 0.5, and nextafter(max_score, +inf)",
        "prediction_rule": "Fake when p_fake >= threshold",
        "qualification": (
            "Threshold-tuned validation scores are optimistic. "
            "Additional JPEG compression of C23-derived crops. "
            "PSNR compares decoded RGB crops; JPEG bytes are crop sizes, not video bitrate."
        ),
        "jpeg": {
            "subsampling": 2, "optimize": False, "progressive": False,
            "pillow": PIL.__version__,
            "jpeg_library": Image.core.jpeglib_version,
        },
        "dinov2_source_commit": preflight["source_commit"],
        "dinov2_checkpoint_sha256": preflight["checkpoint_sha256"],
        "cached_reference_equivalence_samples": len(smoke_rows),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "features.py", "probe_validation.py", "prepare_jpeg.py",
                "metrics.py", "research_metrics.py",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resource_qualification": (
            "Timer starts after loading and reference smoke check. "
            "GPU figures use the PyTorch allocator; system RAM is unmeasured."
        ),
        "representations": reports,
    }
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)

    compact = {
        rep: {
            "alpha": reports[rep]["alpha"],
            "threshold": reports[rep]["threshold"],
            "conditions": {
                condition: {
                    metric: reports[rep]["metrics"][condition][metric]
                    for metric in (
                        "detection_auc", "detection_ba", "attribution_ba",
                        "detection_retention", "attribution_retention",
                    )
                }
                for condition in CONDITIONS
            },
        }
        for rep in REPS
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    print("Saved:", output / "report.json")


if __name__ == "__main__":
    main()
