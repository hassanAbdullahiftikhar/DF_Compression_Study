"""Known-quality ridge adaptation on existing train/validation crops only."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch
from PIL import Image

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import (
    REPS, DIMS, BATCH, sha256, read_csv, write_csv,
    extract_batch, probabilities,
)
from metrics import aggregate, balanced_accuracy, evaluate
from prepare_jpeg import jpeg_sequence
from probe_validation import (
    ALPHAS, weighted_system, solve, softmax,
    prediction_rows, shuffle_video_labels,
)


def fit(train, val, targets, val_rows, condition):
    fake = targets >= 0
    det_system = weighted_system(train, fake.astype(int), 2)
    src_system = weighted_system(train[fake], targets[fake], 4)
    best, search = None, []

    for alpha in ALPHAS:
        det_coef = solve(det_system, alpha)
        src_coef = solve(src_system, alpha)
        predictions = prediction_rows(
            val_rows,
            softmax(val @ det_coef),
            softmax(val @ src_coef),
        )
        scores = evaluate(predictions, threshold=0.5)[condition]
        score = (scores["detection_ba"] + scores["attribution_ba"]) / 2
        entry = {
            "alpha": alpha,
            "selection_score": float(score),
            "validation": scores,
        }
        search.append(entry)
        if best is None or score > best["selection_score"] + 1e-12:
            best = {
                **entry,
                "det_coef": det_coef,
                "src_coef": src_coef,
                "predictions": predictions,
            }
    return best, search


def threshold_for(rows, condition):
    videos = aggregate(rows)
    assert len(videos) == 699
    assert {row["condition"] for row in videos} == {condition}
    labels = np.array([row["label"] for row in videos])
    scores = np.array([row["p_fake"] for row in videos])
    candidates = np.unique(np.r_[
        scores, 0.5, np.nextafter(scores.max(), np.inf)
    ])
    best_score, threshold = -1.0, None

    for candidate in candidates:
        score = balanced_accuracy(labels, (scores >= candidate).astype(int), 2)
        better = score > best_score + 1e-12
        tied = abs(score - best_score) <= 1e-12
        if better or (
            tied and (
                threshold is None or
                (abs(candidate - 0.5), candidate) <
                (abs(threshold - 0.5), threshold)
            )
        ):
            best_score, threshold = score, float(candidate)
    return threshold


def compact(best):
    return {
        "alpha": best["alpha"],
        "selection_score": best["selection_score"],
        **{
            name: best["validation"][name]
            for name in ("detection_auc", "detection_ba", "attribution_ba")
        },
    }


def main(quality):
    condition = f"jpeg_q{quality}"
    output = Path(
        f"experiments/runs/c23_main_matched_q{quality}_probes_validation"
    )
    if output.exists():
        raise FileExistsError(output)

    manifest = Path("experiments/data/c23_main/train_val_reference.csv")
    rows = read_manifest(manifest)
    checked = audit(rows, official_dir="experiments/data/splits")
    assert len(rows) == 34317
    assert {row["split"] for row in rows} == {"train", "val"}
    assert all(row["condition"] == "ref" for row in rows)

    train_indices = np.array([
        i for i, row in enumerate(rows) if row["split"] == "train"
    ])
    val_indices = np.array([
        i for i, row in enumerate(rows) if row["split"] == "val"
    ])
    assert len(train_indices) == 28735 and len(val_indices) == 5582
    val_rows = [dict(rows[i], condition=condition) for i in val_indices]
    targets = np.array([
        SOURCES.index(rows[i]["source"]) - 1 for i in train_indices
    ])
    video_ids = [rows[i]["video_id"] for i in train_indices]

    frozen_path = Path(
        "experiments/results/c23_main_probe_compression_validation/report.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["test_scores_used"] is False
    assert frozen["manifest_sha256"] == sha256(manifest)
    assert condition in frozen["conditions"]
    for name in ("features.py", "prepare_jpeg.py", "probe_validation.py"):
        assert sha256(Path(__file__).with_name(name)) == frozen["helper_sha256"][name]

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    preflight = json.loads(Path(
        "experiments/results/dinov2_preflight.json"
    ).read_text(encoding="utf-8"))
    checkpoint = Path(preflight["checkpoint_path"])
    assert sha256(checkpoint) == preflight["checkpoint_sha256"]
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

    matrices = {
        rep: np.empty((len(rows), DIMS[rep]), dtype=np.float32)
        for rep in REPS
    }

    for start in range(0, len(rows), BATCH):
        batch_rows = rows[start:start + BATCH]
        images = []
        for row in batch_rows:
            with Image.open(row["reference_path"]) as handle:
                clean = handle.convert("RGB")
            assert clean.size == (224, 224)
            images.append(jpeg_sequence(clean, [quality], subsampling=2)[1])
        features = extract_batch(images, model)
        end = start + len(batch_rows)
        for rep in REPS:
            matrices[rep][start:end] = features[rep]
        if end % 800 == 0 or end == len(rows):
            print(json.dumps({
                "condition": condition,
                "crops_completed": end,
                "crops_total": len(rows),
            }), flush=True)

    torch.cuda.synchronize()
    del model
    torch.cuda.empty_cache()

    for rep in REPS:
        assert np.isfinite(matrices[rep]).all()
        baseline = frozen["representations"][rep]
        original_run = Path(f"experiments/runs/c23_main_{rep}_probe_validation")
        parameter_path = original_run / "probe_parameters.npz"
        assert sha256(parameter_path) == baseline["parameters_sha256"]
        with np.load(parameter_path, allow_pickle=False) as saved:
            parameters = {name: saved[name].copy() for name in saved.files}

        prediction_path = frozen_path.parent / rep / "validation_predictions.csv"
        assert sha256(prediction_path) == baseline["validation_predictions_sha256"]
        stored = {
            row["sample_id"]: row for row in read_csv(prediction_path)
            if row["condition"] == condition
        }
        sample_indices = val_indices[:BATCH]
        det, src = probabilities(matrices[rep][sample_indices], parameters)
        expected = np.array([
            [float(stored[rows[i]["sample_id"]]["p_fake"]), *[
                float(stored[rows[i]["sample_id"]][f"p_source_{k}"])
                for k in range(4)
            ]]
            for i in sample_indices
        ])
        assert np.allclose(
            np.column_stack([det[:, 1], src]), expected,
            atol=1e-5, rtol=0,
        ), f"Compressed prediction mismatch: {rep}"

    print(json.dumps({
        "status": "passed_existing_compressed_prediction_equivalence",
        "condition": condition, "samples": BATCH,
        "representations": list(REPS),
    }), flush=True)

    output.mkdir(parents=True)
    reports = {}
    for rep in REPS:
        train_raw = matrices[rep][train_indices].astype(np.float64)
        val_raw = matrices[rep][val_indices].astype(np.float64)
        mean = train_raw.mean(axis=0)
        scale = train_raw.std(axis=0)
        scale[scale < 1e-12] = 1.0
        train = np.column_stack([
            (train_raw - mean) / scale, np.ones(len(train_raw))
        ])
        val = np.column_stack([
            (val_raw - mean) / scale, np.ones(len(val_raw))
        ])
        del train_raw, val_raw

        best, search = fit(train, val, targets, val_rows, condition)
        shuffled = []
        for seed in range(101, 106):
            shuffled_targets = shuffle_video_labels(video_ids, targets, seed)
            control, _ = fit(train, val, shuffled_targets, val_rows, condition)
            shuffled.append({"seed": seed, **compact(control)})
            print(json.dumps({
                "representation": rep, "shuffle_completed": seed
            }), flush=True)
            del control

        threshold = threshold_for(best["predictions"], condition)
        tuned = evaluate(best["predictions"], threshold=threshold)[condition]
        destination = output / rep
        destination.mkdir()
        prediction_path = destination / "validation_predictions.csv"
        write_csv(prediction_path, best["predictions"])
        parameter_path = destination / "probe_parameters.npz"
        np.savez_compressed(
            parameter_path,
            scaler_mean=mean, scaler_scale=scale,
            det_coef=best["det_coef"], src_coef=best["src_coef"],
        )
        reports[rep] = {
            "real_labels": compact(best),
            "alpha_search": search,
            "threshold": threshold,
            "threshold_validation": tuned,
            "reference_trained_validation": {
                name: frozen["representations"][rep]["metrics"][condition][name]
                for name in ("detection_auc", "detection_ba", "attribution_ba")
            },
            "shuffled_labels": shuffled,
            "parameters_sha256": sha256(parameter_path),
            "validation_predictions_sha256": sha256(prediction_path),
        }
        print(json.dumps({
            "representation": rep,
            "matched_quality": reports[rep]["real_labels"],
            "threshold": threshold,
            "threshold_validation": tuned,
            "reference_trained_validation": reports[rep]["reference_trained_validation"],
        }, indent=2, allow_nan=False), flush=True)
        del train, val, best

    record = {
        "status": "validation_only_known_quality_adaptation",
        "training_condition": condition,
        "validation_condition": condition,
        "test_scores_used": False,
        "manifest_audit": checked,
        "manifest_sha256": sha256(manifest),
        "source_order": list(SOURCES[1:]),
        "classifier": "Separate class-weighted ridge heads",
        "scaling": "Training-compressed-feature mean and population standard deviation",
        "alpha_selection": "Mean validation detection/source BA at threshold 0.5",
        "alpha_ties": "Smaller alpha; tolerance 1e-12",
        "threshold_rule": "Maximize video detection BA on matched-quality validation",
        "threshold_ties": "Closest to 0.5, then smaller threshold; tolerance 1e-12",
        "scope_note": (
            "Planned E02 known-quality control, prepared after exploratory "
            "reference-probe validation and before test scoring."
        ),
        "qualification": (
            "Validation-selected results are optimistic. Matched-quality "
            "adaptation is distinct from reference-trained compression retention. "
            "Original and adapted probes use different threshold selection domains; "
            "AUROC and fake-only attribution do not depend on the detection threshold. "
            "Five shuffles are diagnostics, not a formal significance test."
        ),
        "reference_probe_report_sha256": sha256(frozen_path),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "evaluate_probe_compression.py", "probe_validation.py",
                "features.py", "prepare_jpeg.py", "metrics.py",
            )
        },
        "dinov2_source_commit": preflight["source_commit"],
        "dinov2_checkpoint_sha256": preflight["checkpoint_sha256"],
        "temporary_numeric_feature_bytes": sum(value.nbytes for value in matrices.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resource_qualification": (
            "Timer includes backbone loading, extraction, fitting and per-probe writes "
            "after input audits. GPU figures use the PyTorch allocator; system RAM is unmeasured."
        ),
        "representations": reports,
    }
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=int, choices=(80, 40), default=40)
    main(parser.parse_args().quality)
