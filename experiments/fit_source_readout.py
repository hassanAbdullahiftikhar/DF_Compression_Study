"""Registered readout R for control C2: frozen-encoder ridge source probe.

Applied identically to F3 (source-supervised paired) and C2-BIN (binary-only
paired) encoders. Features are the 512-d pooled ResNet-18 outputs at
reference, Q80 and Q40; the ridge head is fitted on fake training crops
pooled across the three views and selected on validation only.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import read_csv, sha256, write_csv
from metrics import evaluate
from model import TwoHeadResNet
from probe_validation import ALPHAS, prediction_rows, softmax, solve, weighted_system
from train_paired import PairedCrops

QUALITIES = (0, 80, 40)
CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
SEEDS = (42, 43, 44)
# Same batch as the training-time predictions: cuDNN TF32 convolution
# algorithms vary with batch size, so 8 reproduces stored outputs exactly.
BATCH = 8
OUTPUT = Path("experiments/runs/c23_control_source_readout_validation")


def encoders():
    main = json.loads(Path("experiments/results/main_validation_thresholds_seeds42_44.json").read_text(encoding="utf-8"))
    control = json.loads(Path("experiments/results/control_validation_thresholds.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        name = f"c23_main_paired_w1_seed{seed}"
        yield f"f3_seed{seed}", Path("experiments/runs") / name, main["runs"][name]
    for seed in SEEDS:
        name = f"c23_control_c2_bin_seed{seed}"
        yield f"c2_bin_seed{seed}", Path("experiments/runs") / name, control["runs"][name]


def extract(model, rows, quality, device):
    """Pooled encoder features plus native head probabilities, in row order."""
    data = PairedCrops(rows, [quality], subsampling=2, fixed_quality=quality)
    features, p_fake, p_source = [], [], []
    for _, images, _, _, _ in DataLoader(data, batch_size=BATCH, num_workers=0):
        with torch.inference_mode():
            pooled = model.encoder(images.to(device))
            features.append(pooled.cpu().numpy())
            p_fake.append(model.detection(pooled).softmax(1)[:, 1].cpu().numpy())
            p_source.append(model.attribution(pooled).softmax(1).cpu().numpy())
    return np.concatenate(features), np.concatenate(p_fake), np.concatenate(p_source)


def fit_readout(train_x, train_y, val_x, val_rows, native_fake):
    system = weighted_system(train_x, train_y, 4)
    best, search = None, []
    for alpha in ALPHAS:
        coef = solve(system, alpha)
        predictions = []
        for condition in CONDITIONS:
            source = softmax(val_x[condition] @ coef)
            detection = np.column_stack([1 - native_fake[condition], native_fake[condition]])
            predictions += prediction_rows(val_rows[condition], detection, source)
        scores = evaluate(predictions, threshold=0.5)
        score = float(np.mean([scores[c]["attribution_ba"] for c in CONDITIONS]))
        search.append({"alpha": alpha, "selection_score": score,
                       "attribution_ba": {c: scores[c]["attribution_ba"] for c in CONDITIONS}})
        # ALPHAS is ascending: exact ties retain the smaller alpha.
        if best is None or score > best["selection_score"] + 1e-12:
            best = {"alpha": alpha, "selection_score": score, "coef": coef, "predictions": predictions,
                    "scores": scores}
    return best, search


def selftest():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 6)) * 3
    labels = np.repeat(np.arange(4), 30)
    features = np.column_stack([centres[labels] + rng.normal(size=(120, 6)), np.ones(120)])
    coef = solve(weighted_system(features, labels, 4), 0.01)
    assert (softmax(features @ coef).argmax(axis=1) == labels).mean() > 0.95
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = Path("experiments/data/c23_main/train_val_reference.csv")
    rows = read_manifest(manifest)
    checked = audit(rows, official_dir="experiments/data/splits")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    assert len(train) == 28735 and len(val) == 5582
    train_sources = np.array([SOURCES.index(r["source"]) - 1 for r in train])
    fake = train_sources >= 0
    val_rows = {c: [dict(r, condition=c) for r in val] for c in CONDITIONS}
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    OUTPUT.mkdir(parents=True)
    reports, started = {}, time.perf_counter()

    for label, run, registered in encoders():
        assert sha256(run / "best.pt") == registered["checkpoint_sha256"]
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        assert tuple(saved["source_order"]) == SOURCES[1:]
        model = TwoHeadResNet(pretrained=False)
        model.load_state_dict(saved["state_dict"])
        model = model.to(device).eval()

        train_x, val_x, native_fake, native_source = [], {}, {}, {}
        for quality, condition in zip(QUALITIES, CONDITIONS):
            features, _, _ = extract(model, train, quality, device)
            train_x.append(features[fake])
            val_x[condition], native_fake[condition], native_source[condition] = extract(model, val, quality, device)

        # Reloaded encoders must reproduce the stored native validation predictions.
        stored = {(r["sample_id"], r["condition"]): r for r in read_csv(run / "validation_predictions.csv")}
        difference = max(
            max(abs(float(stored[(r["sample_id"], c)]["p_fake"]) - native_fake[c][i]),
                *[abs(float(stored[(r["sample_id"], c)][f"p_source_{k}"]) - native_source[c][i, k]) for k in range(4)])
            for c in CONDITIONS for i, r in enumerate(val)
        )
        assert difference < 1e-5, f"Native prediction mismatch {difference}: {label}"
        del model
        torch.cuda.empty_cache()

        raw = np.concatenate(train_x).astype(np.float64)
        targets = np.tile(train_sources[fake], len(QUALITIES))
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design = lambda values: np.column_stack([(values.astype(np.float64) - mean) / scale, np.ones(len(values))])
        best, search = fit_readout(design(raw), targets, {c: design(v) for c, v in val_x.items()},
                                   val_rows, native_fake)
        destination = OUTPUT / label
        destination.mkdir()
        write_csv(destination / "validation_predictions.csv", best["predictions"])
        np.savez_compressed(destination / "readout_parameters.npz", scaler_mean=mean, scaler_scale=scale,
                            src_coef=best["coef"])
        native = evaluate([
            row for c in CONDITIONS for row in prediction_rows(
                val_rows[c], np.column_stack([1 - native_fake[c], native_fake[c]]), native_source[c])
        ], threshold=0.5)
        reports[label] = {
            "encoder_run": run.as_posix(),
            "checkpoint_sha256": registered["checkpoint_sha256"],
            "native_prediction_max_abs_difference": float(difference),
            "train_fake_crops_per_view": int(fake.sum()),
            "alpha": best["alpha"],
            "selection_score": best["selection_score"],
            "alpha_search": search,
            "readout_attribution_ba": {c: best["scores"][c]["attribution_ba"] for c in CONDITIONS},
            "native_head_attribution_ba": {c: native[c]["attribution_ba"] for c in CONDITIONS},
            "parameters_sha256": sha256(destination / "readout_parameters.npz"),
            "validation_predictions_sha256": sha256(destination / "validation_predictions.csv"),
        }
        print(json.dumps({label: {k: reports[label][k] for k in (
            "alpha", "readout_attribution_ba", "native_head_attribution_ba", "native_prediction_max_abs_difference")}}),
            flush=True)

    record = {
        "status": "validation_only_registered_control",
        "control": "C2 readout R (frozen-encoder ridge source probe)",
        "register": "research/comparison_register.md v1.0 section 5",
        "test_scores_used": False,
        "manifest_audit": checked,
        "manifest_sha256": sha256(manifest),
        "features": "512-d pooled ResNet-18 encoder output; eval mode; float32; batch 8 as in training-time prediction",
        "deviation_note": "First attempt (batch 64) stopped at the native-prediction check (max difference "
                          "0.0014 from TF32 convolution algorithm choice); no outputs were kept. Rerun at batch 8.",
        "training_views": "Fake training crops at ref, Q80 and Q40 (Pillow 4:2:0), pooled",
        "scaling": "Mean and population SD of the pooled training features",
        "alpha_selection": "Mean validation fake-only attribution BA over ref/Q80/Q40; smaller alpha on ties",
        "p_fake": "Native detection head of the same encoder (unchanged)",
        "source_order": list(SOURCES[1:]),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n)) for n in (
            "probe_validation.py", "train_paired.py", "model.py", "metrics.py", "prepare_jpeg.py")},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "qualification": "Validation-selected and optimistic. A linear readout measures linearly accessible "
                         "source information, not all information in the encoder.",
        "encoders": reports,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", OUTPUT / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
