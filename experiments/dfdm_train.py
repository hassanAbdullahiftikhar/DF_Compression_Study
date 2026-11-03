"""DFDM RGB models D-F1 / D-F2 (ResNet-18) and D-EB4-F1 / D-EB4-F2 (research/dfdm_register.md section 5).

--backbone resnet18|effb4 --recipe f1|f2 --seed N : train one run (validation-only selection).
--thresholds : verify all 12 runs and freeze their detection thresholds (F1 rule).
Recipe as FF++: 20 epochs, AdamW 3e-4 / 1e-4, 8 pairs per batch, AMP, workers 0,
inverse-count class weights, pair weight 0; f2 adds a random Q80/Q40 second view.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfdm_lib import (GENERATORS, ROOT, VAL_CONDITIONS, PairedCrops, TwoHead, aggregate, choose_threshold, evaluate,
                      predict, read_csv, read_rows, sha256, source_index, write_csv)
from model import paired_loss

CONFIG = ROOT / "experiments/configs/pilot.json"
RECIPES = {"f1": [0], "f2": [80, 40]}
BACKBONES = ("resnet18", "effb4")
SEEDS = (42, 43, 44)
THRESHOLDS = ROOT / "experiments/results/dfdm_validation_thresholds.json"


def run_dir(backbone, recipe, seed):
    return ROOT / f"experiments/runs/dfdm_{backbone}_{recipe}_seed{seed}"


def selection_score(scores):
    return float(np.mean([(v["detection_ba"] + v["attribution_ba"]) / 2 for v in scores.values()]))


def train(backbone, recipe, seed):
    output = run_dir(backbone, recipe, seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    crops_report = json.loads((ROOT / "experiments/data/dfdm_crops/report.json").read_text(encoding="utf-8"))
    train_rows, val_rows = read_rows("crf0", "train"), read_rows("crf0", "val")
    assert {r["source"] for r in train_rows} == {"real", *GENERATORS}
    config = json.loads(CONFIG.read_text())
    settings = config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(PairedCrops(train_rows, RECIPES[recipe]), batch_size=settings["microbatch_pairs"],
                        shuffle=True, num_workers=settings["workers"])
    model = TwoHead(backbone, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    labels = np.array([source_index(r["source"]) for r in train_rows])
    det_counts = torch.tensor([(labels < 0).sum(), (labels >= 0).sum()], device=device, dtype=torch.float32)
    src_counts = torch.tensor([(labels == k).sum() for k in range(len(GENERATORS))], device=device, dtype=torch.float32)
    det_w, src_w = det_counts.sum() / (2 * det_counts), src_counts.sum() / (len(GENERATORS) * src_counts)
    torch.cuda.reset_peak_memory_stats()
    started, history, best = time.perf_counter(), [], -float("inf")
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        for clean, degraded, detection, source, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                loss = paired_loss(model, clean.to(device), degraded.to(device), detection.to(device),
                                   source.to(device), 0.0, det_w, src_w, settings["lambda_source"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        predictions = predict(model, val_rows, [80, 40], device)
        scores = evaluate(predictions)
        score = selection_score(scores)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "selection_score": score,
                        "validation": scores})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "backbone": backbone, "recipe": recipe, "seed": seed,
                        "generators": list(GENERATORS), "epoch": epoch + 1}, output / "best.pt")
            path = output / "validation_predictions.csv"
            path.unlink(missing_ok=True)
            write_csv(path, predictions)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    selected = max(history, key=lambda e: e["selection_score"])
    metadata = {"status": "completed_dfdm_training", "register": "research/dfdm_register.md v1.0",
                "backbone": backbone, "recipe": recipe, "seed": seed, "train_crops": len(train_rows),
                "selected_epoch": selected["epoch"], "selection_score": selected["selection_score"],
                "checkpoint_selection": "Max mean validation (det BA + attr BA)/2 over ref/Q80/Q40 at 0.5; earliest on ties",
                "test_scores_used": False, "config_sha256": sha256(CONFIG),
                "crops_report_sha256": sha256(ROOT / "experiments/data/dfdm_crops/report.json"),
                "manifest_sha256": crops_report["manifests_sha256"]["crf0"], "script_sha256": sha256(Path(__file__)),
                "lib_sha256": sha256(Path(__file__).with_name("dfdm_lib.py")),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_sha256": sha256(output / "best.pt"),
                "validation_predictions_sha256": sha256(output / "validation_predictions.csv")}
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({k: metadata[k] for k in ("backbone", "recipe", "seed", "selected_epoch", "selection_score",
                                                "elapsed_minutes")}))


def freeze_thresholds():
    if THRESHOLDS.exists():
        raise FileExistsError(THRESHOLDS)
    runs = {}
    for backbone in BACKBONES:
        for recipe in RECIPES:
            for seed in SEEDS:
                run = run_dir(backbone, recipe, seed)
                meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
                assert meta["test_scores_used"] is False and meta["checkpoint_sha256"] == sha256(run / "best.pt")
                assert meta["validation_predictions_sha256"] == sha256(run / "validation_predictions.csv")
                assert meta["script_sha256"] == sha256(Path(__file__))
                history = json.loads((run / "history.json").read_text(encoding="utf-8"))
                assert len(history) == 20 and max(history, key=lambda e: e["selection_score"])["epoch"] == meta["selected_epoch"]
                rows = read_csv(run / "validation_predictions.csv")
                threshold, score = choose_threshold(aggregate(rows), VAL_CONDITIONS)
                model_id = f"D-{'EB4-' if backbone == 'effb4' else ''}{recipe.upper()}_s{seed}"
                runs[model_id] = {"run": run.relative_to(ROOT).as_posix(), "threshold": threshold,
                                  "mean_validation_detection_ba": score, "selected_epoch": meta["selected_epoch"],
                                  "checkpoint_sha256": meta["checkpoint_sha256"],
                                  "validation_predictions_sha256": meta["validation_predictions_sha256"],
                                  "validation": evaluate(rows, threshold)}
    record = {"status": "dfdm_validation_thresholds", "rule": "Max mean validation video detection BA over ref/Q80/Q40",
              "test_scores_used": False, "script_sha256": sha256(Path(__file__)), "runs": runs}
    with THRESHOLDS.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: (v["selected_epoch"], round(v["threshold"], 4)) for k, v in runs.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--recipe", choices=sorted(RECIPES))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--thresholds", action="store_true")
    args = parser.parse_args()
    if args.thresholds:
        freeze_thresholds()
    elif args.backbone and args.recipe and args.seed:
        train(args.backbone, args.recipe, args.seed)
    else:
        parser.error("--backbone --recipe --seed, or --thresholds")
