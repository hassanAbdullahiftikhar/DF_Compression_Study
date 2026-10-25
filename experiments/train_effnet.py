"""Extension X6 (register Amendment 2): EfficientNet-B4 replication of the F1 and F2 recipes.

Only the backbone changes. Data pipeline, two-view loss (paired_loss with
pair weight 0), optimizer, schedule, checkpoint selection (mean validation
detection/attribution BA over ref/Q80/Q40 at 0.5) and stored predictions
follow train_paired.py. Input stays 224 px (EfficientNet-B4 native: 380 px).
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from common import SOURCES, audit, read_manifest
from metrics import evaluate
from model import paired_loss
from train_paired import PairedCrops, predict

RECIPES = {"f1": True, "f2": False}  # reference_only
MANIFEST = Path("experiments/data/c23_main/reference.csv")
CONFIG = Path("experiments/configs/pilot.json")
FEATURES = 1792


class EffNetTwoHead(nn.Module):
    """Same heads and attribute names as model.TwoHeadResNet, on an EfficientNet-B4 encoder."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.classifier = nn.Identity()
        self.detection = nn.Linear(FEATURES, 2)
        self.attribution = nn.Linear(FEATURES, 4)
        self.projection = nn.Sequential(nn.Linear(FEATURES, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, images):
        features = self.encoder(images)
        return self.detection(features), self.attribution(features), self.projection(features)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_score(scores):
    return float(np.mean([(r["detection_ba"] + r["attribution_ba"]) / 2 for r in scores.values()]))


def selftest():
    model = EffNetTwoHead(pretrained=False).eval()
    with torch.no_grad():
        det, src, proj = model(torch.randn(2, 3, 224, 224))
    assert det.shape == (2, 2) and src.shape == (2, 4) and proj.shape == (2, 128)
    assert model.encoder(torch.randn(1, 3, 224, 224)).shape == (1, FEATURES)
    assert selection_score({"ref": {"detection_ba": 0.8, "attribution_ba": 0.6}}) == 0.7
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main(recipe, seed):
    reference_only = RECIPES[recipe]
    output = Path(f"experiments/runs/c23_ext_effb4_{recipe}_seed{seed}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    assembly = json.loads(Path("experiments/data/c23_main/assembly.json").read_text(encoding="utf-8-sig"))
    assert sha256(MANIFEST) == assembly["reference_manifest_sha256"]
    rows = read_manifest(MANIFEST)
    audit(rows, official_dir="experiments/data/splits")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    config = json.loads(CONFIG.read_text())
    config["seed"] = seed
    assert torch.cuda.is_available()
    output.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    settings = config["training"]
    batch = settings["microbatch_pairs"]
    qualities = [0] if reference_only else config["jpeg"]["mixed_train_qualities"]
    subsampling = config["jpeg"]["subsampling"]
    loader = DataLoader(PairedCrops(train, qualities, subsampling), batch_size=batch,
                        shuffle=True, num_workers=settings["workers"])
    model = EffNetTwoHead(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    counts = torch.tensor([sum(r["source"] == "real" for r in train),
                           sum(r["source"] != "real" for r in train)], device=device, dtype=torch.float32)
    det_weights = counts.sum() / (2 * counts)
    source_counts = torch.tensor([sum(r["source"] == s for r in train) for s in SOURCES[1:]],
                                 device=device, dtype=torch.float32)
    src_weights = source_counts.sum() / (4 * source_counts)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history, best = [], -float("inf")
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        for clean, degraded, detection, source, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                loss = paired_loss(model, clean.to(device), degraded.to(device), detection.to(device),
                                   source.to(device), 0.0, det_weights, src_weights, settings["lambda_source"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        predictions = predict(model, val, config["jpeg"]["mixed_train_qualities"], subsampling, batch, device)
        scores = evaluate(predictions, threshold=0.5)
        score = selection_score(scores)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "selection_score": score,
                  "validation": scores}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": config, "pair_weight": 0.0,
                        "pretrained": True, "reference_only": reference_only, "source_order": SOURCES[1:],
                        "epoch": epoch + 1, "backbone": "efficientnet_b4", "recipe": recipe}, output / "best.pt")
            with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.cuda.synchronize()
    selected = max(history, key=lambda e: e["selection_score"])
    metadata = {
        "status": "completed_post_freeze_extension_training",
        "extension": "X6 EfficientNet-B4, research/comparison_register.md Amendment 2",
        "recipe": recipe, "seed": seed, "config": config, "reference_only": reference_only, "pair_weight": 0.0,
        "backbone": "torchvision efficientnet_b4 IMAGENET1K_V1, classifier=Identity, 1792-d", "input_px": 224,
        "train_frames": len(train), "optimizer_updates": len(loader) * settings["epochs"],
        "checkpoint_selection": "Max mean validation (detection BA + attribution BA)/2 over ref/Q80/Q40 at 0.5; earliest on ties",
        "selected_epoch": selected["epoch"], "selection_score": selected["selection_score"],
        "test_scores_used": False,
        "manifest_sha256": sha256(MANIFEST), "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n))
                          for n in ("train_paired.py", "model.py", "metrics.py", "prepare_jpeg.py", "common.py")},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "validation_predictions_sha256": sha256(output / "validation_predictions.csv"),
    }
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    print(json.dumps({k: metadata[k] for k in ("recipe", "seed", "selected_epoch", "selection_score", "elapsed_minutes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=sorted(RECIPES))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.recipe and args.seed:
        main(args.recipe, args.seed)
    else:
        parser.error("--recipe and --seed are required")
