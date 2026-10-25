"""Registered RGB controls C1-DET, C1-ATT and C2-BIN (validation-only selection).

Reuses the F1/F3 data pipeline, model, optimizer and schedule from
train_paired.py; only the loss weights, training subset and checkpoint
selection score differ, as declared in research/comparison_register.md.
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

from common import SOURCES, audit, read_manifest
from metrics import evaluate
from model import TwoHeadResNet, paired_loss
from train_paired import PairedCrops, predict

CONTROLS = {
    # reference_only, pair_weight, detection_weight, source_weight, fake_only, selection
    "c1_det": (True, 0.0, 1.0, 0.0, False, "detection_ba"),
    "c1_att": (True, 0.0, 0.0, 1.0, True, "attribution_ba"),
    "c2_bin": (False, 1.0, 1.0, 0.0, False, "detection_ba"),
}
MANIFEST = Path("experiments/data/c23_main/reference.csv")
CONFIG = Path("experiments/configs/pilot.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control_loss(model, clean, degraded, detection, source, pair_weight,
                 detection_weights, source_weights, detection_weight, source_weight):
    """paired_loss with explicit task weights; a zero weight skips that loss."""
    outputs = model(clean), model(degraded)
    total = outputs[0][0].sum() * 0
    if detection_weight:
        total = total + detection_weight * sum(
            nn.functional.cross_entropy(o[0], detection, weight=detection_weights) for o in outputs) / 2
    mask = source >= 0
    if source_weight and mask.any():
        total = total + source_weight * sum(
            nn.functional.cross_entropy(o[1][mask], source[mask], weight=source_weights) for o in outputs) / 2
    alignment = (1 - nn.functional.cosine_similarity(outputs[0][2], outputs[1][2])).mean()
    return total + pair_weight * alignment


def selection_score(scores, metric):
    return float(np.mean([values[metric] for values in scores.values()]))


def selftest():
    torch.manual_seed(0)
    model = TwoHeadResNet(pretrained=False).eval()
    clean, degraded = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    detection, source = torch.tensor([0, 1, 1, 1]), torch.tensor([-1, 0, 2, 3])
    weights = torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.2, 0.9, 1.1])
    with torch.no_grad():
        for pair_weight in (0.0, 1.0):
            expected = paired_loss(model, clean, degraded, detection, source, pair_weight, *weights, 1.0)
            actual = control_loss(model, clean, degraded, detection, source, pair_weight, *weights, 1.0, 1.0)
            assert torch.allclose(expected, actual, atol=1e-6)
        # A zero detection weight must ignore an undefined (infinite) real-class weight.
        fake_only = control_loss(model, clean[1:], degraded[1:], detection[1:], source[1:], 0.0,
                                 torch.tensor([float("inf"), 1.0]), weights[1], 0.0, 1.0)
        assert torch.isfinite(fake_only)
    assert selection_score({"ref": {"detection_ba": 0.6}, "jpeg_q40": {"detection_ba": 0.8}}, "detection_ba") == 0.7
    print(json.dumps({"status": "passed", "synthetic_only": True,
                      "checks": ["control_loss equals paired_loss at unit task weights",
                                 "zero task weight skips undefined class weights", "selection score"]}))


def main(control, seed):
    reference_only, pair_weight, detection_weight, source_weight, fake_only, metric = CONTROLS[control]
    output = Path(f"experiments/runs/c23_control_{control}_seed{seed}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    assembly = json.loads(Path("experiments/data/c23_main/assembly.json").read_text(encoding="utf-8-sig"))
    assert sha256(MANIFEST) == assembly["reference_manifest_sha256"]
    rows = read_manifest(MANIFEST)
    audit(rows, official_dir="experiments/data/splits")
    train = [r for r in rows if r["split"] == "train" and (not fake_only or r["source"] != "real")]
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
    model = TwoHeadResNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    counts = torch.tensor([sum(r["source"] == "real" for r in train),
                           sum(r["source"] != "real" for r in train)], device=device, dtype=torch.float32)
    det_weights = counts.sum() / (2 * counts) if detection_weight else None
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
                loss = control_loss(model, clean.to(device), degraded.to(device), detection.to(device),
                                    source.to(device), pair_weight, det_weights, src_weights,
                                    detection_weight, source_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        predictions = predict(model, val, config["jpeg"]["mixed_train_qualities"], subsampling, batch, device)
        scores = evaluate(predictions, threshold=0.5)
        score = selection_score(scores, metric)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "selection_score": score, "validation": scores}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": config, "pair_weight": pair_weight,
                        "pretrained": True, "reference_only": reference_only, "source_order": SOURCES[1:],
                        "epoch": epoch + 1, "control": control, "detection_weight": detection_weight,
                        "source_weight": source_weight, "fake_only_training": fake_only}, output / "best.pt")
            with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    selected = max(history, key=lambda epoch: epoch["selection_score"])
    metadata = {
        "status": "completed_registered_control_training",
        "control": control,
        "register": "research/comparison_register.md v1.0 section 5",
        "seed": seed,
        "config": config,
        "reference_only": reference_only,
        "pair_weight": pair_weight,
        "detection_loss_weight": detection_weight,
        "source_loss_weight": source_weight,
        "fake_only_training": fake_only,
        "train_frames": len(train),
        "optimizer_updates": len(loader) * settings["epochs"],
        "checkpoint_selection": f"Max mean validation {metric} over ref/Q80/Q40 at threshold 0.5; earliest epoch on ties",
        "selected_epoch": selected["epoch"],
        "selection_score": selected["selection_score"],
        "test_scores_used": False,
        "manifest_sha256": sha256(MANIFEST),
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {n: sha256(Path(__file__).with_name(n))
                          for n in ("train_paired.py", "model.py", "metrics.py", "prepare_jpeg.py", "common.py")},
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_minutes": seconds / 60,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "validation_predictions_sha256": sha256(output / "validation_predictions.csv"),
        "qualification": "Validation-selected; system RAM unmeasured. A zero-weight task head is untrained "
                         "and its outputs are not interpreted.",
    }
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    print(json.dumps({k: metadata[k] for k in ("control", "seed", "selected_epoch", "selection_score",
                                                "elapsed_minutes", "peak_reserved_bytes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=sorted(CONTROLS))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.control and args.seed:
        main(args.control, args.seed)
    else:
        parser.error("--control and --seed are required")
