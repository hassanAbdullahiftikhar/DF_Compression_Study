"""Small two-view RGB baseline; reference PNG manifest required, held-out quality untouched."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor, normalize
from common import SOURCES, audit, read_manifest
from metrics import evaluate
from model import TwoHeadResNet, paired_loss
from prepare_jpeg import jpeg_sequence


def tensor(image):
    return normalize(to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class PairedCrops(Dataset):
    def __init__(self, rows, qualities, subsampling, fixed_quality=None):
        self.rows, self.qualities = rows, qualities
        self.subsampling, self.fixed_quality = subsampling, fixed_quality

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["reference_path"]) as handle:
            clean = handle.convert("RGB")
        quality = random.choice(self.qualities) if self.fixed_quality is None else self.fixed_quality
        degraded = clean if quality == 0 else jpeg_sequence(clean, [quality], self.subsampling)[1]
        source = SOURCES.index(row["source"]) - 1
        return tensor(clean), tensor(degraded), int(source >= 0), source, index


def predict(model, rows, qualities, subsampling, batch, device):
    model.eval()
    predictions = []
    for quality in [0, *qualities]:
        data = PairedCrops(rows, qualities, subsampling, fixed_quality=quality)
        for _, images, labels, sources, indices in DataLoader(data, batch_size=batch, num_workers=0):
            with torch.no_grad():
                output = model(images.to(device))
                det = output[0].softmax(1)[:, 1].cpu().numpy()
                src = output[1].softmax(1).cpu().numpy()
            for i, index in enumerate(indices.tolist()):
                row = rows[index]
                predictions.append({"sample_id": row["sample_id"], "video_id": row["video_id"],
                    "group_id": row["group_id"], "condition": "ref" if quality == 0 else f"jpeg_q{quality}",
                    "label": int(labels[i]), "source_index": int(sources[i]), "p_fake": float(det[i]),
                    **{f"p_source_{k}": float(src[i, k]) for k in range(4)}})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/pilot.json"))
    parser.add_argument("--official-splits", required=True)
    parser.add_argument("--pair-weight", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true", help="Downloads official ImageNet weights if absent")
    parser.add_argument("--reference-only", action="store_true", help="Two identical reference views for baseline")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rows = read_manifest(args.manifest)
    audit(rows, official_dir=args.official_splits)
    if any(r["condition"] != "ref" for r in rows):
        raise ValueError("Use reference-only manifest; JPEG training is generated on demand")
    train, val = ([r for r in rows if r["split"] == split] for split in ("train", "val"))
    for subset in (train, val):
        if {r["source"] for r in subset} != set(SOURCES):
            raise ValueError("Training and validation must each contain all five sources")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config["training"]
    batch = settings["microbatch_pairs"]
    qualities = [0] if args.reference_only else config["jpeg"]["mixed_train_qualities"]
    subsampling = config["jpeg"]["subsampling"]
    data = PairedCrops(train, qualities, subsampling)
    loader = DataLoader(data, batch_size=batch, shuffle=True, num_workers=settings["workers"])
    model = TwoHeadResNet(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = torch.tensor([sum(r["source"] == "real" for r in train),
                           sum(r["source"] != "real" for r in train)], device=device, dtype=torch.float32)
    det_weights = counts.sum() / (2 * counts)
    source_counts = torch.tensor([sum(r["source"] == source for r in train) for source in SOURCES[1:]],
                                  device=device, dtype=torch.float32)
    src_weights = source_counts.sum() / (4 * source_counts)
    history, best = [], -float("inf")
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        for clean, degraded, detection, source, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = paired_loss(model, clean.to(device), degraded.to(device), detection.to(device),
                                   source.to(device), args.pair_weight, det_weights, src_weights,
                                   settings["lambda_source"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        predictions = predict(model, val, config["jpeg"]["mixed_train_qualities"], subsampling, batch, device)
        scores = evaluate(predictions, threshold=0.5)
        # Select using only reference and training-domain validation qualities.
        score = float(np.mean([(r["detection_ba"] + r["attribution_ba"]) / 2 for r in scores.values()]))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": scores}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "pair_weight": args.pair_weight, "pretrained": args.pretrained,
                        "reference_only": args.reference_only, "source_order": SOURCES[1:],
                        "epoch": epoch + 1}, output / "best.pt")
            with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({"torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device), "arguments": vars(args),
        "config": config, "note": "0.5 validation threshold is illustrative; choose/freeze before final test."}, indent=2))


if __name__ == "__main__":
    main()
