"""Write held-out predictions from a locally trained paired baseline."""
import argparse
import csv
from pathlib import Path
import torch
from common import SOURCES, audit, read_manifest
from model import TwoHeadResNet
from train_paired import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--official-splits", required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    audit(rows, official_dir=args.official_splits)
    test = [row for row in rows if row["split"] == "test"]
    if not test or any(row["condition"] != "ref" for row in rows):
        raise ValueError("Use reference manifest containing test examples")
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Output exists")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only load the local project's own checkpoint, not untrusted downloaded pickles.
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if tuple(saved["source_order"]) != SOURCES[1:]:
        raise ValueError("Unexpected checkpoint source label order")
    model = TwoHeadResNet(pretrained=False).to(device)
    model.load_state_dict(saved["state_dict"])
    config = saved["config"]
    predictions = predict(model, test, config["jpeg"]["evaluation_qualities"],
                          config["jpeg"]["subsampling"], config["training"]["microbatch_pairs"], device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} frame predictions; aggregate at video level for analysis")
