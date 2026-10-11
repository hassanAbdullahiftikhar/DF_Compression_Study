"""Frozen DINOv2-small features from existing training/validation crops."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor
from common import audit, read_manifest, write_manifest


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    manifest = Path(args.manifest)
    output = Path(args.output)

    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    if args.batch < 1:
        raise ValueError("Batch size must be positive")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    rows = read_manifest(manifest)
    checked = audit(rows, official_dir=args.official_splits)
    if {r["split"] for r in rows} != {"train", "val"}:
        raise ValueError("This extraction requires training/validation only")
    if any(r["condition"] != "ref" for r in rows):
        raise ValueError("Reference crops required")

    preflight = json.loads(Path(
        "experiments/results/dinov2_preflight.json"
    ).read_text(encoding="utf-8"))

    commit = preflight["source_commit"]
    checkpoint = Path(preflight["checkpoint_path"])
    if sha256(checkpoint) != preflight["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the preflight record")

    repository = (
        Path(torch.hub.get_dir()) / f"facebookresearch_dinov2_{commit}"
    )
    if not repository.is_dir():
        raise FileNotFoundError(f"Cached source missing: {repository}")

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

    # Load cached source and weights directly, without another download.
    model = torch.hub.load(
        str(repository),
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    model = model.to(device="cuda", dtype=torch.float32)

    output.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 384),
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            end = min(start + args.batch, len(rows))
            images = []

            for row in rows[start:end]:
                with Image.open(row["reference_path"]) as handle:
                    image = handle.convert("RGB")
                    if image.size != (224, 224):
                        raise ValueError(f"Unexpected crop: {row['sample_id']}")
                    images.append(normalize(
                        to_tensor(image),
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ))

            features = model(torch.stack(images).to("cuda"))
            if tuple(features.shape) != (end - start, 384):
                raise ValueError(f"Unexpected feature shape: {features.shape}")
            values = features.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite features")

            matrix[start:end] = values

            if end % 800 == 0 or end == len(rows):
                print(json.dumps({
                    "crops_completed": end,
                    "crops_total": len(rows),
                }), flush=True)

    matrix.flush()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    write_manifest(output / "rows.csv", rows)

    metadata = {
        "status": "completed_frozen_feature_extraction",
        "representation": "dinov2",
        "model": "dinov2_vits14",
        "feature": "Final normalized class token",
        "dimensions": 384,
        "rows": len(rows),
        "dtype": "float32",
        "batch": args.batch,
        "manifest_audit": checked,
        "manifest_sha256": sha256(manifest),
        "features_sha256": sha256(output / "features.npy"),
        "source_commit": commit,
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "preprocessing": preflight["preprocessing"],
        "normalization": "ImageNet RGB normalization; probe scaling uses training only",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seconds": seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "feature_file_bytes": (output / "features.npy").stat().st_size,
        "qualification": (
            "Frozen backbone; common 224x224 study crops; no additional geometry. "
            "Timer includes decoding and cache writes, excludes model loading. "
            "No test predictions or classifier fitting."
        ),
    }
    with (output / "feature_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--official-splits", required=True)
    parser.add_argument("--batch", type=int, default=8)
    main(parser.parse_args())
