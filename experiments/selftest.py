"""Synthetic pipeline verification; these are not deepfake research results."""
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from common import SOURCES, audit, read_manifest, write_manifest
from features import block_dct, dct_basis, descriptor, haar
from metrics import auc, evaluate
from prepare_jpeg import jpeg_sequence, prepare


def expect_failure(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Invalid manifest accepted")


def main():
    rng = np.random.default_rng(42)
    basis = dct_basis()
    np.testing.assert_allclose(basis @ basis.T, np.eye(8), atol=1e-12)
    gray = rng.normal(size=(224, 224))
    np.testing.assert_allclose(np.square(block_dct(gray)).sum(), np.square(gray).sum(), rtol=1e-12)
    np.testing.assert_allclose(sum(np.square(band).sum() for band in haar(gray)),
                               np.square(gray).sum(), rtol=1e-12)
    image = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
    first = jpeg_sequence(image, [80])[0]
    assert first == jpeg_sequence(image, [80])[0]
    assert first != jpeg_sequence(image, [80, 80])[0]
    assert jpeg_sequence(image, [80, 40])[0] != jpeg_sequence(image, [40, 80])[0]
    dimensions = {}
    for representation in ("rgb", "fft", "dct", "wavelet", "residual"):
        vector = descriptor(image, representation)
        assert np.isfinite(vector).all() and vector.ndim == 1
        dimensions[representation] = len(vector)
    assert auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    predictions = []
    for index, source in enumerate(SOURCES):
        row = dict(video_id=source, condition="ref", group_id="g", label=int(index > 0),
                   source_index=index-1, p_fake=0.9 if index else 0.1,
                   **{f"p_source_{k}": float(k == index-1) for k in range(4)})
        predictions.extend([row, dict(row)])
    scores = evaluate(predictions)["ref"]
    assert scores["detection_ba"] == scores["attribution_ba"] == 1
    assert scores["videos"] == 5
    # Temporary directory is generated directly under this workspace and checked before use/cleanup.
    base = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="synthetic_check_", dir=base) as temporary:
        root = Path(temporary).resolve()
        assert root.parent == base and root.name.startswith("synthetic_check_")
        rows = []
        for split in ("train", "val", "test"):
            image_path = root / f"{split}.png"
            image.save(image_path)
            for source in SOURCES:
                rows.append(dict(sample_id=f"{split}-{source}-0", path=str(image_path),
                    reference_path=str(image_path), split=split, group_id=f"g-{split}",
                    video_id=f"{split}-{source}", original_ids=split, source=source,
                    frame_index="0", condition="ref", history="synthetic; not real research data"))
        audit(rows)
        duplicate = copy.deepcopy(rows)
        duplicate.append(dict(rows[0]))
        expect_failure(lambda: audit(duplicate))
        leakage = copy.deepcopy(rows)
        leakage[5]["original_ids"] = "train"
        expect_failure(lambda: audit(leakage))
        clusters = copy.deepcopy(rows)
        clusters[1]["group_id"] = "different"
        expect_failure(lambda: audit(clusters))
        manifest = root / "reference.csv"
        write_manifest(manifest, rows)
        config = json.loads((base / "configs/pilot.json").read_text())
        report = prepare(manifest, root / "jpeg", config)
        assert report["rows"] == 135 and report["samples"] == 15
        variants = read_manifest(root / "jpeg/manifest.csv")
        broken = copy.deepcopy(variants)
        broken[15]["source"] = "FaceSwap"
        expect_failure(lambda: audit(broken))
        expect_failure(lambda: prepare(manifest, root / "jpeg", config))
    print(json.dumps({"status": "passed", "synthetic_only": True,
        "checks": ["orthonormal DCT and energy conservation", "Haar energy conservation",
                   "deterministic JPEG", "history order and repeated-pass differences",
                   "finite descriptor dimensions", "AUROC tie handling", "video probability aggregation",
                   "duplicate and split leakage rejection", "consistent lineage clusters",
                   "paired variant label consistency", "output overwrite rejection"],
        "jpeg_manifest_rows": report["rows"], "descriptor_dimensions": dimensions}, indent=2))


if __name__ == "__main__":
    main()
