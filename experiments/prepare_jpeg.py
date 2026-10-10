"""Deterministic crop JPEG/recompression; starts from fixed reference PNG crops."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import PIL
from PIL import Image
from common import audit, read_manifest, write_manifest


def jpeg_sequence(image, qualities, subsampling=2):
    current = image.convert("RGB")
    payload = None
    for quality in qualities:
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be 1–100")
        buffer = io.BytesIO()
        current.save(buffer, format="JPEG", quality=quality, subsampling=subsampling,
                     optimize=False, progressive=False)
        payload = buffer.getvalue()
        with Image.open(io.BytesIO(payload)) as decoded:
            current = decoded.convert("RGB").copy()
    return payload, current


def prepare(manifest, output, config, official_dir=None):
    rows = read_manifest(manifest)
    audit(rows, official_dir=official_dir)
    if any(row["condition"] != "ref" for row in rows):
        raise ValueError("Input must contain one reference row per sample")
    output = Path(output).resolve()
    conditions = [(f"jpeg_q{q}", [q]) for q in config["jpeg"]["evaluation_qualities"]]
    conditions += [("jpeg_" + "_".join(f"q{q}" for q in sequence), sequence)
                   for sequence in config["jpeg"]["history_sequences"]]
    if len({name for name, _ in conditions}) != len(conditions):
        raise ValueError("Duplicate compression conditions")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; refusing to overwrite")
    # Check all input contracts before writing outputs.
    for row in rows:
        if Path(row["path"]).resolve() != Path(row["reference_path"]).resolve():
            raise ValueError("Reference row path and reference_path differ")
        with Image.open(row["path"]) as image:
            if image.format != "PNG" or image.size != (config["input_size"],) * 2:
                raise ValueError("Expected already prepared fixed-size reference PNG crops")
    output.mkdir(parents=True, exist_ok=True)
    variants, records = list(rows), []
    for row in rows:
        with Image.open(row["reference_path"]) as source:
            image = source.convert("RGB")
        reference_hash = hashlib.sha256(Path(row["reference_path"]).read_bytes()).hexdigest()
        token = hashlib.sha256(row["sample_id"].encode()).hexdigest()
        for condition, sequence in conditions:
            payload, _ = jpeg_sequence(image, sequence, config["jpeg"]["subsampling"])
            destination = output / condition / f"{token}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
            variant = dict(row, path=str(destination), condition=condition,
                           history=json.dumps({"prior": row["history"], "jpeg": sequence,
                                               "subsampling": config["jpeg"]["subsampling"]}))
            variants.append(variant)
            records.append({"sample_id": row["sample_id"], "condition": condition,
                            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                            "reference_sha256": reference_hash})
    audit(variants, official_dir=official_dir)
    write_manifest(output / "manifest.csv", variants)
    metadata = {"pillow_version": PIL.__version__,
                "jpeg_library_version": Image.core.jpeglib_version,
                "input_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                "config": config, "records": records}
    (output / "compression_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return audit(variants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/pilot.json"))
    parser.add_argument("--official-splits")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(json.dumps(prepare(args.manifest, args.output, config, args.official_splits), indent=2))
