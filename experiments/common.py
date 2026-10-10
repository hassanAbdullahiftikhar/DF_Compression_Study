"""Manifest contract and split checks; no image or ML dependencies."""
import csv
import json
from pathlib import Path

SOURCES = ("real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FIELDS = ("sample_id", "path", "reference_path", "split", "group_id", "video_id",
          "original_ids", "source", "frame_index", "condition", "history")


def read_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no samples")
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)


def audit(rows, check_files=True, official_dir=None):
    seen, groups, videos, originals, samples = set(), {}, {}, {}, {}
    official = {}
    if official_dir:
        for split in ("train", "val", "test"):
            pairs = json.loads((Path(official_dir) / f"{split}.json").read_text())
            for pair in pairs:
                for original in pair:
                    original = str(original)
                    if original in official and official[original] != split:
                        raise ValueError(f"Official original appears in two splits: {original}")
                    official[original] = split
    for row in rows:
        split = row["split"]
        if split not in ("train", "val", "test") or row["source"] not in SOURCES:
            raise ValueError(f"Invalid split/source: {row['sample_id']}")
        if any(not row[field] for field in FIELDS if field != "history"):
            raise ValueError(f"Empty required field: {row['sample_id']}")
        if int(row["frame_index"]) < 0:
            raise ValueError("Negative frame index")
        key = (row["sample_id"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate sample/condition: {key}")
        seen.add(key)
        ids = row["original_ids"].split("|")
        for value, table in ((row["group_id"], groups), (row["video_id"], videos)):
            if value in table and table[value] != split:
                raise ValueError(f"Lineage/video leakage: {value}")
            table[value] = split
        for original in ids:
            if original in originals and originals[original] != split:
                raise ValueError(f"Original ID leaks across splits: {original}")
            originals[original] = split
            if official_dir and official.get(original) != split:
                raise ValueError(f"Original {original} does not match official {split} split")
        provenance = tuple(row[field] for field in
                           ("split", "group_id", "video_id", "original_ids", "source",
                            "frame_index", "reference_path"))
        if row["sample_id"] in samples and samples[row["sample_id"]] != provenance:
            raise ValueError(f"Compression variants changed provenance: {row['sample_id']}")
        samples[row["sample_id"]] = provenance
        for field in ("path", "reference_path"):
            path = Path(row[field])
            if not path.is_absolute():
                raise ValueError(f"{field} must be absolute: {path}")
            if check_files and not path.is_file():
                raise ValueError(f"Missing {field}: {path}")
    # A shared original must have one cluster label, even within a split.
    clusters = {}
    for row in rows:
        for original in row["original_ids"].split("|"):
            if original in clusters and clusters[original] != row["group_id"]:
                raise ValueError(f"Shared original has inconsistent group_id: {original}")
            clusters[original] = row["group_id"]
    return {"rows": len(rows), "samples": len(samples), "videos": len(videos),
            "groups": len(groups), "splits": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "val", "test")}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit crop lineage and variant consistency")
    parser.add_argument("manifest")
    parser.add_argument("--official-splits")
    args = parser.parse_args()
    print(json.dumps(audit(read_manifest(args.manifest), official_dir=args.official_splits), indent=2))
