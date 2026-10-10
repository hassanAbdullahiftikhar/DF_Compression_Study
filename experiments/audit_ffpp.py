"""Inventory a C23 mirror, map official provenance groups, and select a balanced pilot."""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import struct
from pathlib import Path

CORE = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mp4_structure(path):
    """Check box bounds/header structure, not full codec-decoding integrity."""
    total, offset, boxes = path.stat().st_size, 0, []
    with path.open("rb") as handle:
        while offset < total:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("Truncated MP4 box header")
            size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extension = handle.read(8)
                if len(extension) != 8:
                    raise ValueError("Truncated extended MP4 box header")
                size, header_size = struct.unpack(">Q", extension)[0], 16
            elif size == 0:
                size = total - offset
            if size < header_size or offset + size > total:
                raise ValueError("Invalid MP4 box bounds")
            boxes.append(kind.decode("ascii", errors="replace"))
            offset += size
    if not {"ftyp", "moov", "mdat"}.issubset(boxes):
        raise ValueError("Missing required MP4 boxes")
    return boxes


def audit_dataset(root, splits, output, seed=42, pilot_groups=(72, 14, 14)):
    root, splits, output = Path(root).resolve(), Path(splits).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output must be new or empty")
    pair_map, group_map = {}, {}
    for split in ("train", "val", "test"):
        pairs = json.loads((splits / f"{split}.json").read_text())
        for pair in pairs:
            group_id = "ffpp_" + "_".join(sorted(pair))
            for original in pair:
                if original in pair_map:
                    raise ValueError("Official splits repeat an original ID")
                pair_map[original], group_map[original] = split, group_id
    metadata_path = root / "csv/FF++_Metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    metadata_map = {row["File Path"].replace("\\", "/"): row for row in metadata}
    if len(metadata_map) != len(metadata):
        raise ValueError("Duplicate paths in mirror metadata")
    missing = [name for name in metadata_map if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Metadata paths missing from dataset: {missing[:10]}")
    inventory, errors = [], []
    folders = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "csv"):
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        folders[folder.name] = {"files": len(files), "bytes": sum(path.stat().st_size for path in files),
                              "extensions": dict(collections.Counter(path.suffix.lower() for path in files))}
        for path in files:
            if folder.name not in CORE:
                continue
            relative = path.relative_to(root).as_posix()
            row = metadata_map.get(relative)
            if row is None:
                errors.append({"path": relative, "error": "No metadata row"})
                continue
            source = "real" if folder.name == "original" else folder.name
            pattern = r"(\d{3})\.mp4" if source == "real" else r"(\d{3})_(\d{3})\.mp4"
            match = re.fullmatch(pattern, path.name)
            if match is None:
                errors.append({"path": relative, "error": "Unrecognized core filename"})
                continue
            ids = match.groups()
            if any(original not in pair_map for original in ids):
                errors.append({"path": relative, "error": "Unknown original ID"})
                continue
            if len({pair_map[i] for i in ids}) != 1 or len({group_map[i] for i in ids}) != 1:
                errors.append({"path": relative, "error": "Input IDs cross official provenance groups"})
                continue
            expected_label = "REAL" if source == "real" else "FAKE"
            if row["Label"] != expected_label:
                errors.append({"path": relative, "error": "Metadata label mismatch"})
                continue
            try:
                mp4_structure(path)
                if min(int(row[k]) for k in ("Frame Count", "Width", "Height")) <= 0:
                    raise ValueError("Nonpositive metadata dimensions/count")
            except (ValueError, OSError) as exc:
                errors.append({"path": relative, "error": str(exc)})
                continue
            inventory.append({"video_id": f"{source}/{path.stem}", "path": str(path),
                "split": pair_map[ids[0]], "group_id": group_map[ids[0]], "source": source,
                "original_ids": "|".join(ids), "frame_count_metadata": int(row["Frame Count"]),
                "width_metadata": int(row["Width"]), "height_metadata": int(row["Height"]),
                "codec_metadata": row["Codec"], "file_size_bytes": path.stat().st_size,
                "reference_condition": "c23_reference",
                "prior_history": "Kaggle FF++ C23 mirror; upstream H.264 C23; byte equivalence to official release unverified"})
    groups = collections.defaultdict(list)
    for row in inventory:
        groups[row["group_id"]].append(row)
    incomplete = []
    for group, rows in groups.items():
        support = collections.Counter(row["source"] for row in rows)
        if any(support[source] != 2 for source in ("real", *CORE[1:])):
            incomplete.append({"group_id": group, "source_counts": dict(support)})
    rng = random.Random(seed)
    selected = set()
    for split, number in zip(("train", "val", "test"), pilot_groups):
        candidates = sorted(group for group, rows in groups.items() if rows[0]["split"] == split
                            and group not in {r["group_id"] for r in incomplete})
        if number > len(candidates):
            raise ValueError(f"Not enough complete {split} groups")
        selected.update(rng.sample(candidates, number))
    pilot = [row for row in inventory if row["group_id"] in selected]
    counts = {split: dict(collections.Counter(row["source"] for row in inventory if row["split"] == split))
              for split in ("train", "val", "test")}
    pilot_counts = {split: dict(collections.Counter(row["source"] for row in pilot if row["split"] == split))
                    for split in ("train", "val", "test")}
    report = {"dataset_root": str(root), "folders": folders, "metadata_rows": len(metadata),
        "metadata_missing_paths": len(missing), "metadata_duplicate_paths": len(metadata)-len(metadata_map),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "core_videos": len(inventory), "core_groups": len(groups), "core_split_counts": counts,
        "errors": errors, "incomplete_groups": incomplete, "pilot_seed": seed,
        "pilot_groups": len(selected), "pilot_videos": len(pilot), "pilot_split_counts": pilot_counts,
        "container_check": "MP4 box/header bounds checked for all core files; sampled codec decoding is a separate check",
        "excluded_from_primary": {name: info for name, info in folders.items() if name not in CORE},
        "reference_scope": "Additional compression from an already H.264 C23-compressed reference"}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "videos_core.csv", inventory)
    write_csv(output / "videos_pilot.csv", pilot)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors or incomplete:
        raise SystemExit("Audit issues require review before training")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--official-splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit_dataset(args.root, args.official_splits, args.output, args.seed)
