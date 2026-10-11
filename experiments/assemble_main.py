"""Assemble the main cohort, reconciling scheduled crops and exclusions."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from common import SOURCES, audit, read_manifest, write_manifest

output = Path("experiments/data/c23_main")
protocol_path = Path("research/main_cohort_specification.md")

if output.exists() and any(output.iterdir()):
    raise ValueError("Use a new or empty main output directory")
if protocol_path.exists():
    raise FileExistsError(f"Already exists: {protocol_path}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

core_path = Path("experiments/data/audit_c23/videos_core.csv")
with core_path.open(encoding="utf-8-sig", newline="") as handle:
    core = list(csv.DictReader(handle))

inventory = {row["video_id"]: row for row in core}
assert len(core) == len(inventory) == 5000

manifest_paths = (
    Path("experiments/data/c23_pilot/reference.csv"),
    Path("experiments/data/c23_main_remaining/reference.csv"),
)
rows = []
for path in manifest_paths:
    rows.extend(read_manifest(path))

assert len(rows) == 39898
assert len({row["sample_id"] for row in rows}) == len(rows)
assert all(row["condition"] == "ref" for row in rows)

summary_paths = (
    Path("experiments/data/c23_smoke/extraction_summary.json"),
    Path("experiments/data/c23_remaining/extraction_summary.json"),
    Path("experiments/data/c23_main_remaining/extraction_summary.json"),
)
summaries = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in summary_paths
]

for summary in summaries[1:]:
    for field in ("model_sha256", "detector", "opencv"):
        assert summary[field] == summaries[0][field], field
    for field in ("frames", "score_threshold", "margin"):
        assert (
            summary["settings"][field]
            == summaries[0]["settings"][field]
        ), field

planned = set()
for video, metadata in inventory.items():
    count = int(metadata["frame_count_metadata"])
    assert count >= 8
    indices = np.rint(np.linspace(0, count - 1, 8)).astype(int)
    planned.update((video, int(index)) for index in indices)
assert len(planned) == 40000

successful = set()
for row in rows:
    metadata = inventory[row["video_id"]]
    for field in ("split", "group_id", "source", "original_ids"):
        assert row[field] == metadata[field], f"Mismatch: {field}"
    key = (row["video_id"], int(row["frame_index"]))
    assert key in planned
    assert row["sample_id"] == f"{key[0]}:frame{key[1]:06d}"
    successful.add(key)
assert len(successful) == len(rows)

failure_paths = (
    Path("experiments/data/c23_smoke/failures.jsonl"),
    Path("experiments/data/c23_remaining/failures.jsonl"),
    Path("experiments/data/c23_main_remaining/failures.jsonl"),
)
failures = []
for path in failure_paths:
    with path.open(encoding="utf-8") as handle:
        failures.extend(json.loads(line) for line in handle if line.strip())

assert len(failures) == 102
failed = set()
for failure in failures:
    assert failure["reason"] == "no_face"
    metadata = inventory[failure["video_id"]]
    for field in ("source", "split"):
        assert failure[field] == metadata[field]
    key = (failure["video_id"], int(failure["frame_index"]))
    assert key in planned
    failed.add(key)

assert len(failed) == len(failures)
assert not successful.intersection(failed)
assert successful.union(failed) == planned, "Unaccounted scheduled frames"

video_counts = Counter(row["video_id"] for row in rows)
excluded = [
    {
        "video_id": row["video_id"],
        "split": row["split"],
        "group_id": row["group_id"],
        "source": row["source"],
        "planned_crops": 8,
        "valid_crops": 0,
        "reason": "All scheduled crops failed fixed-threshold face detection",
    }
    for row in core if row["video_id"] not in video_counts
]
assert len(excluded) == 1
assert len(video_counts) == 4999

order = {"train": 0, "val": 1, "test": 2}
rows.sort(key=lambda row: (
    order[row["split"]], row["video_id"], int(row["frame_index"])
))
checked = audit(rows, official_dir="experiments/data/splits")
train_val = [row for row in rows if row["split"] in ("train", "val")]
audit(train_val, official_dir="experiments/data/splits")

coverage = {}
for source in SOURCES:
    source_inventory = [row for row in core if row["source"] == source]
    valid_crops = sum(row["source"] == source for row in rows)
    valid_videos = sum(
        row["video_id"] in video_counts for row in source_inventory
    )
    coverage[source] = {
        "planned_videos": len(source_inventory),
        "usable_videos": valid_videos,
        "planned_crops": len(source_inventory) * 8,
        "valid_crops": valid_crops,
        "failed_crops": len(source_inventory) * 8 - valid_crops,
    }

frame_histogram = dict(sorted(Counter(
    video_counts.get(row["video_id"], 0) for row in core
).items()))

record = {
    "status": "main_reference_cohort_assembled",
    "audit": checked,
    "planned_crops": len(planned),
    "valid_crops": len(successful),
    "failed_crops": len(failed),
    "crop_coverage": len(successful) / len(planned),
    "video_crop_count_histogram": frame_histogram,
    "excluded_videos": excluded,
    "source_coverage": coverage,
    "support_rule": (
        "Keep valid scheduled reference crops; include videos with >=1 crop; "
        "average available frame probabilities; identical support across "
        "models and compression conditions"
    ),
    "failure_reasons": dict(Counter(f["reason"] for f in failures)),
    "reference_origin": "Already H.264 C23-compressed videos",
    "source_manifests": [
        {"path": str(path), "sha256": sha256(path)}
        for path in manifest_paths
    ],
    "failure_logs": [
        {"path": str(path), "sha256": sha256(path)}
        for path in failure_paths
    ],
    "preprocessing_summaries": [
        {"path": str(path), "sha256": sha256(path)}
        for path in summary_paths
    ],
    "decoder_comparison": json.loads(Path(
        "experiments/data/audit_c23/decoder_compatibility.json"
    ).read_text(encoding="utf-8")),
    "images_copied": 0,
    "qualification": (
        "Exclusions recorded before main classifier test scoring. "
        "Partial videos retained; complete-eight-frame sensitivity remains planned."
    ),
}

output.mkdir(parents=True, exist_ok=True)
write_manifest(output / "reference.csv", rows)
write_manifest(output / "train_val_reference.csv", train_val)

with (output / "excluded_videos.csv").open(
    "x", encoding="utf-8", newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=excluded[0].keys())
    writer.writeheader()
    writer.writerows(excluded)

record["reference_manifest_sha256"] = sha256(output / "reference.csv")
record["train_val_manifest_sha256"] = sha256(
    output / "train_val_reference.csv"
)
with (output / "assembly.json").open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, allow_nan=False)

protocol = """# Main cohort inclusion specification

Recorded after reference preprocessing and before main classifier test scoring.

Use the 5,000-video FF++ C23 core inventory and official provenance splits.
Retain every valid crop from the eight originally scheduled frame positions.
Include videos with at least one valid crop; exclude zero-crop videos.
Average probabilities over available frames, giving each evaluated video equal weight.
Use the same reference sample IDs across all compared models and compression conditions.
Log failures and coverage by source/split. Keep the detector threshold, margin,
sampling positions and crop settings unchanged; do not replace failed frames.

The assembled cohort contains 39,898 crops from 4,999 videos.
Every missing scheduled crop is reconciled with a no-face failure entry.
A complete-eight-frame sensitivity analysis remains planned.

The pilot used validation for model and threshold selection; its results remain exploratory.
Retrain and select final models using the expanded training/validation cohort.
The earlier checkpoint-selection and threshold-selection rules remain in force.
Freeze each model's validation-selected operating threshold across test conditions.

Reference images derive from already H.264 C23-compressed videos.
Mixed decoder versions and the limited 80-crop compatibility comparison remain documented.
Known input-ID/provenance separation does not establish identity/YouTube independence.
See experiments/data/c23_main/assembly.json for counts, coverage and source hashes.
"""
with protocol_path.open("x", encoding="utf-8") as handle:
    handle.write(protocol)

print(json.dumps({
    key: record[key]
    for key in (
        "status", "audit", "crop_coverage", "video_crop_count_histogram",
        "excluded_videos", "source_coverage", "images_copied"
    )
}, indent=2))
