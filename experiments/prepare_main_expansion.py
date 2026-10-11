"""Prepare full-core expansion while reusing every existing pilot crop."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from common import SOURCES, read_manifest

core_path = Path("experiments/data/audit_c23/videos_core.csv")
reference_path = Path("experiments/data/c23_pilot/reference.csv")
remaining_path = Path("experiments/data/audit_c23/videos_main_remaining.csv")
record_path = Path("experiments/data/audit_c23/main_preparation.json")
qa_path = Path("experiments/results/crop_qa_pilot.json")
qa_image = Path("experiments/results/crop_qa_pilot.png")

for path in (remaining_path, record_path):
    if path.exists():
        raise FileExistsError(f"Already exists: {path}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with core_path.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    core = list(reader)
    fields = reader.fieldnames

core_by_video = {row["video_id"]: row for row in core}
assert len(core) == len(core_by_video) == 5000
assert {row["source"] for row in core} == set(SOURCES)

references = read_manifest(reference_path)
assert len(references) == 8000
assert all(row["condition"] == "ref" for row in references)

counts = Counter(row["video_id"] for row in references)
assert len(counts) == 1000
assert all(count == 8 for count in counts.values())

for row in references:
    expected = core_by_video[row["video_id"]]
    for field in ("split", "group_id", "source", "original_ids"):
        assert row[field] == expected[field], f"Mismatch: {field}"

remaining = [
    row for row in core if row["video_id"] not in counts
]
assert len(remaining) == 4000
assert Counter(row["split"] for row in remaining) == {
    "train": 2880, "val": 560, "test": 560
}

pilot_groups = {row["group_id"] for row in references}
remaining_groups = {row["group_id"] for row in remaining}
assert len(pilot_groups) == 100
assert len(remaining_groups) == 400
assert not pilot_groups.intersection(remaining_groups)

group_sources = {}
for row in core:
    group_sources.setdefault(row["group_id"], Counter())[row["source"]] += 1
assert all(
    counts_by_source == Counter({source: 2 for source in SOURCES})
    for counts_by_source in group_sources.values()
)

crop_paths = {
    Path(row["reference_path"]).resolve() for row in references
}
assert len(crop_paths) == 8000

crop_bytes = sum(path.stat().st_size for path in crop_paths)
mean_crop_bytes = crop_bytes / len(crop_paths)
additional_crops = len(remaining) * 8
estimated_bytes = int(mean_crop_bytes * additional_crops)

# Planning margin, not a guarantee about unseen image sizes.
planning_budget = int(estimated_bytes * 1.5) + 1024 ** 3
free_bytes = shutil.disk_usage(Path.cwd()).free
gib = 1024 ** 3

qa = json.loads(qa_path.read_text(encoding="utf-8"))
assert qa["sample_count"] == 30
assert qa["status"] == "sampled_crop_sheet_pending_visual_review"

record = {
    "status": "main_inventory_prepared_no_new_crops_extracted",
    "core_videos": len(core),
    "reused_videos": len(counts),
    "reused_crops": len(references),
    "remaining_videos": len(remaining),
    "remaining_groups": len(remaining_groups),
    "remaining_by_split": dict(Counter(r["split"] for r in remaining)),
    "additional_crops_expected": additional_crops,
    "final_crops_expected": 40000,
    "existing_crop_storage_gib": crop_bytes / gib,
    "estimated_additional_crop_storage_gib": estimated_bytes / gib,
    "planning_budget_gib": planning_budget / gib,
    "free_disk_gib": free_bytes / gib,
    "storage_gate_passed": free_bytes >= planning_budget,
    "storage_qualification": (
        "Extrapolated from pilot PNG sizes; budget includes 50% margin "
        "and 1 GiB overhead"
    ),
    "input_sha256": {
        "core_inventory": sha256(core_path),
        "pilot_reference_manifest": sha256(reference_path),
    },
    "images_copied": 0,
    "videos_copied": 0,
    "final_training_note": (
        "Retrain and select using expanded train/validation data; "
        "pilot validation findings remain exploratory"
    ),
}

with remaining_path.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(remaining)

with record_path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, allow_nan=False)

qa["status"] = "sampled_visual_review_passed"
qa["contact_sheet_sha256"] = sha256(qa_image)
qa["visual_review"] = {
    "basis": "Assistant review of user-supplied 30-crop contact sheet",
    "findings": [
        "All sampled crops contain reasonably framed faces",
        "No obvious empty or background-only detection",
        "Different originals, directions and frames appear within groups",
        "Background and text shortcut controls remain necessary"
    ],
    "qualification": "Sampled review only; does not verify every crop or source label"
}
qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

print(json.dumps(record, indent=2))
