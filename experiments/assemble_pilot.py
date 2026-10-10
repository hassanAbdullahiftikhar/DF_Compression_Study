import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from common import audit, read_manifest, write_manifest

cached_path = Path("experiments/data/c23_smoke/reference.csv")
new_path = Path("experiments/data/c23_remaining/reference.csv")
pilot_path = Path("experiments/data/audit_c23/videos_pilot.csv")
output = Path("experiments/data/c23_pilot")

cached = read_manifest(cached_path)
new = read_manifest(new_path)

assert len(cached) == 800 and len(new) == 7200
assert not ({r["sample_id"] for r in cached} &
            {r["sample_id"] for r in new}), "Duplicate samples"

old_summary = json.loads(cached_path.with_name("extraction_summary.json").read_text())
new_summary = json.loads(new_path.with_name("extraction_summary.json").read_text())

for key in ("model_sha256", "detector", "opencv"):
    assert old_summary[key] == new_summary[key], f"Different {key}"
for key in ("frames", "score_threshold", "margin"):
    assert old_summary["settings"][key] == new_summary["settings"][key], f"Different {key}"

with pilot_path.open(encoding="utf-8-sig", newline="") as handle:
    pilot = {r["video_id"]: r for r in csv.DictReader(handle)}

merged = cached + new
assert {r["video_id"] for r in merged} == set(pilot)
assert all(r["condition"] == "ref" for r in merged)

for row in merged:
    expected = pilot[row["video_id"]]
    for key in ("split", "group_id", "source", "original_ids"):
        assert row[key] == expected[key], f"Provenance mismatch: {key}"

assert all(n == 8 for n in Counter(r["video_id"] for r in merged).values())

split_order = {"train": 0, "val": 1, "test": 2}
merged.sort(key=lambda r: (
    split_order[r["split"]], r["video_id"], int(r["frame_index"])
))
checked = audit(merged, official_dir="experiments/data/splits")

output.mkdir(parents=True, exist_ok=True)
assert not (output / "reference.csv").exists(), "Manifest already exists"
assert not (output / "assembly.json").exists(), "Assembly record already exists"

write_manifest(output / "reference.csv", merged)

record = {
    "audit": checked,
    "source_manifests": [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (cached_path, new_path)
    ],
    "cached_pyav": old_summary["pyav"],
    "new_pyav": new_summary["pyav"],
    "decoder_comparison": json.loads(Path(
        "experiments/data/audit_c23/decoder_compatibility.json"
    ).read_text()),
    "images_copied": 0,
    "reference_origin": "Already H.264 C23-compressed video",
    "crop_counts": {
        split: dict(Counter(r["source"] for r in merged if r["split"] == split))
        for split in ("train", "val", "test")
    }
}
(output / "assembly.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))
