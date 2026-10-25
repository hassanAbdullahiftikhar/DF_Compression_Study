"""Record hashes that freeze the comparison register before test scoring."""
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "research/comparison_register.md"
OUTPUT = ROOT / "research/comparison_register_freeze.json"
RECORDS = (
    "experiments/data/c23_main/assembly.json",
    "experiments/data/c23_main/reference.csv",
    "experiments/data/c23_main/train_val_reference.csv",
    "experiments/results/main_validation_thresholds_seeds42_44.json",
    "experiments/results/main_validation_analysis_seeds42_44.json",
    "experiments/results/c23_main_probe_compression_validation/report.json",
    "experiments/results/main_probe_validation_analysis.json",
    "experiments/runs/c23_main_matched_q40_probes_validation/report.json",
    "experiments/runs/c23_main_matched_q80_probes_validation/report.json",
    "experiments/results/eight_frame_sensitivity_validation.json",
)
SOURCES = (
    "common.py", "metrics.py", "research_metrics.py", "model.py",
    "train_paired.py", "train_measured.py", "predict_rgb.py", "prepare_jpeg.py",
    "features.py", "probe_validation.py", "evaluate_probe_compression.py",
    "fit_known_quality_probes.py", "eight_frame_sensitivity.py", "freeze_register.py",
)


def hashes(path):
    raw = path.read_bytes()
    return {
        "working_tree_sha256": hashlib.sha256(raw).hexdigest(),
        "lf_normalized_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def test_prediction_scan():
    """Confirm no stored prediction file contains a test-split video."""
    with (ROOT / RECORDS[1]).open(encoding="utf-8-sig", newline="") as handle:
        split = {row["video_id"]: row["split"] for row in csv.DictReader(handle)}
    scanned, offending = 0, []
    for folder in ("experiments/runs", "experiments/results"):
        for path in sorted((ROOT / folder).rglob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if "video_id" not in (reader.fieldnames or []):
                    if "sample_id" not in (reader.fieldnames or []):
                        continue
                    videos = {row["sample_id"].split(":")[0] for row in reader}
                else:
                    videos = {row["video_id"] for row in reader}
            scanned += 1
            if any(split.get(video) == "test" for video in videos):
                offending.append(path.relative_to(ROOT).as_posix())
    return {"csv_files_scanned": scanned, "files_with_test_videos": offending}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scan = test_prediction_scan()
    if scan["files_with_test_videos"]:
        raise RuntimeError(f"Test predictions already exist: {scan}")
    record = {
        "register": "research/comparison_register.md",
        "register_version": "1.0",
        "frozen_on": date.today().isoformat(),
        "register_sha256": hashes(REGISTER),
        "test_scoring_status": "not_started",
        "test_prediction_scan": scan,
        "frozen_records": {name: hashes(ROOT / name) for name in RECORDS},
        "source_sha256": {f"experiments/{n}": hashes(ROOT / "experiments" / n) for n in SOURCES},
        "qualification": (
            "Freezes the register text and the validation records it cites. "
            "Controls C1-C3 are registered but not yet trained; their checkpoints, "
            "heads and thresholds enter through the G3 model-freeze amendment."
        ),
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": OUTPUT.relative_to(ROOT).as_posix(), **scan,
                      "register_sha256": record["register_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
