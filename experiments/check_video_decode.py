"""Decode a deterministic, split/source-balanced sample; does not prove full-dataset integrity."""
import argparse
import csv
import json
import random
from pathlib import Path
import av


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("output")
    parser.add_argument("--per-cell", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists")
    with open(args.inventory, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng, selected = random.Random(42), []
    for split in ("train", "val", "test"):
        for source in sorted({r["source"] for r in rows}):
            candidates = [row for row in rows if row["split"] == split and row["source"] == source]
            selected.extend(rng.sample(candidates, min(len(candidates), args.per_cell)))
    records = []
    for row in selected:
        record = {"video_id": row["video_id"], "split": row["split"], "source": row["source"]}
        try:
            with av.open(row["path"]) as container:
                stream = container.streams.video[0]
                expected = int(row["frame_count_metadata"])
                targets = {0, expected // 2, expected - 1}
                count, samples = 0, []
                for index, frame in enumerate(container.decode(stream)):
                    count += 1
                    if index in targets:
                        image = frame.to_ndarray(format="rgb24")
                        samples.append({"frame_index": index, "shape": list(image.shape),
                                        "pixel_mean": float(image.mean()), "pts": frame.pts})
                record.update({"status": "ok", "codec": stream.codec_context.name,
                    "decoded_frames": count, "expected_metadata_frames": expected,
                    "count_matches_metadata": count == expected,
                    "dimensions_match_metadata": stream.codec_context.width == int(row["width_metadata"])
                        and stream.codec_context.height == int(row["height_metadata"]),
                    "samples": samples})
                if count == 0 or len(samples) != len(targets):
                    record["status"] = "error"
                    record["error"] = "Could not decode all expected sample positions"
        except Exception as exc:
            record.update({"status": "error", "error": str(exc)})
        records.append(record)
        print(json.dumps({"completed": len(records), "total": len(selected),
                          "video_id": row["video_id"], "status": record["status"]}), flush=True)
    summary = {"decoder": "PyAV", "decoder_version": av.__version__,
        "ffmpeg_library_versions": av.library_versions, "seed": 42, "sampled_videos": len(records),
        "errors": sum(r["status"] != "ok" for r in records),
        "frame_count_mismatches": sum(not r.get("count_matches_metadata", False) for r in records),
        "dimension_mismatches": sum(not r.get("dimensions_match_metadata", False) for r in records),
        "qualification": "Full decoding of selected videos only; other videos had structural checks, not full decoding",
        "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
