"""Extension X5 (register Amendment 2): whole-video H.264/H.265 re-encoding of the test videos.

--extract : one decode pass per test video feeds six in-memory encoders
            (libx264 CRF 18/23/32/40, libx265 CRF 23/32; preset medium,
            GOP 48, 3 B-frames, yuv420p, 4 threads). Each re-encoded stream is
            decoded, and the scheduled frames are cropped with the stored
            crop_box and resized to 224 px (Lanczos), exactly as
            extract_reference_faces.py did. The same pass checks that the
            original stream reproduces every stored reference crop pixel for
            pixel. Crops are saved as PNG; no video file is written.
"""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from common import read_manifest
from fit_border_probes import load_crop_metadata

MANIFEST = Path("experiments/data/c23_main/reference.csv")
INVENTORY = Path("experiments/data/audit_c23/videos_core.csv")
OUTPUT = Path("experiments/data/c23_codec_test")
SETTINGS = (("libx264", 18), ("libx264", 23), ("libx264", 32), ("libx264", 40), ("libx265", 23), ("libx265", 32))
CONDITIONS = tuple(f"{'h264' if c == 'libx264' else 'h265'}_crf{q}" for c, q in SETTINGS)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop(frame, box):
    """extract_reference_faces.py path: PyAV BGR24, OpenCV RGB, PIL crop, 224 px Lanczos."""
    rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).crop(tuple(box)).resize((224, 224), Image.Resampling.LANCZOS)


def open_encoder(codec, crf, template, rate):
    buffer = io.BytesIO()
    container = av.open(buffer, "w", format="mp4")
    stream = container.add_stream(codec, rate=rate)
    stream.width, stream.height, stream.pix_fmt = template.width, template.height, "yuv420p"
    options = {"crf": str(crf), "preset": "medium"}
    if codec == "libx264":
        options.update({"g": "48", "bf": "3"})
    else:
        options["x265-params"] = "keyint=48:bframes=3:pools=4:frame-threads=2:log-level=error"
    stream.options = options
    stream.thread_count = 4
    stream.codec_context.time_base = 1 / rate
    return buffer, container, stream


def process_video(path, wanted, reference_paths):
    """Return {condition: {frame_index: PIL crop}}, frame count, bytes per setting, reference check."""
    with av.open(path) as source:
        video = source.streams.video[0]
        rate = video.average_rate
        encoders, count, mismatches = None, 0, []
        for index, frame in enumerate(source.decode(video)):
            if index in wanted:
                stored = np.asarray(Image.open(reference_paths[index]).convert("RGB"))
                if not np.array_equal(np.asarray(crop(frame, wanted[index])), stored):
                    mismatches.append(index)
            if encoders is None:
                encoders = [open_encoder(c, q, video.codec_context, rate) for c, q in SETTINGS]
            yuv = frame.reformat(format="yuv420p")
            yuv.pts, yuv.time_base = index, 1 / rate
            for _, container, stream in encoders:
                for packet in stream.encode(yuv):
                    container.mux(packet)
            count = index + 1
    crops, sizes = {}, {}
    for condition, (buffer, container, stream) in zip(CONDITIONS, encoders):
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        sizes[condition] = buffer.getbuffer().nbytes
        buffer.seek(0)
        crops[condition], decoded = {}, 0
        with av.open(buffer) as encoded:
            for index, frame in enumerate(encoded.decode(video=0)):
                if index in wanted:
                    crops[condition][index] = crop(frame, wanted[index])
                decoded = index + 1
        assert decoded == count, f"Frame count changed in {condition}: {decoded} != {count}"
    return crops, count, sizes, mismatches, float(rate)


def extract():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = [r for r in read_manifest(MANIFEST) if r["split"] == "test"]
    assert len(rows) == 5581
    metadata = load_crop_metadata()
    with INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
        inventory = {r["video_id"]: r for r in csv.DictReader(handle)}
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    assert len(by_video) == 700
    OUTPUT.mkdir(parents=True)
    for condition in CONDITIONS:
        (OUTPUT / condition).mkdir()
    records, videos, started = [], [], time.perf_counter()
    for number, (video_id, members) in enumerate(sorted(by_video.items()), 1):
        wanted = {int(r["frame_index"]): metadata[r["sample_id"]]["crop_box"] for r in members}
        paths = {int(r["frame_index"]): r["reference_path"] for r in members}
        path = inventory[video_id]["path"]
        assert all(metadata[r["sample_id"]]["video_path"] == path for r in members)
        tick = time.perf_counter()
        crops, count, sizes, mismatches, fps = process_video(path, wanted, paths)
        if mismatches:
            raise RuntimeError(f"Reference crops not reproduced for {video_id}: frames {mismatches}")
        for row in members:
            index = int(row["frame_index"])
            token = hashlib.sha256(row["sample_id"].encode()).hexdigest()
            reference = np.asarray(Image.open(row["reference_path"]).convert("RGB"), dtype=np.float64)
            for condition in CONDITIONS:
                destination = OUTPUT / condition / f"{token}.png"
                crops[condition][index].save(destination, format="PNG")
                mse = float(np.square(np.asarray(crops[condition][index], dtype=np.float64) - reference).mean())
                records.append({"sample_id": row["sample_id"], "video_id": video_id, "condition": condition,
                                "path": str(destination.resolve()),
                                "crop_psnr_db": 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else None})
        frames = int(inventory[video_id]["frame_count_metadata"])
        videos.append({"video_id": video_id, "decoded_frames": count, "metadata_frames": frames,
                       "seconds": time.perf_counter() - tick,
                       **{f"{c}_bytes": sizes[c] for c in CONDITIONS},
                       **{f"{c}_kbps": 8 * sizes[c] / 1000 / (count / fps) for c in CONDITIONS}})
        if number % 25 == 0 or number == len(by_video):
            print(json.dumps({"videos_done": number, "elapsed_minutes": (time.perf_counter() - started) / 60}),
                  flush=True)
    for name, table in (("crops.csv", records), ("videos.csv", videos)):
        with (OUTPUT / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    record = {
        "status": "post_freeze_extension_codec_crops",
        "extension": "X5, research/comparison_register.md Amendment 2",
        "settings": [{"condition": n, "codec": c, "crf": q} for n, (c, q) in zip(CONDITIONS, SETTINGS)],
        "encoder_options": "preset medium, GOP 48, 3 B-frames, yuv420p, 4 threads; libx265 via x265-params",
        "pyav": av.__version__,
        "reference_crops_reproduced": len(rows),
        "frame_count_preserved": True,
        "crop_rule": "stored crop_box, 224 px Lanczos, PyAV BGR24 then OpenCV RGB; no redetection",
        "manifest_sha256": sha256(MANIFEST),
        "crops_sha256": sha256(OUTPUT / "crops.csv"),
        "videos_sha256": sha256(OUTPUT / "videos.csv"),
        "script_sha256": sha256(Path(__file__)),
        "elapsed_hours": (time.perf_counter() - started) / 3600,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", OUTPUT / "report.json")


def selftest():
    image = np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    out = crop(frame, (10, 5, 50, 45))
    assert out.size == (224, 224)
    assert len(CONDITIONS) == 6 and CONDITIONS[0] == "h264_crf18" and CONDITIONS[-1] == "h265_crf32"
    print(json.dumps({"status": "passed", "synthetic_only": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.extract:
        extract()
    else:
        parser.error("--extract or --selftest required")
