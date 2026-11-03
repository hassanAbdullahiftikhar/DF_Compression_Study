"""DFDM + Celeb-DF crops (research/dfdm_register.md sections 3-4 and Amendment D-1).

For every distinct Celeb-real source video:
  1. decode the original file once; at the 8 scheduled frames
     (rint(linspace(0, n-1, 8)), n = DFDM frame count), detect the face with
     the FF++ YuNet settings and build the FF++ square box (margin 1.3);
  2. feed every frame to in-memory x264 encoders with DFDM's own option string
     (qp=0 lossless always; CRF 10 / CRF 23 for test-split reals only), so real
     and fake share their final encode;
  3. crop the re-encoded real and all five DFDM fakes of each clip using that
     source, with the same boxes (no generator-specific geometry).
Crops: lossless reference for all splits; CRF 10 / CRF 23 (held-out test
conditions) for the test split only. Failed detections drop that frame for all
six videos and are logged, never replaced.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from codec_sweep import crop
from extract_reference_faces import square_crop_box

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments/data/audit_dfdm/videos.csv"
CELEB_REAL = ROOT / "CelebDF_V2/Celeb-real"
YUNET = ROOT / "experiments/models/face_detection_yunet_2023mar.onnx"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
OUTPUT = ROOT / "experiments/data/dfdm_crops"
LEVELS = {"crf0": "ref", "crf10": "h264_crf10", "crf23": "h264_crf23"}
# DFDM's x264 option string (read from its files); verified to reproduce it exactly
# apart from thread counts, on top of x264's ultrafast preset.
X264_BASE = ("cabac=0:ref=16:no-deblock=1:partitions=none:me=dia:subme=0:mixed-refs=0:merange=16:chroma-me=1:"
             "8x8dct=0:cqm=flat:deadzone-inter=21:deadzone-intra=11:nr=0:bframes=0:weightp=0:keyint=250:"
             "min-keyint=25:scenecut=0:mbtree=0:aq-mode=0:sliced-threads=0")
X264_MODE = {"crf0": "qp=0:psy=0:trellis=0:fast-pskip=0",
             "crf10": "crf=10:psy-rd=1.0,0.0:trellis=1:fast-pskip=1:qcomp=0.6:qpmin=0:qpmax=69:qpstep=4:ipratio=1.4",
             "crf23": "crf=23:psy-rd=1.0,0.0:trellis=1:fast-pskip=1:qcomp=0.6:qpmin=0:qpmax=69:qpstep=4:ipratio=1.4"}
FRAMES, MARGIN, SCORE = 8, 1.3, 0.9


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule(n):
    return sorted(set(np.rint(np.linspace(0, n - 1, min(FRAMES, n))).astype(int).tolist()))


def detector():
    assert sha256(YUNET) == YUNET_SHA
    cv2.setNumThreads(4)
    return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), SCORE, 0.3, 5000,
                                     cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)


def detect(model, bgr):
    """FF++ extract_reference_faces.py procedure: downscale to 640, largest face, confidence tie-break."""
    height, width = bgr.shape[:2]
    scale = min(1.0, 640 / max(width, height))
    dw, dh = max(32, round(width * scale)), max(32, round(height * scale))
    small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    model.setInputSize((dw, dh))
    _, faces = model.detect(small)
    if faces is None or len(faces) == 0:
        return None, 0
    face = max(faces, key=lambda f: (float(f[2] * f[3]), float(f[14]), -float(f[0]), -float(f[1]))).copy()
    face[[0, 2, 4, 6, 8, 10, 12]] *= width / dw
    face[[1, 3, 5, 7, 9, 11, 13]] *= height / dh
    return face, len(faces)


def encoder(level, template, rate):
    buffer = io.BytesIO()
    container = av.open(buffer, "w", format="mp4")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = template.width, template.height, "yuv444p"
    stream.options = {"preset": "ultrafast", "x264-params": X264_BASE + ":" + X264_MODE[level]}
    stream.codec_context.time_base = 1 / rate
    return buffer, container, stream


def decode_crops(source, wanted):
    """Crops at the wanted {index: box} from a path or an in-memory buffer; also returns the frame count."""
    crops, count = {}, 0
    with av.open(source) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                crops[index] = crop(frame, wanted[index])
            count = index + 1
    return crops, count


def save(image, level, sample_id):
    path = OUTPUT / level / (hashlib.sha256(sample_id.encode()).hexdigest() + ".png")
    image.save(path, format="PNG")
    return path


def selftest():
    assert schedule(447) == [0, 64, 127, 191, 255, 319, 382, 446]
    assert schedule(5) == [0, 1, 2, 3, 4]
    assert len(X264_MODE) == 3 and "qp=0" in X264_MODE["crf0"]
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with AUDIT.open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    files = {(r["clip_id"], r["generator"], r["level"]): r for r in audit}
    clips = {r["clip_id"]: r for r in audit if r["level"] == "crf0" and r["generator"] == "faceswap"}
    generators = sorted({r["generator"] for r in audit})
    sources = collections.defaultdict(list)
    for clip_id in sorted(clips):
        target, _, number = clip_id.split("_")
        sources[f"{target}_{number}"].append(clip_id)
    assert len(sources) == 415
    model = detector()
    for level in LEVELS:
        (OUTPUT / level).mkdir(parents=True)
    rows, metadata, failures, started = [], [], [], time.perf_counter()
    for number, (real_name, clip_ids) in enumerate(sorted(sources.items()), 1):
        first = clips[clip_ids[0]]
        split, n = first["split"], int(first["frame_count_metadata"])
        assert all(clips[c]["split"] == split and int(clips[c]["frame_count_metadata"]) == n for c in clip_ids)
        levels = list(LEVELS) if split == "test" else ["crf0"]
        indices = schedule(n)
        boxes, encoders, count = {}, None, 0
        with av.open(str(CELEB_REAL / f"{real_name}.mp4")) as source:
            stream = source.streams.video[0]
            stream.thread_type = "AUTO"
            rate = stream.average_rate
            for index, frame in enumerate(source.decode(stream)):
                if index in indices:
                    bgr = frame.to_ndarray(format="bgr24")
                    face, found = detect(model, bgr)
                    if face is None:
                        failures.append({"real": real_name, "clips": clip_ids, "split": split,
                                         "frame_index": index, "reason": "no_face"})
                    else:
                        box = square_crop_box(face, bgr.shape[1], bgr.shape[0], MARGIN)
                        boxes[index] = list(box)
                        metadata.append({"real": real_name, "clips": clip_ids, "split": split, "frame_index": index,
                                         "face_xywh": [float(v) for v in face[:4]], "confidence": float(face[14]),
                                         "faces_detected": found, "crop_box": list(box),
                                         "frame_size": [bgr.shape[1], bgr.shape[0]]})
                if encoders is None:
                    encoders = {lv: encoder(lv, stream.codec_context, rate) for lv in levels}
                yuv = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24").reformat(format="yuv444p")
                yuv.pts, yuv.time_base = index, 1 / rate
                for _, container, out in encoders.values():
                    for packet in out.encode(yuv):
                        container.mux(packet)
                count = index + 1
        assert count >= n, (real_name, count, n)
        for level, (buffer, container, out) in encoders.items():
            for packet in out.encode():
                container.mux(packet)
            container.close()
            buffer.seek(0)
            real_crops, decoded = decode_crops(buffer, boxes)
            assert decoded == count, (real_name, level, decoded, count)
            video_id = f"real/{real_name}"
            group = f"dfdm_{clip_ids[0].rsplit('_', 1)[0]}"
            for index, image in real_crops.items():
                sample_id = f"{video_id}:frame{index:06d}"
                rows.append({"sample_id": sample_id, "path": str(save(image, level, sample_id)), "split": split,
                             "group_id": group, "video_id": video_id, "clip_id": clip_ids[0], "source": "real",
                             "frame_index": index, "condition": LEVELS[level], "level": level})
            for clip_id in clip_ids:
                for generator in generators:
                    record = files[(clip_id, generator, level)]
                    fake_crops, decoded = decode_crops(record["path"], boxes)
                    assert decoded == n, (clip_id, generator, level, decoded, n)
                    video_id = f"{generator}/{clip_id}"
                    for index, image in fake_crops.items():
                        sample_id = f"{video_id}:frame{index:06d}"
                        rows.append({"sample_id": sample_id, "path": str(save(image, level, sample_id)), "split": split,
                                     "group_id": f"dfdm_{clip_id.rsplit('_', 1)[0]}", "video_id": video_id,
                                     "clip_id": clip_id, "source": generator, "frame_index": index,
                                     "condition": LEVELS[level], "level": level})
        if number % 25 == 0 or number == len(sources):
            print(json.dumps({"reals_done": number, "of": len(sources), "crops": len(rows), "failures": len(failures),
                              "minutes": round((time.perf_counter() - started) / 60, 1)}), flush=True)
    for level in LEVELS:
        subset = [r for r in rows if r["level"] == level]
        with (OUTPUT / f"manifest_{level}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    for name, values in (("crop_metadata.jsonl", metadata), ("failures.jsonl", failures)):
        with (OUTPUT / name).open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value) + "\n")
    counts = {level: dict(collections.Counter(f"{r['split']}:{'real' if r['source'] == 'real' else 'fake'}"
                                              for r in rows if r["level"] == level)) for level in LEVELS}
    report = {
        "status": "dfdm_crops_no_training", "register": "research/dfdm_register.md v1.0 + Amendment D-1",
        "reals": len(sources), "frames_scheduled_per_video": FRAMES, "crops": len(rows), "failed_frames": len(failures),
        "counts": counts, "detector": "YuNet 2023mar", "detector_sha256": YUNET_SHA, "score_threshold": SCORE,
        "margin": MARGIN, "box_source": "original Celeb-real frame; same box for real and all five fakes",
        "x264": {"preset": "ultrafast", "params_base": X264_BASE, "modes": X264_MODE, "pyav": av.__version__},
        "manifests_sha256": {lv: sha256(OUTPUT / f"manifest_{lv}.csv") for lv in LEVELS},
        "audit_sha256": sha256(AUDIT), "script_sha256": sha256(Path(__file__)),
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("reals", "crops", "failed_frames", "counts", "elapsed_minutes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
