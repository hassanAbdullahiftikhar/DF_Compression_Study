"""Fixed C23-reference crops with frozen YuNet; never silently replaces failed samples."""
import argparse
import collections
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
from common import audit, write_manifest


def selected_rows(rows, groups_per_split):
    if groups_per_split is None:
        return rows
    selected = set()
    for split, number in zip(("train", "val", "test"), groups_per_split):
        groups = sorted({r["group_id"] for r in rows if r["split"] == split})
        selected.update(groups[:number])
    return [r for r in rows if r["group_id"] in selected]


def square_crop_box(face, image_width, image_height, margin):
    x, y, width, height = face[:4]
    side = min(int(math.ceil(max(width, height) * margin)), image_width, image_height)
    if side <= 0:
        raise ValueError("Invalid detected face")
    left = min(max(int(math.floor(x + width/2 - side/2)), 0), image_width-side)
    top = min(max(int(math.floor(y + height/2 - side/2)), 0), image_height-side)
    return left, top, left+side, top+side


def contact_sheet(rows, destination):
    sources = ("real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    canvas = Image.new("RGB", (5*224, 2*256), "white")
    draw = ImageDraw.Draw(canvas)
    for column, source in enumerate(sources):
        candidates = [r for r in rows if r["source"] == source and r["split"] == "train"]
        if not candidates:
            continue
        video = candidates[0]["video_id"]
        frames = [r for r in candidates if r["video_id"] == video]
        chosen = [frames[0], frames[len(frames)//2]]
        for index, row in enumerate(chosen):
            with Image.open(row["path"]) as image:
                canvas.paste(image.convert("RGB"), (column*224, index*256+32))
            draw.text((column*224+4, index*256+4), f"{source} | frame {row['frame_index']}", fill="black")
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--official-splits", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--groups-per-split", type=int, nargs=3, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=1.3)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    if args.frames < 1 or args.margin < 1 or not 0 < args.score_threshold < 1:
        raise ValueError("Invalid sampling/detector settings")
    with open(args.inventory, encoding="utf-8", newline="") as handle:
        videos = selected_rows(list(csv.DictReader(handle)), args.groups_per_split)
    if not videos:
        raise ValueError("No selected videos")
    cv2.setNumThreads(4)
    detector = cv2.FaceDetectorYN.create(str(Path(args.model).resolve()), "", (320, 320),
        args.score_threshold, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    output.mkdir(parents=True, exist_ok=True)
    crops = output / "crops"
    crops.mkdir()
    references, records, failures = [], [], []
    start = time.perf_counter()
    for video_number, row in enumerate(videos):
        count = int(row["frame_count_metadata"])
        indices = set(np.rint(np.linspace(0, count-1, min(args.frames, count))).astype(int).tolist())
        found = set()
        try:
            with av.open(row["path"]) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                for frame_index, frame in enumerate(container.decode(stream)):
                    if frame_index not in indices:
                        continue
                    found.add(frame_index)
                    bgr = frame.to_ndarray(format="bgr24")
                    height, width = bgr.shape[:2]
                    scale = min(1.0, 640/max(width, height))
                    detection_width, detection_height = max(32, round(width*scale)), max(32, round(height*scale))
                    small = cv2.resize(bgr, (detection_width, detection_height), interpolation=cv2.INTER_AREA)
                    detector.setInputSize((detection_width, detection_height))
                    _, faces = detector.detect(small)
                    if faces is None or len(faces) == 0:
                        failures.append({"video_id": row["video_id"], "source": row["source"],
                            "split": row["split"], "frame_index": frame_index, "reason": "no_face"})
                        continue
                    face = max(faces, key=lambda f: (float(f[2]*f[3]), float(f[14]), -float(f[0]), -float(f[1]))).copy()
                    face[[0,2,4,6,8,10,12]] *= width/detection_width
                    face[[1,3,5,7,9,11,13]] *= height/detection_height
                    box = square_crop_box(face, width, height, args.margin)
                    sample_id = f"{row['video_id']}:frame{frame_index:06d}"
                    destination = crops / (hashlib.sha256(sample_id.encode()).hexdigest() + ".png")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb).crop(box).resize((224,224), Image.Resampling.LANCZOS)
                    image.save(destination, format="PNG")
                    history = json.dumps({"reference_origin": "FF++ C23 mirror",
                        "upstream": row["prior_history"], "crop": list(box), "resize": "224x224 Lanczos",
                        "alignment": "none", "decoder": "PyAV BGR24 then OpenCV RGB"})
                    references.append({"sample_id": sample_id, "path": str(destination),
                        "reference_path": str(destination), "split": row["split"],
                        "group_id": row["group_id"], "video_id": row["video_id"],
                        "original_ids": row["original_ids"], "source": row["source"],
                        "frame_index": str(frame_index), "condition": "ref", "history": history})
                    records.append({"sample_id": sample_id, "video_path": row["path"],
                        "frame_index": frame_index, "pts": frame.pts, "time_base": str(frame.time_base),
                        "original_dimensions": [width,height], "detection_dimensions": [detection_width,detection_height],
                        "face_xywh": [float(v) for v in face[:4]], "confidence": float(face[14]),
                        "faces_detected": len(faces), "selection": "largest area; confidence tie-break",
                        "crop_box": list(box), "source": row["source"], "split": row["split"],
                        "png_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
                    if frame_index >= max(indices):
                        break
        except Exception as exc:
            failures.append({"video_id": row["video_id"], "source": row["source"],
                             "split": row["split"], "reason": "video_or_detector_error", "error": str(exc)})
        for missing in sorted(indices-found):
            failures.append({"video_id": row["video_id"], "source": row["source"],
                             "split": row["split"], "frame_index": missing, "reason": "missing_frame"})
        if (video_number+1) % 10 == 0 or video_number+1 == len(videos):
            print(json.dumps({"videos_completed": video_number+1, "videos_selected": len(videos),
                              "crops": len(references), "failures": len(failures)}), flush=True)
    if not references:
        raise ValueError("No reference crops produced")
    manifest_audit = audit(references, official_dir=args.official_splits)
    write_manifest(output / "reference.csv", references)
    for name, values in (("crop_metadata.jsonl", records), ("failures.jsonl", failures)):
        with (output/name).open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value)+"\n")
    summary = {"status": "preprocessing_smoke_not_classifier_results", "videos_selected": len(videos),
        "crops": len(references), "failures": len(failures), "failure_reasons": dict(collections.Counter(
            r["reason"] for r in failures)), "crop_counts": {split: dict(collections.Counter(
                r["source"] for r in references if r["split"] == split)) for split in ("train","val","test")},
        "seconds": time.perf_counter()-start, "manifest_audit": manifest_audit,
        "detector": "YuNet 2023mar", "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "opencv": cv2.__version__, "pyav": av.__version__, "numpy": np.__version__,
        "settings": vars(args), "reference_origin": "Already H.264 C23-compressed videos",
        "qualification": "Failures are logged, not replaced; training-classifier accuracy is unmeasured"}
    (output/"extraction_summary.json").write_text(json.dumps(summary, indent=2))
    contact_sheet(references, output/"train_crop_preview.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
