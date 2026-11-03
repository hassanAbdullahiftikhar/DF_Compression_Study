"""Extension E10 (comparison_register.md Amendment 6): frozen FF++ detectors on Celeb-DF v2 test.

--extract : FF++ crop protocol on the 518 official test videos (YuNet, 8 uniform
            frames, square box margin 1.3, 224 px Lanczos; failures logged).
--score   : frozen F1-F3 (ResNet-18) and EB4-F1/F2 at frozen FF++ thresholds;
            ref + JPEG Q95/Q80/Q60/Q40/Q20 (in memory), batch 8; one pass.
--analyze : detection AUROC and BA with 2,000 identity-group bootstrap draws (seed 4242).
Detection only: Celeb-DF has one manipulation method.
"""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
import torch

from analyze_ext import WeightedAUC
from analyze_test import interval
from dfdm_crops import detect, detector, schedule
from extract_reference_faces import square_crop_box
from codec_sweep import crop
from metrics import auc, balanced_accuracy
from prepare_jpeg import jpeg_sequence
from train_paired import tensor

ROOT = Path(__file__).resolve().parents[1]
CDF = ROOT / "CelebDF_V2"
CROPS = ROOT / "experiments/data/celebdf_test_crops"
SCORES = ROOT / "experiments/runs/e10_celebdf_test"
ANALYSIS = ROOT / "experiments/results/e10_celebdf_analysis.json"
QUALITIES = (0, 95, 80, 60, 40, 20)
SEEDS = (42, 43, 44)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_list():
    rows = []
    for line in (CDF / "List_of_testing_videos.txt").read_text().splitlines():
        if line.strip():
            flag, rel = line.split()
            name = Path(rel).stem
            real = flag == "1"
            # Identity group: the target celebrity (YouTube-real videos are their own group).
            group = f"yt_{name}" if rel.startswith("YouTube") else f"cdf_{name.split('_')[0]}"
            rows.append({"video": rel, "label": 0 if real else 1, "group": group})
    assert len(rows) == 518
    return rows


def extract():
    if CROPS.exists():
        raise FileExistsError(CROPS)
    model = detector()
    CROPS.mkdir(parents=True)
    records, failures, started = [], [], time.perf_counter()
    for number, video in enumerate(test_list(), 1):
        with av.open(str(CDF / video["video"])) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            wanted = set(schedule(stream.frames))
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    bgr = frame.to_ndarray(format="bgr24")
                    face, _ = detect(model, bgr)
                    if face is None:
                        failures.append({**video, "frame_index": index, "reason": "no_face"})
                        continue
                    box = square_crop_box(face, bgr.shape[1], bgr.shape[0], 1.3)
                    sample = f"{video['video']}:frame{index:06d}"
                    path = CROPS / (hashlib.sha256(sample.encode()).hexdigest() + ".png")
                    crop(frame, box).save(path, format="PNG")
                    records.append({**video, "sample_id": sample, "frame_index": index, "path": str(path)})
                if index >= max(wanted):
                    break
        if number % 100 == 0:
            print(json.dumps({"videos": number, "crops": len(records), "failures": len(failures)}), flush=True)
    with (CROPS / "manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"status": "e10_celebdf_crops", "videos": 518, "crops": len(records), "failed_frames": len(failures),
              "videos_without_crops": sorted({v["video"] for v in test_list()} - {r["video"] for r in records}),
              "manifest_sha256": sha256(CROPS / "manifest.csv"), "script_sha256": sha256(Path(__file__)),
              "minutes": (time.perf_counter() - started) / 60}
    (CROPS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("crops", "failed_frames", "videos_without_crops", "minutes")}))


def models():
    """Frozen FF++ detectors with their frozen thresholds and artifact hashes."""
    from registered_models import rgb_models
    out = {m["id"]: ("rgb", m["run"], m["checkpoint_sha256"], m["threshold"]) for m in rgb_models()
           if m["family"] in ("F1", "F2", "F3")}
    manifest = json.loads((ROOT / "research/extension_freeze_manifest.json").read_text(encoding="utf-8"))
    for model_id, entry in manifest["models"].items():
        if entry["kind"] == "effb4":
            (path, digest), = entry["artifacts_sha256"].items()
            out[model_id] = ("effb4", Path(path).parent, digest, entry["threshold"])
    return out


def score():
    from freeze_models import require_model_freeze
    from freeze_ext_models import require_extension_freeze
    from score_ext import load_effb4
    from score_rgb import load_model
    text = (ROOT / "research/comparison_register.md").read_text(encoding="utf-8")
    assert "Amendment 6" in text and sha256(Path(__file__)) in text, "Amendment 6 must cite this script's hash"
    require_model_freeze(ROOT / "experiments/score_rgb.py")
    require_extension_freeze(ROOT / "experiments/score_ext.py")
    if SCORES.exists():
        raise FileExistsError(SCORES)
    with (CROPS / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    SCORES.mkdir(parents=True)
    out, started = {}, time.perf_counter()
    for model_id, (kind, run, digest, threshold) in models().items():
        model = load_model(run, digest, device) if kind == "rgb" else load_effb4(ROOT / run, digest, device)
        model.eval()
        predictions = []
        for quality in QUALITIES:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                images = []
                for r in batch:
                    from PIL import Image
                    with Image.open(r["path"]) as handle:
                        image = handle.convert("RGB")
                    images.append(tensor(jpeg_sequence(image, [quality], subsampling=2)[1] if quality else image))
                with torch.no_grad():
                    p_fake = model(torch.stack(images).to(device))[0].softmax(1)[:, 1].cpu().numpy()
                for r, p in zip(batch, p_fake):
                    predictions.append({"video": r["video"], "group": r["group"], "label": r["label"],
                                        "condition": "ref" if quality == 0 else f"jpeg_q{quality}", "p_fake": float(p)})
        del model
        torch.cuda.empty_cache()
        path = SCORES / f"{model_id}.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        out[model_id] = {"threshold": threshold, "predictions_sha256": sha256(path)}
        print(json.dumps({model_id: out[model_id]}), flush=True)
    report = {"status": "e10_celebdf_scores", "register": "comparison_register.md Amendment 6",
              "crops_report_sha256": sha256(CROPS / "report.json"), "script_sha256": sha256(Path(__file__)),
              "elapsed_seconds": time.perf_counter() - started, "models": out}
    (SCORES / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def analyze():
    if ANALYSIS.exists():
        raise FileExistsError(ANALYSIS)
    report = json.loads((SCORES / "report.json").read_text(encoding="utf-8"))
    conditions = ["ref" if q == 0 else f"jpeg_q{q}" for q in QUALITIES]
    per_model, canonical = {}, None
    for model_id, entry in report["models"].items():
        path = SCORES / f"{model_id}.csv"
        assert sha256(path) == entry["predictions_sha256"]
        frames = defaultdict(list)
        meta = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for r in csv.DictReader(handle):
                frames[(r["video"], r["condition"])].append(float(r["p_fake"]))
                meta[r["video"]] = (r["group"], int(r["label"]))
        videos = sorted(meta)
        canonical = canonical or videos
        assert videos == canonical
        per_model[model_id] = {c: np.array([np.mean(frames[(v, c)]) for v in videos]) for c in conditions}
    groups = np.array([meta[v][0] for v in canonical])
    labels = np.array([meta[v][1] for v in canonical])
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    rng = np.random.default_rng(4242)
    draws = [np.bincount(np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))]),
                         minlength=len(canonical)).astype(float) for _ in range(2000)]
    results = {}
    for model_id, scores in per_model.items():
        threshold = report["models"][model_id]["threshold"]
        results[model_id] = {}
        for c in conditions:
            s = scores[c]
            correct = ((s >= threshold) == labels).astype(float)
            calc = WeightedAUC(s, labels)

            def ba(w):
                return np.mean([(correct * w * m).sum() / (w * m).sum() for m in (labels == 0, labels == 1)])

            results[model_id][c] = {
                "detection_auc": interval(np.array([calc(w) for w in draws]), auc(labels, s)),
                "detection_ba": interval(np.array([ba(w) for w in draws]),
                                         balanced_accuracy(labels, (s >= threshold).astype(int), 2)),
                "fake_coverage": float((s[labels == 1] >= threshold).mean()),
                "real_specificity": float((s[labels == 0] < threshold).mean())}
    families = {"F1": "F1", "F2": "F2", "F3": "F3", "EB4-F1": "EB4-F1", "EB4-F2": "EB4-F2"}
    summary = {fam: {c: {m: float(np.mean([results[f"{fam}_s{s}"][c][m]["point"] for s in SEEDS]))
                         for m in ("detection_auc", "detection_ba")} for c in conditions} for fam in families}
    record = {"status": "e10_celebdf_external_detection", "register": "comparison_register.md Amendment 6",
              "videos": len(canonical), "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()),
              "groups": len(members), "qualification": "Descriptive external test; frozen FF++ models and thresholds.",
              "seed_means": summary, "models": results, "script_sha256": sha256(Path(__file__)),
              "scores_report_sha256": sha256(SCORES / "report.json")}
    with ANALYSIS.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("extract", "score", "analyze"):
        group.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    {"extract": extract, "score": score, "analyze": analyze}[next(k for k, v in vars(args).items() if v)]()
