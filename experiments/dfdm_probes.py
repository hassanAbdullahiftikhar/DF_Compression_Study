"""DFDM probes and controls (research/dfdm_register.md sections 5 and 7).

Probes (class-weighted ridge; FF++ F4/F6 procedures, five generator classes):
  P-DINO-REF, P-CLIP-REF : trained at ref; penalty by mean val det/src BA at ref;
                           threshold maximizes mean val video detection BA over ref/Q80/Q40.
  P-DINO-Q40, P-CLIP-Q40 : trained and selected at Q40 (known-quality heads).
Controls (trained and evaluated at ref):
  C1-DCT  : detection-only ridge on the F4 block-DCT descriptor (compression history).
  C2-MASK : DINOv2 on face-masked crops (fit_border_probes mask; background leakage).

--fit     : validation only (features of train/val crops)
--verify  : re-extract validation features end to end; stored probabilities must match within 1e-6
--test    : one test pass (after the D-G3 freeze)
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch

from clip_probes import load_clip, extract as clip_extract
from dfdm_lib import (GENERATORS, N_SRC, ROOT, VAL_CONDITIONS, aggregate, choose_threshold, evaluate, load_image,
                      prediction_row, read_csv, read_rows, sha256, source_index, write_csv)
from evaluate_probe_compression import extract_batch
from fit_border_probes import apply_mask, load_dinov2, mask_box
from probe_validation import ALPHAS, softmax, solve, weighted_system

OUTPUT = ROOT / "experiments/runs/dfdm_probes_validation"
VERIFY = ROOT / "experiments/results/dfdm_probes_verification_validation.json"
TEST_OUTPUT = ROOT / "experiments/runs/dfdm_probes_test"
QUALITIES = {"ref": 0, "jpeg_q80": 80, "jpeg_q40": 40}
PROBES = {  # id: (feature, training condition, threshold conditions, detection only)
    "P-DINO-REF": ("dinov2", "ref", VAL_CONDITIONS, False),
    "P-DINO-Q40": ("dinov2", "jpeg_q40", ("jpeg_q40",), False),
    "P-CLIP-REF": ("clip", "ref", VAL_CONDITIONS, False),
    "P-CLIP-Q40": ("clip", "jpeg_q40", ("jpeg_q40",), False),
    "C1-DCT": ("dct", "ref", ("ref",), True),
    "C2-MASK": ("dinov2_masked", "ref", ("ref",), False),
}
BATCH = 8


def boxes():
    """Face-mask box per (clip, frame): the real frame's detection, shared by real and fakes."""
    table = {}
    with (ROOT / "experiments/data/dfdm_crops/crop_metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for clip in record["clips"]:
                table[(clip, int(record["frame_index"]))] = mask_box(record)
    return table


def dino_features(model, rows, quality, masks=None):
    """DINOv2 and block-DCT descriptors (evaluate_probe_compression.extract_batch), batch 8."""
    out = {"dinov2": np.empty((len(rows), 384), np.float32), "dct": np.empty((len(rows), 128), np.float32)}
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        images = [load_image(r["path"], quality) for r in batch]
        if masks is not None:
            images = [apply_mask(im, masks[(r["clip_id"], int(r["frame_index"]))]) for im, r in zip(images, batch)]
        feats = extract_batch(images, model)
        for name in out:
            out[name][start:start + len(batch)] = feats[name]
    return out


def extract_all(rows, conditions, split_name):
    """Features per condition for dinov2, dct, clip; masked dinov2 at ref only."""
    model, _ = load_dinov2()
    masks = boxes()
    feats = {}
    for condition in conditions:
        f = dino_features(model, rows, QUALITIES.get(condition, 0))
        feats[("dinov2", condition)], feats[("dct", condition)] = f["dinov2"], f["dct"]
        print(json.dumps({"dinov2_done": split_name, "condition": condition}), flush=True)
    feats[("dinov2_masked", "ref")] = dino_features(model, rows, 0, masks)["dinov2"]
    del model
    torch.cuda.empty_cache()
    clip = load_clip()
    clip_rows = [dict(r, reference_path=r["path"]) for r in rows]
    for condition in conditions:
        feats[("clip", condition)] = clip_extract(clip, clip_rows, QUALITIES.get(condition, 0))
        print(json.dumps({"clip_done": split_name, "condition": condition}), flush=True)
    del clip
    torch.cuda.empty_cache()
    return feats


def design(values, mean, scale):
    return np.column_stack([(values.astype(np.float64) - mean) / scale, np.ones(len(values))])


def predictions(rows, condition, values, params, detection_only):
    x = design(values, params["scaler_mean"], params["scaler_scale"])
    det = softmax(x @ params["det_coef"])
    src = softmax(x @ params["src_coef"]) if not detection_only else np.full((len(rows), N_SRC), 1.0 / N_SRC)
    return [prediction_row(r, condition, det[i, 1], src[i]) for i, r in enumerate(rows)]


def fit_probe(train_x, train_rows, val_values, val_rows, condition, detection_only):
    labels = np.array([source_index(r["source"]) for r in train_rows])
    fake = labels >= 0
    det_sys = weighted_system(train_x, fake.astype(int), 2)
    src_sys = weighted_system(train_x[fake], labels[fake], N_SRC)
    best, search = None, []
    for alpha in ALPHAS:
        params = {"det_coef": solve(det_sys, alpha), "src_coef": solve(src_sys, alpha)}
        x = val_values
        det, src = softmax(x @ params["det_coef"]), softmax(x @ params["src_coef"])
        rows = [prediction_row(r, condition, det[i, 1], src[i]) for i, r in enumerate(val_rows)]
        scores = evaluate(rows)[condition]
        score = scores["detection_ba"] if detection_only else (scores["detection_ba"] + scores["attribution_ba"]) / 2
        search.append({"alpha": alpha, "score": float(score), "validation": scores})
        if best is None or score > best[0] + 1e-12:
            best = (float(score), alpha, params)
    return best, search


def fit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train, val = read_rows("crf0", "train"), read_rows("crf0", "val")
    started = time.perf_counter()
    feats = {"train": extract_all(train, ("ref", "jpeg_q40"), "train"), "val": extract_all(val, VAL_CONDITIONS, "val")}
    OUTPUT.mkdir(parents=True)
    reports = {}
    for probe, (feature, condition, threshold_conditions, detection_only) in PROBES.items():
        raw = feats["train"][(feature, condition)].astype(np.float64)
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale < 1e-12] = 1.0
        val_x = design(feats["val"][(feature, condition)], mean, scale)
        (score, alpha, params), search = fit_probe(design(raw, mean, scale), train, val_x, val, condition, detection_only)
        params.update({"scaler_mean": mean, "scaler_scale": scale})
        conditions = threshold_conditions if feature in ("dinov2", "clip") else ("ref",)
        stored = [row for c in conditions for row in predictions(val, c, feats["val"][(feature, c)], params, detection_only)]
        threshold, threshold_score = choose_threshold(aggregate(stored), threshold_conditions)
        destination = OUTPUT / probe
        destination.mkdir()
        np.savez_compressed(destination / "probe_parameters.npz", **params)
        write_csv(destination / "validation_predictions.csv", stored)
        reports[probe] = {"feature": feature, "training_condition": condition, "detection_only": detection_only,
                          "alpha": alpha, "selection_score": score, "alpha_search": search, "threshold": threshold,
                          "mean_validation_detection_ba_at_threshold": threshold_score,
                          "validation_at_threshold": evaluate(stored, threshold),
                          "parameters_sha256": sha256(destination / "probe_parameters.npz"),
                          "validation_predictions_sha256": sha256(destination / "validation_predictions.csv")}
        print(json.dumps({probe: {"alpha": alpha, "threshold": round(threshold, 4), "validation": {
            c: {k: round(v[k], 4) for k in ("detection_auc", "detection_ba", "attribution_ba")}
            for c, v in reports[probe]["validation_at_threshold"].items()}}}), flush=True)
    record = {"status": "dfdm_validation_only_probes", "register": "research/dfdm_register.md v1.0",
              "test_scores_used": False, "elapsed_minutes": (time.perf_counter() - started) / 60,
              "script_sha256": sha256(Path(__file__)), "lib_sha256": sha256(Path(__file__).with_name("dfdm_lib.py")),
              "probes": reports}
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("Saved:", OUTPUT / "report.json")


def load_params(probe):
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    path = OUTPUT / probe / "probe_parameters.npz"
    assert sha256(path) == report["probes"][probe]["parameters_sha256"]
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}, report["probes"][probe]


def verify():
    if VERIFY.exists():
        raise FileExistsError(VERIFY)
    val = read_rows("crf0", "val")
    feats = extract_all(val, VAL_CONDITIONS, "val")
    results = {}
    for probe, (feature, condition, threshold_conditions, detection_only) in PROBES.items():
        params, entry = load_params(probe)
        conditions = threshold_conditions if feature in ("dinov2", "clip") else ("ref",)
        rescored = {(r["sample_id"], r["condition"]): r for c in conditions
                    for r in predictions(val, c, feats[(feature, c)], params, detection_only)}
        stored = read_csv(OUTPUT / probe / "validation_predictions.csv")
        assert len(stored) == len(rescored)
        worst = max(abs(float(s[f]) - rescored[(s["sample_id"], s["condition"])][f]) for s in stored
                    for f in ("p_fake", *[f"p_source_{k}" for k in range(N_SRC)]))
        results[probe] = {"max_abs_probability_difference": worst, "passed": worst <= 1e-6}
        print(json.dumps({probe: results[probe]}), flush=True)
    record = {"status": "dfdm_probe_verification", "tolerance": 1e-6, "all_passed": all(r["passed"] for r in results.values()),
              "script_sha256": sha256(Path(__file__)), "probes": results}
    with VERIFY.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    if not record["all_passed"]:
        raise SystemExit("DFDM probe verification failed")


def score_test():
    from dfdm_freeze import require_dfdm_freeze
    digest = require_dfdm_freeze(Path(__file__))
    if TEST_OUTPUT.exists():
        raise FileExistsError(TEST_OUTPUT)
    test = read_rows("crf0", "test")
    codec = {lv: read_rows(lv, "test") for lv in ("crf10", "crf23")}
    for lv, rows in codec.items():
        assert [r["video_id"] for r in rows] == [r["video_id"] for r in test]
    dino, _ = load_dinov2()
    masks = boxes()
    clip = load_clip()
    outputs = {p: [] for p in PROBES}
    params = {p: load_params(p)[0] for p in PROBES}
    started = time.perf_counter()
    jobs = [("ref", test, 0)] + [(f"jpeg_q{q}", test, q) for q in (95, 80, 60, 40, 20)] + \
           [("h264_crf10", codec["crf10"], 0), ("h264_crf23", codec["crf23"], 0)]
    for condition, rows, quality in jobs:
        f = dino_features(dino, rows, quality)
        f["clip"] = clip_extract(clip, [dict(r, reference_path=r["path"]) for r in rows], quality)
        if condition == "ref":
            f["dinov2_masked"] = dino_features(dino, rows, 0, masks)["dinov2"]
        for probe, (feature, _, _, detection_only) in PROBES.items():
            if feature in f:
                outputs[probe] += predictions(rows, condition, f[feature], params[probe], detection_only)
        print(json.dumps({"condition_done": condition, "minutes": round((time.perf_counter() - started) / 60, 1)}), flush=True)
    TEST_OUTPUT.mkdir(parents=True)
    reports = {}
    for probe, rows in outputs.items():
        (TEST_OUTPUT / probe).mkdir()
        write_csv(TEST_OUTPUT / probe / "test_predictions.csv", rows)
        reports[probe] = {"rows": len(rows), "test_predictions_sha256": sha256(TEST_OUTPUT / probe / "test_predictions.csv")}
    record = {"status": "dfdm_probe_test_scores", "dfdm_freeze_manifest_sha256": digest,
              "script_sha256": sha256(Path(__file__)), "elapsed_seconds": time.perf_counter() - started, "probes": reports}
    with (TEST_OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("Saved:", TEST_OUTPUT / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("fit", "verify", "test"):
        group.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    {"fit": fit, "verify": verify, "test": score_test}[next(k for k, v in vars(args).items() if v)]()
