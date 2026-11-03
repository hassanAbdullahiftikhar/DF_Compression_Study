"""Extension X8 (register Amendments 7-8): I-JEPA ViT-H/14 ridge probes.

--prepare : extract the target encoder from Meta's official I-JEPA checkpoint into a
            small state-dict file and record both hashes.
--fit     : validation only. I-JEPA features for train/val at ref/Q80/Q40; X8-REF,
            X8-Q80 and X8-Q40 probes with the F4/F5/F6 procedures.
--verify  : re-extract validation features end to end; stored probabilities must
            be reproduced within 1e-6.
--freeze  : write research/jepa_freeze_manifest.json (Amendment 8 must cite it).
--test    : one test pass at six JPEG and six X5 codec conditions.
--analyze : the Amendment 7 analysis, on the same 2,000 group draws (seed 4242).

The encoder is a plain-PyTorch copy of the I-JEPA ViT (facebookresearch/ijepa,
src/models/vision_transformer.py), loaded with strict key matching; no extra
packages are needed. It runs in float32 with TF32 off and deterministic kernels.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms.functional import to_tensor

from common import SOURCES, audit, read_manifest
from evaluate_probe_compression import choose_threshold, probabilities
from fit_known_quality_probes import compact, fit, threshold_for
from metrics import aggregate, auc, evaluate
from prepare_jpeg import jpeg_sequence
from probe_validation import prediction_rows, shuffle_video_labels

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path.home() / ".cache" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
ENCODER = Path.home() / ".cache" / "ijepa" / "IN1K-vit.h.14-300e.target_encoder.pt"
ENCODER_RECORD = Path("experiments/results/ext_jepa_encoder_extraction.json")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DIM, DEPTH, HEADS, PATCH = 1280, 32, 16, 14
BATCH = 8
TRAIN_VAL = Path("experiments/data/c23_main/train_val_reference.csv")
MANIFEST = Path("experiments/data/c23_main/reference.csv")
CACHE = Path("experiments/features/c23_ext_jepa_trainval")
FIT_OUTPUT = Path("experiments/runs/c23_ext_jepa_probes_validation")
VERIFY_OUTPUT = Path("experiments/results/ext_jepa_verification_validation.json")
FREEZE = ROOT / "research/jepa_freeze_manifest.json"
REGISTER = ROOT / "research/comparison_register.md"
TEST_OUTPUT = Path("experiments/runs/c23_ext_jepa_test")
ANALYSIS = Path("experiments/results/ext_jepa_analysis_test.json")
CLIP_TEST = Path("experiments/runs/c23_ext_clip_test")
TRACKS = {"X8-REF": 0, "X8-Q80": 80, "X8-Q40": 40}
VAL_QUALITIES = (0, 80, 40)
TEST_QUALITIES = (0, 95, 80, 60, 40, 20)
SCRIPTS = ("jepa_probes.py", "analyze_test.py", "analyze_ext.py", "codec_sweep.py", "score_ext.py", "metrics.py",
           "research_metrics.py", "probe_validation.py", "fit_known_quality_probes.py",
           "evaluate_probe_compression.py", "prepare_jpeg.py", "registered_models.py", "common.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_of(quality):
    return "ref" if quality == 0 else f"jpeg_q{quality}"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, DIM * 3)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * (c // HEADS) ** -0.5).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(DIM, DIM * 4), nn.GELU(), nn.Linear(DIM * 4, DIM)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(DIM, eps=1e-6), Attention()
        self.norm2, self.mlp = nn.LayerNorm(DIM, eps=1e-6), Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """I-JEPA ViT-H/14 at 224 px: 256 patch tokens, fixed sin-cos positions, no class token."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // PATCH) ** 2, DIM), requires_grad=False)
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def prepare():
    if ENCODER.exists() or ENCODER_RECORD.exists():
        raise FileExistsError(ENCODER if ENCODER.exists() else ENCODER_RECORD)
    checkpoint_sha = sha256(CHECKPOINT)
    saved = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    state = {k.removeprefix("module."): v.float().clone() for k, v in saved["target_encoder"].items()}
    model = VisionTransformer()
    model.load_state_dict(state, strict=True)
    torch.save(state, ENCODER)
    record = {"status": "jepa_target_encoder_extracted", "source_url": CHECKPOINT_URL,
              "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": CHECKPOINT.stat().st_size,
              "checkpoint_keys": sorted(saved), "epoch": saved.get("epoch"), "used": "target_encoder",
              "parameters": int(sum(v.numel() for v in state.values())),
              "encoder_sha256": sha256(ENCODER), "encoder_bytes": ENCODER.stat().st_size,
              "script_sha256": sha256(Path(__file__))}
    with ENCODER_RECORD.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


def encoder_sha():
    return json.loads(ENCODER_RECORD.read_text(encoding="utf-8"))["encoder_sha256"]


def load_jepa():
    assert sha256(ENCODER) == encoder_sha(), "I-JEPA encoder changed"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = VisionTransformer()
    model.load_state_dict(torch.load(ENCODER, map_location="cpu", weights_only=True), strict=True)
    return model.to("cuda").eval()


def extract(model, rows, quality=0):
    """1280-d average-pooled final-layer I-JEPA tokens for rows (reference_path), optionally JPEG-compressed first."""
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    out = np.empty((len(rows), DIM), dtype=np.float32)
    for start in range(0, len(rows), BATCH):
        images = []
        for row in rows[start:start + BATCH]:
            with Image.open(row["reference_path"]) as handle:
                image = handle.convert("RGB")
            assert image.size == (224, 224)
            images.append(to_tensor(jpeg_sequence(image, [quality], subsampling=2)[1] if quality else image))
        x = (torch.stack(images).to("cuda") - mean) / std
        with torch.inference_mode():
            out[start:start + len(images)] = model(x).mean(dim=1).cpu().numpy()
    assert np.isfinite(out).all()
    return out


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_params(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def predictions(rows, condition, values, params):
    det, src = probabilities(values, params)
    return prediction_rows([dict(r, condition=condition) for r in rows], det, src)


def fit_all():
    if FIT_OUTPUT.exists() or CACHE.exists():
        raise FileExistsError(FIT_OUTPUT if FIT_OUTPUT.exists() else CACHE)
    rows = read_manifest(TRAIN_VAL)
    checked = audit(rows, official_dir="experiments/data/splits")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    assert len(train) == 28735 and len(val) == 5582
    model = load_jepa()
    started = time.perf_counter()
    CACHE.mkdir(parents=True)
    feats = {}
    for quality in VAL_QUALITIES:
        for split, subset in (("train", train), ("val", val)):
            feats[(split, quality)] = extract(model, subset, quality)
            np.save(CACHE / f"{split}_{condition_of(quality)}.npy", feats[(split, quality)])
            print(json.dumps({"extracted": split, "condition": condition_of(quality),
                              "minutes": (time.perf_counter() - started) / 60}), flush=True)
    extract_minutes = (time.perf_counter() - started) / 60
    del model
    torch.cuda.empty_cache()
    targets = np.array([SOURCES.index(r["source"]) - 1 for r in train])
    video_ids = [r["video_id"] for r in train]
    FIT_OUTPUT.mkdir(parents=True)
    reports = {}
    for track, quality in TRACKS.items():
        condition = condition_of(quality)
        raw = feats[("train", quality)].astype(np.float64)
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design = lambda x: np.column_stack([(x.astype(np.float64) - mean) / scale, np.ones(len(x))])
        train_x = design(raw)
        val_rows = [dict(r, condition=condition) for r in val]
        best, search = fit(train_x, design(feats[("val", quality)]), targets, val_rows, condition)
        params = {"scaler_mean": mean, "scaler_scale": scale, "det_coef": best["det_coef"], "src_coef": best["src_coef"]}
        if track == "X8-REF":
            stored = [r for q in VAL_QUALITIES for r in predictions(val, condition_of(q), feats[("val", q)], params)]
            threshold, threshold_score = choose_threshold(aggregate(stored))
        else:
            stored = best["predictions"]
            threshold, threshold_score = threshold_for(stored, condition), None
        shuffled = []
        for seed in range(101, 106):
            control, _ = fit(train_x, design(feats[("val", quality)]), shuffle_video_labels(video_ids, targets, seed),
                             val_rows, condition)
            shuffled.append({"seed": seed, **compact(control)})
        destination = FIT_OUTPUT / track
        destination.mkdir()
        np.savez_compressed(destination / "probe_parameters.npz", **params)
        write_csv(destination / "validation_predictions.csv", stored)
        reports[track] = {"training_condition": condition, "real_labels": compact(best), "alpha_search": search,
                          "threshold": threshold, "mean_validation_detection_ba_at_threshold": threshold_score,
                          "validation_at_threshold": evaluate(stored, threshold),
                          "shuffled_labels": shuffled,
                          "parameters_sha256": sha256(destination / "probe_parameters.npz"),
                          "validation_predictions_sha256": sha256(destination / "validation_predictions.csv")}
        print(json.dumps({track: {"alpha": best["alpha"], "threshold": threshold,
                                  "validation": {c: {k: round(v[k], 4) for k in ("detection_auc", "detection_ba", "attribution_ba")}
                                                 for c, v in reports[track]["validation_at_threshold"].items()}}}), flush=True)
    record = {"status": "validation_only_post_freeze_extension", "extension": "X8 I-JEPA ViT-H/14, Amendment 7",
              "test_scores_used": False, "manifest_audit": checked, "manifest_sha256": sha256(TRAIN_VAL),
              "encoder_sha256": encoder_sha(), "feature": "target encoder, mean of 256 final-layer tokens, 1280-d, float32",
              "feature_cache_bytes": sum(p.stat().st_size for p in CACHE.glob("*.npy")),
              "extract_minutes": extract_minutes, "script_sha256": sha256(Path(__file__)),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "tracks": reports}
    with (FIT_OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print("Saved:", FIT_OUTPUT / "report.json")


def verify():
    if VERIFY_OUTPUT.exists():
        raise FileExistsError(VERIFY_OUTPUT)
    report = json.loads((FIT_OUTPUT / "report.json").read_text(encoding="utf-8"))
    val = [r for r in read_manifest(TRAIN_VAL) if r["split"] == "val"]
    model = load_jepa()
    feats = {q: extract(model, val, q) for q in VAL_QUALITIES}
    results = {}
    for track, quality in TRACKS.items():
        entry = report["tracks"][track]
        path = FIT_OUTPUT / track / "probe_parameters.npz"
        assert sha256(path) == entry["parameters_sha256"]
        params = load_params(path)
        qualities = VAL_QUALITIES if track == "X8-REF" else (quality,)
        rescored = {(r["sample_id"], r["condition"]): r for q in qualities
                    for r in predictions(val, condition_of(q), feats[q], params)}
        stored = read_csv(FIT_OUTPUT / track / "validation_predictions.csv")
        assert len(stored) == len(rescored)
        worst = max(abs(float(s[k]) - rescored[(s["sample_id"], s["condition"])][k])
                    for s in stored for k in ("p_fake", "p_source_0", "p_source_1", "p_source_2", "p_source_3"))
        results[track] = {"max_abs_probability_difference": worst, "passed": worst <= 1e-6}
        print(json.dumps({track: results[track]}), flush=True)
    record = {"status": "extension_jepa_verification_validation", "tolerance": 1e-6,
              "all_passed": all(r["passed"] for r in results.values()), "script_sha256": sha256(Path(__file__)),
              "tracks": results}
    with VERIFY_OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    if not record["all_passed"]:
        raise SystemExit("I-JEPA verification failed")


def freeze():
    if FREEZE.exists():
        raise FileExistsError(FREEZE)
    verification = json.loads(VERIFY_OUTPUT.read_text(encoding="utf-8"))
    assert verification["all_passed"]
    report = json.loads((FIT_OUTPUT / "report.json").read_text(encoding="utf-8"))
    models = {}
    for track in TRACKS:
        path = (FIT_OUTPUT / track / "probe_parameters.npz").as_posix()
        assert sha256(path) == report["tracks"][track]["parameters_sha256"]
        models[track] = {"threshold": report["tracks"][track]["threshold"], "parameters": path,
                         "parameters_sha256": report["tracks"][track]["parameters_sha256"]}
    extraction = json.loads(ENCODER_RECORD.read_text(encoding="utf-8"))
    record = {"status": "jepa_freeze_manifest_before_test_scoring", "register": "Amendments 7 and 8",
              "frozen_on": date.today().isoformat(), "checkpoint_sha256": extraction["checkpoint_sha256"],
              "encoder_sha256": extraction["encoder_sha256"], "models": models,
              "fit_report_sha256": sha256(FIT_OUTPUT / "report.json"), "verification_sha256": sha256(VERIFY_OUTPUT),
              "scripts_sha256": {f"experiments/{n}": sha256(ROOT / "experiments" / n) for n in SCRIPTS}}
    with FREEZE.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": FREEZE.relative_to(ROOT).as_posix(), "sha256": sha256(FREEZE)}))


def require_jepa_freeze():
    if not FREEZE.exists():
        raise RuntimeError("I-JEPA freeze manifest missing")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    digest = sha256(FREEZE)
    text = REGISTER.read_text(encoding="utf-8")
    if "Amendment 8" not in text or digest not in text:
        raise RuntimeError("Register Amendment 8 citing the I-JEPA manifest is missing")
    for path, expected in manifest["scripts_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Frozen code changed: {path}"
    for entry in manifest["models"].values():
        assert sha256(entry["parameters"]) == entry["parameters_sha256"]
    return manifest, digest


def score_test():
    from score_ext import codec_rows
    manifest, digest = require_jepa_freeze()
    if TEST_OUTPUT.exists():
        raise FileExistsError(TEST_OUTPUT)
    rows = read_manifest(MANIFEST)
    audit(rows, official_dir="experiments/data/splits")
    test = [r for r in rows if r["split"] == "test"]
    assert len(test) == 5581
    codec, _ = codec_rows()
    params = {t: load_params(e["parameters"]) for t, e in manifest["models"].items()}
    model = load_jepa()
    started, outputs = time.perf_counter(), {t: [] for t in TRACKS}
    jobs = [(condition_of(q), test, q) for q in TEST_QUALITIES] + [(c, r, 0) for c, r in codec.items()]
    for condition, subset, quality in jobs:
        feats = extract(model, subset, quality)
        for track in TRACKS:
            outputs[track] += predictions(subset, condition, feats, params[track])
        print(json.dumps({"condition_done": condition, "minutes": (time.perf_counter() - started) / 60}), flush=True)
    TEST_OUTPUT.mkdir(parents=True)
    reports = {}
    for track, rows_out in outputs.items():
        (TEST_OUTPUT / track).mkdir()
        write_csv(TEST_OUTPUT / track / "test_predictions.csv", rows_out)
        reports[track] = {"rows": len(rows_out), "test_predictions_sha256": sha256(TEST_OUTPUT / track / "test_predictions.csv")}
    record = {"status": "extension_jepa_test_scores", "jepa_freeze_manifest_sha256": digest,
              "conditions": [j[0] for j in jobs], "script_sha256": sha256(Path(__file__)),
              "elapsed_seconds": time.perf_counter() - started, "models": reports}
    with (TEST_OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("Saved:", TEST_OUTPUT / "report.json")


def analyze():
    from analyze_ext import WeightedAUC
    from analyze_test import BOOT_SEED, DRAWS, K, METRICS, REENACT, SWAP, Measures, bootstrap_p, interval
    from codec_sweep import CONDITIONS as CODEC
    manifest, digest = require_jepa_freeze()
    if ANALYSIS.exists():
        raise FileExistsError(ANALYSIS)
    jepa_report = json.loads((TEST_OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert jepa_report["jepa_freeze_manifest_sha256"] == digest
    files = defaultdict(list)
    for track, entry in jepa_report["models"].items():
        path = TEST_OUTPUT / track / "test_predictions.csv"
        assert sha256(path) == entry["test_predictions_sha256"]
        files[track].append(path)
    for folder in (Path("experiments/runs/c23_test_probes"), Path("experiments/runs/c23_ext_test_codec")):
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        path = folder / "F4_dinov2" / "test_predictions.csv"
        assert sha256(path) == report["models"]["F4_dinov2"]["test_predictions_sha256"]
        files["F4_dinov2"].append(path)
    clip_report = json.loads((CLIP_TEST / "report.json").read_text(encoding="utf-8"))
    path = CLIP_TEST / "X7-REF" / "test_predictions.csv"
    assert sha256(path) == clip_report["models"]["X7-REF"]["test_predictions_sha256"]
    files["X7-REF"].append(path)
    thresholds = {t: e["threshold"] for t, e in manifest["models"].items()}
    registered = json.loads(Path("experiments/results/test_analysis.json").read_text(encoding="utf-8"))
    thresholds["F4_dinov2"] = json.loads(Path("research/model_freeze_manifest.json").read_text(encoding="utf-8"))["models"]["F4_dinov2"]["threshold"]
    thresholds["X7-REF"] = json.loads(Path("research/clip_freeze_manifest.json").read_text(encoding="utf-8"))["models"]["X7-REF"]["threshold"]
    clip_analysis = json.loads(Path("experiments/results/ext_clip_analysis_test.json").read_text(encoding="utf-8"))
    conditions = [condition_of(q) for q in TEST_QUALITIES] + list(CODEC)
    C = {c: i for i, c in enumerate(conditions)}
    tables, canonical = {}, None
    for name, paths in files.items():
        by = defaultdict(dict)
        for v in aggregate([r for p in paths for r in read_csv(p)]):
            by[v["condition"]][v["video_id"]] = v
        first = by["ref"]
        meta = [(v, first[v]["group_id"], first[v]["label"], first[v]["source_index"]) for v in sorted(first)]
        canonical = canonical or meta
        assert meta == canonical and set(by) == set(conditions), name
        tables[name] = by
    vids = [m[0] for m in canonical]
    groups = np.array([m[1] for m in canonical])
    labels = np.array([m[2] for m in canonical])
    sources = np.array([m[3] for m in canonical])
    fake, family = labels == 1, np.isin(sources, REENACT).astype(int)
    names = list(tables)
    arrays = {t: np.full((len(names), len(conditions), len(vids)), np.nan) for t in ("det", "src", "fam", "wswap", "wreen")}
    scores = {}
    for i, name in enumerate(names):
        for c in conditions:
            p_fake = np.array([tables[name][c][v]["p_fake"] for v in vids])
            p = np.array([[tables[name][c][v][f"p_source_{k}"] for k in range(4)] for v in vids])
            scores[(name, c)] = (p_fake, p)
            arrays["det"][i, C[c]] = (p_fake >= thresholds[name]) == labels
            arrays["src"][i, C[c]] = (p.argmax(axis=1) == sources) & fake
            arrays["fam"][i, C[c]] = ((p[:, list(REENACT)].sum(1) > p[:, list(SWAP)].sum(1)).astype(int) == family) & fake
            for task, (a, b) in (("wswap", SWAP), ("wreen", REENACT)):
                arrays[task][i, C[c]] = (np.where(p[:, b] > p[:, a], b, a) == sources) & np.isin(sources, (a, b))
    measure = Measures(arrays, labels, sources, C["ref"])
    point = measure(np.ones(len(vids)))
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    calculators = {c: (WeightedAUC(scores[("X8-REF", c)][0], labels),
                       [WeightedAUC(scores[("X8-REF", c)][1][fake, k], (sources[fake] == k).astype(int)) for k in range(4)])
                   for c in conditions}

    def auc_vals(w):
        vals = {c: (d(w), float(np.mean([a(w[fake]) for a in att]))) for c, (d, att) in calculators.items()}
        ref = vals["ref"]
        out = {}
        for c, (d, a) in vals.items():
            s_det = (d - 0.5) / (ref[0] - 0.5) if ref[0] - 0.5 >= 0.05 else np.nan
            s_att = (a - 0.5) / (ref[1] - 0.5) if ref[1] - 0.5 >= 0.05 else np.nan
            out[c] = (d, a, s_att - s_det)
        return out

    auc_point = auc_vals(np.ones(len(vids)))
    samples = np.full((DRAWS, len(names), len(conditions), len(METRICS)), np.nan)
    auc_samples = {c: np.full((DRAWS, 3), np.nan) for c in conditions}
    rng = np.random.default_rng(BOOT_SEED)
    for d in range(DRAWS):
        idx = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        w = np.bincount(idx, minlength=len(vids)).astype(float)
        samples[d] = measure(w)
        for c, v in auc_vals(w).items():
            auc_samples[c][d] = v
    row = {n: i for i, n in enumerate(names)}

    def single(name, c, m):
        return interval(samples[:, row[name], C[c], K[m]], point[row[name], C[c], K[m]]), samples[:, row[name], C[c], K[m]]

    def paired(a, b, c, m):
        return interval(samples[:, row[a], C[c], K[m]] - samples[:, row[b], C[c], K[m]],
                        point[row[a], C[c], K[m]] - point[row[b], C[c], K[m]])

    worst = 0.0
    for c in conditions[:6]:
        for m in ("detection_ba", "attribution_ba", "attribution_minus_detection_retention"):
            mine, theirs = single("F4_dinov2", c, m)[0], registered["model_intervals"]["F4_dinov2"][c][m]
            if mine["ci95"] is not None or theirs["ci95"] is not None:
                worst = max(worst, abs(mine["point"] - theirs["point"]), *[abs(x - y) for x, y in zip(mine["ci95"], theirs["ci95"])])
    clip_worst = 0.0
    for c in conditions:
        mine, theirs = single("X7-REF", c, "attribution_minus_detection_retention")[0], clip_analysis["curves"]["X7-REF"][c]["attribution_minus_detection_retention"]
        if mine["ci95"] is not None or theirs["ci95"] is not None:
            clip_worst = max(clip_worst, abs(mine["point"] - theirs["point"]), *[abs(x - y) for x, y in zip(mine["ci95"], theirs["ci95"])])
    assert worst < 1e-10 and clip_worst < 1e-10, (worst, clip_worst)
    delta, values = single("X8-REF", "jpeg_q20", "attribution_minus_detection_retention")
    record = {
        "status": "post_freeze_extension_jepa_test_analysis", "register": "Amendments 7 and 8",
        "qualification": "Post-freeze extension designed after registered, X1-X7, DFDM and E10 results; descriptive.",
        "registered_F4_dinov2_reproduced_max_abs_difference": worst,
        "X7_REF_reproduced_max_abs_difference": clip_worst,
        "curves": {n: {c: {m: single(n, c, m)[0] for m in ("detection_ba", "attribution_ba", "detection_retention",
                                                             "attribution_retention", "attribution_minus_detection_retention")}
                       for c in conditions} for n in names},
        "auroc_points": {n: {c: auc(labels, scores[(n, c)][0]) for c in conditions} for n in names},
        "X8_REF_delta_q20": {**delta, "bootstrap_p": bootstrap_p(values, delta), "note": "outside the P1-PROBE Holm family"},
        "X8_REF_delta_auc": {c: {k: interval(auc_samples[c][:, i], auc_point[c][i]) for i, k in
                                 enumerate(("detection_auc", "attribution_auc", "delta_auc"))} for c in conditions},
        "adaptation": {f"{t}_minus_X8-REF_{c}": {"attribution_ba": paired(t, "X8-REF", c, "attribution_ba"),
                                                  "auroc_point_difference": auc(labels, scores[(t, c)][0]) - auc(labels, scores[("X8-REF", c)][0])}
                       for t, c in (("X8-Q80", "jpeg_q80"), ("X8-Q40", "jpeg_q40"))},
        "codec_delta": {c: single("X8-REF", c, "attribution_minus_detection_retention")[0] for c in CODEC},
        "X8_REF_minus_F4_dinov2": {c: {m: paired("X8-REF", "F4_dinov2", c, m) for m in ("detection_ba", "attribution_ba")}
                                   for c in ("ref", "jpeg_q20")},
        "X8_REF_minus_X7_REF": {c: {m: paired("X8-REF", "X7-REF", c, m) for m in ("detection_ba", "attribution_ba")}
                                for c in ("ref", "jpeg_q20")},
        "provenance": {"jepa_freeze_manifest_sha256": digest, "jepa_test_report_sha256": sha256(TEST_OUTPUT / "report.json"),
                       "script_sha256": sha256(Path(__file__))},
    }
    with ANALYSIS.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"delta_q20": (record["X8_REF_delta_q20"]["point"], record["X8_REF_delta_q20"]["ci95"]),
                      "delta_auc_q20": record["X8_REF_delta_auc"]["jpeg_q20"]["delta_auc"],
                      "reproduced": (worst, clip_worst)}, indent=2))
    print("Saved:", ANALYSIS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "fit", "verify", "freeze", "test", "analyze"):
        group.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit_all, "verify": verify, "freeze": freeze, "test": score_test, "analyze": analyze}[
        next(k for k, v in vars(args).items() if v)]()
