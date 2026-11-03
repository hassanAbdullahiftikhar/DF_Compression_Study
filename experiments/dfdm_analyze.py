"""DFDM registered analysis (research/dfdm_register.md sections 6-7), run once after D-G4 scoring.

Provenance-group bootstrap: 2,000 draws, seed 4242, ordered identity pairs as
groups, paired across models. Attribution chance 0.2 (five generators).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_ext import WeightedAUC
from analyze_test import bootstrap_p, excludes_zero, interval
from dfdm_lib import N_SRC, ROOT, aggregate, read_csv, sha256

OUTPUT = ROOT / "experiments/results/dfdm_analysis_test.json"
REPORTS = (ROOT / "experiments/runs/dfdm_rgb_test", ROOT / "experiments/runs/dfdm_probes_test")
CONDITIONS = ("ref", "jpeg_q95", "jpeg_q80", "jpeg_q60", "jpeg_q40", "jpeg_q20", "h264_crf10", "h264_crf23")
DRAWS, SEED, SEEDS = 2000, 4242, (42, 43, 44)
B_DET, B_ATT = 0.5, 1 / N_SRC


def retention(x, ref, chance):
    margin = ref - chance
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(margin >= 0.05, (x - chance) / margin, np.nan)


def selftest():
    assert np.isnan(retention(np.array([0.8]), np.array([0.52]), 0.5)).all()
    assert np.isclose(retention(np.array([0.6]), np.array([1.0]), 0.2)[0], 0.5)
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    from dfdm_freeze import MANIFEST, require_dfdm_freeze
    digest = require_dfdm_freeze(Path(__file__))
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {}
    for folder in REPORTS:
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert report["dfdm_freeze_manifest_sha256"] == digest
        for model_id, entry in report.get("models", report.get("probes", {})).items():
            path = folder / model_id / "test_predictions.csv"
            assert sha256(path) == entry["test_predictions_sha256"], path
            paths[model_id] = path
    assert set(paths) == set(manifest["models"])
    tables, canonical = {}, None
    for model_id, path in paths.items():
        by = defaultdict(dict)
        for v in aggregate(read_csv(path)):
            by[v["condition"]][v["video_id"]] = v
        first = by["ref"]
        meta = [(v, first[v]["group_id"], first[v]["label"], first[v]["source_index"]) for v in sorted(first)]
        canonical = canonical or meta
        assert meta == canonical, model_id
        tables[model_id] = by
    vids = [m[0] for m in canonical]
    groups = np.array([m[1] for m in canonical])
    labels = np.array([m[2] for m in canonical])
    sources = np.array([m[3] for m in canonical])
    fake = labels == 1
    names = list(tables)
    available = {n: [c for c in CONDITIONS if c in tables[n]] for n in names}
    det = np.full((len(names), len(CONDITIONS), len(vids)), np.nan)
    src = np.full_like(det, np.nan)
    calculators = {}
    for i, n in enumerate(names):
        threshold = manifest["models"][n]["threshold"]
        for c in available[n]:
            ci = CONDITIONS.index(c)
            p_fake = np.array([tables[n][c][v]["p_fake"] for v in vids])
            p = np.array([[tables[n][c][v][f"p_source_{k}"] for k in range(N_SRC)] for v in vids])
            det[i, ci] = (p_fake >= threshold) == labels
            src[i, ci] = (p.argmax(axis=1) == sources) & fake
            calculators[(n, c)] = (WeightedAUC(p_fake, labels),
                                   [WeightedAUC(p[fake, k], (sources[fake] == k).astype(int)) for k in range(N_SRC)])
    det_masks = [labels == 0, labels == 1]
    src_masks = [sources == k for k in range(N_SRC)]
    ref = CONDITIONS.index("ref")

    def measure(w):
        def ba(array, masks):
            flat = array.reshape(-1, array.shape[-1])
            parts = []
            for m in masks:
                total = (w * m).sum()
                parts.append(flat @ (w * m) / total if total > 0 else np.full(len(flat), np.nan))
            return np.mean(parts, axis=0).reshape(array.shape[:2])
        d, a = ba(det, det_masks), ba(src, src_masks)
        sd, sa = retention(d, d[:, [ref]], B_DET), retention(a, a[:, [ref]], B_ATT)
        aucs = np.full((len(names), len(CONDITIONS), 2), np.nan)
        for (n, c), (dc, ac) in calculators.items():
            aucs[names.index(n), CONDITIONS.index(c)] = (dc(w), np.mean([x(w[fake]) for x in ac]))
        s_auc_d = retention(aucs[..., 0], aucs[:, [ref], 0], 0.5)
        s_auc_a = retention(aucs[..., 1], aucs[:, [ref], 1], 0.5)
        return np.stack([d, a, sd, sa, sa - sd, aucs[..., 0], aucs[..., 1], s_auc_a - s_auc_d], axis=-1)

    K = {m: i for i, m in enumerate(("detection_ba", "attribution_ba", "detection_retention", "attribution_retention",
                                     "delta", "detection_auc", "attribution_auc", "delta_auc"))}
    point = measure(np.ones(len(vids)))
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    samples = np.full((DRAWS, *point.shape), np.nan)
    rng = np.random.default_rng(SEED)
    for draw in range(DRAWS):
        idx = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        samples[draw] = measure(np.bincount(idx, minlength=len(vids)).astype(float))
    row = {n: i for i, n in enumerate(names)}

    def single(n, c, m):
        a = (row[n], CONDITIONS.index(c), K[m])
        return interval(samples[(slice(None), *a)], point[a]), samples[(slice(None), *a)]

    def paired(left, right, c, m):
        a, b = (row[left], CONDITIONS.index(c), K[m]), (row[right], CONDITIONS.index(c), K[m])
        return interval(samples[(slice(None), *a)] - samples[(slice(None), *b)], point[a] - point[b])

    def rule(entries):
        signs = {np.sign(e["point"]) for e in entries if e["point"] is not None}
        if not all(excludes_zero(e) for e in entries) or len(signs) != 1:
            return "direction not established"
        return "attribution retained more than detection" if signs == {1.0} else "attribution retained less than detection"

    rgb = [n for n in names if manifest["models"][n]["kind"] == "rgb"]
    p1 = {fam: {f"{fam}_s{s}": single(f"{fam}_s{s}", "jpeg_q20", "delta")[0] for s in SEEDS} for fam in ("D-F1", "D-EB4-F1")}
    curves = {n: {c: {m: single(n, c, m)[0] for m in K} for c in available[n]} for n in names}
    seed_means = {fam: {c: {m: (float(np.mean([point[row[f"{fam}_s{s}"], CONDITIONS.index(c), K[m]] for s in SEEDS])),
                                float(np.std([point[row[f"{fam}_s{s}"], CONDITIONS.index(c), K[m]] for s in SEEDS], ddof=1)))
                            for m in ("detection_ba", "attribution_ba", "detection_auc", "attribution_auc")}
                        for c in CONDITIONS} for fam in ("D-F1", "D-F2", "D-EB4-F1", "D-EB4-F2")}
    s3 = {f"{label}_seed{s}": {c: {m: paired(f"{l}_s{s}", f"{r}_s{s}", c, m) for m in ("detection_ba", "attribution_ba")}
                               for c in ("ref", "jpeg_q95", "jpeg_q60", "jpeg_q20", "h264_crf23")}
          for label, l, r in (("F2_minus_F1", "D-F2", "D-F1"), ("EB4F2_minus_EB4F1", "D-EB4-F2", "D-EB4-F1")) for s in SEEDS}
    probes = {}
    for p in ("P-DINO-REF", "P-CLIP-REF"):
        delta, values = single(p, "jpeg_q20", "delta")
        probes[p] = {"delta_q20": {**delta, "bootstrap_p": bootstrap_p(values, delta)},
                     "delta_auc_q20": single(p, "jpeg_q20", "delta_auc")[0]}
    for q40, refp in (("P-DINO-Q40", "P-DINO-REF"), ("P-CLIP-Q40", "P-CLIP-REF")):
        probes[f"{q40}_minus_{refp}_at_q40"] = paired(q40, refp, "jpeg_q40", "attribution_ba")
    confusion = {}
    for s in SEEDS:
        n = f"D-F1_s{s}"
        for c in ("ref", "jpeg_q20"):
            p = np.array([[tables[n][c][v][f"p_source_{k}"] for k in range(N_SRC)] for v in vids])[fake]
            m = np.zeros((N_SRC, N_SRC), int)
            np.add.at(m, (sources[fake], p.argmax(axis=1)), 1)
            confusion[f"{n}_{c}"] = m.tolist()
    c1 = single("C1-DCT", "ref", "detection_ba")[0]
    record = {
        "status": "dfdm_registered_test_analysis", "register": "research/dfdm_register.md v1.0 + amendments",
        "qualification": ("Single test pass after the D-G3 freeze. Intervals condition on validation-selected models and "
                          "thresholds; 19 ordered identity-pair groups make them coarse."),
        "support": {"videos": len(vids), "real": int((~fake).sum()), "fake": int(fake.sum()), "groups": len(members)},
        "D_P1": {fam: {"per_seed": e, "reading": rule(list(e.values()))} for fam, e in p1.items()},
        "D_S1_delta_auc_q20": {n: single(n, "jpeg_q20", "delta_auc")[0] for n in rgb if "-F1_" in n},
        "D_S2_codec_delta": {n: {c: single(n, c, "delta")[0] for c in ("h264_crf10", "h264_crf23")} for n in rgb if "-F1_" in n},
        "D_S3_augmentation": s3,
        "D_S4_probes": probes,
        "D_S5_confusion": {"generators": ["faceswap", "lightweight", "iae", "dfaker", "dfl_h128"], "matrices": confusion},
        "D_C1_compression_history": {"detection_ba_ref": c1, "material": c1["ci95"] is not None and c1["ci95"][0] > 0.60,
                                     "rule": "material if the 95% lower bound of detection BA exceeds 0.60"},
        "D_C2_face_masked": {m: single("C2-MASK", "ref", m)[0] for m in ("detection_ba", "attribution_ba")},
        "curves": curves, "seed_means": seed_means,
        "provenance": {"dfdm_freeze_manifest_sha256": digest, "script_sha256": sha256(Path(__file__)),
                       "predictions_sha256": {k: sha256(v) for k, v in paths.items()}},
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"D_P1": {f: v["reading"] for f, v in record["D_P1"].items()},
                      "D_C1_material": record["D_C1_compression_history"]["material"]}, indent=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
