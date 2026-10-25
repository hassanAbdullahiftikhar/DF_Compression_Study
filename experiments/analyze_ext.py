"""Extension analysis (register Amendment 2): X1-X6 on test, run once after the extension freeze.

Reuses the registered analysis machinery (analyze_test.Measures, the same
2,000 provenance-group draws with seed 4242) so extension contrasts are paired
with the registered ones. Before reporting, it reproduces the registered
test intervals of F1 exactly. Every output is a post-freeze extension or an
exploratory re-analysis; registered readings are not changed.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_test import BOOT_SEED, DRAWS, K, METRICS, REENACT, SWAP, Measures, excludes_zero, interval, retention
from codec_sweep import CONDITIONS as CODEC, OUTPUT as CODEC_DIR
from metrics import aggregate, auc
from registered_models import REPS, SEEDS, TEST_CONDITIONS, all_models

OUTPUT = Path("experiments/results/ext_analysis_test.json")
REGISTERED = Path("experiments/results/test_analysis.json")
SCORE_DIRS = (Path("experiments/runs/c23_test_rgb"), Path("experiments/runs/c23_test_probes"),
              Path("experiments/runs/c23_ext_test_jpeg"), Path("experiments/runs/c23_ext_test_codec"))
X1B = Path("experiments/runs/c23_ext_x1b_quality_thresholds_validation/report.json")
JPEG = TEST_CONDITIONS
CONDITIONS = (*JPEG, *CODEC, "metadata")
CALIBRATED = ("F1", "F2", "F3", "C1-DET", "C2-BIN")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeightedAUC:
    """AUROC under integer video weights; sort order and tie groups fixed once."""

    def __init__(self, scores, labels):
        order = np.argsort(scores, kind="mergesort")
        self.order, self.positive = order, labels[order] == 1
        _, self.starts = np.unique(scores[order], return_index=True)

    def __call__(self, weights):
        w = weights[self.order]
        wp, wn = np.where(self.positive, w, 0.0), np.where(self.positive, 0.0, w)
        gp, gn = np.add.reduceat(wp, self.starts), np.add.reduceat(wn, self.starts)
        total = wp.sum() * wn.sum()
        if total == 0:
            return np.nan
        return float((gp * (np.cumsum(gn) - gn + 0.5 * gn)).sum() / total)


def selftest():
    rng = np.random.default_rng(0)
    scores, labels = rng.random(50).round(1), rng.integers(0, 2, 50)
    assert abs(WeightedAUC(scores, labels)(np.ones(50)) - auc(labels, scores)) < 1e-12
    weights = rng.integers(0, 3, 50).astype(float)
    repeated = np.repeat(np.arange(50), weights.astype(int))
    assert abs(WeightedAUC(scores, labels)(weights) - auc(labels[repeated], scores[repeated])) < 1e-12
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def load():
    from freeze_ext_models import require_extension_freeze
    freeze = require_extension_freeze(Path(__file__))
    paths, provenance = defaultdict(list), {}
    for folder in SCORE_DIRS:
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        provenance[folder.as_posix()] = sha256(folder / "report.json")
        if "extension_freeze_manifest_sha256" in report:
            assert report["extension_freeze_manifest_sha256"] == freeze
        for model_id, entry in report["models"].items():
            path = folder / model_id / "test_predictions.csv"
            assert sha256(path) == entry["test_predictions_sha256"], path
            paths[model_id].append(path)
    return paths, freeze, provenance


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths, freeze, provenance = load()
    manifest = json.loads(Path("research/extension_freeze_manifest.json").read_text(encoding="utf-8"))
    thresholds = {m["id"]: m["threshold"] for m in all_models()}
    tasks = {m["id"]: m["tasks"] for m in all_models()}
    for model_id, entry in manifest["models"].items():
        thresholds[model_id] = entry["threshold"]
        tasks[model_id] = ("detection", "attribution")
    x1b = json.loads(X1B.read_text(encoding="utf-8"))
    assert sha256(X1B) == manifest["x1b_report_sha256"]

    tables, canonical = {}, None
    for model_id, files in paths.items():
        rows = [r for path in files for r in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))]
        by_condition = defaultdict(dict)
        for video in aggregate(rows):
            by_condition[video["condition"]][video["video_id"]] = video
        first = next(iter(by_condition.values()))
        meta = [(v, first[v]["group_id"], first[v]["label"], first[v]["source_index"]) for v in sorted(first)]
        canonical = canonical or meta
        assert meta == canonical, model_id
        tables[model_id] = by_condition
    video_ids = [m[0] for m in canonical]
    groups_of = np.array([m[1] for m in canonical])
    labels = np.array([m[2] for m in canonical])
    sources = np.array([m[3] for m in canonical])
    fake, family = labels == 1, np.isin(sources, REENACT).astype(int)
    V, C = len(video_ids), {c: i for i, c in enumerate(CONDITIONS)}

    def ordered(model_id, condition):
        table = tables[model_id][condition]
        p_fake = np.array([table[v]["p_fake"] for v in video_ids])
        p = np.array([[table[v][f"p_source_{k}"] for k in range(4)] for v in video_ids])
        return p_fake, p

    ids = list(tables) + [f"C1-SEP_s{s}" for s in SEEDS] + [f"{f}-CAL_s{s}" for f in CALIBRATED for s in SEEDS]
    row_of = {name: i for i, name in enumerate(ids)}
    arrays = {t: np.full((len(ids), len(CONDITIONS), V), np.nan) for t in ("det", "src", "fam", "wswap", "wreen")}
    for model_id in tables:
        i = row_of[model_id]
        for condition in tables[model_id]:
            p_fake, p = ordered(model_id, condition)
            c = C[condition]
            if "detection" in tasks[model_id]:
                arrays["det"][i, c] = (p_fake >= thresholds[model_id]) == labels
            if "attribution" in tasks[model_id]:
                arrays["src"][i, c] = (p.argmax(axis=1) == sources) & fake
                guess = (p[:, list(REENACT)].sum(axis=1) > p[:, list(SWAP)].sum(axis=1)).astype(int)
                arrays["fam"][i, c] = (guess == family) & fake
                for task, (a, b) in (("wswap", SWAP), ("wreen", REENACT)):
                    arrays[task][i, c] = (np.where(p[:, b] > p[:, a], b, a) == sources) & np.isin(sources, (a, b))
    for s in SEEDS:
        arrays["det"][row_of[f"C1-SEP_s{s}"]] = arrays["det"][row_of[f"C1-DET_s{s}"]]
        for task in ("src", "fam", "wswap", "wreen"):
            arrays[task][row_of[f"C1-SEP_s{s}"]] = arrays[task][row_of[f"C1-ATT_s{s}"]]
        for family_name in CALIBRATED:
            base, cal = f"{family_name}_s{s}", f"{family_name}-CAL_s{s}"
            for task in arrays:
                arrays[task][row_of[cal]] = arrays[task][row_of[base]]
            for condition, threshold in x1b["models"][base]["thresholds"].items():
                p_fake, _ = ordered(base, condition)
                arrays["det"][row_of[cal], C[condition]] = (p_fake >= threshold) == labels

    measure = Measures(arrays, labels, sources, C["ref"])
    point = measure(np.ones(V))
    members = [np.flatnonzero(groups_of == g) for g in sorted(set(groups_of))]
    assert len(members) == 70

    # X1a threshold-free AUROC models.
    auc_models = [f"F1_s{s}" for s in SEEDS] + [f"F4_{r}" for r in REPS] + [f"C1-SEP_s{s}" for s in SEEDS] + \
                 [f"EB4-F1_s{s}" for s in SEEDS]
    auc_conditions = [c for c in (*JPEG, *CODEC)]
    calculators = {}
    for name in auc_models:
        det_source = name.replace("C1-SEP", "C1-DET")
        att_source = name.replace("C1-SEP", "C1-ATT")
        for condition in auc_conditions:
            if condition not in tables[det_source]:
                continue
            p_fake, _ = ordered(det_source, condition)
            _, p = ordered(att_source, condition)
            calculators[(name, condition)] = (
                WeightedAUC(p_fake, labels),
                [WeightedAUC(p[fake, k], (sources[fake] == k).astype(int)) for k in range(4)])

    def auc_values(weights):
        out = {}
        for (name, condition), (det, att) in calculators.items():
            out[(name, condition)] = (det(weights), float(np.mean([a(weights[fake]) for a in att])))
        return out

    def auc_delta(values):
        result = {}
        for (name, condition), (det, att) in values.items():
            ref_det, ref_att = values[(name, "ref")]
            s_det = retention(np.array(det), np.array(ref_det), 0.5)
            s_att = retention(np.array(att), np.array(ref_att), 0.5)
            result[(name, condition)] = (float(det), float(att), float(s_det), float(s_att), float(s_att - s_det))
        return result

    auc_point = auc_delta(auc_values(np.ones(V)))
    samples = np.full((DRAWS, len(ids), len(CONDITIONS), len(METRICS)), np.nan)
    auc_samples = {key: np.full((DRAWS, 5), np.nan) for key in calculators}
    rng = np.random.default_rng(BOOT_SEED)
    for draw in range(DRAWS):
        indices = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        weights = np.bincount(indices, minlength=V).astype(float)
        samples[draw] = measure(weights)
        for key, values in auc_delta(auc_values(weights)).items():
            auc_samples[key][draw] = values
        if (draw + 1) % 500 == 0:
            print(json.dumps({"bootstrap_draws_completed": draw + 1}), flush=True)

    def single(name, condition, metric):
        return interval(samples[:, row_of[name], C[condition], K[metric]], point[row_of[name], C[condition], K[metric]])

    def paired(left, right, condition, metric):
        a, b = (row_of[left], C[condition], K[metric]), (row_of[right], C[condition], K[metric])
        return interval(samples[:, a[0], a[1], a[2]] - samples[:, b[0], b[1], b[2]], point[a] - point[b])

    # Consistency: the registered F1 intervals must be reproduced exactly.
    registered = json.loads(REGISTERED.read_text(encoding="utf-8"))
    worst = 0.0
    for s in SEEDS:
        for condition in JPEG:
            for metric in ("detection_ba", "attribution_ba", "attribution_minus_detection_retention"):
                mine, theirs = single(f"F1_s{s}", condition, metric), registered["model_intervals"][f"F1_s{s}"][condition][metric]
                assert (mine["ci95"] is None) == (theirs["ci95"] is None)
                if mine["ci95"]:
                    worst = max(worst, abs(mine["point"] - theirs["point"]),
                                *[abs(x - y) for x, y in zip(mine["ci95"], theirs["ci95"])])
    assert worst < 1e-10, worst

    def auc_entry(name, condition):
        keys = ("detection_auc", "attribution_auc", "detection_auc_retention", "attribution_auc_retention",
                "delta_auc")
        return {k: interval(auc_samples[(name, condition)][:, i], auc_point[(name, condition)][i])
                for i, k in enumerate(keys)}

    x1a = {name: {c: auc_entry(name, c) for c in auc_conditions if (name, c) in calculators} for name in auc_models}
    x1b_results = {}
    for family_name in CALIBRATED:
        for s in SEEDS:
            cal = f"{family_name}-CAL_s{s}"
            x1b_results[cal] = {c: {"detection_ba": single(cal, c, "detection_ba"),
                                    "detection_retention": single(cal, c, "detection_retention"),
                                    **({"attribution_minus_detection_retention":
                                        single(cal, c, "attribution_minus_detection_retention")}
                                       if family_name in ("F1", "F2", "F3") else {})}
                                for c in ("jpeg_q95", "jpeg_q60", "jpeg_q20")}
    cal_entries = [x1b_results[f"F1-CAL_s{s}"]["jpeg_q20"]["attribution_minus_detection_retention"] for s in SEEDS]
    signs = {np.sign(e["point"]) for e in cal_entries if e["point"] is not None}
    cal_reading = ("direction not established" if not all(excludes_zero(e) for e in cal_entries) or len(signs) != 1
                   else "attribution retained more" if signs == {1.0} else "attribution retained less")
    x1c = {}
    for name in [f"{f}_s{s}" for f in ("F1", "F2", "F3", "EB4-F1", "EB4-F2") for s in SEEDS]:
        x1c[name] = {}
        for condition in (*JPEG, *CODEC):
            if condition in tables[name]:
                p_fake, _ = ordered(name, condition)
                x1c[name][condition] = {"mean_p_fake_real": float(p_fake[~fake].mean()),
                                        "mean_p_fake_fake": float(p_fake[fake].mean()),
                                        "fake_coverage_at_frozen_threshold": float((p_fake[fake] >= thresholds[name]).mean())}

    x2 = {}
    for s in SEEDS:
        readout, native = f"X2-F1_s{s}", f"F1_s{s}"
        x2[f"seed{s}"] = {
            "readout_minus_native_q20": {m: paired(readout, native, "jpeg_q20", m) for m in ("detection_ba", "attribution_ba")},
            "readout": {c: {m: single(readout, c, m) for m in ("detection_ba", "attribution_ba")} for c in JPEG},
            "auroc_point": {name: auc(labels, ordered(name, "jpeg_q20")[0]) for name in (readout, native)},
        }

    rng_seeds = np.random.default_rng(BOOT_SEED + 1)
    seed_draws = rng_seeds.integers(0, len(SEEDS), size=(DRAWS, len(SEEDS)))

    def pooled(values_by_seed, points_by_seed):
        values = np.stack(values_by_seed, axis=1)                 # draws x seeds
        drawn = np.take_along_axis(values, seed_draws, axis=1).mean(axis=1)
        return interval(drawn, float(np.mean(points_by_seed)))

    def seed_series(name_of, condition, metric, right_of=None):
        values, points = [], []
        for s in SEEDS:
            a = (row_of[name_of(s)], C[condition], K[metric])
            v, p = samples[:, a[0], a[1], a[2]], point[a]
            if right_of:
                b = (row_of[right_of(s)], C[condition], K[metric])
                v, p = v - samples[:, b[0], b[1], b[2]], p - point[b]
            values.append(v)
            points.append(p)
        return pooled(values, points)

    x3 = {"F1_delta_q20": seed_series(lambda s: f"F1_s{s}", "jpeg_q20", "attribution_minus_detection_retention"),
          **{f"{label}_{metric}_q20": seed_series(lambda s, l=left: f"{l}_s{s}", "jpeg_q20", metric,
                                                  lambda s, r=right: f"{r}_s{s}")
             for label, left, right in (("F3_minus_F2", "F3", "F2"), ("F2_minus_F1", "F2", "F1"),
                                        ("EB4F2_minus_EB4F1", "EB4-F2", "EB4-F1"))
             for metric in ("detection_ba", "attribution_ba")},
          "EB4F1_delta_q20": seed_series(lambda s: f"EB4-F1_s{s}", "jpeg_q20", "attribution_minus_detection_retention")}

    meta_p = ordered("C3b", "metadata")[1]
    meta_correct = meta_p.argmax(axis=1) == sources
    raw = np.where(fake, 1 / np.clip(meta_p[np.arange(V), np.clip(sources, 0, 3)], 1e-12, None), 0.0)
    raw = np.minimum(raw, np.quantile(raw[fake], 0.99))
    balance = np.zeros(V)
    for k in range(4):
        mask = sources == k
        balance[mask] = raw[mask] / raw[mask].mean()
    x4_models = [f"{f}_s{s}" for f in ("F1", "F2", "F3", "C1-ATT", "EB4-F1", "EB4-F2") for s in SEEDS] + \
                [f"F4_{r}" for r in REPS]

    def x4_stats(weights):
        out = {}
        for name in x4_models:
            for condition in ("ref", "jpeg_q20"):
                correct = arrays["src"][row_of[name], C[condition]]
                w_ok, w_bad = weights * fake * meta_correct, weights * fake * ~meta_correct
                split = correct @ w_ok / w_ok.sum() - correct @ w_bad / w_bad.sum()
                plain = np.mean([correct @ (weights * (sources == k)) / (weights * (sources == k)).sum() for k in range(4)])
                balanced = np.mean([correct @ (weights * balance * (sources == k)) / (weights * balance * (sources == k)).sum()
                                    for k in range(4)])
                out[(name, condition)] = (float(split), float(balanced - plain))
        return out

    x4_point = x4_stats(np.ones(V))
    x4_samples = defaultdict(list)
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(DRAWS):
        indices = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        for key, values in x4_stats(np.bincount(indices, minlength=V).astype(float)).items():
            x4_samples[key].append(values)
    x4 = {name: {c: {"accuracy_meta_correct_minus_wrong": interval(np.array(x4_samples[(name, c)])[:, 0], x4_point[(name, c)][0]),
                     "balanced_minus_unweighted_attribution_ba": interval(np.array(x4_samples[(name, c)])[:, 1], x4_point[(name, c)][1])}
                 for c in ("ref", "jpeg_q20")} for name in x4_models}

    codec_models = [m for m in tables if CODEC[0] in tables[m]]
    x5 = {name: {c: {metric: single(name, c, metric) for metric in METRICS
                     if np.isfinite(point[row_of[name], C[c], K[metric]]) or metric.endswith("ba")}
                 for c in CODEC} for name in codec_models}
    x5_contrasts = {f"seed{s}": {label: {c: {m: paired(f"{l}_s{s}", f"{r}_s{s}", c, m) for m in ("detection_ba", "attribution_ba")}
                                         for c in CODEC}
                                 for label, l, r in (("F2_minus_F1", "F2", "F1"), ("F3_minus_F2", "F3", "F2"),
                                                     ("EB4F2_minus_EB4F1", "EB4-F2", "EB4-F1"))}
                    for s in SEEDS}
    crops = list(csv.DictReader((CODEC_DIR / "crops.csv").open(encoding="utf-8-sig", newline="")))
    videos = list(csv.DictReader((CODEC_DIR / "videos.csv").open(encoding="utf-8-sig", newline="")))
    distortion = {c: {"mean_crop_psnr_db": float(np.mean([float(r["crop_psnr_db"]) for r in crops
                                                          if r["condition"] == c and r["crop_psnr_db"] not in ("", "None")])),
                      "mean_kbps": float(np.mean([float(v[f"{c}_kbps"]) for v in videos]))} for c in CODEC}
    p1_eb4 = {f"EB4-F1_s{s}": single(f"EB4-F1_s{s}", "jpeg_q20", "attribution_minus_detection_retention") for s in SEEDS}
    signs = {np.sign(e["point"]) for e in p1_eb4.values() if e["point"] is not None}
    eb4_reading = ("direction not established" if not all(excludes_zero(e) for e in p1_eb4.values()) or len(signs) != 1
                   else "attribution retained more" if signs == {1.0} else "attribution retained less")
    x6 = {"P1_rule_at_q20": {"per_seed": p1_eb4, "reading_descriptive": eb4_reading},
          "curves": {name: {c: {m: single(name, c, m) for m in ("detection_ba", "attribution_ba")} for c in JPEG}
                     for name in [f"EB4-{r}_s{s}" for r in ("F1", "F2") for s in SEEDS]},
          "F2_minus_F1_q20": {f"seed{s}": {m: paired(f"EB4-F2_s{s}", f"EB4-F1_s{s}", "jpeg_q20", m)
                                           for m in ("detection_ba", "attribution_ba")} for s in SEEDS}}

    p1_codec = {c: {f"F1_s{s}": single(f"F1_s{s}", c, "attribution_minus_detection_retention") for s in SEEDS} for c in CODEC}
    record = {
        "status": "post_freeze_extension_and_exploratory_test_analysis",
        "register": "research/comparison_register.md Amendments 2 and 3",
        "qualification": ("Post-freeze extensions and exploratory re-analyses, designed after the registered test "
                          "results were seen. They do not change the registered readings. Intervals condition on "
                          "selected models and thresholds; X3 additionally resamples seeds."),
        "bootstrap": {"draws": DRAWS, "seed": BOOT_SEED, "seed_resampling_seed": BOOT_SEED + 1,
                      "registered_F1_intervals_reproduced_max_abs_difference": worst},
        "X1a_threshold_free": x1a,
        "X1b_per_quality_calibration": {"models": x1b_results, "F1_q20_reading_descriptive": cal_reading},
        "X1c_score_shift": x1c,
        "X2_q20_readout": x2,
        "X3_seed_pooled": x3,
        "X4_metadata_cue": x4,
        "X5_codec": {"models": x5, "contrasts": x5_contrasts, "F1_delta": p1_codec, "distortion": distortion},
        "X6_efficientnet_b4": x6,
        "provenance": {"extension_freeze_manifest_sha256": freeze, "score_reports_sha256": provenance,
                       "registered_analysis_sha256": sha256(REGISTERED), "script_sha256": sha256(Path(__file__))},
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"X1b_F1_q20": cal_reading, "X6_P1_rule": eb4_reading, "X3": {k: (v["point"], v["ci95"]) for k, v in x3.items()}},
                     indent=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
