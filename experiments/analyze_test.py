"""Registered test analysis (gate G4): research/comparison_register.md sections 6-8.

--split test : the single registered analysis of the test predictions.
               Requires the G3 model-freeze manifest and both test score reports.
--split val  : dry run of the identical code on the stored validation
               predictions. Q40 stands in for Q20 and Q80 for Q60/Q95 because
               validation has no held-out qualities. It is a code check: it
               must reproduce the recorded validation intervals, and its numbers
               are not results.

All models share each bootstrap draw (2,000 provenance-group draws, seed
4242), so model differences are paired. Subset supports reuse the drawn
groups with their multiplicity, as in eight_frame_sensitivity.py.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import SOURCES
from eight_frame_sensitivity import complete_support
from metrics import aggregate, evaluate
from registered_models import REPS, SEEDS, TEST_CONDITIONS, VAL_CONDITIONS, all_models
from research_metrics import summarize

MANIFEST = Path("experiments/data/c23_main/reference.csv")
DRAWS, BOOT_SEED, ALPHA = 2000, 4242, 0.05
SUPPORTS = ("full", "complete_video", "complete_group")
METRICS = ("detection_ba", "attribution_ba", "detection_retention", "attribution_retention",
           "attribution_minus_detection_retention", "family_ba", "within_swap_ba", "within_reenactment_ba",
           "family_retention", "within_swap_retention", "within_reenactment_retention")
K = {name: i for i, name in enumerate(METRICS)}
SWAP, REENACT = (0, 2), (1, 3)  # Deepfakes, Face2Face, FaceSwap, NeuralTextures
MATERIAL = {"attribution_ba": 0.30, "detection_ba": 0.55}
PLAN = {
    "test": {"conditions": (*TEST_CONDITIONS, "metadata"), "primary": "jpeg_q20", "s1": ("jpeg_q60", "jpeg_q95"),
             "s2": ("jpeg_q20", "jpeg_q60", "jpeg_q95", "ref"), "c3c_retention": ("jpeg_q20", "jpeg_q60"),
             "output": Path("experiments/results/test_analysis.json")},
    "val": {"conditions": (*VAL_CONDITIONS, "metadata"), "primary": "jpeg_q40", "s1": ("jpeg_q80",),
            "s2": ("jpeg_q40", "jpeg_q80", "ref"), "c3c_retention": ("jpeg_q40", "jpeg_q80"),
            "output": Path("experiments/results/test_analysis_dryrun_validation.json")},
}
TEST_SCORES = (Path("experiments/runs/c23_test_rgb"), Path("experiments/runs/c23_test_probes"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retention(values, reference, chance):
    """Vectorized research_metrics.retention: NaN when the reference is < chance + 0.05."""
    margin = reference - chance
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(margin >= 0.05, (values - chance) / margin, np.nan)


class Measures:
    """Balanced accuracies from per-video correctness arrays under video weights."""

    def __init__(self, arrays, labels, sources, ref_index):
        self.arrays, self.ref = arrays, ref_index
        fake = labels == 1
        family = np.isin(sources, REENACT).astype(int)
        self.classes = {
            "det": [labels == 0, labels == 1],
            "src": [sources == k for k in range(4)],
            "fam": [fake & (family == 0), fake & (family == 1)],
            "wswap": [sources == SWAP[0], sources == SWAP[1]],
            "wreen": [sources == REENACT[0], sources == REENACT[1]],
        }

    def __call__(self, weights):
        out = {}
        for task, masks in self.classes.items():
            array = self.arrays[task]
            flat = array.reshape(-1, array.shape[-1])
            means = []
            for mask in masks:
                w = weights * mask
                total = w.sum()
                means.append(flat @ w / total if total > 0 else np.full(len(flat), np.nan))
            out[task] = np.mean(means, axis=0).reshape(array.shape[:2])
        result = np.full((*out["det"].shape, len(METRICS)), np.nan)
        result[..., K["detection_ba"]] = out["det"]
        result[..., K["attribution_ba"]] = out["src"]
        result[..., K["family_ba"]] = out["fam"]
        result[..., K["within_swap_ba"]] = out["wswap"]
        result[..., K["within_reenactment_ba"]] = out["wreen"]
        for metric, source, chance in (("detection_retention", "det", 0.5), ("attribution_retention", "src", 0.25),
                                       ("family_retention", "fam", 0.5), ("within_swap_retention", "wswap", 0.5),
                                       ("within_reenactment_retention", "wreen", 0.5)):
            result[..., K[metric]] = retention(out[source], out[source][:, [self.ref]], chance)
        result[..., K["attribution_minus_detection_retention"]] = (
            result[..., K["attribution_retention"]] - result[..., K["detection_retention"]])
        return result


def interval(values, point):
    valid = values[np.isfinite(values)]
    defined = point is not None and np.isfinite(point)
    usable = defined and len(valid) > 0 and len(valid) >= int(np.ceil(0.95 * len(values)))
    return {"point": float(point) if defined else None,
            "ci95": np.quantile(valid, [0.025, 0.975], method="linear").tolist() if usable else None,
            "valid_draws": int(len(valid)), "undefined_draws": int(len(values) - len(valid)),
            "interval_available": bool(usable)}


def excludes_zero(entry):
    return entry["ci95"] is not None and (entry["ci95"][0] > 0 or entry["ci95"][1] < 0)


def bootstrap_p(values, entry):
    """Two-sided p = min(1, 2 min(P(d<=0), P(d>=0))); p = 1 when undefined or unavailable."""
    if not entry["interval_available"]:
        return 1.0
    valid = values[np.isfinite(values)]
    return float(min(1.0, 2 * min(np.mean(valid <= 0), np.mean(valid >= 0))))


def holm(p_values, alpha=ALPHA):
    order = sorted(p_values, key=lambda name: (p_values[name], name))
    rejected, active = {}, True
    for rank, name in enumerate(order):
        bound = alpha / (len(order) - rank)
        active = active and p_values[name] <= bound
        rejected[name] = {"p": p_values[name], "holm_bound": bound, "rejected": active}
    return rejected


def selftest():
    labels = np.array([0, 1, 1, 1, 1] * 2)
    sources = np.array([-1, 0, 1, 2, 3] * 2)
    ones = np.ones((1, 2, 10))
    arrays = {t: ones.copy() for t in ("det", "src", "fam", "wswap", "wreen")}
    arrays["det"][0, 1] = labels == 0          # condition 1: every fake missed
    arrays["src"][0, 1] = sources == 0         # condition 1: only Deepfakes right
    result = Measures(arrays, labels, sources, 0)(np.ones(10))
    assert result[0, 1, K["detection_ba"]] == 0.5 and result[0, 1, K["attribution_ba"]] == 0.25
    assert result[0, 1, K["detection_retention"]] == 0 and result[0, 1, K["attribution_retention"]] == 0
    weighted = Measures(arrays, labels, sources, 0)(np.r_[np.full(5, 2.0), np.zeros(5)])
    assert np.allclose(weighted[..., :2], result[..., :2])
    assert np.isnan(retention(np.array([0.8]), np.array([0.54]), 0.5)).all()
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03})["a"]["rejected"]
    assert not holm({"a": 0.01, "b": 0.04, "c": 0.03})["c"]["rejected"]
    entry = interval(np.r_[np.full(1900, 0.1), np.full(100, np.nan)], 0.1)
    assert entry["interval_available"] and bootstrap_p(np.full(2000, 0.1), entry) == 0
    assert not interval(np.r_[np.full(1800, 0.1), np.full(200, np.nan)], 0.1)["interval_available"]
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def load_predictions(models, split):
    paths, reports = {}, {}
    if split == "val":
        for m in models:
            assert sha256(m["validation_predictions"]) == m["validation_predictions_sha256"], m["id"]
            paths[m["id"]] = m["validation_predictions"]
        return paths, {}
    from freeze_models import require_model_freeze
    freeze = require_model_freeze(Path(__file__))
    for folder in TEST_SCORES:
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "registered_test_scores" and report["model_freeze_manifest_sha256"] == freeze
        reports[folder.as_posix()] = sha256(folder / "report.json")
        for model_id, entry in report["models"].items():
            path = folder / model_id / "test_predictions.csv"
            assert sha256(path) == entry["test_predictions_sha256"], path
            paths[model_id] = path
    assert set(paths) == {m["id"] for m in models}
    return paths, {"model_freeze_manifest_sha256": freeze, "score_report_sha256": reports}


def main(split):
    plan = PLAN[split]
    output = plan["output"]
    if output.exists():
        raise FileExistsError(output)
    conditions = list(plan["conditions"])
    C = {c: i for i, c in enumerate(conditions)}
    models = all_models()
    paths, provenance = load_predictions(models, split)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    complete_videos, complete_groups = complete_support(manifest_rows, split)

    canonical, videos_by_model = None, {}
    for m in models:
        with Path(paths[m["id"]]).open(encoding="utf-8-sig", newline="") as handle:
            videos = aggregate(list(csv.DictReader(handle)))
        videos_by_model[m["id"]] = videos
        by_condition = defaultdict(dict)
        for v in videos:
            by_condition[v["condition"]][v["video_id"]] = v
        first = by_condition[next(iter(by_condition))]
        meta = [(vid, first[vid]["group_id"], first[vid]["label"], first[vid]["source_index"]) for vid in sorted(first)]
        if canonical is None:
            canonical = meta
        assert meta == canonical, f"Video support differs: {m['id']}"
        assert all(set(d) == set(first) for d in by_condition.values())
        m["by_condition"] = by_condition
    video_ids = [row[0] for row in canonical]
    groups_of = np.array([row[1] for row in canonical])
    labels = np.array([row[2] for row in canonical])
    sources = np.array([row[3] for row in canonical])
    V = len(canonical)
    assert V == (700 if split == "test" else 699)

    # Model rows, plus the C1 "separate system" (detection from C1-DET, attribution from C1-ATT).
    ids = [m["id"] for m in models] + [f"C1-SEP_s{s}" for s in SEEDS]
    row_of = {name: i for i, name in enumerate(ids)}
    arrays = {t: np.full((len(ids), len(conditions), V), np.nan) for t in ("det", "src", "fam", "wswap", "wreen")}
    fake, family = labels == 1, np.isin(sources, REENACT).astype(int)
    for m in models:
        i = row_of[m["id"]]
        for condition, table in m["by_condition"].items():
            ordered = [table[v] for v in video_ids]
            p_fake = np.array([r["p_fake"] for r in ordered])
            p = np.array([[r[f"p_source_{k}"] for k in range(4)] for r in ordered])
            c = C[condition]
            if "detection" in m["tasks"]:
                arrays["det"][i, c] = (p_fake >= m["threshold"]) == labels
            if "attribution" in m["tasks"]:
                arrays["src"][i, c] = (p.argmax(axis=1) == sources) & fake
                guess_family = (p[:, list(REENACT)].sum(axis=1) > p[:, list(SWAP)].sum(axis=1)).astype(int)
                arrays["fam"][i, c] = (guess_family == family) & fake
                for task, (first, second) in (("wswap", SWAP), ("wreen", REENACT)):
                    guess = np.where(p[:, second] > p[:, first], second, first)
                    arrays[task][i, c] = (guess == sources) & np.isin(sources, (first, second))
    for s in SEEDS:
        arrays["det"][row_of[f"C1-SEP_s{s}"]] = arrays["det"][row_of[f"C1-DET_s{s}"]]
        for task in ("src", "fam", "wswap", "wreen"):
            arrays[task][row_of[f"C1-SEP_s{s}"]] = arrays[task][row_of[f"C1-ATT_s{s}"]]

    measure = Measures(arrays, labels, sources, C["ref"])
    masks = {"full": np.ones(V),
             "complete_video": np.array([v in complete_videos for v in video_ids], dtype=float),
             "complete_group": np.array([g in complete_groups for g in groups_of], dtype=float)}
    points = {s: measure(masks[s]) for s in SUPPORTS}
    group_ids = sorted(set(groups_of))
    members = [np.flatnonzero(groups_of == g) for g in group_ids]
    assert len(members) == 70
    samples = np.full((DRAWS, len(SUPPORTS), len(ids), len(conditions), len(METRICS)), np.nan)
    rng = np.random.default_rng(BOOT_SEED)
    for draw in range(DRAWS):
        indices = np.concatenate([members[int(i)] for i in rng.integers(0, len(members), size=len(members))])
        weights = np.bincount(indices, minlength=V).astype(float)
        for s, support in enumerate(SUPPORTS):
            samples[draw, s] = measure(weights * masks[support])
        if (draw + 1) % 500 == 0:
            print(json.dumps({"bootstrap_draws_completed": draw + 1}), flush=True)

    def single(name, condition, metric, support="full"):
        s = SUPPORTS.index(support)
        values = samples[:, s, row_of[name], C[condition], K[metric]]
        return interval(values, points[support][row_of[name], C[condition], K[metric]]), values

    def paired(left, right, condition, metric, support="full", left_metric=None, right_metric=None):
        s = SUPPORTS.index(support)
        a = (row_of[left], C[condition], K[left_metric or metric])
        b = (row_of[right], C[condition], K[right_metric or metric])
        values = samples[:, s, a[0], a[1], a[2]] - samples[:, s, b[0], b[1], b[2]]
        return interval(values, points[support][a] - points[support][b]), values

    available = {m["id"]: set(m["by_condition"]) for m in models}
    for s in SEEDS:
        available[f"C1-SEP_s{s}"] = available[f"C1-DET_s{s}"]
    task_metrics = {m["id"]: m["tasks"] for m in models}
    for s in SEEDS:
        task_metrics[f"C1-SEP_s{s}"] = ("detection", "attribution")

    def metric_names(name):
        tasks = task_metrics[name]
        names = []
        if "detection" in tasks:
            names += ["detection_ba", "detection_retention"]
        if "attribution" in tasks:
            names += ["attribution_ba", "attribution_retention", "family_ba", "within_swap_ba",
                      "within_reenactment_ba", "family_retention", "within_swap_retention",
                      "within_reenactment_retention"]
        if len(tasks) == 2:
            names.append("attribution_minus_detection_retention")
        return names

    model_intervals = {name: {c: {metric: single(name, c, metric)[0] for metric in metric_names(name)}
                              for c in conditions if c in available[name]} for name in ids}

    # Section 6: primary contrast.
    primary = plan["primary"]
    p1_rgb = {f"F1_s{s}": single(f"F1_s{s}", primary, "attribution_minus_detection_retention")[0] for s in SEEDS}
    signs = {np.sign(e["point"]) for e in p1_rgb.values() if e["point"] is not None}
    rgb_direction = ("direction not established" if not all(excludes_zero(e) for e in p1_rgb.values()) or len(signs) != 1
                     else "attribution retained more than detection" if signs == {1.0}
                     else "attribution retained less than detection")
    p1_probe, p_values = {}, {}
    for rep in REPS:
        entry, values = single(f"F4_{rep}", primary, "attribution_minus_detection_retention")
        p_values[rep] = bootstrap_p(values, entry)
        p1_probe[rep] = {**entry, "bootstrap_p": p_values[rep]}
    holm_result = holm(p_values)
    for rep in REPS:
        p1_probe[rep]["holm"] = holm_result[rep]
        p1_probe[rep]["holm_supported_sign"] = (
            ("positive" if p1_probe[rep]["point"] > 0 else "negative") if holm_result[rep]["rejected"] else None)

    # Section 7: secondary contrasts.
    s1 = {c: {**{f"F1_s{s}": single(f"F1_s{s}", c, "attribution_minus_detection_retention")[0] for s in SEEDS},
              **{f"F4_{rep}": single(f"F4_{rep}", c, "attribution_minus_detection_retention")[0] for rep in REPS}}
          for c in plan["s1"]}
    s2, s2_decision = {}, {}
    for s in SEEDS:
        s2[f"seed{s}"] = {
            label: {c: {metric: paired(left, right, c, metric)[0] for metric in ("detection_ba", "attribution_ba")}
                    for c in conditions if c != "metadata" and c in available[left]}
            for label, left, right in (("F3_minus_F2", f"F3_s{s}", f"F2_s{s}"), ("F2_minus_F1", f"F2_s{s}", f"F1_s{s}"))}
        gain = s2[f"seed{s}"]["F3_minus_F2"][primary]["attribution_ba"]
        reference = s2[f"seed{s}"]["F3_minus_F2"]["ref"]["attribution_ba"]
        s2_decision[f"seed{s}"] = {"primary_attribution_gain_excludes_zero_positive":
                                   gain["ci95"] is not None and gain["ci95"][0] > 0,
                                   "reference_attribution_lower_bound_above_minus_0_02":
                                   reference["ci95"] is not None and reference["ci95"][0] > -0.02}
    benefit = all(all(v.values()) for v in s2_decision.values())
    s3 = {}
    for adapted, condition in (("F5", "jpeg_q80"), ("F6", "jpeg_q40")):
        for rep in REPS:
            left, right = f"{adapted}_{rep}", f"F4_{rep}"
            auroc = {name: evaluate([v for v in videos_by_model[name] if v["condition"] == condition],
                                    threshold=0.5)[condition]["detection_auc"] for name in (left, right)}
            s3[f"{adapted}_minus_F4_{rep}_{condition}"] = {
                "auroc_point_difference": auroc[left] - auroc[right], "auroc": auroc,
                "attribution_ba": paired(left, right, condition, "attribution_ba")[0]}
    s4, s5 = {}, {}
    for s in SEEDS:
        cs = [c for c in conditions if c != "metadata" and c in available[f"F1_s{s}"]]
        s4[f"seed{s}"] = {
            "C1-DET_minus_F1_detection_ba": {c: paired(f"C1-DET_s{s}", f"F1_s{s}", c, "detection_ba")[0] for c in cs},
            "C1-ATT_minus_F1_attribution_ba": {c: paired(f"C1-ATT_s{s}", f"F1_s{s}", c, "attribution_ba")[0] for c in cs},
            "separate_minus_shared_delta": paired(f"C1-SEP_s{s}", f"F1_s{s}", primary,
                                                  "attribution_minus_detection_retention")[0],
            "separate_system_delta": single(f"C1-SEP_s{s}", primary, "attribution_minus_detection_retention")[0],
        }
        s5[f"seed{s}"] = {
            "C2-R_minus_F3-R_attribution_ba": {c: paired(f"C2-R_s{s}", f"F3-R_s{s}", c, "attribution_ba")[0] for c in cs},
            "C2-BIN_minus_F3_detection_ba": {c: paired(f"C2-BIN_s{s}", f"F3_s{s}", c, "detection_ba")[0] for c in cs},
            "F3_native_attribution_ba": {c: single(f"F3_s{s}", c, "attribution_ba")[0] for c in cs},
        }
    border = {}
    for rep in REPS:
        border[rep] = {c: {metric: single(f"C3a_{rep}", c, metric)[0] for metric in ("detection_ba", "attribution_ba")}
                       for c in conditions if c in available[f"C3a_{rep}"]}
        border[rep]["material_flag_ref"] = {
            metric: border[rep]["ref"][metric]["ci95"] is not None
            and border[rep]["ref"][metric]["ci95"][0] > bound for metric, bound in MATERIAL.items()}
    metadata = {metric: single("C3b", "metadata", metric)[0] for metric in ("detection_ba", "attribution_ba")}
    metadata["material_flag"] = {metric: metadata[metric]["ci95"] is not None
                                 and metadata[metric]["ci95"][0] > bound for metric, bound in MATERIAL.items()}
    c3c_metrics = ("family_ba", "within_swap_ba", "within_reenactment_ba")
    c3c = {name: {c: {metric: model_intervals[name][c][metric] for metric in c3c_metrics}
                  for c in model_intervals[name] if c != "metadata"}
           for name in ids if "attribution" in task_metrics[name] and not name.startswith("C1-SEP")
           and name != "C3b"}
    for name in c3c:
        if "ref" in available[name]:
            c3c[name]["retention"] = {c: {metric: model_intervals[name][c][metric] for metric in (
                "family_retention", "within_swap_retention", "within_reenactment_retention")}
                for c in plan["c3c_retention"] if c in available[name]}

    # S7: complete-eight supports for section 6 and S2.
    s7, changes = {}, []
    checks = [(f"P1_F1_s{s}", lambda sup, s=s: single(f"F1_s{s}", primary, "attribution_minus_detection_retention", sup)[0])
              for s in SEEDS]
    checks += [(f"P1_F4_{rep}", lambda sup, rep=rep: single(f"F4_{rep}", primary,
                                                            "attribution_minus_detection_retention", sup)[0])
               for rep in REPS]
    for s in SEEDS:
        for label, left, right in (("F3_minus_F2", f"F3_s{s}", f"F2_s{s}"), ("F2_minus_F1", f"F2_s{s}", f"F1_s{s}")):
            for c in plan["s2"]:
                for metric in ("detection_ba", "attribution_ba"):
                    checks.append((f"S2_seed{s}_{label}_{c}_{metric}",
                                   lambda sup, l=left, r=right, c=c, m=metric: paired(l, r, c, m, sup)[0]))
    for key, function in checks:
        s7[key] = {sup: function(sup) for sup in SUPPORTS}
        full = s7[key]["full"]
        for sup in SUPPORTS[1:]:
            sub = s7[key][sup]
            flipped = (full["point"] is not None and sub["point"] is not None
                       and np.sign(full["point"]) != np.sign(sub["point"]))
            if excludes_zero(full) != excludes_zero(sub) or flipped:
                changes.append({"contrast": key, "support": sup, "full": full, "subset": sub})

    # Exploratory: full point metrics and seed summaries.
    exploratory = {}
    for m in models:
        videos = videos_by_model[m["id"]]
        threshold = m["threshold"] if m["threshold"] is not None else 0.5
        exploratory[m["id"]] = (summarize(videos, threshold) if "ref" in available[m["id"]]
                                else evaluate(videos, threshold))
    seed_summary = {}
    for family in ("F1", "F2", "F3", "C1-DET", "C1-ATT", "C2-BIN", "F3-R", "C2-R", "C1-SEP"):
        names = [f"{family}_s{s}" for s in SEEDS]
        seed_summary[family] = {}
        for c in conditions:
            if c not in available[names[0]]:
                continue
            seed_summary[family][c] = {}
            for metric in metric_names(names[0]):
                values = [points["full"][row_of[n], C[c], K[metric]] for n in names]
                ok = all(np.isfinite(values))
                seed_summary[family][c][metric] = {
                    "values_by_seed": [float(v) if np.isfinite(v) else None for v in values],
                    "mean": float(np.mean(values)) if ok else None,
                    "sample_sd": float(np.std(values, ddof=1)) if ok else None}

    distortion = None
    if split == "test":
        with (TEST_SCORES[1] / "distortion.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        distortion = {}
        for c in TEST_CONDITIONS[1:]:
            subset = [r for r in rows if r["condition"] == c]
            psnr = [float(r["rgb_psnr_db"]) for r in subset if r["rgb_psnr_db"] not in ("", "None")]
            distortion[c] = {"crops": len(subset), "mean_jpeg_bytes": float(np.mean([float(r["jpeg_bytes"]) for r in subset])),
                             "mean_rgb_psnr_db": float(np.mean(psnr)), "lossless_crops": len(subset) - len(psnr),
                             "mean_luma_ssim": float(np.mean([float(r["luma_ssim"]) for r in subset]))}

    cross_check = cross_check_validation(model_intervals, points, row_of, C, s2, ids) if split == "val" else None

    record = {
        "status": "registered_test_analysis" if split == "test" else "analysis_code_dry_run_on_validation",
        "register": "research/comparison_register.md v1.0 sections 6-8",
        "split": split,
        "qualification": (
            "Held-out test analysis of models, thresholds and contrasts frozen before test scoring. Intervals "
            "condition on the selected models and thresholds and omit selection and training-seed uncertainty. "
            "Additional JPEG compression of C23-derived crops; attribution is closed-set among four FF++ methods. "
            "A lower score does not show that information was physically or irreversibly erased."
            if split == "test" else
            "Code check only. Validation has no held-out qualities: Q40 stands in for Q20 and Q80 for Q60/Q95. "
            "These numbers are validation-selected and are not test results."),
        "condition_plan": {k: v for k, v in plan.items() if k != "output"},
        "bootstrap": {"draws": DRAWS, "seed": BOOT_SEED, "unit": "provenance_group", "groups": len(members),
                      "videos": V, "interval": "95% percentile, linear quantiles",
                      "minimum_valid_draw_fraction": 0.95, "paired_across_models": True},
        "support": {"full": V, "complete_video": int(masks["complete_video"].sum()),
                    "complete_group_videos": int(masks["complete_group"].sum()),
                    "complete_groups": len(complete_groups)},
        "primary_contrast": {
            "condition": primary,
            "P1_RGB": {"per_seed": p1_rgb, "rule": "Direction only if all three seed intervals exclude zero with the same sign",
                       "reading": rgb_direction},
            "P1_PROBE": {"per_probe": p1_probe, "family": list(REPS), "alpha": ALPHA,
                         "rule": "Two-sided bootstrap p, Holm over the fixed six; undefined counts as p = 1"},
        },
        "secondary": {
            "S1_delta_other_qualities": s1,
            "S2_training": {"per_seed": s2, "decision_inputs": s2_decision,
                            "reading": "paired-learning benefit" if benefit else "no reliable benefit over augmentation"},
            "S3_known_quality_adaptation": s3,
            "S4_task_sharing": s4,
            "S5_source_supervision": s5,
            "S6_shortcut_and_content": {"C3a_border": border, "C3b_metadata": metadata,
                                        "material_rule": "95% lower bound: attribution BA > 0.30, detection BA > 0.55",
                                        "C3c": c3c},
            "S7_support_sensitivity": {"contrasts": s7, "status_changes": changes},
        },
        "model_intervals": model_intervals,
        "exploratory": {"label": "Exploratory (register section 8)", "point_metrics": exploratory,
                        "seed_summary": seed_summary, "distortion": distortion,
                        "not_computed": ["Label-shuffle diagnostics (validation-only fits; shuffled heads not saved)",
                                         "Embedding views (illustration only)"]},
        "validation_cross_check": cross_check,
        "provenance": {**provenance, "predictions_sha256": {k: sha256(v) for k, v in paths.items()},
                       "manifest_sha256": sha256(MANIFEST), "script_sha256": sha256(Path(__file__)),
                       "helper_sha256": {n: sha256(Path(__file__).with_name(n)) for n in (
                           "metrics.py", "research_metrics.py", "registered_models.py",
                           "eight_frame_sensitivity.py")},
                       "numpy": np.__version__},
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps({"P1_RGB": {k: (v["point"], v["ci95"]) for k, v in p1_rgb.items()}, "P1_RGB_reading": rgb_direction,
                      "P1_PROBE": {k: (v["point"], v["ci95"], v["bootstrap_p"], v["holm"]["rejected"])
                                   for k, v in p1_probe.items()},
                      "S2_reading": record["secondary"]["S2_training"]["reading"],
                      "S7_status_changes": len(changes), "validation_cross_check": cross_check}, indent=2))
    print("Saved:", output)


def cross_check_validation(model_intervals, points, row_of, C, s2, ids):
    """Dry run only: the same code must reproduce the recorded validation analyses."""
    rgb = json.loads(Path("experiments/results/main_validation_analysis_seeds42_44.json").read_text(encoding="utf-8"))
    probe = json.loads(Path("experiments/results/main_probe_validation_analysis.json").read_text(encoding="utf-8"))
    eight = json.loads(Path("experiments/results/eight_frame_sensitivity_validation.json").read_text(encoding="utf-8"))
    content = json.loads(Path("experiments/results/content_decomposition_validation.json").read_text(encoding="utf-8"))
    worst, compared = 0.0, 0

    def same(mine, theirs):
        nonlocal worst, compared
        assert (mine["ci95"] is None) == (theirs["ci95"] is None), (mine, theirs)
        assert (mine["point"] is None) == (theirs["point"] is None), (mine, theirs)
        if mine["point"] is not None:
            worst = max(worst, abs(mine["point"] - theirs["point"]))
        if mine["ci95"] is not None:
            worst = max(worst, *[abs(a - b) for a, b in zip(mine["ci95"], theirs["ci95"])])
        compared += 1

    modes = {"F1": "reference", "F2": "augmented", "F3": "paired_w1"}
    for family, mode in modes.items():
        for s in SEEDS:
            for c in VAL_CONDITIONS:
                for metric in ("detection_ba", "attribution_ba", "detection_retention", "attribution_retention",
                               "attribution_minus_detection_retention"):
                    same(model_intervals[f"{family}_s{s}"][c][metric], rgb["intervals"][f"c23_main_{mode}_seed{s}"][c][metric])
    for s in SEEDS:
        for label, recorded in (("F3_minus_F2", "pair_w1_minus_augmentation"), ("F2_minus_F1", "augmentation_minus_reference")):
            for c in VAL_CONDITIONS:
                for metric in ("detection_ba", "attribution_ba"):
                    same(s2[f"seed{s}"][label][c][metric], rgb["paired_differences"][f"{recorded}_seed{s}"][c][metric])
    for rep in REPS:
        for c in VAL_CONDITIONS:
            for metric, intervals in probe["intervals"][rep][c].items():
                if metric in model_intervals[f"F4_{rep}"][c]:
                    same(model_intervals[f"F4_{rep}"][c][metric], intervals)
    for name, recorded in [(f"{f}_s{s}", f"c23_main_{m}_seed{s}") for f, m in modes.items() for s in SEEDS] + \
            [(f"F4_{rep}", f"reference_probe_{rep}") for rep in REPS]:
        for c in VAL_CONDITIONS:
            for metric in ("detection_ba", "attribution_ba"):
                for support in SUPPORTS[1:]:
                    entry = eight["models"][recorded][c][metric][support]
                    value = points[support][row_of[name], C[c], K[metric]]
                    worst = max(worst, abs(value - entry["point"]))
                    compared += 1
    for name, recorded in [(f"F1_s{s}", f"rgb_reference_seed{s}") for s in SEEDS] + \
            [(f"C1-ATT_s{s}", f"c1_att_seed{s}") for s in SEEDS] + [(f"F4_{rep}", f"reference_probe_{rep}") for rep in REPS]:
        for c in VAL_CONDITIONS:
            for metric in ("family_ba", "within_swap_ba", "within_reenactment_ba"):
                mine = model_intervals[name][c][metric]["point"]
                worst = max(worst, abs(mine - content["models"][recorded][c][metric]))
                compared += 1
    assert worst < 1e-10, worst
    return {"values_compared": compared, "max_abs_difference": worst,
            "sources": ["main_validation_analysis_seeds42_44.json (intervals, paired differences)",
                        "main_probe_validation_analysis.json (intervals)",
                        "eight_frame_sensitivity_validation.json (subset points)",
                        "content_decomposition_validation.json (C3c points)"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.split:
        main(args.split)
    else:
        parser.error("--split or --selftest required")
