"""Registered control C3c: content-matched design check and attribution decomposition.

Family BA separates identity-swap {Deepfakes, FaceSwap} from reenactment
{Face2Face, NeuralTextures} using summed source probabilities. Within-family
BA compares the two methods inside each true family. All are descriptive and
computed from stored validation predictions; nothing is refitted.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate_probe_compression import sha256
from metrics import aggregate, balanced_accuracy
from research_metrics import retention

SWAP, REENACT = (0, 2), (1, 3)  # Deepfakes, Face2Face, FaceSwap, NeuralTextures
OUTPUT = Path("experiments/results/content_decomposition_validation.json")
REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")


def models():
    runs = Path("experiments/runs")
    probe_results = Path("experiments/results/c23_main_probe_compression_validation")
    listing = {}
    for seed in (42, 43, 44):
        for mode in ("reference", "augmented", "paired_w1"):
            listing[f"rgb_{mode}_seed{seed}"] = runs / f"c23_main_{mode}_seed{seed}/validation_predictions.csv"
        listing[f"c1_att_seed{seed}"] = runs / f"c23_control_c1_att_seed{seed}/validation_predictions.csv"
        for encoder in ("f3", "c2_bin"):
            listing[f"readout_{encoder}_seed{seed}"] = (
                runs / f"c23_control_source_readout_validation/{encoder}_seed{seed}/validation_predictions.csv")
    for rep in REPS:
        listing[f"reference_probe_{rep}"] = probe_results / rep / "validation_predictions.csv"
        listing[f"border_probe_{rep}"] = runs / f"c23_control_border_probes_validation/{rep}/validation_predictions.csv"
        for quality in (80, 40):
            listing[f"matched_q{quality}_probe_{rep}"] = (
                runs / f"c23_main_matched_q{quality}_probes_validation/{rep}/validation_predictions.csv")
    return listing


def decompose(videos):
    fakes = [v for v in videos if v["label"] == 1]
    sources = np.array([v["source_index"] for v in fakes])
    probabilities = np.array([[v[f"p_source_{k}"] for k in range(4)] for v in fakes])
    true_family = np.isin(sources, REENACT).astype(int)
    predicted_family = (probabilities[:, list(REENACT)].sum(axis=1) > probabilities[:, list(SWAP)].sum(axis=1)).astype(int)
    result = {
        "four_way_attribution_ba": balanced_accuracy(sources, probabilities.argmax(axis=1), 4),
        "family_ba": balanced_accuracy(true_family, predicted_family, 2),
    }
    for name, (first, second) in (("within_swap_ba", SWAP), ("within_reenactment_ba", REENACT)):
        inside = np.isin(sources, (first, second))
        truth = (sources[inside] == second).astype(int)
        # Ties go to the first method, matching argmax ordering.
        guess = (probabilities[inside, second] > probabilities[inside, first]).astype(int)
        result[name] = balanced_accuracy(truth, guess, 2)
    return result


def design_check():
    with Path("experiments/data/audit_c23/videos_core.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = defaultdict(set)
    for row in rows:
        if row["source"] != "real":
            methods[(row["split"], row["video_id"].split("/")[1])].add(row["source"])
    with Path("experiments/data/c23_main/reference.csv").open(encoding="utf-8-sig", newline="") as handle:
        usable = {r["video_id"] for r in csv.DictReader(handle)}
    report = {}
    for split in ("train", "val", "test"):
        pairs = [pair for (s, pair) in methods if s == split]
        report[split] = {
            "fake_pair_ids": len(pairs),
            "pairs_with_all_four_methods": sum(len(methods[(split, p)]) == 4 for p in pairs),
            "pairs_with_all_four_usable_videos": sum(
                all(f"{m}/{p}" in usable for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"))
                for p in pairs),
        }
    return report


def selftest():
    videos = []
    for source in range(4):
        p = [0.05] * 4
        p[source] = 0.85
        videos.append({"label": 1, "source_index": source, **{f"p_source_{k}": p[k] for k in range(4)}})
    perfect = decompose(videos)
    assert all(value == 1 for value in perfect.values()), perfect
    # Swapping within families keeps family accuracy and breaks within-family accuracy.
    swapped = [dict(v, source_index={0: 2, 2: 0, 1: 3, 3: 1}[v["source_index"]]) for v in videos]
    result = decompose(swapped)
    assert result["family_ba"] == 1 and result["within_swap_ba"] == 0 and result["within_reenactment_ba"] == 0
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    results, hashes = {}, {}
    for name, path in models().items():
        hashes[name] = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            videos = aggregate(list(csv.DictReader(handle)))
        by_condition = defaultdict(list)
        for video in videos:
            by_condition[video["condition"]].append(video)
        results[name] = {c: decompose(v) for c, v in sorted(by_condition.items())}
        assert all(len(v) == 699 for v in by_condition.values())
        if "ref" in results[name]:
            reference = results[name]["ref"]
            for condition, values in results[name].items():
                values["retention"] = {
                    "family": retention(values["family_ba"], reference["family_ba"], 0.5),
                    "within_swap": retention(values["within_swap_ba"], reference["within_swap_ba"], 0.5),
                    "within_reenactment": retention(values["within_reenactment_ba"], reference["within_reenactment_ba"], 0.5),
                    "four_way": retention(values["four_way_attribution_ba"], reference["four_way_attribution_ba"], 0.25),
                }
    record = {
        "status": "validation_descriptive_registered_control",
        "control": "C3c content-matched decomposition",
        "register": "research/comparison_register.md v1.0 section 5",
        "test_scores_used": False,
        "source_order": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        "families": {"identity_swap": ["Deepfakes", "FaceSwap"], "reenactment": ["Face2Face", "NeuralTextures"]},
        "chance": {"family": 0.5, "within_family": 0.5, "four_way": 0.25},
        "design_check": design_check(),
        "retention_rule": "Unclipped chance-adjusted BA retention; undefined when reference is < chance + 0.05",
        "prediction_sha256": hashes,
        "script_sha256": sha256(Path(__file__)),
        "qualification": "Descriptive. Family separation can reflect identity change rather than "
                         "method-specific traces; within-family BA isolates finer method cues.",
        "models": results,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps(record["design_check"], indent=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
