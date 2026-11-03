"""DFDM RGB scorer: --verify (gate D-G2, validation) and --test (gate D-G4, one pass).

Uses dfdm_lib.predict at batch 8, exactly as the stored validation predictions
were made. Test conditions: ref, JPEG Q95/Q80/Q60/Q40/Q20 (in memory) and
DFDM's native H.264 CRF 10 / CRF 23 crops.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from dfdm_lib import (GENERATORS, N_SRC, ROOT, TwoHead, predict, read_csv, read_rows, sha256, write_csv)

THRESHOLDS = ROOT / "experiments/results/dfdm_validation_thresholds.json"
VERIFY = ROOT / "experiments/results/dfdm_rgb_verification_validation.json"
TEST_OUTPUT = ROOT / "experiments/runs/dfdm_rgb_test"
FIELDS = ("p_fake", *[f"p_source_{k}" for k in range(N_SRC)])


def load(entry, device):
    path = ROOT / entry["run"] / "best.pt"
    assert sha256(path) == entry["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(saved["generators"]) == GENERATORS
    model = TwoHead(saved["backbone"], pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def verify():
    if VERIFY.exists():
        raise FileExistsError(VERIFY)
    runs = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["runs"]
    val = read_rows("crf0", "val")
    torch.backends.cudnn.benchmark = False
    device, results = torch.device("cuda"), {}
    for model_id, entry in runs.items():
        model = load(entry, device)
        rescored = {(r["sample_id"], r["condition"]): r for r in predict(model, val, [80, 40], device)}
        del model
        torch.cuda.empty_cache()
        stored = read_csv(ROOT / entry["run"] / "validation_predictions.csv")
        assert len(stored) == len(rescored)
        worst = max(abs(float(s[f]) - rescored[(s["sample_id"], s["condition"])][f]) for s in stored for f in FIELDS)
        results[model_id] = {"max_abs_probability_difference": worst, "passed": worst <= 1e-6}
        print(json.dumps({model_id: results[model_id]}), flush=True)
    record = {"status": "dfdm_rgb_verification", "tolerance": 1e-6, "all_passed": all(r["passed"] for r in results.values()),
              "script_sha256": sha256(Path(__file__)), "models": results}
    with VERIFY.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    if not record["all_passed"]:
        raise SystemExit("DFDM RGB verification failed")


def score_test():
    from dfdm_freeze import require_dfdm_freeze
    digest = require_dfdm_freeze(Path(__file__))
    if TEST_OUTPUT.exists():
        raise FileExistsError(TEST_OUTPUT)
    runs = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["runs"]
    test = read_rows("crf0", "test")
    codec = {"h264_crf10": read_rows("crf10", "test"), "h264_crf23": read_rows("crf23", "test")}
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    TEST_OUTPUT.mkdir(parents=True)
    reports, started = {}, time.perf_counter()
    for model_id, entry in runs.items():
        model = load(entry, device)
        rows = predict(model, test, [95, 80, 60, 40, 20], device)
        for condition, codec_rows in codec.items():
            rows += predict(model, codec_rows, [], device, condition_names={0: condition})
        del model
        torch.cuda.empty_cache()
        (TEST_OUTPUT / model_id).mkdir()
        write_csv(TEST_OUTPUT / model_id / "test_predictions.csv", rows)
        reports[model_id] = {"rows": len(rows), "test_predictions_sha256": sha256(TEST_OUTPUT / model_id / "test_predictions.csv")}
        print(json.dumps({model_id: reports[model_id]}), flush=True)
    record = {"status": "dfdm_rgb_test_scores", "dfdm_freeze_manifest_sha256": digest, "script_sha256": sha256(Path(__file__)),
              "elapsed_seconds": time.perf_counter() - started, "models": reports}
    with (TEST_OUTPUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("Saved:", TEST_OUTPUT / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--test", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else score_test()
