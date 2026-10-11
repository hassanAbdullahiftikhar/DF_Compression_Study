"""Run the paired trainer and record timing, GPU memory and selected results."""
import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest")
parser.add_argument("output")
parser.add_argument("trainer_arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()

script = Path(__file__).resolve().parent / "train_paired.py"
sys.path.insert(0, str(script.parent))

import common
assert Path(common.__file__).resolve() == script.parent / "common.py"

import torch

manifest = Path(args.manifest).resolve()
output = Path(args.output).resolve()
if output.exists() and any(output.iterdir()):
    raise ValueError("Training output must be new or empty")

manifest_hash = sha256(manifest)
assembly_path = manifest.parent / "assembly.json"
if assembly_path.exists():
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    expected = assembly.get("reference_manifest_sha256")
    if expected is not None and manifest.name == "reference.csv":
        assert manifest_hash == expected, "Manifest changed after assembly"

assert torch.cuda.is_available(), "CUDA unavailable"
sys.argv = [
    str(script), str(manifest), str(output), *args.trainer_arguments
]

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
runpy.run_path(str(script), run_name="__main__")
torch.cuda.synchronize()
seconds = time.perf_counter() - started

resources = {
    "status": "completed_measured_main_training",
    "elapsed_seconds": seconds,
    "elapsed_minutes": seconds / 60,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "manifest_sha256": manifest_hash,
    "trainer_sha256": sha256(script),
    "runner_sha256": sha256(Path(__file__)),
    "qualification": (
        "Training/validation runtime; GPU figures use the PyTorch allocator. "
        "System RAM and other processes are not measured."
    ),
}
with (output / "resource_usage.json").open("x", encoding="utf-8") as handle:
    json.dump(resources, handle, indent=2, allow_nan=False)

history = json.loads((output / "history.json").read_text())

def selection_score(epoch):
    return sum(
        (scores["detection_ba"] + scores["attribution_ba"]) / 2
        for scores in epoch["validation"].values()
    ) / len(epoch["validation"])

best = max(history, key=selection_score)
print(json.dumps({
    "epochs_completed": len(history),
    "first_train_loss": history[0]["train_loss"],
    "last_train_loss": history[-1]["train_loss"],
    "selected_epoch": best["epoch"],
    "selection_score": selection_score(best),
    "selected_validation": best["validation"],
    "resources": resources,
}, indent=2, allow_nan=False))
