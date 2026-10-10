"""Synthetic real forward/backward GPU test; no dataset or pretrained download required."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8, help="Number of pairs")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="gpu_benchmark.json")
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1:
        raise ValueError("Positive batch and steps required")
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Output already exists")
    try:
        import torch
        from model import TwoHeadResNet, paired_loss
    except ImportError as exc:
        print(json.dumps({"status": "not_run", "reason": str(exc),
                          "next_step": "Install the documented compatible research environment"}, indent=2))
        return 2
    if not torch.cuda.is_available():
        print(json.dumps({"status": "not_run", "reason": "CUDA unavailable"}, indent=2))
        return 2
    model = TwoHeadResNet().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    clean = torch.randn(args.batch, 3, 224, 224, device="cuda")
    degraded = torch.randn_like(clean)
    labels = torch.arange(args.batch, device="cuda") % 2
    sources = torch.arange(args.batch, device="cuda") % 4
    sources[labels == 0] = -1
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda"):
            loss = paired_loss(model, clean, degraded, labels, sources)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result = {"status": "measured_synthetic_training", "gpu": torch.cuda.get_device_name(),
              "capability": torch.cuda.get_device_capability(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "batch_pairs": args.batch,
              "steps": args.steps, "seconds_per_step": seconds / args.steps,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "qualification": "Excludes disk decoding, JPEG generation, validation and real dataset variability"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
