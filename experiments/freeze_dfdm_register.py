"""Freeze the DFDM second-dataset register before any crop extraction, fitting or scoring."""
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "research/dfdm_register.md"
OUTPUT = ROOT / "research/dfdm_register_freeze.json"
INPUTS = ("experiments/data/audit_dfdm/videos.csv", "experiments/data/audit_dfdm/audit.json",
          "CelebDF_V2/List_of_testing_videos.txt", "experiments/audit_dfdm.py")


def hashes(path):
    raw = Path(path).read_bytes()
    return {"working_tree_sha256": hashlib.sha256(raw).hexdigest(),
            "lf_normalized_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    # Nothing downstream may exist yet: no crops, no runs, no results.
    downstream = [p for pattern in ("experiments/data/dfdm*", "experiments/runs/*dfdm*", "experiments/results/*dfdm*",
                                    "experiments/features/*dfdm*")
                  for p in ROOT.glob(pattern) if p.name != "audit_dfdm"]
    if downstream:
        raise RuntimeError(f"DFDM outputs already exist: {downstream}")
    record = {
        "register": "research/dfdm_register.md", "register_version": "1.0", "frozen_on": date.today().isoformat(),
        "register_sha256": hashes(REGISTER),
        "status": "no DFDM crops, models or predictions existed at freeze time",
        "inputs_sha256": {name: hashes(ROOT / name) for name in INPUTS},
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"saved": OUTPUT.relative_to(ROOT).as_posix(),
                      "register_sha256": record["register_sha256"]["working_tree_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
