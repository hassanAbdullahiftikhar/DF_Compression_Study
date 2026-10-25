"""DFDM inventory and identity-disjoint split (Tier 3 preparation; reads videos, trains nothing).

Every DFDM file is named idT_idS_NNNN_<k><model>.mp4: target identity T,
source identity S, clip NNNN, generator k. The identity graph (T-S edges)
splits into connected components; each component goes wholly to one split,
so no celebrity appears in two splits. Fixed rule: largest component ->
train, middle -> test, smallest -> val.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
DFDM = ROOT / "DFDM"
LEVELS = {"crf0": "CRF0/DFDM_cfr0", "crf10": "CRF10/DFDM_cfr10", "crf23": "CRF23/DFDM_cfr23"}
MODELS = {"1original": "faceswap", "2lightweight": "lightweight", "3iae": "iae", "4dfaker": "dfaker", "5dfl-h128": "dfl_h128"}
NAME = re.compile(r"^id(\d+)_id(\d+)_(\d+)_(\d[a-z0-9-]+)\.mp4$")
OUTPUT = ROOT / "experiments/data/audit_dfdm"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def components(clips):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for target, source, _ in clips:
        parent[find(target)] = find(source)
    groups = collections.defaultdict(list)
    for clip in clips:
        groups[find(clip[0])].append(clip)
    return sorted(groups.values(), key=len, reverse=True)


def split_of(clips):
    ordered = components(clips)
    assert len(ordered) == 3, f"Expected three identity components, found {len(ordered)}"
    rule = dict(zip(("train", "test", "val"), ordered))
    return {clip: split for split, members in rule.items() for clip in members}


def selftest():
    clips = [("1", "2", "0"), ("2", "3", "0"), ("4", "5", "0"), ("4", "5", "1"), ("6", "7", "0")]
    parts = components(clips)
    assert [len(p) for p in parts] == [2, 2, 1]
    mapping = split_of(clips)
    assert {mapping[c] for c in clips[:2]} == {"train"} and mapping[clips[4]] == "val"
    assert NAME.match("id0_id1_0000_5dfl-h128.mp4").groups() == ("0", "1", "0000", "5dfl-h128")
    print(json.dumps({"status": "passed", "synthetic_only": True}))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows, problems = [], []
    for level, folder in LEVELS.items():
        files = sorted((DFDM / folder).glob("*.mp4"))
        assert len(files) == 2150, (level, len(files))
        for path in files:
            target, source, clip, model = NAME.match(path.name).groups()
            assert model in MODELS, path.name
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                context = stream.codec_context
                row = {"level": level, "path": str(path), "file": path.name, "clip_id": f"id{target}_id{source}_{clip}",
                       "target_id": target, "source_id": source, "generator": MODELS[model],
                       "frame_count_metadata": stream.frames, "width": context.width, "height": context.height,
                       "fps": float(stream.average_rate), "codec": context.name, "pix_fmt": context.pix_fmt,
                       "file_size_bytes": path.stat().st_size, "sha256": sha256(path)}
            rows.append(row)
        print(json.dumps({"level_done": level, "videos": len(files)}), flush=True)
    clips = sorted({(r["target_id"], r["source_id"], r["clip_id"].rsplit("_", 1)[1]) for r in rows})
    mapping = split_of(clips)
    for row in rows:
        row["split"] = mapping[(row["target_id"], row["source_id"], row["clip_id"].rsplit("_", 1)[1])]
        row["group_id"] = f"dfdm_id{row['target_id']}_id{row['source_id']}"
    by_clip = collections.defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    for clip_id, members in by_clip.items():
        if len(members) != 15:
            problems.append({"clip_id": clip_id, "problem": f"{len(members)} files instead of 15"})
        for field in ("frame_count_metadata", "width", "height", "fps"):
            values = {r[field] for r in members}
            if len(values) != 1:
                problems.append({"clip_id": clip_id, "problem": f"{field} differs", "values": sorted(map(str, values))})
    ids = {split: set() for split in ("train", "val", "test")}
    for row in rows:
        ids[row["split"]] |= {row["target_id"], row["source_id"]}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "videos.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "dfdm_inventory_no_training",
        "videos": len(rows), "clips": len(by_clip), "levels": list(LEVELS), "generators": list(MODELS.values()),
        "split_rule": "Identity-graph components; largest -> train, middle -> test, smallest -> val",
        "splits": {s: {"identities": len(ids[s]), "clips": len({r["clip_id"] for r in rows if r["split"] == s}),
                       "videos_per_level": sum(r["split"] == s for r in rows) // 3} for s in ids},
        "identity_overlap_between_splits": 0,
        "consistency_problems": problems,
        "formats": dict(collections.Counter(f"{r['level']}:{r['codec']}/{r['pix_fmt']}" for r in rows)),
        "resolutions": dict(collections.Counter(f"{r['width']}x{r['height']}" for r in rows).most_common(6)),
        "frames": {"min": min(r["frame_count_metadata"] for r in rows), "max": max(r["frame_count_metadata"] for r in rows)},
        "total_bytes": sum(r["file_size_bytes"] for r in rows),
        "videos_sha256": sha256(OUTPUT / "videos.csv"), "script_sha256": sha256(Path(__file__)), "pyav": av.__version__,
        "provenance_note": "DFDM as downloaded from the authors' repository; per-file SHA-256 recorded at audit time.",
    }
    (OUTPUT / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("videos", "clips", "splits", "formats", "frames")}, indent=2))
    print(json.dumps({"consistency_problems": len(problems), "examples": problems[:5]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest() if args.selftest else main()
