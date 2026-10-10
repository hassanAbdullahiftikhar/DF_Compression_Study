"""Fetch small public FF++ split lists, without downloading or modifying dataset media."""
import argparse
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    payloads, metadata, seen = {}, {}, {}
    for split in ("train", "val", "test"):
        url = f"https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split}.json"
        request = urllib.request.Request(url, headers={"User-Agent": "FFpp-research-audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        pairs = json.loads(payload)
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Invalid official split JSON")
        for pair in pairs:
            if len(pair) != 2 or any(not isinstance(x, str) or not x.isdigit() for x in pair):
                raise ValueError("Expected two original video IDs per split entry")
            for original in pair:
                if original in seen:
                    raise ValueError(f"Official split lists repeat original {original}")
                seen[original] = split
        payloads[split] = payload
        metadata[split] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                           "pairs": len(pairs), "original_ids": 2 * len(pairs)}
    output.mkdir(parents=True, exist_ok=True)
    for split, payload in payloads.items():
        with (output / f"{split}.json").open("xb") as handle:
            handle.write(payload)
    (output / "provenance.json").write_text(json.dumps({
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_ref": "master; exact downloaded content identified by SHA256", "splits": metadata}, indent=2))
    print(json.dumps({"output": str(output), "total_original_ids": len(seen), "splits": metadata}, indent=2))


if __name__ == "__main__":
    main()
