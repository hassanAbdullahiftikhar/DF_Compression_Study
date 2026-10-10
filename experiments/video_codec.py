"""Plan one full-video re-encode; execute only when explicitly requested via --execute."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--codec", choices=["libx264", "libx265"], default="libx264")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.crf <= 51:
        raise ValueError("CRF must be in 0–51")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        raise ValueError("Output must be a new file distinct from input")
    if not source.is_file():
        raise ValueError("Input video does not exist")
    command = [shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-n", "-i", str(source),
               "-map", "0:v:0", "-an", "-c:v", args.codec, "-preset", "medium",
               "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-g", "48", "-bf", "3", "-threads", "4"]
    if args.codec == "libx265":
        command += ["-x265-params", "pools=4:frame-threads=2"]
    command.append(str(output))
    print(json.dumps({"command": command, "execute": args.execute,
                      "note": "Whole-video encoding; audit frame timing, dimensions and distortion afterwards."}, indent=2))
    if args.execute:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install compatible FFmpeg/FFprobe first")
        metadata_path = output.with_suffix(output.suffix + ".metadata.json")
        if metadata_path.exists():
            raise ValueError("Metadata file already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        def probe(path):
            return json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True))
        input_info = probe(source)
        subprocess.run(command, check=True)
        metadata_path.write_text(json.dumps({"command": command, "input": input_info,
            "output": probe(output), "ffmpeg_version": subprocess.check_output(
                ["ffmpeg", "-version"], text=True)}, indent=2), encoding="utf-8")
