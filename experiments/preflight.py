"""Read-only environment audit; does not install or change global software."""
import ctypes
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def capture(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "scipy", "scikit-learn",
                 "opencv-python-headless", "PyWavelets", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(Path(__file__).resolve().parent)
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": packages, "free_disk_bytes": disk.free,
              "ffmpeg_path": shutil.which("ffmpeg"), "ffprobe_path": shutil.which("ffprobe"),
              "gpu": capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])}
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            class Memory(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
                    "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                result["ram_total_bytes"] = memory.total_physical
                result["ram_available_bytes"] = memory.available_physical
        except (OSError, AttributeError) as exc:
            result["system_info_error"] = str(exc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
