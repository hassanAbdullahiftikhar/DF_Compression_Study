"""Small deterministic descriptors for pipeline/probe baselines, not paper reproductions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import audit, read_manifest, write_manifest


def dct_basis(n=8):
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    matrix = np.cos(np.pi / n * (x + 0.5) * k) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


def block_dct(gray):
    h, w = gray.shape
    if h % 8 or w % 8:
        raise ValueError("Block DCT dimensions must be multiples of 8")
    blocks = gray.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    basis = dct_basis()
    return np.einsum("ki,abij,lj->abkl", basis, blocks, basis)


def pool(array, grid=16):
    h, w = array.shape[:2]
    if h % grid or w % grid:
        raise ValueError("Pooling dimensions must be divisible by grid")
    return array.reshape(grid, h // grid, grid, w // grid, *array.shape[2:]).mean((1, 3))


def stats(array):
    return np.array([array.mean(), array.std(), np.abs(array).mean(),
                     np.square(array).mean(), *np.quantile(array, [0.1, 0.5, 0.9])])


def haar(gray):
    a, b, c, d = gray[::2, ::2], gray[::2, 1::2], gray[1::2, ::2], gray[1::2, 1::2]
    return [(a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2]


def laplacian(gray):
    # Defined 3x3 high-pass filter; deliberately not named SRM or camera noise.
    padded = np.pad(gray, 1, mode="reflect")
    return (-4 * gray + padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:])


def descriptor(image, representation):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if representation == "rgb":
        value = pool(rgb).ravel()
    elif representation == "fft":
        spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean(), norm="ortho"))
        value = pool(np.log1p(np.abs(spectrum))).ravel()
    elif representation == "dct":
        transformed = np.log1p(np.abs(block_dct(gray - 0.5)))
        value = np.concatenate([transformed.mean((0, 1)).ravel(), transformed.std((0, 1)).ravel()])
    elif representation == "wavelet":
        value = np.concatenate([stats(band) for band in haar(gray)])
    elif representation == "residual":
        residual = laplacian(gray)
        value = np.concatenate([pool(residual).ravel(), stats(residual)])
    else:
        raise ValueError("Representation must be rgb, fft, dct, wavelet, residual")
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite descriptor")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--representation", choices=["rgb", "fft", "dct", "wavelet", "residual"], required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    audit(rows)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(rows[0]["path"]) as image:
        dimensions = len(descriptor(image, args.representation))
    # Disk-backed array avoids holding images or all descriptors in RAM.
    matrix = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), dimensions))
    for index, row in enumerate(rows):
        with Image.open(row["path"]) as image:
            matrix[index] = descriptor(image, args.representation)
    matrix.flush()
    write_manifest(output / "rows.csv", rows)
    (output / "feature_metadata.json").write_text(json.dumps({
        "representation": args.representation, "dimensions": dimensions,
        "rows": len(rows), "normalization": "none; fit scaler on training split only",
        "qualification": "deterministic descriptor baseline, not F3Net, SRM, or DINOv2"}, indent=2))
    print(json.dumps({"rows": len(rows), "dimensions": dimensions}))
