"""Shared pieces for the DFDM study (research/dfdm_register.md): labels, data, models, metrics.

Mirrors the FF++ code with five generator classes; the frozen FF++ scripts
(4-class) are not modified. Video scores average frame probabilities.
"""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_B4_Weights, ResNet18_Weights, efficientnet_b4, resnet18
from torchvision.transforms.functional import normalize, to_tensor

from metrics import auc, balanced_accuracy
from prepare_jpeg import jpeg_sequence

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "experiments/data/dfdm_crops"
GENERATORS = ("faceswap", "lightweight", "iae", "dfaker", "dfl_h128")
N_SRC = len(GENERATORS)
VAL_CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
JPEG_TEST = (95, 80, 60, 40, 20)
CODEC_TEST = ("h264_crf10", "h264_crf23")
PROB_FIELDS = ("p_fake", *[f"p_source_{k}" for k in range(N_SRC)])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_index(source):
    return -1 if source == "real" else GENERATORS.index(source)


def read_rows(level="crf0", split=None):
    with (CROPS / f"manifest_{level}.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if split is None or r["split"] == split]


def tensor(image):
    return normalize(to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_image(path, quality=0):
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    assert image.size == (224, 224)
    return jpeg_sequence(image, [quality], subsampling=2)[1] if quality else image


class PairedCrops(Dataset):
    """train_paired.PairedCrops with five-way source labels."""

    def __init__(self, rows, qualities, fixed_quality=None):
        self.rows, self.qualities, self.fixed = rows, qualities, fixed_quality

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import random
        row = self.rows[index]
        clean = load_image(row["path"])
        quality = random.choice(self.qualities) if self.fixed is None else self.fixed
        degraded = clean if quality == 0 else jpeg_sequence(clean, [quality], subsampling=2)[1]
        source = source_index(row["source"])
        return tensor(clean), tensor(degraded), int(source >= 0), source, index


class TwoHead(nn.Module):
    """ResNet-18 or EfficientNet-B4 encoder with detection, 5-way source and projection heads."""

    def __init__(self, backbone, pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder.fc, width = nn.Identity(), 512
        elif backbone == "effb4":
            self.encoder = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder.classifier, width = nn.Identity(), 1792
        else:
            raise ValueError(backbone)
        self.detection = nn.Linear(width, 2)
        self.attribution = nn.Linear(width, N_SRC)
        self.projection = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, images):
        features = self.encoder(images)
        return self.detection(features), self.attribution(features), self.projection(features)


def prediction_row(row, condition, p_fake, p_source):
    source = source_index(row["source"])
    return {"sample_id": row["sample_id"], "video_id": row["video_id"], "group_id": row["group_id"],
            "condition": condition, "label": int(source >= 0), "source_index": source, "p_fake": float(p_fake),
            **{f"p_source_{k}": float(p_source[k]) for k in range(N_SRC)}}


def predict(model, rows, qualities, device, batch=8, condition_names=None):
    """Quality 0 first, then each JPEG quality; batch 8 as in the FF++ scorers."""
    from torch.utils.data import DataLoader
    model.eval()
    out = []
    for quality in [0, *qualities]:
        condition = (condition_names or {}).get(quality, "ref" if quality == 0 else f"jpeg_q{quality}")
        data = PairedCrops(rows, list(qualities) or [0], fixed_quality=quality)
        for _, images, _, _, indices in DataLoader(data, batch_size=batch, num_workers=0):
            with torch.no_grad():
                det, src, _ = model(images.to(device))
                p_fake = det.softmax(1)[:, 1].cpu().numpy()
                p_src = src.softmax(1).cpu().numpy()
            for i, index in enumerate(indices.tolist()):
                out.append(prediction_row(rows[index], condition, p_fake[i], p_src[i]))
    return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["video_id"], row["condition"])].append(row)
    videos = []
    for (video, condition), frames in groups.items():
        first = frames[0]
        assert len({(int(r["label"]), int(r["source_index"]), r["group_id"]) for r in frames}) == 1
        videos.append({"video_id": video, "condition": condition, "group_id": first["group_id"],
                       "label": int(first["label"]), "source_index": int(first["source_index"]),
                       **{f: float(np.mean([float(r[f]) for r in frames])) for f in PROB_FIELDS}})
    return videos


def evaluate(rows, threshold=0.5, videos=None):
    videos = videos if videos is not None else aggregate(rows)
    results = {}
    for condition in sorted({v["condition"] for v in videos}):
        subset = [v for v in videos if v["condition"] == condition]
        y = np.array([v["label"] for v in subset])
        scores = np.array([v["p_fake"] for v in subset])
        fakes = [v for v in subset if v["label"] == 1]
        s_true = np.array([v["source_index"] for v in fakes])
        s_pred = np.array([np.argmax([v[f"p_source_{k}"] for k in range(N_SRC)]) for v in fakes])
        results[condition] = {"videos": len(subset), "fake_videos": len(fakes), "detection_auc": auc(y, scores),
                              "detection_ba": balanced_accuracy(y, (scores >= threshold).astype(int), 2),
                              "attribution_ba": balanced_accuracy(s_true, s_pred, N_SRC)}
    return results


def choose_threshold(videos, conditions):
    """Maximize mean video detection BA over conditions; ties: closest to 0.5, then smaller (1e-12)."""
    data = []
    for condition in conditions:
        subset = [v for v in videos if v["condition"] == condition]
        data.append((np.array([v["label"] for v in subset]), np.array([v["p_fake"] for v in subset])))
    scores = np.concatenate([s for _, s in data])
    best_score, best = -1.0, None
    for t in np.unique(np.r_[scores, 0.5, np.nextafter(scores.max(), np.inf)]):
        score = float(np.mean([balanced_accuracy(y, (s >= t).astype(int), 2) for y, s in data]))
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and
                                          (abs(t - 0.5), t) < (abs(best - 0.5), best)):
            best_score, best = score, float(t)
    return best, best_score


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selftest():
    videos = [{"video_id": f"v{i}", "condition": "ref", "label": int(i > 0), "source_index": i - 1,
               "group_id": "g", "p_fake": 0.9 if i else 0.1,
               **{f"p_source_{k}": float(k == i - 1) for k in range(N_SRC)}} for i in range(6)]
    result = evaluate(None, 0.5, videos)["ref"]
    assert result["detection_ba"] == 1 and result["attribution_ba"] == 1
    threshold, score = choose_threshold(videos, ("ref",))
    assert score == 1 and 0.1 < threshold <= 0.9
    model = TwoHead("resnet18").eval()
    with torch.no_grad():
        det, src, proj = model(torch.randn(2, 3, 224, 224))
    assert src.shape == (2, N_SRC)
    print({"status": "passed", "synthetic_only": True})


if __name__ == "__main__":
    selftest()
