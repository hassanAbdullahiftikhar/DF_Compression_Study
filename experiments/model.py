"""Ordinary ResNet-18 multi-task paired baseline; no novelty claim."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class TwoHeadResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.detection = nn.Linear(512, 2)
        self.attribution = nn.Linear(512, 4)
        self.projection = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, images):
        features = self.encoder(images)
        return self.detection(features), self.attribution(features), self.projection(features)


def paired_loss(model, clean, degraded, detection, source, pair_weight=0.1,
                detection_weights=None, source_weights=None, source_loss_weight=1.0):
    clean_outputs, degraded_outputs = model(clean), model(degraded)
    det_loss = sum(nn.functional.cross_entropy(outputs[0], detection, weight=detection_weights)
                   for outputs in (clean_outputs, degraded_outputs)) / 2
    mask = source >= 0
    src_loss = clean_outputs[1].sum() * 0
    if mask.any():
        src_loss = sum(nn.functional.cross_entropy(outputs[1][mask], source[mask], weight=source_weights)
                       for outputs in (clean_outputs, degraded_outputs)) / 2
    alignment = (1 - nn.functional.cosine_similarity(clean_outputs[2], degraded_outputs[2])).mean()
    return det_loss + source_loss_weight * src_loss + pair_weight * alignment
