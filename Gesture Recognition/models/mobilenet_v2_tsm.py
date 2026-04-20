import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from .tsm import TemporalShift

class MobileNetV2TSM(nn.Module):
    def __init__(self, num_classes=27, num_segments=8, pretrained=True):
        super().__init__()
        self.num_segments = num_segments
        base_model = mobilenet_v2(weights="DEFAULT" if pretrained else None)

        features = []
        for layer in base_model.features:
            features.append(TemporalShift(layer, num_segments=num_segments))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_model.last_channel, num_classes)

    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(b, t, -1).mean(dim=1)

        x = self.classifier(x)
        return x