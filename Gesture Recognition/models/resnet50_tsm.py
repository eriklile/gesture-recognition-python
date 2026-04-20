import torch
import torch.nn as nn
from torchvision.models import resnet50
from .tsm import TemporalShift

class ResNet50TSM(nn.Module):
    def __init__(self, num_classes=27, num_segments=8, pretrained=True):
        super().__init__()
        self.num_segments = num_segments
        base_model = resnet50(weights="DEFAULT" if pretrained else None)

        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

        self.layer1 = TemporalShift(base_model.layer1, num_segments=num_segments)
        self.layer2 = TemporalShift(base_model.layer2, num_segments=num_segments)
        self.layer3 = TemporalShift(base_model.layer3, num_segments=num_segments)
        self.layer4 = TemporalShift(base_model.layer4, num_segments=num_segments)

        self.pool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = x.view(b, t, -1).mean(dim=1)
        x = self.fc(x)
        return x