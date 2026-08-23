import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FaceSpoofMaskClassifier(nn.Module):
    """ResNet18 transfer-learning classifier for real/spoof/masked faces."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = resnet18(weights=weights)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)
