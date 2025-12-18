import torch
import torch.nn as nn
import torchvision.models as models

class FaceSpoofMaskClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(FaceSpoofMaskClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)