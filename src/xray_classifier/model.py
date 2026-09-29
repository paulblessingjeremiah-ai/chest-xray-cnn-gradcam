"""CNN model definition using transfer learning."""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=4):
    """Build a ResNet18 model pretrained on ImageNet, adapted for our classes."""
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
