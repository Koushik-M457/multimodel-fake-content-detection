import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_resnet50(num_classes=2, freeze_backbone=False):
    # Load pretrained ResNet-50 (ImageNet)
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
