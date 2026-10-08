"""
training/model_factory.py
==========================
Shared model creation utility for the multi-model plant detection system.
Supports swappable backbone architectures:
    - efficientnet_b0 (default)
    - efficientnet_b2
    - efficientnet_b3
    - resnet50
    - convnext_tiny

Usage:
    from training.model_factory import create_model, freeze_backbone, unfreeze_backbone
"""

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = [
    'efficientnet_b0',
    'efficientnet_b2',
    'efficientnet_b3',
    'resnet50',
    'convnext_tiny',
]


def create_model(backbone_name: str, num_classes: int, pretrained: bool = True):
    """
    Create a model with the specified backbone and a custom classifier head.

    Args:
        backbone_name: One of SUPPORTED_BACKBONES
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        (model, in_features) tuple
    """
    backbone_name = backbone_name.lower()

    if backbone_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f'Unsupported backbone: {backbone_name}. Choose from: {SUPPORTED_BACKBONES}')

    return model


def freeze_backbone(model, backbone_name: str):
    """
    Freeze all layers except the classifier head.
    Allows fast warmup training of the head only.
    """
    backbone_name = backbone_name.lower()
    if 'efficientnet' in backbone_name:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    elif backbone_name == 'resnet50':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    elif backbone_name == 'convnext_tiny':
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f'  Backbone frozen: {frozen}/{total} parameter groups frozen')


def unfreeze_backbone(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(1 for p in model.parameters())
    print(f'  Backbone unfrozen: all {total} parameter groups trainable')


def get_classifier_params(model, backbone_name: str):
    """Return (classifier_params, backbone_params) for differential learning rates."""
    backbone_name = backbone_name.lower()
    if 'efficientnet' in backbone_name:
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    elif backbone_name == 'resnet50':
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    elif backbone_name == 'convnext_tiny':
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    else:
        head_params = list(model.parameters())
        backbone_params = []
    return head_params, backbone_params
