"""Classification components
"""

import torch
import torch.nn as nn


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        from .vgg11 import VGG11
        from .layers import CustomDropout
        
        self.encoder = VGG11(in_channels=in_channels)
        
        # Global Average Pooling: collapses spatial dims to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight head: 512-dim GAP vector -> logits
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        x = self.encoder(x, return_features=False)
        x = self.avgpool(x)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.classifier(x)
        return x
