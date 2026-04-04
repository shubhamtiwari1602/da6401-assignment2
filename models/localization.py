"""Localization modules
"""

import torch
import torch.nn as nn

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        from .vgg11 import VGG11
        from .layers import CustomDropout
        self.encoder = VGG11(in_channels=in_channels)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head: 512-dim GAP vector -> 4 box coords
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Scale outputs to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        x = self.encoder(x, return_features=False)
        x = self.avgpool(x)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.regressor(x)    # [B, 4] in [0, 1]
        return x * 224.0
