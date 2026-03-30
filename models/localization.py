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
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Regression head for 4 coordinates
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid() # Scale outputs to [0, 1] for normalized box coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        x = self.encoder(x, return_features=False)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x * 224.0
