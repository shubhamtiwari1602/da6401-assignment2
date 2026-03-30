"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        from .vgg11 import VGG11
        self.encoder = VGG11(in_channels=in_channels)
        
        # Decoder 1 (bottleneck 512, f5 512) -> 512
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 2 (dec1 512, f4 512) -> 512
        self.upconv2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 3 (dec2 256, f3 256) -> 256
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 4 (dec3 128, f2 128) -> 128
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 5 (dec4 64, f1 64) -> 64
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction head
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.encoder(x, return_features=True)
        
        f1 = features["f1"]
        f2 = features["f2"]
        f3 = features["f3"]
        f4 = features["f4"]
        f5 = features["f5"]
        
        d1 = self.upconv1(bottleneck)
        d1 = torch.cat([d1, f5], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, f4], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, f3], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, f2], dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.upconv5(d4)
        d5 = torch.cat([d5, f1], dim=1)
        d5 = self.dec5(d5)
        
        out = self.final_conv(d5)
        return out
