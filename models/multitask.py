"""Unified multi-task model
"""

import torch
import torch.nn as nn

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        from .vgg11 import VGG11
        from .layers import CustomDropout
        
        self.encoder = VGG11(in_channels=in_channels)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 1. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds)
        )
        
        # 2. Localization Head
        self.localizer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        
        # 3. Segmentation Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.seg_conv = nn.Conv2d(64, seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Classification & Localization
        flat_feats = torch.flatten(self.avgpool(bottleneck), 1)
        cls_logits = self.classifier(flat_feats)
        loc_coords = self.localizer(flat_feats)
        
        # Segmentation
        d1 = self.upconv1(bottleneck)
        d1 = torch.cat([d1, features["f5"]], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, features["f4"]], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, features["f3"]], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, features["f2"]], dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.upconv5(d4)
        d5 = torch.cat([d5, features["f1"]], dim=1)
        d5 = self.dec5(d5)
        
        seg_logits = self.seg_conv(d5)
        
        return {
            "classification": cls_logits,
            "localization": loc_coords,
            "segmentation": seg_logits
        }
