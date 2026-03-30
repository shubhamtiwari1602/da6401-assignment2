"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        """
        super().__init__()
        
        self.encoder = VGG11(in_channels=in_channels)
        
        # Instantiate separate models to load heads
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        
        # Load weights reliably
        def _load_if_exists(model, path):
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location="cpu"))
        
        _load_if_exists(classifier, classifier_path)
        _load_if_exists(localizer, localizer_path)
        _load_if_exists(unet, unet_path)
        
        # Copy the backbone from the classifier (serves as the shared initialized backbone)
        self.encoder.load_state_dict(classifier.encoder.state_dict())
        
        # Classification Head
        self.cls_pool = classifier.avgpool
        self.cls_head = classifier.classifier
        
        # Localization Head
        self.loc_pool = localizer.avgpool
        self.loc_head = localizer.regressor
        
        # Segmentation Decoder
        self.upconv1 = unet.upconv1
        self.dec1 = unet.dec1
        self.upconv2 = unet.upconv2
        self.dec2 = unet.dec2
        self.upconv3 = unet.upconv3
        self.dec3 = unet.dec3
        self.upconv4 = unet.upconv4
        self.dec4 = unet.dec4
        self.upconv5 = unet.upconv5
        self.dec5 = unet.dec5
        self.seg_final = unet.final_conv
        
    def forward(self, x: torch.Tensor):
        bottleneck, f = self.encoder(x, return_features=True)
        
        p = self.cls_pool(bottleneck)
        p = torch.flatten(p, 1)
        
        cls_logits = self.cls_head(p)
        # Note: Localizer inherently outputs [0,1] followed by scale * 224 in its own forward
        # In multitask we must do the scale here since we copied the regressor directly
        loc_coords = self.loc_head(p) * 224.0
        
        d = self.dec1(torch.cat([self.upconv1(bottleneck), f["f5"]], dim=1))
        d = self.dec2(torch.cat([self.upconv2(d), f["f4"]], dim=1))
        d = self.dec3(torch.cat([self.upconv3(d), f["f3"]], dim=1))
        d = self.dec4(torch.cat([self.upconv4(d), f["f2"]], dim=1))
        d = self.dec5(torch.cat([self.upconv5(d), f["f1"]], dim=1))
        
        seg_logits = self.seg_final(d)
        
        return {
            "classification": cls_logits,
            "localization": loc_coords,
            "segmentation": seg_logits
        }
