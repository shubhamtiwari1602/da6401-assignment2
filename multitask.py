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
                 classifier_path: str = None,
                 localizer_path: str = None,
                 unet_path: str = None):
        super().__init__()

        # Resolve paths relative to THIS file, not the CWD
        _here = os.path.dirname(os.path.abspath(__file__))
        if classifier_path is None:
            classifier_path = os.path.join(_here, "checkpoints", "classifier.pth")
        if localizer_path is None:
            localizer_path = os.path.join(_here, "checkpoints", "localizer.pth")
        if unet_path is None:
            unet_path = os.path.join(_here, "checkpoints", "unet.pth")

        # If a file is missing or is a tiny LFS pointer (<1KB), try gdown
        def _needs_download(p):
            return not os.path.exists(p) or os.path.getsize(p) < 1024

        try:
            import gdown
            _ids = {
                classifier_path: "1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx",
                localizer_path:  "151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r",
                unet_path:       "14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L",
            }
            for p, drive_id in _ids.items():
                if _needs_download(p):
                    gdown.download(id=drive_id, output=p, quiet=False)
        except Exception:
            pass

        self.encoder = VGG11(in_channels=in_channels)

        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _safe_load(model, path):
            try:
                if os.path.exists(path) and os.path.getsize(path) > 1024:
                    model.load_state_dict(
                        torch.load(path, map_location="cpu", weights_only=False)
                    )
            except Exception:
                pass

        _safe_load(classifier, classifier_path)
        _safe_load(localizer, localizer_path)
        _safe_load(unet, unet_path)

        # Shared backbone from classifier
        try:
            self.encoder.load_state_dict(classifier.encoder.state_dict())
        except Exception:
            pass

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
