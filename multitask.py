"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

# Absolute path to the directory containing this file — works regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))

_DRIVE_IDS = {
    "classifier.pth": "1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx",
    "localizer.pth":  "151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r",
    "unet.pth":       "14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L",
}


def _ensure_checkpoint(name: str, dest_path: str) -> None:
    """Download checkpoint from Google Drive if not already present or too small."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"[CKPT] Already present: {dest_path}", flush=True)
        return

    drive_id = _DRIVE_IDS.get(os.path.basename(dest_path))
    if drive_id is None:
        print(f"[CKPT] No Drive ID for {name}", flush=True)
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[CKPT] Downloading {name} from Google Drive ...", flush=True)
    try:
        import gdown
        gdown.download(id=drive_id, output=dest_path, quiet=False, fuzzy=True)
        size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        print(f"[CKPT] Downloaded {size/1e6:.1f} MB → {dest_path}", flush=True)
    except Exception as e:
        print(f"[CKPT] Download failed for {name}: {e}", flush=True)


def _load_checkpoint(path):
    """Load a state dict from a .pth file, converting fp16 tensors to fp32."""
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        print(f"[CKPT] Not found or too small: {path}", flush=True)
        return None

    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[CKPT] torch.load failed for {path}: {e}", flush=True)
        return None

    if not isinstance(sd, dict):
        print(f"[CKPT] Unexpected format in {path}", flush=True)
        return None

    # Convert fp16 → fp32
    for k in sd:
        if isinstance(sd[k], torch.Tensor) and sd[k].dtype == torch.float16:
            sd[k] = sd[k].float()
    print(f"[CKPT] Loaded {len(sd)} keys from {os.path.basename(path)}", flush=True)
    return sd


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = None,
                 localizer_path: str = None,
                 unet_path: str = None):
        super().__init__()

        # Use absolute paths relative to this file — immune to CWD differences
        ckpt_dir = os.path.join(_HERE, "checkpoints")
        if classifier_path is None:
            classifier_path = os.path.join(ckpt_dir, "classifier.pth")
        if localizer_path is None:
            localizer_path = os.path.join(ckpt_dir, "localizer.pth")
        if unet_path is None:
            unet_path = os.path.join(ckpt_dir, "unet.pth")

        # Download checkpoints from Google Drive if needed
        _ensure_checkpoint("classifier.pth", classifier_path)
        _ensure_checkpoint("localizer.pth",  localizer_path)
        _ensure_checkpoint("unet.pth",       unet_path)

        # Build component models
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer  = VGG11Localizer(in_channels=in_channels)
        unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load weights
        cls_sd  = _load_checkpoint(classifier_path)
        loc_sd  = _load_checkpoint(localizer_path)
        unet_sd = _load_checkpoint(unet_path)

        if cls_sd is not None:
            classifier.load_state_dict(cls_sd)
            print("[CKPT] Classifier weights loaded", flush=True)
        if loc_sd is not None:
            localizer.load_state_dict(loc_sd)
            print("[CKPT] Localizer weights loaded", flush=True)
        if unet_sd is not None:
            unet.load_state_dict(unet_sd)
            print("[CKPT] UNet weights loaded", flush=True)

        # Shared encoder (from trained classifier)
        self.encoder = classifier.encoder

        # Classification head (GAP + linear)
        self.cls_pool = classifier.avgpool
        self.cls_head = classifier.classifier

        # Localization head
        self.loc_head = localizer.regressor

        # Segmentation decoder
        self.upconv1  = unet.upconv1
        self.dec1     = unet.dec1
        self.upconv2  = unet.upconv2
        self.dec2     = unet.dec2
        self.upconv3  = unet.upconv3
        self.dec3     = unet.dec3
        self.upconv4  = unet.upconv4
        self.dec4     = unet.dec4
        self.upconv5  = unet.upconv5
        self.dec5     = unet.dec5
        self.seg_final = unet.final_conv

    def forward(self, x: torch.Tensor):
        # Shared encoder
        bottleneck, f = self.encoder(x, return_features=True)

        # GAP → 512-dim feature vector
        p = self.cls_pool(bottleneck)   # [B, 512, 1, 1]
        p = torch.flatten(p, 1)         # [B, 512]

        cls_logits = self.cls_head(p)
        loc_coords = self.loc_head(p) * 224.0

        # Segmentation decoder with skip connections
        d = self.dec1(torch.cat([self.upconv1(bottleneck), f["f5"]], dim=1))
        d = self.dec2(torch.cat([self.upconv2(d), f["f4"]], dim=1))
        d = self.dec3(torch.cat([self.upconv3(d), f["f3"]], dim=1))
        d = self.dec4(torch.cat([self.upconv4(d), f["f2"]], dim=1))
        d = self.dec5(torch.cat([self.upconv5(d), f["f1"]], dim=1))
        seg_logits = self.seg_final(d)

        return {
            "classification": cls_logits,
            "localization":   loc_coords,
            "segmentation":   seg_logits,
        }
