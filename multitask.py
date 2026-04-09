"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

_DRIVE_IDS = {
    "classifier.pth": "1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx",
    "localizer.pth":  "151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r",
    "unet.pth":       "14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L",
}


def _download(name, dest):
    """Try gdown; return True if dest has a valid file afterwards."""
    drive_id = _DRIVE_IDS.get(name)
    if not drive_id:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import gdown
        gdown.download(id=drive_id, output=dest, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"[CKPT] gdown failed for {name}: {e}", flush=True)
    return os.path.exists(dest) and os.path.getsize(dest) > 1024


def _resolve(rel_path):
    """Return a writable path for the checkpoint.

    Priority:
      1. rel_path  (relative, as TA instructs — works when CWD = submission dir)
      2. __file__-based absolute path  (works regardless of CWD)
      3. /tmp/      (works even if submission dir is read-only)
    """
    name = os.path.basename(rel_path)

    # Already present at relative path?
    if os.path.exists(rel_path) and os.path.getsize(rel_path) > 1024:
        print(f"[CKPT] Found at relative path: {rel_path}", flush=True)
        return rel_path

    # Try absolute path next to multitask.py
    abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", name)
    if os.path.exists(abs_path) and os.path.getsize(abs_path) > 1024:
        print(f"[CKPT] Found at absolute path: {abs_path}", flush=True)
        return abs_path

    # Download — prefer relative path, fall back to absolute, then /tmp
    for dest in [rel_path, abs_path, os.path.join("/tmp", "a2ckpts", name)]:
        try:
            parent = os.path.dirname(dest) or "."
            os.makedirs(parent, exist_ok=True)
            # Write-test
            tp = os.path.join(parent, ".wtest")
            open(tp, "w").close()
            os.remove(tp)
        except Exception:
            continue  # directory not writable, try next
        if _download(name, dest):
            return dest

    print(f"[CKPT] Could not obtain {name} — will use random weights", flush=True)
    return rel_path   # return something; _load_checkpoint will return None


def _load_checkpoint(path):
    """Load a state dict from a .pth file, fp16 → fp32."""
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        print(f"[CKPT] Not found or too small: {path}", flush=True)
        return None
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[CKPT] torch.load failed: {e}", flush=True)
        return None
    if not isinstance(sd, dict):
        print(f"[CKPT] Unexpected format in {path}", flush=True)
        return None
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

        # Default relative paths (TA-specified convention)
        if classifier_path is None:
            classifier_path = "checkpoints/classifier.pth"
        if localizer_path is None:
            localizer_path = "checkpoints/localizer.pth"
        if unet_path is None:
            unet_path = "checkpoints/unet.pth"

        # Download from Google Drive if not already present
        import gdown
        os.makedirs("checkpoints", exist_ok=True)
        gdown.download(id="1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx", output=classifier_path, quiet=False)
        gdown.download(id="151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r", output=localizer_path, quiet=False)
        gdown.download(id="14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L", output=unet_path, quiet=False)

        # Resolve to an actually-readable path (handles CWD & permission issues)
        classifier_path = _resolve(classifier_path)
        localizer_path  = _resolve(localizer_path)
        unet_path       = _resolve(unet_path)

        # Build component models
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer  = VGG11Localizer(in_channels=in_channels)
        unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load weights
        cls_sd  = _load_checkpoint(classifier_path)
        loc_sd  = _load_checkpoint(localizer_path)
        unet_sd = _load_checkpoint(unet_path)

        if cls_sd is not None:
            try:
                classifier.load_state_dict(cls_sd)
                print("[CKPT] Classifier weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] Classifier load_state_dict failed: {e}", flush=True)
        if loc_sd is not None:
            try:
                localizer.load_state_dict(loc_sd)
                print("[CKPT] Localizer weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] Localizer load_state_dict failed: {e}", flush=True)
        if unet_sd is not None:
            try:
                unet.load_state_dict(unet_sd)
                print("[CKPT] UNet weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] UNet load_state_dict failed: {e}", flush=True)

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
