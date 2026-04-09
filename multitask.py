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


def _download_gdown(file_id, dest):
    """Download via gdown."""
    try:
        import gdown
        gdown.download(id=file_id, output=dest, quiet=False, fuzzy=True)
        return os.path.exists(dest) and os.path.getsize(dest) > 1024
    except Exception as e:
        print(f"[CKPT] gdown failed: {e}", flush=True)
        return False


def _download_urllib(file_id, dest):
    """Fallback download via urllib (no extra deps)."""
    try:
        import urllib.request
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        print(f"[CKPT] urllib fallback for {os.path.basename(dest)}...", flush=True)
        urllib.request.urlretrieve(url, dest)
        return os.path.exists(dest) and os.path.getsize(dest) > 1024
    except Exception as e:
        print(f"[CKPT] urllib failed: {e}", flush=True)
        return False


def _ensure_checkpoint(name, dest):
    """Make sure checkpoint exists at dest; download if needed."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"[CKPT] Found: {dest}", flush=True)
        return dest

    # Also check next to this script
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", name)
    if os.path.exists(alt) and os.path.getsize(alt) > 1024:
        print(f"[CKPT] Found at alt path: {alt}", flush=True)
        return alt

    drive_id = _DRIVE_IDS.get(name)
    if not drive_id:
        return dest

    # Try downloading to multiple locations
    for target in [dest, alt, os.path.join("/tmp", "a2ckpts", name)]:
        try:
            parent = os.path.dirname(target) or "."
            os.makedirs(parent, exist_ok=True)
        except Exception:
            continue

        if _download_gdown(drive_id, target):
            return target
        if _download_urllib(drive_id, target):
            return target

    print(f"[CKPT] Could not obtain {name} — will use random weights", flush=True)
    return dest


def _load_checkpoint(path):
    """Load a state dict from a .pth file, fp16 → fp32."""
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        print(f"[CKPT] Not found or too small: {path}", flush=True)
        return None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
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
    """Shared-backbone multi-task model.

    Each task head keeps its own trained encoder so that
    the pretrained weights are applied consistently.
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = None,
                 localizer_path: str = None,
                 unet_path: str = None):
        super().__init__()

        # Default relative paths (TA convention)
        if classifier_path is None:
            classifier_path = "checkpoints/classifier.pth"
        if localizer_path is None:
            localizer_path = "checkpoints/localizer.pth"
        if unet_path is None:
            unet_path = "checkpoints/unet.pth"

        os.makedirs("checkpoints", exist_ok=True)

        # Ensure checkpoints are available (download if needed)
        classifier_path = _ensure_checkpoint("classifier.pth", classifier_path)
        localizer_path  = _ensure_checkpoint("localizer.pth",  localizer_path)
        unet_path       = _ensure_checkpoint("unet.pth",       unet_path)

        # Build the three sub-models
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer  = VGG11Localizer(in_channels=in_channels)
        self.unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load weights into each sub-model independently
        cls_sd  = _load_checkpoint(classifier_path)
        loc_sd  = _load_checkpoint(localizer_path)
        unet_sd = _load_checkpoint(unet_path)

        if cls_sd is not None:
            try:
                self.classifier.load_state_dict(cls_sd)
                print("[CKPT] Classifier weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] Classifier load_state_dict failed: {e}", flush=True)

        if loc_sd is not None:
            try:
                self.localizer.load_state_dict(loc_sd)
                print("[CKPT] Localizer weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] Localizer load_state_dict failed: {e}", flush=True)

        if unet_sd is not None:
            try:
                self.unet.load_state_dict(unet_sd)
                print("[CKPT] UNet weights loaded", flush=True)
            except Exception as e:
                print(f"[CKPT] UNet load_state_dict failed: {e}", flush=True)

    def forward(self, x: torch.Tensor):
        """Single forward pass returning all three task outputs.

        Returns:
            dict with keys:
              'classification' : [B, num_breeds] logits
              'localization'   : [B, 4] box in pixel space [cx, cy, w, h]
              'segmentation'   : [B, seg_classes, H, W] logits
        """
        cls_logits = self.classifier(x)
        loc_coords = self.localizer(x)
        seg_logits = self.unet(x)

        return {
            "classification": cls_logits,
            "localization":   loc_coords,
            "segmentation":   seg_logits,
        }
