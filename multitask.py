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
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        import gdown
        gdown.download(id=drive_id, output=dest, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"[CKPT] gdown failed for {name}: {e}", flush=True)
    return os.path.exists(dest) and os.path.getsize(dest) > 1024


def _resolve(rel_path):
    """Return a readable path for the checkpoint.

    Priority:
      1. rel_path  (relative — works when CWD = submission dir)
      2. __file__-based absolute path
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
    """Shared-backbone multi-task model.

    Each task head is paired with its own trained encoder so that
    the pretrained weights are applied consistently.  A single
    forward pass runs all three sub-models and returns a dict.
    """

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
        if not (os.path.exists(classifier_path) and os.path.getsize(classifier_path) > 1024):
            gdown.download(id="1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx", output=classifier_path, quiet=False)
        if not (os.path.exists(localizer_path) and os.path.getsize(localizer_path) > 1024):
            gdown.download(id="151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r", output=localizer_path, quiet=False)
        if not (os.path.exists(unet_path) and os.path.getsize(unet_path) > 1024):
            gdown.download(id="14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L", output=unet_path, quiet=False)

        # Resolve to actually-readable paths
        classifier_path = _resolve(classifier_path)
        localizer_path  = _resolve(localizer_path)
        unet_path       = _resolve(unet_path)

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
        cls_logits = self.classifier(x)          # [B, num_breeds]
        loc_coords = self.localizer(x)            # [B, 4]  pixel-space (VGG11Localizer already ×224)
        seg_logits = self.unet(x)                 # [B, seg_classes, H, W]

        return {
            "classification": cls_logits,
            "localization":   loc_coords,
            "segmentation":   seg_logits,
        }
