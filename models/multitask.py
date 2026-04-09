"""Unified multi-task model (models package version)
"""

import os
import torch
import torch.nn as nn


_DRIVE_IDS = {
    "classifier.pth": "1sDZTJ3SVxFUqVxVgXCPzajSEQwVeViKx",
    "localizer.pth":  "151gfnQk97XDx6KJ0wtPkDCOZkAF1Se7r",
    "unet.pth":       "14JQvAPxmd9-UeWKcE6vskYfIfUh5y47L",
}


def _download_gdown(file_id, dest):
    try:
        import gdown
        gdown.download(id=file_id, output=dest, quiet=False, fuzzy=True)
        return os.path.exists(dest) and os.path.getsize(dest) > 1024
    except Exception as e:
        print(f"[CKPT] gdown failed: {e}", flush=True)
        return False


def _download_urllib(file_id, dest):
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
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "checkpoints", name)
    if os.path.exists(alt) and os.path.getsize(alt) > 1024:
        return alt
    drive_id = _DRIVE_IDS.get(name)
    if not drive_id:
        return dest
    for target in [dest, alt, os.path.join("/tmp", "a2ckpts", name)]:
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        except Exception:
            continue
        if _download_gdown(drive_id, target):
            return target
        if _download_urllib(drive_id, target):
            return target
    print(f"[CKPT] Could not obtain {name}", flush=True)
    return dest


def _load_checkpoint(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(sd, dict):
        return None
    for k in sd:
        if isinstance(sd[k], torch.Tensor) and sd[k].dtype == torch.float16:
            sd[k] = sd[k].float()
    return sd


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = None,
                 localizer_path: str = None,
                 unet_path: str = None):
        super().__init__()
        from .vgg11 import VGG11
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet

        if classifier_path is None:
            classifier_path = "checkpoints/classifier.pth"
        if localizer_path is None:
            localizer_path = "checkpoints/localizer.pth"
        if unet_path is None:
            unet_path = "checkpoints/unet.pth"

        os.makedirs("checkpoints", exist_ok=True)

        classifier_path = _ensure_checkpoint("classifier.pth", classifier_path)
        localizer_path  = _ensure_checkpoint("localizer.pth",  localizer_path)
        unet_path       = _ensure_checkpoint("unet.pth",       unet_path)

        # Build complete sub-models (each with their own encoder)
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer  = VGG11Localizer(in_channels=in_channels)
        self.unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load pretrained weights
        for model, sd_path, label in [
            (self.classifier, classifier_path, "Classifier"),
            (self.localizer,  localizer_path,  "Localizer"),
            (self.unet,       unet_path,       "UNet"),
        ]:
            sd = _load_checkpoint(sd_path)
            if sd is not None:
                try:
                    model.load_state_dict(sd)
                    print(f"[CKPT] {label} weights loaded", flush=True)
                except Exception as e:
                    print(f"[CKPT] {label} load_state_dict failed: {e}", flush=True)

    def forward(self, x: torch.Tensor):
        cls_logits = self.classifier(x)
        loc_coords = self.localizer(x)
        seg_logits = self.unet(x)

        return {
            "classification": cls_logits,
            "localization":   loc_coords,
            "segmentation":   seg_logits,
        }
