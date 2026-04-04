"""Unified multi-task model
"""

import os
import sys
import io
import glob
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def _log(msg):
    print(f"[CKPT] {msg}", file=sys.stderr, flush=True)


def _find_checkpoint_dir():
    """Search multiple candidate locations for the checkpoints directory."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
        "/autograder/source/checkpoints",
        "/autograder/submission/checkpoints",
        os.path.join(os.getcwd(), "checkpoints"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            contents = os.listdir(d)
            _log(f"Found dir {d} -> {contents}")
            # Check if it has actual checkpoint data (not just metadata)
            if any(f.endswith('.pth') or '.part_' in f for f in contents):
                return d
        else:
            _log(f"Not found: {d}")
    # Fall back to first candidate
    _log(f"Falling back to {candidates[0]}")
    return candidates[0]


def _load_checkpoint(path):
    """Load a checkpoint, handling split files and fp16->fp32 conversion."""
    _log(f"Loading: {path}")

    # Check for split parts first (e.g. classifier.pth.part_aa, part_ab, ...)
    parts = sorted(glob.glob(path + ".part_*"))
    _log(f"  Split parts found: {len(parts)} {parts}")

    if parts:
        buf = bytearray()
        for p in parts:
            sz = os.path.getsize(p)
            _log(f"  Reading part {os.path.basename(p)} ({sz} bytes)")
            with open(p, "rb") as f:
                buf.extend(f.read())
        _log(f"  Total buffer: {len(buf)} bytes")
        sd = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)
    elif os.path.exists(path):
        sz = os.path.getsize(path)
        _log(f"  Single file exists: {sz} bytes")
        if sz > 1024:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        else:
            _log(f"  TOO SMALL ({sz} bytes) - likely LFS pointer, skipping")
            return None
    else:
        _log(f"  FILE NOT FOUND: {path}")
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            _log(f"  Dir contents: {os.listdir(parent)}")
        else:
            _log(f"  Parent dir doesn't exist: {parent}")
        return None

    # Convert fp16 tensors back to fp32
    fp16_count = 0
    for k in sd:
        if isinstance(sd[k], torch.Tensor) and sd[k].dtype == torch.float16:
            sd[k] = sd[k].float()
            fp16_count += 1
    _log(f"  Loaded {len(sd)} keys, converted {fp16_count} fp16->fp32")
    return sd


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = None,
                 localizer_path: str = None,
                 unet_path: str = None):
        super().__init__()

        # Find checkpoints directory
        ckpt_dir = _find_checkpoint_dir()
        _log(f"Using checkpoint dir: {ckpt_dir}")

        if classifier_path is None:
            classifier_path = os.path.join(ckpt_dir, "classifier.pth")
        if localizer_path is None:
            localizer_path = os.path.join(ckpt_dir, "localizer.pth")
        if unet_path is None:
            unet_path = os.path.join(ckpt_dir, "unet.pth")

        self.encoder = VGG11(in_channels=in_channels)

        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        try:
            cls_sd = _load_checkpoint(classifier_path)
            if cls_sd is not None:
                classifier.load_state_dict(cls_sd)
                _log("Classifier loaded OK")
            else:
                _log("WARNING: Classifier checkpoint not loaded")
        except Exception as e:
            _log(f"ERROR loading classifier: {e}")

        try:
            loc_sd = _load_checkpoint(localizer_path)
            if loc_sd is not None:
                localizer.load_state_dict(loc_sd)
                _log("Localizer loaded OK")
            else:
                _log("WARNING: Localizer checkpoint not loaded")
        except Exception as e:
            _log(f"ERROR loading localizer: {e}")

        try:
            unet_sd = _load_checkpoint(unet_path)
            if unet_sd is not None:
                unet.load_state_dict(unet_sd)
                _log("UNet loaded OK")
            else:
                _log("WARNING: UNet checkpoint not loaded")
        except Exception as e:
            _log(f"ERROR loading unet: {e}")

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
