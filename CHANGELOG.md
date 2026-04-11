# Autograder Fix Changelog

## Score Progression

| Version | Score | Date | Key Change |
|---------|-------|------|-----------|
| v1 (original) | 20/50 | Apr 8 | Only arch/dropout/IoU tests pass |
| v2 | 33/50 | Apr 9 | Fixed shared-backbone bug in `multitask.py` |
| v3 | 44/50 | Apr 9 | Retrained with fixed training pipeline (no RandomCrop, fp32, 50 epochs) |
| v4 (pending) | TBD | Apr 9 | Wider localizer, SmoothL1+IoU, Dice+CE, 80 epochs |

---

## v2: Fix Shared-Backbone Bug (20 → 33 pts)

### Problem

`MultiTaskPerceptionModel.forward()` routed all three task heads through a **single shared encoder** (the classifier's backbone). But each sub-model was trained independently with its **own encoder** — so feeding the localizer head and UNet decoder features from the wrong encoder produced garbage:

- **Classification F1 = 0.0000** → random predictions
- **Localization Acc@IoU=0.5 = 0.0%** → mismatched features
- **Segmentation Dice = 0.16** → near-random mask predictions

### Fix — `multitask.py` (top-level) and `models/multitask.py`

Rewrote `MultiTaskPerceptionModel` to keep all three sub-models fully intact, each with their own trained encoder:

```python
# Before (broken): shared encoder, decomposed heads
self.encoder = classifier.encoder          # only classifier's encoder
self.cls_head = classifier.classifier
self.loc_head = localizer.regressor        # gets wrong features!
# ...decoder layers from unet...          # gets wrong features!

# After (fixed): independent sub-models
self.classifier = VGG11Classifier(...)     # own encoder
self.localizer  = VGG11Localizer(...)      # own encoder
self.unet       = VGG11UNet(...)           # own encoder

def forward(self, x):
    return {
        "classification": self.classifier(x),
        "localization":   self.localizer(x),
        "segmentation":   self.unet(x),
    }
```

### Additional Fixes

- **Removed hard `import gdown` dependency** — wrapped in try/except with `urllib` fallback, so the model doesn't crash if gdown is unavailable on the autograder
- **Added `weights_only=False`** to `torch.load()` — newer PyTorch versions default to `True`, which refuses to load checkpoints containing non-tensor objects

### Tests Newly Passed

| Test | Score | Metric |
|------|-------|--------|
| 4.1a Classification F1 > 0.3 | 5/5 | F1 = 0.3889 |
| 4.3a Segmentation Dice > 0.3 | 5/5 | Dice = 0.5563 |
| 4.3b Segmentation Dice > 0.5 | 3/3 | Dice = 0.5563 |

---

## v3: Fix Training Pipeline (33 → 44 pts)

### Root Cause: `RandomCrop` Breaks Bbox & Mask Alignment

The training transform did `Resize(256) → RandomCrop(224)`, but:

1. **Bbox targets** were computed relative to a direct 224×224 resize (in `pets_dataset.py`). The random crop offset was **never subtracted** from the bbox coordinates → localizer trained on misaligned image-bbox pairs
2. **Segmentation masks** were resized to 224×224 independently, not cropped in sync with the image → UNet trained on misaligned image-mask pairs

Training metrics confirmed the problem:
- Classifier val_acc = **32.4%** → Localizer val_loss = **7989** → UNet val_dice = **0.5511**

### Changes in `train.py`

| Setting | Before | After |
|---------|--------|-------|
| Train transform | `Resize(256)+RandomCrop(224)` | `Resize(224,224)` |
| Weight format | fp16 via `save_fp16()` | fp32 via `torch.save()` |
| Localizer MSE | Raw pixel-space (loss ~8000) | Normalized `preds/224, boxes/224` |
| Optimizer | `Adam` | `AdamW` + gradient clipping |
| Classifier loss | Plain CE | CE + label smoothing (0.1) |
| Default epochs | 30 | 50 |

### Tests Newly Passed

| Test | Score | Metric |
|------|-------|--------|
| 4.1b Classification F1 > 0.5 | 3/3 | F1 = 0.7333 |
| 4.2a Localization IoU@0.5 > 60% | 5/5 | Acc = 70.0% |

---

## v4: Targeted Model Improvements (retraining in progress)

### Remaining Failures After v3

| Test | Required | Actual | Gap |
|------|----------|--------|-----|
| 4.1c F1 > 0.8 | 0.80 | 0.73 | -0.07 |
| 4.2b IoU@0.75 > 40% | 40% | 20% | -20% |
| 4.3c Dice > 0.8 | 0.80 | 0.77 | -0.04 |

### Changes

#### Localizer — `models/localization.py`

Wider regressor head to improve box precision:

```python
# Before: narrow 64-neuron bottleneck
nn.Linear(512, 256) → nn.Linear(256, 64) → nn.Linear(64, 4)

# After: wider with dropout regularization
nn.Linear(512, 512) → CustomDropout(0.3) → nn.Linear(512, 256) → nn.Linear(256, 4)
```

#### Training — `train.py`

| Setting | v3 | v4 |
|---------|-----|-----|
| Localizer loss | MSE + IoU | **SmoothL1 + 2×IoU** (emphasize IoU) |
| UNet loss | CE only | **CE + Soft Dice** (directly optimizes metric) |
| Augmentation | HFlip + ColorJitter | + **RandomRotation(10°)** |
| Encoder unfreeze | At 1/2 epochs | At **1/3** (more fine-tuning time) |
| Epochs | 50 | **80** |

#### Inference — `multitask.py`

Added **Test-Time Augmentation** (horizontal flip + average) for all 3 tasks.

### Retraining

Kaggle kernel v7 running on T4 GPU — expected ~3-4 hours.

---

## Files Modified

| File | Changes |
|------|---------|
| `multitask.py` | Independent sub-models, urllib fallback, weights_only=False, TTA |
| `models/multitask.py` | Same as above (both import paths covered) |
| `models/localization.py` | Wider regressor head (512→512→256→4 + dropout) |
| `train.py` | RandomCrop fix, fp32, SmoothL1+2×IoU, CE+Dice, AdamW, 80 epochs |
| `kaggle_kernel/train_kaggle.py` | Updated to v3 (80 epochs) |
