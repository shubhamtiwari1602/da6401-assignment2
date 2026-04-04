"""Training entrypoint — separate per-task training
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import torchvision.transforms as T

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset


# Standard ImageNet stats
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Training transform: augmentation to reduce overfitting
TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

# Validation/test transform: no augmentation
VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


def save_fp16(state_dict, path):
    """Save state dict in float16 to keep file size small (~half of fp32)."""
    fp16_sd = {k: (v.half() if v.is_floating_point() else v) for k, v in state_dict.items()}
    torch.save(fp16_sd, path)


# ──────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────

def train_classifier(args, device):
    train_ds = OxfordIIITPetDataset(root="./data", split="trainval", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = OxfordIIITPetDataset(root="./data", split="test",     download=True, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"[Classifier] Epoch {epoch+1}/{args.epochs}"):
            imgs   = batch["image"].to(device)
            labels = batch["class_label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        scheduler.step()

        train_acc = correct / total
        val_loss, val_acc = _eval_classifier(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        wandb.log({"epoch": epoch+1, "cls/train_loss": train_loss/len(train_loader),
                   "cls/train_acc": train_acc, "cls/val_loss": val_loss, "cls/val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_fp16(model.state_dict(), "checkpoints/classifier.pth")
            print(f"  Saved classifier.pth fp16 (val_acc={val_acc:.4f})")

    print(f"Best classifier val_acc: {best_val_acc:.4f}")


def _eval_classifier(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["class_label"].to(device)
            logits = model(imgs)
            total_loss += loss_fn(logits, labels).item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


# ──────────────────────────────────────────────────────────────
# Localizer
# ──────────────────────────────────────────────────────────────

def train_localizer(args, device):
    train_ds = OxfordIIITPetDataset(root="./data", split="trainval", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = OxfordIIITPetDataset(root="./data", split="test",     download=True, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # Load pretrained encoder from classifier if available
    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        cls_sd = torch.load(cls_ckpt, map_location=device, weights_only=False)
        encoder_sd = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_sd)
        print("Loaded pretrained encoder from classifier.pth")

        # Freeze encoder for first half of training, then unfreeze
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_fn    = nn.MSELoss()
    iou_fn    = IoULoss()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Unfreeze encoder halfway through
        if epoch == args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            print("  Unfroze encoder for fine-tuning")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Localizer] Epoch {epoch+1}/{args.epochs}"):
            imgs   = batch["image"].to(device)
            boxes  = batch["bbox_target"].to(device)
            optimizer.zero_grad()
            preds  = model(imgs)
            loss   = mse_fn(preds, boxes) + iou_fn(preds, boxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        val_loss = _eval_localizer(model, val_loader, mse_fn, iou_fn, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} | val_loss={val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "loc/train_loss": train_loss/len(train_loader), "loc/val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_fp16(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  Saved localizer.pth fp16 (val_loss={val_loss:.4f})")

    print(f"Best localizer val_loss: {best_val_loss:.4f}")


def _eval_localizer(model, loader, mse_fn, iou_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            boxes = batch["bbox_target"].to(device)
            preds = model(imgs)
            total_loss += (mse_fn(preds, boxes) + iou_fn(preds, boxes)).item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────
# U-Net Segmentation
# ──────────────────────────────────────────────────────────────

def train_unet(args, device):
    train_ds = OxfordIIITPetDataset(root="./data", split="trainval", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = OxfordIIITPetDataset(root="./data", split="test",     download=True, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Load pretrained encoder from classifier if available
    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        cls_sd = torch.load(cls_ckpt, map_location=device, weights_only=False)
        encoder_sd = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_sd)
        print("Loaded pretrained encoder from classifier.pth")

        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_fn     = nn.CrossEntropyLoss()

    best_val_dice = 0.0
    for epoch in range(args.epochs):
        if epoch == args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            print("  Unfroze encoder for fine-tuning")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[UNet] Epoch {epoch+1}/{args.epochs}"):
            imgs  = batch["image"].to(device)
            masks = batch["segmentation_mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        val_loss, val_dice = _eval_unet(model, val_loader, ce_fn, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        wandb.log({"epoch": epoch+1, "seg/train_loss": train_loss/len(train_loader),
                   "seg/val_loss": val_loss, "seg/val_dice": val_dice})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_fp16(model.state_dict(), "checkpoints/unet.pth")
            print(f"  Saved unet.pth fp16 (val_dice={val_dice:.4f})")

    print(f"Best unet val_dice: {best_val_dice:.4f}")


def _eval_unet(model, loader, ce_fn, device):
    model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["segmentation_mask"].to(device)
            logits = model(imgs)
            total_loss += ce_fn(logits, masks).item()
            preds = logits.argmax(1)
            for cls in range(3):
                tp = ((preds == cls) & (masks == cls)).sum().float()
                fp = ((preds == cls) & (masks != cls)).sum().float()
                fn = ((preds != cls) & (masks == cls)).sum().float()
                total_dice += (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
            n += 1
    return total_loss / len(loader), total_dice / (n * 3)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(project="DA6401-Assignment2", name=args.run_name, config=vars(args))

    if args.task in ("classifier", "all"):
        print("\n=== Training Classifier ===")
        train_classifier(args, device)

    if args.task in ("localizer", "all"):
        print("\n=== Training Localizer ===")
        train_localizer(args, device)

    if args.task in ("unet", "all"):
        print("\n=== Training U-Net ===")
        train_unet(args, device)

    wandb.finish()
    print("\nDone. Checkpoints saved to checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=str, default="all",
                        choices=["classifier", "localizer", "unet", "all"],
                        help="Which model to train")
    parser.add_argument("--run_name",   type=str, default="full_pipeline")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dropout_p",  type=float, default=0.5)
    args = parser.parse_args()
    main(args)
