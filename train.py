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

# Training transform: NO RandomCrop (breaks bbox/mask alignment)
TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

# Validation/test transform
VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


# ──────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────

def train_classifier(args, device):
    train_ds = OxfordIIITPetDataset(root="./data", split="trainval", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = OxfordIIITPetDataset(root="./data", split="test",     download=True, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            torch.save(model.state_dict(), "checkpoints/classifier.pth")
            print(f"  Saved classifier.pth (val_acc={val_acc:.4f})")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # Load pretrained encoder from classifier if available
    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        cls_sd = torch.load(cls_ckpt, map_location=device, weights_only=False)
        encoder_sd = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        # Handle fp16 → fp32 for older checkpoints
        for k in encoder_sd:
            if isinstance(encoder_sd[k], torch.Tensor) and encoder_sd[k].dtype == torch.float16:
                encoder_sd[k] = encoder_sd[k].float()
        model.encoder.load_state_dict(encoder_sd)
        print("Loaded pretrained encoder from classifier.pth")

        # Freeze encoder for first half of training, then unfreeze
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_fn    = nn.MSELoss()
    iou_fn    = IoULoss()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Unfreeze encoder halfway through
        if epoch == args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            print("  Unfroze encoder for fine-tuning")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Localizer] Epoch {epoch+1}/{args.epochs}"):
            imgs   = batch["image"].to(device)
            boxes  = batch["bbox_target"].to(device)
            optimizer.zero_grad()
            preds  = model(imgs)
            # Normalize coords to [0,1] for MSE to keep scale reasonable
            mse_loss = mse_fn(preds / 224.0, boxes / 224.0)
            iou_loss = iou_fn(preds, boxes)
            loss = mse_loss + iou_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        val_loss, val_iou = _eval_localizer(model, val_loader, mse_fn, iou_fn, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        wandb.log({"epoch": epoch+1, "loc/train_loss": train_loss/len(train_loader),
                   "loc/val_loss": val_loss, "loc/val_iou": val_iou})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  Saved localizer.pth (val_loss={val_loss:.4f})")

    print(f"Best localizer val_loss: {best_val_loss:.4f}")


def _eval_localizer(model, loader, mse_fn, iou_fn, device):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            boxes = batch["bbox_target"].to(device)
            preds = model(imgs)
            mse_loss = mse_fn(preds / 224.0, boxes / 224.0)
            iou_loss = iou_fn(preds, boxes)
            total_loss += (mse_loss + iou_loss).item()
            # Compute actual IoU for logging
            total_iou += (1.0 - iou_loss.item())
            n += 1
    return total_loss / len(loader), total_iou / n


# ──────────────────────────────────────────────────────────────
# U-Net Segmentation
# ──────────────────────────────────────────────────────────────

def train_unet(args, device):
    train_ds = OxfordIIITPetDataset(root="./data", split="trainval", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = OxfordIIITPetDataset(root="./data", split="test",     download=True, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Load pretrained encoder from classifier if available
    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        cls_sd = torch.load(cls_ckpt, map_location=device, weights_only=False)
        encoder_sd = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        for k in encoder_sd:
            if isinstance(encoder_sd[k], torch.Tensor) and encoder_sd[k].dtype == torch.float16:
                encoder_sd[k] = encoder_sd[k].float()
        model.encoder.load_state_dict(encoder_sd)
        print("Loaded pretrained encoder from classifier.pth")

        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_fn     = nn.CrossEntropyLoss()

    best_val_dice = 0.0
    for epoch in range(args.epochs):
        if epoch == args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        val_loss, val_dice = _eval_unet(model, val_loader, ce_fn, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        wandb.log({"epoch": epoch+1, "seg/train_loss": train_loss/len(train_loader),
                   "seg/val_loss": val_loss, "seg/val_dice": val_dice})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "checkpoints/unet.pth")
            print(f"  Saved unet.pth (val_dice={val_dice:.4f})")

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
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dropout_p",  type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
