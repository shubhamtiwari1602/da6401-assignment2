"""Training entrypoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import os

from multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset

def train_epoch(model, dataloader, optimizer, metrics_history, device, alpha=1.0, beta=10.0, gamma=1.0):
    model.train()
    running_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    iou_loss_fn = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Training"):
        imgs = batch["image"].to(device)
        cls_targets = batch["class_label"].to(device)
        bbox_targets = batch["bbox_target"].to(device)
        seg_targets = batch["segmentation_mask"].to(device)
        
        optimizer.zero_grad()
        out = model(imgs)
        
        loss_cls = ce_loss_fn(out["classification"], cls_targets)
        loss_box = mse_loss_fn(out["localization"], bbox_targets) + iou_loss_fn(out["localization"], bbox_targets)
        loss_seg = seg_loss_fn(out["segmentation"], seg_targets)
        
        total_loss = alpha * loss_cls + beta * loss_box + gamma * loss_seg
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        wandb.log({
            "train/loss": total_loss.item(),
            "train/loss_cls": loss_cls.item(),
            "train/loss_box": loss_box.item(),
            "train/loss_seg": loss_seg.item()
        })
        
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    iou_loss_fn = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            imgs = batch["image"].to(device)
            cls_targets = batch["class_label"].to(device)
            bbox_targets = batch["bbox_target"].to(device)
            seg_targets = batch["segmentation_mask"].to(device)
            
            out = model(imgs)
            
            loss_cls = ce_loss_fn(out["classification"], cls_targets)
            loss_box = mse_loss_fn(out["localization"], bbox_targets) + iou_loss_fn(out["localization"], bbox_targets)
            loss_seg = seg_loss_fn(out["segmentation"], seg_targets)
            
            total_loss = loss_cls + 10.0 * loss_box + loss_seg
            running_loss += total_loss.item()
            
            wandb.log({
                "val/loss": total_loss.item(),
                "val/loss_cls": loss_cls.item(),
                "val/loss_box": loss_box.item(),
                "val/loss_seg": loss_seg.item()
            })
            
    return running_loss / len(dataloader)

def save_individual_checkpoints(model, save_dir="checkpoints"):
    """Extract and save individual model checkpoints from a trained multitask model.

    Remaps multitask state-dict keys to match the standalone model formats
    expected by VGG11Classifier, VGG11Localizer, and VGG11UNet.
    """
    sd = model.state_dict()

    cls_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            cls_sd[k] = v
        elif k.startswith("cls_pool."):
            cls_sd[k.replace("cls_pool.", "avgpool.")] = v
        elif k.startswith("cls_head."):
            cls_sd[k.replace("cls_head.", "classifier.")] = v
    torch.save(cls_sd, os.path.join(save_dir, "classifier.pth"))

    loc_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            loc_sd[k] = v
        elif k.startswith("loc_pool."):
            loc_sd[k.replace("loc_pool.", "avgpool.")] = v
        elif k.startswith("loc_head."):
            loc_sd[k.replace("loc_head.", "regressor.")] = v
    torch.save(loc_sd, os.path.join(save_dir, "localizer.pth"))

    unet_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder.") or k.startswith("upconv") or k.startswith("dec"):
            unet_sd[k] = v
        elif k.startswith("seg_final."):
            unet_sd[k.replace("seg_final.", "final_conv.")] = v
    torch.save(unet_sd, os.path.join(save_dir, "unet.pth"))

    print(f"Individual checkpoints saved to {save_dir}/")


def main(args):
    wandb.init(project="DA6401-Assignment2", name=args.run_name, config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = OxfordIIITPetDataset(root="./data", split="trainval", download=True)
    val_dataset = OxfordIIITPetDataset(root="./data", split="test", download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = MultiTaskPerceptionModel().to(device)
    
    # Optional logic for fine-tuning experiment
    if args.fine_tune_strategy == "strict":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.fine_tune_strategy == "partial":
        for name, param in model.encoder.named_parameters():
            if "block5" in name or "block4" in name:
                param.requires_grad = True # unfreeze later
            else:
                param.requires_grad = False
                
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, {}, device)
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_epoch_loss": train_loss, "val_epoch_loss": val_loss})
        
        # Save full multitask checkpoint
        torch.save(model.state_dict(), f"checkpoints/{args.run_name}_latest.pth")

    # Save individual checkpoints for submission
    save_individual_checkpoints(model, save_dir="checkpoints")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--fine_tune_strategy", type=str, choices=["none", "strict", "partial", "full"], default="none")
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    main(args)