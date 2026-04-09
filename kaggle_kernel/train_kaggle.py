"""Kaggle training script for DA6401 Assignment 2.
Runs on T4 GPU (sm_75, compatible with Kaggle's installed PyTorch).
Clones the repo, trains all 3 models, saves checkpoints to /kaggle/working/.
"""
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/shubhamtiwari1602/da6401-assignment2.git"
REPO_DIR = "/kaggle/working/repo"
OUT_DIR  = "/kaggle/working"

# ── Verify GPU ─────────────────────────────────────────────────────────────
import torch
print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("WARNING: No GPU — training on CPU (slow)", flush=True)

# ── Install W&B ────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)

# ── Clone repo ─────────────────────────────────────────────────────────────
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
os.makedirs(REPO_DIR, exist_ok=True)
subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
os.makedirs("checkpoints", exist_ok=True)

os.environ["WANDB_MODE"] = "offline"

# ── Training ───────────────────────────────────────────────────────────────
# num_workers=0 avoids "storage not resizable" bug with pin_memory in this Kaggle env
# v3: 80 epochs, SmoothL1+IoU for localizer, CE+Dice for UNet
tasks = [
    f"{sys.executable} train.py --task classifier --epochs 80 --batch_size 64 --lr 3e-4 --run_name cls_v3 --num_workers 0",
    f"{sys.executable} train.py --task localizer  --epochs 80 --batch_size 64 --lr 3e-4 --run_name loc_v3 --num_workers 0",
    f"{sys.executable} train.py --task unet       --epochs 80 --batch_size 32 --lr 3e-4 --run_name unet_v3 --num_workers 0",
]
for cmd in tasks:
    print(f"\n{'='*60}\n{cmd}\n{'='*60}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# ── Copy outputs ───────────────────────────────────────────────────────────
for name in ["classifier.pth", "localizer.pth", "unet.pth"]:
    src = os.path.join(REPO_DIR, "checkpoints", name)
    dst = os.path.join(OUT_DIR, name)
    shutil.copy(src, dst)
    print(f"[OUTPUT] {name}: {os.path.getsize(dst)/1e6:.1f} MB", flush=True)

print("\nDone. All checkpoints in /kaggle/working/", flush=True)
