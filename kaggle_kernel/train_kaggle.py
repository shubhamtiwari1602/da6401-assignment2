"""Kaggle training script for DA6401 Assignment 2.
Installs PyTorch 2.3.1 (supports P100/sm_60), clones the repo,
trains all 3 models, and saves checkpoints to /kaggle/working/.
"""
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/shubhamtiwari1602/da6401-assignment2.git"
REPO_DIR = "/kaggle/working/repo"
OUT_DIR  = "/kaggle/working"

# ── Install compatible PyTorch (2.3.1 supports P100/sm_60) ────────────────
print("=== Installing PyTorch 2.3.1 (P100-compatible) ===", flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.3.1+cu121",
    "torchvision==0.18.1+cu121",
    "--index-url", "https://download.pytorch.org/whl/cu121",
], check=True)

# Verify GPU is now visible
import importlib
import torch
importlib.reload(torch)  # pick up new install
print(f"PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── Install W&B ────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)

# ── Clone repo ─────────────────────────────────────────────────────────────
os.makedirs(REPO_DIR, exist_ok=True)
subprocess.run(
    ["git", "clone", "--depth=1", REPO_URL, REPO_DIR],
    check=True,
)
os.chdir(REPO_DIR)
os.makedirs("checkpoints", exist_ok=True)

# Run W&B in offline mode (no API key needed in kernel)
os.environ["WANDB_MODE"] = "offline"

# ── Training ───────────────────────────────────────────────────────────────
# num_workers=0 avoids shared-memory issues if GPU isn't available
nw = 2 if torch.cuda.is_available() else 0
tasks = [
    f"{sys.executable} train.py --task classifier --epochs 30 --batch_size 64 --lr 3e-4 --run_name cls_v1 --num_workers {nw}",
    f"{sys.executable} train.py --task localizer  --epochs 30 --batch_size 64 --lr 3e-4 --run_name loc_v1 --num_workers {nw}",
    f"{sys.executable} train.py --task unet       --epochs 30 --batch_size 32 --lr 3e-4 --run_name unet_v1 --num_workers {nw}",
]
for cmd in tasks:
    print(f"\n{'='*60}\n{cmd}\n{'='*60}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# ── Copy to Kaggle output dir ───────────────────────────────────────────────
for name in ["classifier.pth", "localizer.pth", "unet.pth"]:
    src = os.path.join(REPO_DIR, "checkpoints", name)
    dst = os.path.join(OUT_DIR, name)
    shutil.copy(src, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"[OUTPUT] {name}: {size_mb:.1f} MB", flush=True)

print("\nAll checkpoints saved to /kaggle/working/", flush=True)
