"""Kaggle training script for DA6401 Assignment 2.
Clones the repo, trains all 3 models, and saves checkpoints to /kaggle/working/.
"""
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/shubhamtiwari1602/da6401-assignment2.git"
REPO_DIR = "/kaggle/working/repo"
OUT_DIR  = "/kaggle/working"

# ── Setup ──────────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)

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
tasks = [
    f"{sys.executable} train.py --task classifier --epochs 30 --batch_size 64 --lr 3e-4 --run_name cls_v1",
    f"{sys.executable} train.py --task localizer  --epochs 30 --batch_size 64 --lr 3e-4 --run_name loc_v1",
    f"{sys.executable} train.py --task unet       --epochs 30 --batch_size 32 --lr 3e-4 --run_name unet_v1",
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
