#!/usr/bin/env bash
# Run this on a fresh AWS instance after copying the repo.
# Assumes Ubuntu 22.04, Python 3.11, CUDA 12.1 already present.
set -e

REPO="$HOME/wvsignclassification"

echo "=== Creating directory structure ==="
mkdir -p "$REPO/data/processed/lisa/train/images"
mkdir -p "$REPO/data/processed/lisa/train/labels"
mkdir -p "$REPO/data/processed/lisa_4class/train/labels"
mkdir -p "$REPO/data/processed/mtsd/train/images"
mkdir -p "$REPO/data/processed/mtsd/train/labels"
mkdir -p "$REPO/data/processed/mtsd/val/images"
mkdir -p "$REPO/data/processed/mtsd/val/labels"
mkdir -p "$REPO/checkpoints"
mkdir -p "$REPO/runs"

echo "=== Creating lisa_4class images symlink ==="
ln -sf "$REPO/data/processed/lisa/train/images" \
       "$REPO/data/processed/lisa_4class/train/images"

echo "=== Installing Python packages ==="
pip install --upgrade pip
# PyTorch with CUDA 12.1
pip install torch==2.12.0+cu121 torchvision==0.27.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
# Remaining training deps
pip install -r "$REPO/requirements_aws.txt"

echo "=== Fixing config paths ==="
sed -i "s|C:\\\\Users\\\\jrj00048\\\\Projects\\\\wvsignclassification|$REPO|g" \
    "$REPO/configs/mtsd.yaml" \
    "$REPO/configs/data.yaml"

echo ""
echo "=== Setup complete. Next steps: ==="
echo "  1. Copy processed data into $REPO/data/processed/"
echo "     (lisa/, lisa_4class/train/labels/, mtsd/)"
echo "  2. Copy phase1_mtsd_best.pt into $REPO/checkpoints/"
echo "  3. Fix _junction() in train_phase2_lisa.py (see checklist)"
echo "  4. Run:  python scripts/train_phase2_lisa.py --batch 32"
