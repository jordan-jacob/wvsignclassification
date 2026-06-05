#!/usr/bin/env bash
# Run this on a fresh AWS instance after copying the repo.
# Assumes Ubuntu 22.04, Python 3.11, CUDA 12.1 already present.
set -e

REPO="$HOME/wvsignclassification"

# --- S3 restore (optional) ---
# Set S3_BUCKET before running to pull processed data from S3 instead of
# uploading from Windows each time.  Leave empty to skip.
S3_BUCKET=""  # e.g. "my-wv-sign-data"

if [ -n "$S3_BUCKET" ]; then
  echo "=== Restoring data from s3://$S3_BUCKET ==="
  aws s3 sync "s3://$S3_BUCKET/data/processed/" "$REPO/data/processed/" \
      --exclude "*.cache"
  aws s3 sync "s3://$S3_BUCKET/checkpoints/"    "$REPO/checkpoints/"
  aws s3 sync "s3://$S3_BUCKET/configs/"        "$REPO/configs/"
fi

echo "=== Creating directory structure ==="
mkdir -p "$REPO/data/processed/lisa/train/images"
mkdir -p "$REPO/data/processed/lisa/train/labels"
mkdir -p "$REPO/data/processed/lisa_4class/train/labels"
mkdir -p "$REPO/data/processed/mtsd/train/images"
mkdir -p "$REPO/data/processed/mtsd/train/labels"
mkdir -p "$REPO/data/processed/mtsd/val/images"
mkdir -p "$REPO/data/processed/mtsd/val/labels"
mkdir -p "$REPO/data/processed/mtsd_coarse/train/labels"
mkdir -p "$REPO/data/processed/mtsd_coarse/val/labels"
mkdir -p "$REPO/checkpoints"
mkdir -p "$REPO/runs"

echo "=== Creating symlinks (replace Windows junctions) ==="
# lisa_4class images → lisa images (no data copy)
ln -sfn "$REPO/data/processed/lisa/train/images" \
        "$REPO/data/processed/lisa_4class/train/images"
# mtsd_coarse images → mtsd images (no data copy; labels differ)
ln -sfn "$REPO/data/processed/mtsd/train/images" \
        "$REPO/data/processed/mtsd_coarse/train/images"
ln -sfn "$REPO/data/processed/mtsd/val/images" \
        "$REPO/data/processed/mtsd_coarse/val/images"

echo "=== Installing Python packages ==="
pip install --upgrade pip
# PyTorch with CUDA 12.1
pip install torch==2.12.0+cu121 torchvision==0.27.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
# Remaining training deps
pip install -r "$REPO/requirements_aws.txt"

# ── Sparse R-CNN environment (separate from main env) ────────────────────────
# Run with INSTALL_DETECTRON2=1 to also set up the detectron2/SparseR-CNN env.
# This takes ~45 min on a fresh instance. It MUST be a separate conda env —
# detectron2 conflicts with the Ultralytics torch version in this venv.
if [ "${INSTALL_DETECTRON2:-0}" = "1" ]; then
  echo "=== Setting up detectron2_env (Sparse R-CNN) ==="
  conda create -n detectron2_env python=3.10 -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate detectron2_env
  pip install torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu121
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  pip install pycocotools tqdm pillow pyyaml
  # Clone SparseR-CNN adjacent to this repo (required by configs/sparse_rcnn_lisa.yaml)
  cd "$HOME"
  git clone https://github.com/PeizeSun/SparseR-CNN
  cd SparseR-CNN
  pip install -r requirements.txt
  cd "$REPO"
  conda deactivate
  echo ""
  echo "detectron2_env ready. To run Sparse R-CNN:"
  echo "  conda activate detectron2_env"
  echo "  python scripts/coco_convert.py --dataset lisa"
  echo "  python scripts/train_sparse_rcnn.py --smoke   # verify data loads"
  echo "  python scripts/train_sparse_rcnn.py           # full training"
fi

echo "=== Fixing config paths ==="
sed -i "s|C:\\\\Users\\\\jrj00048\\\\Projects\\\\wvsignclassification|$REPO|g" \
    "$REPO/configs/mtsd.yaml" \
    "$REPO/configs/data.yaml"

echo ""
echo "=== Setup complete. Next steps: ==="
if [ -n "$S3_BUCKET" ]; then
  echo "  Data restored from S3. Verify with:"
  echo "    ls data/processed/lisa/train/images | wc -l   # expect ~6618"
  echo "    ls data/processed/mtsd/train/images | wc -l   # expect ~36589"
else
  echo "  1. Either set S3_BUCKET and re-run, OR manually copy:"
  echo "     data/processed/lisa/, data/processed/mtsd/, data/processed/mtsd_coarse/"
  echo "     checkpoints/phase1_mtsd_best.pt"
fi
echo "  See TRAINING_COMMANDS.md for the full run order and exact commands."
