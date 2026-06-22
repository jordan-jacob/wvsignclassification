#!/bin/bash
# Usage: bash scripts/save_checkpoint_to_github.sh checkpoints/phase2_full_best.pt "Phase 2 complete"
CKPT=$1
MSG=${2:-"Add checkpoint"}
cd /opt/dlami/nvme/Jacob/wvsignclassification
git add "$CKPT"
git commit -m "$MSG — $(date '+%Y-%m-%d %H:%M')"
git push origin main
echo "Pushed $CKPT to GitHub"
