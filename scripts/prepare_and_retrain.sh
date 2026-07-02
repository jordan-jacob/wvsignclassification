#!/usr/bin/env bash
# prepare_and_retrain.sh
# Run on AWS after wv_merged_new has been transferred.
# Updates CLASS_REMAP in prepare_wv_data.py, swaps in the new merged dataset,
# runs preflight, then retrains Phase 3 (production + augmented).
#
# Usage:
#   bash scripts/prepare_and_retrain.sh

set -euo pipefail
REPO="/opt/dlami/nvme/Jacob/wvsignclassification"
cd "$REPO"
source /home/ubuntu/Jacob/Jacob/bin/activate

echo "=== Step 1: Verify wv_merged_new was transferred ==="
LABELS=$(find data/wv_merged_new/labels -name "*.txt" 2>/dev/null | wc -l)
IMAGES=$(find data/wv_merged_new/images_src -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
if [ "$LABELS" -eq 0 ]; then
  echo "ERROR: data/wv_merged_new/labels is empty — transfer first"
  exit 1
fi
echo "  labels: $LABELS  images: $IMAGES"

echo ""
echo "=== Step 2: Swap wv_merged ==="
if [ -d data/wv_merged_old ]; then
  echo "  WARN: data/wv_merged_old already exists — removing before swap"
  rm -rf data/wv_merged_old
fi
if [ -d data/wv_merged ]; then
  mv data/wv_merged data/wv_merged_old
  echo "  old wv_merged backed up to wv_merged_old"
fi
mv data/wv_merged_new data/wv_merged
echo "  wv_merged_new -> wv_merged"

echo ""
echo "=== Step 3: Update CLASS_REMAP in prepare_wv_data.py ==="
# Back up first
cp scripts/prepare_wv_data.py scripts/prepare_wv_data.py.bak

python3 - <<'PYEOF'
from pathlib import Path

script = Path("scripts/prepare_wv_data.py")
text = script.read_text()

new_remap = '''CLASS_REMAP: dict[int, int | None] = {
    0:  0,   # chevron
    1:  1,   # curve
    2:  6,   # damaged              → other
    3:  2,   # deerCrossing
    4:  4,   # guide                → informational (fallback for un-relabeled)
    5:  3,   # highwaySymbol
    6:  4,   # informational
    7:  6,   # intersection         → other
    8:  6,   # laneEnds             → other
    9:  5,   # mileMarker
    10: None, # missing_expected    DROPPED
    11: None, # occluded            DROPPED
    12: 6,   # other
    13: 7,   # pedestrianCrossing
    14: 8,   # railroadCrossing
    15: 6,   # ruralCrossing_other  → other
    16: 6,   # schoolZone           → other
    17: 9,   # speedLimit
    18: 10,  # stop
    19: 11,  # trafficLight
    20: 12,  # warning
    21: 13,  # yield
}'''

import re
text_new = re.sub(
    r"CLASS_REMAP: dict\[int, int \| None\] = \{.*?\}",
    new_remap,
    text,
    flags=re.DOTALL,
)
if text_new == text:
    print("ERROR: CLASS_REMAP pattern not found — patch manually")
    raise SystemExit(1)
script.write_text(text_new)
print("  CLASS_REMAP updated (22-class raw → 14-class training)")
PYEOF

echo ""
echo "=== Step 4: Update configs/wv.yaml ==="
cat > configs/wv.yaml <<'YAML'
names:
- chevron
- curve
- deerCrossing
- highwaySymbol
- informational
- mileMarker
- other
- pedestrianCrossing
- railroadCrossing
- speedLimit
- stop
- trafficLight
- warning
- yield
nc: 14
path: /opt/dlami/nvme/Jacob/wvsignclassification/data/processed/wv
train: train/images
val: val/images
YAML
echo "  configs/wv.yaml written (nc=14)"

echo ""
echo "=== Step 5: Rebuild processed dataset ==="
python scripts/prepare_wv_data.py --merged
if [ $? -ne 0 ]; then
  echo "ERROR: prepare_wv_data.py failed"
  exit 1
fi

echo ""
echo "=== Step 6: Preflight check ==="
python scripts/preflight_check.py configs/wv.yaml
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED — fix above before training"
  exit 1
fi

echo ""
echo "=== Step 7: Commit updated scripts ==="
git add scripts/prepare_wv_data.py configs/wv.yaml
git commit -m "Update CLASS_REMAP and taxonomy to 14-class WVDOH-aligned scheme"
git push origin main

echo ""
echo "=== All prep complete — launching Phase 3 retrain ==="
nohup bash -c '
source /home/ubuntu/Jacob/Jacob/bin/activate
cd /opt/dlami/nvme/Jacob/wvsignclassification

echo "[$(date)] Phase 3: WV fine-tuning (production)..."
python scripts/train_phase3_wv.py --production
if [ $? -ne 0 ]; then echo "[$(date)] PHASE 3 FAILED"; exit 1; fi
bash scripts/save_checkpoint_to_github.sh \
  checkpoints/phase3_wv_best.pt "Phase 3 WV 14-class complete"
if [ $? -ne 0 ]; then echo "[$(date)] CHECKPOINT PUSH FAILED"; exit 1; fi

echo "[$(date)] Phase 3 augmented..."
python scripts/train_phase3_wv.py --augmented
if [ $? -ne 0 ]; then echo "[$(date)] PHASE 3 AUGMENTED FAILED"; exit 1; fi
bash scripts/save_checkpoint_to_github.sh \
  checkpoints/phase3_wv_augmented_best.pt "Phase 3 augmented 14-class complete"
if [ $? -ne 0 ]; then echo "[$(date)] AUGMENTED CHECKPOINT PUSH FAILED"; exit 1; fi

echo "[$(date)] ALL TRAINING COMPLETE"
' >> logs/full_honest_run.log 2>&1 &

echo "Training PID: $!"
echo $! > logs/full_run.pid
sleep 30
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
tail -15 logs/full_honest_run.log
