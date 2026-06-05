# AWS Weekend Run — Operator Handoff

## Instance Requirements

| Field | Spec |
|---|---|
| Instance type | `g4dn.xlarge` minimum (1× T4 GPU, 16 GB VRAM) |
| AMI | AWS Deep Learning AMI (PyTorch 2.x, Ubuntu 22.04) |
| EBS root volume | 60 GB minimum (80 GB recommended) |
| CUDA | 12.1 (pre-installed in Deep Learning AMI) |
| Conda | pre-installed in Deep Learning AMI |

---

## Step 1 — Clone repo and restore data

```bash
# Set your S3 bucket name here (leave empty to copy data manually)
S3_BUCKET="my-wv-sign-data"

git clone https://github.com/jordan-jacob/wvsignclassification
cd wvsignclassification
```

---

## Step 2 — Environment setup

**Main YOLO environment** (run this first, always):
```bash
bash aws_setup.sh
```
- Installs PyTorch 2.12 + CUDA 12.1, Ultralytics, and all deps
- If `S3_BUCKET` is set in the script, syncs processed data from S3 automatically
- Creates directory structure and symlinks for LISA/MTSD

**Sparse R-CNN environment** (optional, adds ~45 min):
```bash
INSTALL_DETECTRON2=1 bash aws_setup.sh
```
- Creates a separate `detectron2_env` conda env (do NOT install detectron2 into the main env)
- Clones PeizeSun/SparseR-CNN adjacent to this repo (`~/SparseR-CNN/`)
- detectron2 was NOT tested on Windows — install on AWS only

---

## Step 3 — Verify data loaded correctly

```bash
ls data/processed/lisa/train/images  | wc -l   # expect 6618
ls data/processed/mtsd/train/images  | wc -l   # expect 36589
ls data/processed/mtsd/val/images    | wc -l   # expect 5320
ls checkpoints/phase1_mtsd_best.pt            # expect file present
```

---

## Step 4 — Weekend training run order

**Use `--production` for all runs.** Each step requires the checkpoint from the previous step.  
See `TRAINING_COMMANDS.md` for exact commands.

### Time estimate (single g4dn.xlarge)

| Step | Max epochs | Expected early stop | Est. time |
|---|---|---|---|
| Phase 1 MTSD | 200 (patience=50) | ~ep 100 | ~14 hr |
| Phase 2 YOLOv8m | 300 (patience=50) | ~ep 150 | ~4 hr |
| Phase 2 YOLOv11m | 300 (patience=50) | ~ep 150 | ~4 hr |
| FND | 50 (patience=10) | ~ep 30 | ~1 hr |
| Sparse R-CNN | 90 000 iters | — | ~10 hr |
| Eval + buffer | — | — | ~2 hr |
| **Total** | | | **~35 hr** |

35 hours fits a weekend run comfortably. Phase 2 runs are sequential on a single GPU
(~8 hr combined). If a second instance is available, they can run in parallel (~4 hr).

### Sequencing constraint

**Phase 1 must complete before Phase 2 starts** — Phase 2 requires `checkpoints/phase1_mtsd_best.pt`.  
Phase 2 YOLOv8m and YOLOv11m are independent of each other and can run on separate instances.

**Sequential (single GPU, recommended):**
```bash
python scripts/train_phase1_mtsd.py --production          # ~14 hr
python scripts/train_phase2_lisa.py --production           # ~4 hr
python scripts/train_phase2_yolo11.py --production         # ~4 hr
python scripts/train_fnd.py --checkpoint checkpoints/phase2_4class_best.pt --production   # ~1 hr
```

**Parallel option (two instances):**  
After Phase 1 completes and `phase1_mtsd_best.pt` is synced to S3, start Phase 2 YOLOv8m on
instance A and Phase 2 YOLOv11m on instance B simultaneously. Saves ~4 hr.

### 1. Phase 1 — MTSD pre-training (~14 hr)
```bash
python scripts/train_phase1_mtsd.py --production
```
Output: `checkpoints/phase1_mtsd_best.pt`  
*Script prints "Early stop: epoch X/200" or "Training complete: 200 epochs" when done.*

### 2. Phase 2 — YOLOv8m LISA fine-tune (~4 hr)
*Requires Phase 1 checkpoint.*
```bash
python scripts/train_phase2_lisa.py --production
```
Output: `checkpoints/phase2_full_best.pt`, `checkpoints/phase2_4class_best.pt`

### 3. Phase 2 — YOLOv11m LISA fine-tune (~4 hr)
*Requires Phase 1 checkpoint. Can run concurrently with Step 2 on a second instance.*
```bash
python scripts/train_phase2_yolo11.py --production
```
Output: `checkpoints/phase2_yolo11_full_best.pt`, `checkpoints/phase2_yolo11_4class_best.pt`

### 4. FND training (~1 hr)
*Requires Phase 2 4-class checkpoint.*
```bash
python scripts/train_fnd.py \
    --checkpoint checkpoints/phase2_4class_best.pt \
    --production
```
Output: `checkpoints/fnd_classifier.pt`

### 5. Sparse R-CNN (~10 hr)
*Requires detectron2_env (Step 2 above) and COCO JSON conversion.*
```bash
conda activate detectron2_env
python scripts/coco_convert.py --dataset lisa
python scripts/train_sparse_rcnn.py --smoke    # confirm data loads (~5 min)
python scripts/train_sparse_rcnn.py
conda deactivate
```
Output: `runs/sparse_rcnn/model_final.pth`

### 6. Model comparison (final, needs all checkpoints)
```bash
python scripts/compare_models.py \
    --models yolov8:checkpoints/phase2_full_best.pt \
             yolo11:checkpoints/phase2_yolo11_full_best.pt \
    --data configs/lisa.yaml --split val --with-fnd
```
Output: `runs/comparison/results.md`

---

## Step 5 — Copy results back to S3

```bash
S3_BUCKET="my-wv-sign-data"   # same bucket used for data upload
aws s3 sync runs/        s3://$S3_BUCKET/runs/
aws s3 sync checkpoints/ s3://$S3_BUCKET/checkpoints/
```

---

## Results location

| Artifact | Path |
|---|---|
| Phase 1 checkpoint | `checkpoints/phase1_mtsd_best.pt` |
| Phase 2 YOLOv8 checkpoints | `checkpoints/phase2_full_best.pt`, `checkpoints/phase2_4class_best.pt` |
| Phase 2 YOLOv11 checkpoints | `checkpoints/phase2_yolo11_full_best.pt`, `checkpoints/phase2_yolo11_4class_best.pt` |
| FND classifier | `checkpoints/fnd_classifier.pt` |
| Sparse R-CNN weights | `runs/sparse_rcnn/model_final.pth` |
| Comparison table | `runs/comparison/results.md` |
| Per-run training logs | `runs/phase*/` |

---

## Troubleshooting

**OOM on g4dn.xlarge (16 GB VRAM):**  
Reduce batch size: `--batch 8` or `--batch 4`

**Phase 2 fails with "phase1_mtsd_best.pt not found":**  
Phase 1 must complete first. Check `checkpoints/` for the file.

**Sparse R-CNN OOM:**  
Edit `configs/sparse_rcnn_lisa.yaml`: set `SOLVER.IMS_PER_BATCH: 8`

**Config path error in Sparse R-CNN:**  
Ensure SparseR-CNN is cloned at `~/SparseR-CNN/` (adjacent to this repo).  
The `_BASE_` path in `configs/sparse_rcnn_lisa.yaml` resolves relative to the config file.

**Data count mismatch:**  
Re-run `bash aws_setup.sh` with `S3_BUCKET` set. The `aws s3 sync` is idempotent.
