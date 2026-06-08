# Training Commands

---

## TODAY'S VALIDATION RUN (4-class, storage-constrained)

Storage budget: 11.82 GB / 20 GB (MTSD subset 6.67 GB + LISA 3 GB + labels/weights/runs 2.15 GB).
8,000-image MTSD subset drawn from 15,529 eligible images (stop: 1,178 boxes, yield: 2,205,
warning: 5,531, speed-limit: 3,271).

### On your LOCAL machine first — generate subset and upload

```bash
# Build the MTSD 4-class subset (already done locally — skip if data/processed/mtsd_4class_subset/ exists)
python scripts/subset_mtsd.py --4class-only --n-images 8000

# Upload subset to S3 (images + labels only, ~6.7 GB)
python scripts/upload_to_s3.py --bucket <name> --upload-mtsd-subset

# Also upload LISA 4-class labels and configs
python scripts/upload_to_s3.py --bucket <name> --upload-lisa --upload-configs
```

### On AWS — download data and train

```bash
# Download data
aws s3 sync s3://<bucket>/data/processed/mtsd_4class_subset data/processed/mtsd_4class_subset/
aws s3 sync s3://<bucket>/data/processed/lisa              data/processed/lisa/
aws s3 sync s3://<bucket>/configs/                         configs/

# Rebuild 4-class LISA labels (fast, ~1 min — creates lisa_4class/ from lisa/)
python scripts/train_phase2_lisa.py --smoke  # will call ensure_lisa_4class() and exit after smoke

# Step 1 — Phase 1 pre-training on MTSD 4-class subset (~3-4 hrs on T4)
python scripts/train_phase1_mtsd.py --production --4class-only
# Est: 45-60 sec/epoch × ~80 ep early stop = ~1-1.5 hr; 200 ep ceiling = ~3-4 hrs

# Step 2 — Phase 2 fine-tune on LISA 4-class (~1-2 hrs on T4)
python scripts/train_phase2_lisa.py --production --4class-only
# Est: 300 ep ceiling, early stop likely ~100-150 ep = ~1-2 hrs

# Step 3 — Check recall threshold (~5 min)
python scripts/check_recall_threshold.py \
    --checkpoint checkpoints/phase2_4class_best.pt \
    --threshold 0.65
# Exit 0 = threshold met
# Exit 1 = one or more classes below threshold (recommend requesting more storage)
# Exit 2 = per-class passed but weighted overall failed
```

---

Run in this order. Each step requires the checkpoint from the previous step.

---

## 0. Smoke tests (run first — verify environment, ~5 min total)

```bash
python scripts/train_phase1_mtsd.py --smoke
python scripts/train_phase2_lisa.py --smoke
python scripts/train_phase2_yolo11.py --smoke
python scripts/train_fnd.py --checkpoint checkpoints/phase2_4class_best.pt --smoke
python scripts/coco_convert.py --dataset lisa --smoke
```

---

## 1. Phase 1 — MTSD pre-training (~14 hr, early stop ~100 ep)

```bash
python scripts/train_phase1_mtsd.py --production
```

Output: `checkpoints/phase1_mtsd_best.pt`

---

## 2. Phase 2 — YOLOv8m LISA fine-tune (~4 hr, early stop ~150 ep)

```bash
python scripts/train_phase2_lisa.py --production
```

Output: `checkpoints/phase2_full_best.pt`, `checkpoints/phase2_4class_best.pt`

---

## 3. Phase 2 — YOLOv11m LISA fine-tune (~4 hr, early stop ~150 ep)

```bash
python scripts/train_phase2_yolo11.py --production
```

Output: `checkpoints/phase2_yolo11_full_best.pt`, `checkpoints/phase2_yolo11_4class_best.pt`

---

## 4. FND training (~1 hr)

```bash
python scripts/train_fnd.py \
    --checkpoint checkpoints/phase2_4class_best.pt \
    --production
```

Output: `checkpoints/fnd_classifier.pt`

---

## 5. Sparse R-CNN (~10 hr)

```bash
conda activate detectron2_env
python scripts/coco_convert.py --dataset lisa
python scripts/train_sparse_rcnn.py --smoke    # ~5 min — verify data loads
python scripts/train_sparse_rcnn.py
conda deactivate
```

Output: `runs/sparse_rcnn/model_final.pth`

---

## 6. Model comparison (run after all checkpoints exist)

```bash
python scripts/compare_models.py \
    --models yolov8:checkpoints/phase2_full_best.pt \
             yolo11:checkpoints/phase2_yolo11_full_best.pt \
    --data configs/lisa.yaml --split val --with-fnd
```

To include Sparse R-CNN:
```bash
conda activate detectron2_env
python scripts/compare_models.py \
    --models yolov8:checkpoints/phase2_full_best.pt \
             yolo11:checkpoints/phase2_yolo11_full_best.pt \
             sparse_rcnn:configs/sparse_rcnn_lisa.yaml:runs/sparse_rcnn/model_final.pth \
    --data configs/lisa.yaml --split val
conda deactivate
```

Output: `runs/comparison/results.md`

---

## Copy results to S3

```bash
S3_BUCKET="my-wv-sign-data"
aws s3 sync runs/        s3://$S3_BUCKET/runs/
aws s3 sync checkpoints/ s3://$S3_BUCKET/checkpoints/
```
