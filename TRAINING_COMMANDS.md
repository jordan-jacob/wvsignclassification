# Training Commands

Run these in order. Each step requires the checkpoint from the previous step.

---

## Smoke tests (run first to verify environment)

```bash
python scripts/train_phase1_mtsd.py --smoke
python scripts/train_phase2_lisa.py --smoke
python scripts/train_phase2_yolo11.py --smoke
python scripts/train_fnd.py --checkpoint checkpoints/phase2_4class_best.pt --smoke
python scripts/coco_convert.py --dataset lisa --smoke
```

---

## Phase 1 — MTSD pre-training

```bash
python scripts/train_phase1_mtsd.py --batch 16
```

Checkpoint: `checkpoints/phase1_mtsd_best.pt`  
Time: ~3–4 hr (g4dn.xlarge)

---

## Phase 2 — LISA fine-tune

```bash
# YOLOv8m
python scripts/train_phase2_lisa.py --batch 16

# YOLOv11m (can run in parallel on a second GPU)
python scripts/train_phase2_yolo11.py --batch 16
```

Checkpoints:
- `checkpoints/phase2_full_best.pt`
- `checkpoints/phase2_4class_best.pt`
- `checkpoints/phase2_yolo11_full_best.pt`
- `checkpoints/phase2_yolo11_4class_best.pt`

Time: ~1–2 hr each

---

## FND training

```bash
python scripts/train_fnd.py \
    --checkpoint checkpoints/phase2_4class_best.pt \
    --epochs 20 --batch 16
```

Checkpoint: `checkpoints/fnd_classifier.pt`  
Time: ~30 min

---

## Sparse R-CNN (detectron2_env)

```bash
conda activate detectron2_env

# Convert data (one-time, ~10 min)
python scripts/coco_convert.py --dataset lisa
python scripts/coco_convert.py --dataset mtsd_coarse  # optional

# Smoke test
python scripts/train_sparse_rcnn.py --smoke

# Full training
python scripts/train_sparse_rcnn.py

conda deactivate
```

Checkpoint: `runs/sparse_rcnn/model_final.pth`  
Time: ~4–6 hr (g4dn.xlarge)

---

## Model comparison

```bash
# YOLO models only
python scripts/compare_models.py \
    --models yolov8:checkpoints/phase2_full_best.pt \
             yolo11:checkpoints/phase2_yolo11_full_best.pt \
    --data configs/lisa.yaml --split val --with-fnd

# Include Sparse R-CNN (requires detectron2_env active)
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
