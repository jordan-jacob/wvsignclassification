# Training Commands

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
