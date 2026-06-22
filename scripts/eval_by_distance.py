#!/usr/bin/env python3
"""
Evaluate phase3_wv_best.pt broken out by box size (proxy for sign distance).

Size buckets based on normalized box area (w*h in 0-1 space):
  tiny:   area < 0.001   (very distant, ~<3% frame width equivalent)
  small:  0.001-0.005    (distant)
  medium: 0.005-0.02     (approaching)
  large:  > 0.02         (close / passing)

Outputs:
  1. AP@50 table:  Class x Size
  2. Recall table: Class x Size  (with GT counts)
  3. "Exclude distant" summary for stop and speedLimit
"""
import argparse
import gc
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "phase3_wv_best.pt"
CONFIG = ROOT / "configs" / "wv.yaml"
VAL_IMAGES = ROOT / "data" / "processed" / "wv" / "val" / "images"
VAL_LABELS = ROOT / "data" / "processed" / "wv" / "val" / "labels"

BUCKETS = ["tiny", "small", "medium", "large"]
IOU_THRESH = 0.5
CONF_THRESH = 0.001  # low threshold to capture full PR curve


def _bucket(area: float) -> str:
    if area < 0.001:
        return "tiny"
    elif area < 0.005:
        return "small"
    elif area < 0.02:
        return "medium"
    return "large"


def _iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes (normalized coords)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xywh_to_xyxy(cx, cy, w, h):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def _load_gt(label_path: Path, nc: int):
    """Returns list of (cls, cx, cy, w, h) from a YOLO label file."""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            if 0 <= cls < nc:
                rows.append((cls, *map(float, parts[1:])))
    return rows


def _compute_ap50(dets, n_gt: int) -> float:
    """Compute AP50 from a list of (conf, is_tp) pairs and total GT count."""
    if n_gt == 0:
        return float("nan")
    if not dets:
        return 0.0
    dets = sorted(dets, key=lambda x: -x[0])
    tp_cum = np.cumsum([int(tp) for _, tp in dets], dtype=float)
    fp_cum = np.cumsum([1 - int(tp) for _, tp in dets], dtype=float)
    prec = tp_cum / (tp_cum + fp_cum)
    rec = tp_cum / n_gt
    # Prepend/append sentinels and make precision monotone decreasing
    rec = np.concatenate([[0.0], rec, [rec[-1]]])
    prec = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


BATCH_SIZE = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=200,
                    help="Random sample of val images to evaluate (default: 200, 0=all)")
    args = ap.parse_args()

    import psutil
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < 4:
        print(f"WARNING: only {available_gb:.1f}GB RAM available — consider closing other processes")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not VAL_IMAGES.exists():
        raise FileNotFoundError(f"Val images not found: {VAL_IMAGES}")

    cfg = yaml.safe_load(CONFIG.read_text())
    class_names: list[str] = cfg["names"]
    nc: int = cfg["nc"]

    from ultralytics import YOLO
    model = YOLO(str(CHECKPOINT))

    # Per (cls, bucket): list of (conf, is_tp) for AP computation
    dets: dict = defaultdict(lambda: defaultdict(list))
    # Per (cls, bucket): GT count
    gt_counts: dict = defaultdict(lambda: defaultdict(int))

    img_files = sorted(VAL_IMAGES.glob("*.*"))
    if args.max_images and len(img_files) > args.max_images:
        random.seed(42)
        img_files = random.sample(img_files, args.max_images)
        print(f"Sampled {args.max_images} of {len(sorted(VAL_IMAGES.glob('*.*')))} val images (--max-images={args.max_images})")
    print(f"Running inference on {len(img_files)} images (CPU, batch={BATCH_SIZE}) ...")

    import torch
    for batch_start in range(0, len(img_files), BATCH_SIZE):
        batch = img_files[batch_start:batch_start + BATCH_SIZE]
        for r in model.predict(
            [str(p) for p in batch],
            conf=CONF_THRESH,
            device="cpu",
            stream=True,
            verbose=False,
        ):
            img_path = Path(r.path)
            label_path = VAL_LABELS / (img_path.stem + ".txt")
            gt = _load_gt(label_path, nc)

            # Count GT by (cls, bucket)
            for cls, cx, cy, w, h in gt:
                gt_counts[cls][_bucket(w * h)] += 1

            # Extract predictions: (cls, cx, cy, w, h, conf)
            boxes = r.boxes
            preds = []
            if boxes is not None and len(boxes):
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    preds.append((int(clss[i]), *xywhn[i], float(confs[i])))

            # Greedy match preds to GT per class (high-conf first)
            gt_matched = [False] * len(gt)
            preds_sorted = sorted(preds, key=lambda x: -x[5])

            for cls_p, cx_p, cy_p, w_p, h_p, conf_p in preds_sorted:
                box_p = _xywh_to_xyxy(cx_p, cy_p, w_p, h_p)
                best_iou = IOU_THRESH
                best_gi = -1
                for gi, (cls_g, cx_g, cy_g, w_g, h_g) in enumerate(gt):
                    if cls_g != cls_p or gt_matched[gi]:
                        continue
                    iou_val = _iou(box_p, _xywh_to_xyxy(cx_g, cy_g, w_g, h_g))
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gi = gi

                if best_gi >= 0:
                    gt_matched[best_gi] = True
                    _, cx_g, cy_g, w_g, h_g = gt[best_gi]
                    bucket = _bucket(w_g * h_g)
                    dets[cls_p][bucket].append((conf_p, True))
                else:
                    # FP: assign to bucket based on pred's own size
                    bucket = _bucket(w_p * h_p)
                    dets[cls_p][bucket].append((conf_p, False))
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("Computing metrics ...\n")

    # Build tables
    ap50 = {c: {b: _compute_ap50(dets[c][b], gt_counts[c][b]) for b in BUCKETS} for c in range(nc)}
    recall = {}
    for c in range(nc):
        recall[c] = {}
        for b in BUCKETS:
            tp = sum(1 for _, is_tp in dets[c][b] if is_tp)
            n = gt_counts[c][b]
            recall[c][b] = tp / n if n > 0 else float("nan")

    CW = 11  # column width

    def _fmt(v):
        return f"{'  -':>{CW}}" if np.isnan(v) else f"{v:>{CW}.4f}"

    def _print_table(title, values, extra_fn=None):
        width = 22 + CW * len(BUCKETS)
        print("=" * width)
        print(f"  {title}")
        print("=" * width)
        header = f"  {'Class':<20}" + "".join(f"{b.capitalize():>{CW}}" for b in BUCKETS)
        print(header)
        print("  " + "-" * (width - 2))
        for c in range(nc):
            row = f"  {class_names[c]:<20}" + "".join(_fmt(values[c][b]) for b in BUCKETS)
            if extra_fn:
                row += extra_fn(c)
            print(row)
        print()

    _print_table("AP@50 per class by box size", ap50)
    _print_table("Recall per class by box size", recall)

    # GT count table
    width = 22 + CW * len(BUCKETS)
    print("=" * width)
    print("  GT box counts by size bucket")
    print("=" * width)
    print(f"  {'Class':<20}" + "".join(f"{b.capitalize():>{CW}}" for b in BUCKETS))
    print("  " + "-" * (width - 2))
    for c in range(nc):
        row = f"  {class_names[c]:<20}" + "".join(f"{gt_counts[c][b]:>{CW}}" for b in BUCKETS)
        print(row)
    print()

    # Summary: recall excluding tiny/small
    focus = ["stop", "speedLimit"]
    print("=" * 60)
    print("  If we exclude tiny/small boxes (distant signs):")
    print("  Recall for medium+large GT only")
    print("=" * 60)
    for name in focus:
        if name not in class_names:
            print(f"  {name}: not found in class list")
            continue
        c = class_names.index(name)
        tp_ml = sum(1 for _, tp in dets[c]["medium"] + dets[c]["large"] if tp)
        n_ml = gt_counts[c]["medium"] + gt_counts[c]["large"]
        r = tp_ml / n_ml if n_ml > 0 else float("nan")
        n_tiny = gt_counts[c]["tiny"]
        n_small = gt_counts[c]["small"]
        r_str = f"{r:.4f}" if not np.isnan(r) else "  -"
        print(f"  {name}: recall={r_str}  (medium+large n={n_ml}, "
              f"excluded tiny={n_tiny}, small={n_small})")
    print()


if __name__ == "__main__":
    main()
