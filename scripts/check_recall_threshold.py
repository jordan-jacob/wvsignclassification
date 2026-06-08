#!/usr/bin/env python3
"""
Validate a 4-class YOLO checkpoint against a per-class recall threshold.

Exit codes:
  0  all classes and overall weighted recall meet threshold
  1  one or more classes below threshold
  2  all per-class recalls pass but overall weighted recall is below threshold
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def _count_class_support(data_yaml_path: Path, nc: int) -> np.ndarray:
    """Count GT boxes per class from the validation label files."""
    cfg = yaml.safe_load(data_yaml_path.read_text())

    base = Path(cfg["path"]) if "path" in cfg else data_yaml_path.parent
    val_key = cfg.get("val", cfg.get("train", "train/images"))

    if val_key.endswith(".txt"):
        val_txt = Path(val_key) if Path(val_key).is_absolute() else base / val_key
        img_paths = [Path(p) for p in val_txt.read_text().splitlines() if p.strip()]
        label_files = [p.parent.parent / "labels" / (p.stem + ".txt") for p in img_paths]
    else:
        img_dir = base / val_key
        label_dir = img_dir.parent.parent / "labels" / img_dir.name
        label_files = list(label_dir.glob("*.txt"))

    counts = np.zeros(nc, dtype=int)
    for f in label_files:
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    ci = int(line.split()[0])
                    if 0 <= ci < nc:
                        counts[ci] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Check per-class recall against a threshold.")
    ap.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    ap.add_argument("--data", default=str(CONFIGS / "lisa_4class.yaml"),
                    help="YOLO data YAML (default: configs/lisa_4class.yaml)")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="Minimum recall required per class (default: 0.65)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: data yaml not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(data_yaml.read_text())
    nc = cfg["nc"]
    class_names = cfg["names"]

    print(f"Checkpoint : {ckpt}")
    print(f"Data       : {data_yaml}")
    print(f"Threshold  : {args.threshold}")
    print()

    model = YOLO(str(ckpt))
    results = model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, verbose=False)

    # results.box.r is per-class recall aligned to results.box.ap_class_index
    ap_idx = list(results.box.ap_class_index)
    r_vals = np.asarray(results.box.r)

    recall = np.zeros(nc, dtype=float)
    for local_i, cls_i in enumerate(ap_idx):
        if cls_i < nc:
            recall[cls_i] = r_vals[local_i]

    support = _count_class_support(data_yaml, nc)
    total = int(support.sum())
    weighted_recall = float(np.sum(recall * support) / total) if total > 0 else 0.0

    print("Per-class recall:")
    below = []
    for i, name in enumerate(class_names):
        status = "OK" if recall[i] >= args.threshold else "FAIL"
        print(f"  {name:<20s}  recall={recall[i]:.4f}  [{status}]  (n={support[i]})")
        if recall[i] < args.threshold:
            below.append(name)

    print()
    print(f"Weighted overall recall: {weighted_recall:.4f}")
    print()

    if below:
        print("THRESHOLD NOT MET — recommend full dataset training")
        print(f"Classes below threshold: {below}")
        sys.exit(1)

    if weighted_recall < args.threshold:
        print("THRESHOLD NOT MET — recommend full dataset training")
        print("Overall weighted recall is below threshold (all per-class recalls passed)")
        sys.exit(2)

    print("All thresholds met.")
    sys.exit(0)


if __name__ == "__main__":
    main()
