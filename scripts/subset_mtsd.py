#!/usr/bin/env python3
"""
Build a stratified 4-class MTSD subset for storage-constrained environments.

--4class-only: include only images with stop/yield/warning/speed-limit
               annotations; strip other-sign (class 4); stratify equally
               across the 4 classes.

Output:
  data/processed/mtsd_4class_subset/train/images  (junction/symlink → mtsd/train/images)
  data/processed/mtsd_4class_subset/train/labels  (filtered coarse label files)
  configs/mtsd_4class_subset.yaml
"""
import argparse
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed"

# Known fixed storage costs on the AWS instance (GB)
_LISA_GB = 3.0
_LABELS_GB = 0.1
_WEIGHTS_GB = 0.05   # yolov8m.pt
_RUNS_GB = 2.0
_FIXED_GB = _LISA_GB + _LABELS_GB + _WEIGHTS_GB + _RUNS_GB
_DATA_BUDGET_GB = 20.0


def _load_fine_to_coarse() -> dict[int, int]:
    fine_names = yaml.safe_load((CONFIGS / "mtsd.yaml").read_text())["names"]
    rules = yaml.safe_load((CONFIGS / "mtsd_to_coarse.yaml").read_text())["rules"]
    mapping: dict[int, int] = {}
    for i, name in enumerate(fine_names):
        coarse = 4  # other-sign default
        for rule in rules:
            if name.startswith(rule["prefix"]):
                coarse = rule["class"]
                break
        mapping[i] = coarse
    return mapping


def _scan_labels(mtsd_dir: Path, split: str) -> tuple[
    dict[int, list[Path]], dict[Path, list[str]]
]:
    """
    Return:
      class_images: {coarse 0-3: [label Paths containing that class]}
      label_cache:  {label Path: [filtered lines, class IDs 0-3 only]}
    """
    labels_fine_dir = mtsd_dir / split / "labels_fine"
    labels_dir = mtsd_dir / split / "labels"

    # If labels_fine/ exists, ensure_coarse_dataset() has already run and
    # labels/ contains 5-class coarse IDs (0-4) — no remapping needed.
    fine_to_coarse: dict[int, int] | None = (
        None if labels_fine_dir.exists() else _load_fine_to_coarse()
    )

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  Scanning {len(label_files):,} label files ...")

    class_images: dict[int, list[Path]] = defaultdict(list)
    label_cache: dict[Path, list[str]] = {}

    for lf in label_files:
        out_lines: list[str] = []
        classes_seen: set[int] = set()

        for line in lf.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            raw_ci = int(parts[0])
            ci = raw_ci if fine_to_coarse is None else fine_to_coarse.get(raw_ci, 4)
            if ci < 4:
                out_lines.append(f"{ci} {' '.join(parts[1:])}")
                classes_seen.add(ci)

        if classes_seen:
            label_cache[lf] = out_lines
            for c in classes_seen:
                class_images[c].append(lf)

    return dict(class_images), label_cache


def _stratified_sample(
    class_images: dict[int, list[Path]], n: int, seed: int
) -> list[Path]:
    """
    Sample n label paths stratified equally across classes 0-3.
    Each class contributes n//4 images (or all it has if fewer).
    Fills to n from remaining eligible images if needed.
    """
    rng = random.Random(seed)
    per_class = n // 4
    selected: set[Path] = set()

    for c in range(4):
        bucket = class_images.get(c, [])
        take = min(per_class, len(bucket))
        selected.update(rng.sample(bucket, take))

    all_eligible = sorted(set().union(*class_images.values()))
    remaining = [p for p in all_eligible if p not in selected]
    rng.shuffle(remaining)
    needed = n - len(selected)
    if needed > 0:
        selected.update(remaining[:needed])

    return sorted(selected)


def _make_junction_or_symlink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())],
            check=True, capture_output=True,
        )
    else:
        link.symlink_to(target.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--4class-only", dest="four_class_only", action="store_true",
                    help="Build stop/yield/warning/speed-limit subset (strips other-sign)")
    ap.add_argument("--n-images", type=int, default=8000,
                    help="Training images to sample (default: 8000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.four_class_only:
        ap.error("Only --4class-only is currently supported")

    mtsd_dir = PROCESSED / "mtsd"
    if not mtsd_dir.exists():
        print(f"ERROR: MTSD data not found at {mtsd_dir}", file=sys.stderr)
        sys.exit(1)

    # --- scan ---
    class_images, label_cache = _scan_labels(mtsd_dir, "train")
    total_eligible = len(set().union(*class_images.values()))
    print(f"Total MTSD images with 4-class annotations: {total_eligible:,}")

    # --- sample ---
    selected = _stratified_sample(class_images, args.n_images, args.seed)
    print(f"Sampled: {len(selected):,} images")

    # --- write label files ---
    out_dir = PROCESSED / "mtsd_4class_subset"
    out_labels = out_dir / "train" / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)

    class_box_counts: dict[int, int] = defaultdict(int)
    for lf in selected:
        lines = label_cache[lf]
        (out_labels / lf.name).write_text("\n".join(lines))
        for line in lines:
            class_box_counts[int(line.split()[0])] += 1

    # --- images junction/symlink ---
    src_images = mtsd_dir / "train" / "images"
    out_images = out_dir / "train" / "images"
    _make_junction_or_symlink(out_images, src_images)

    # --- configs/mtsd_4class_subset.yaml ---
    yaml_cfg = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "train/images",
        "nc": 4,
        "names": ["stop", "yield", "warning", "speed-limit"],
    }
    yaml_path = CONFIGS / "mtsd_4class_subset.yaml"
    yaml_path.write_text(yaml.dump(yaml_cfg, default_flow_style=False))

    # --- disk usage (real measurement) ---
    sample_imgs = sorted(src_images.glob("*.jpg"))[:500]
    avg_bytes = sum(p.stat().st_size for p in sample_imgs) / len(sample_imgs) if sample_imgs else 0.0
    mtsd_est_gb = avg_bytes * len(selected) / 1e9

    class_names = ["stop", "yield", "warning", "speed-limit"]
    print(f"\nClass distribution after filtering:")
    for c, name in enumerate(class_names):
        print(f"  {name:<14s} {class_box_counts[c]:>7,} boxes")

    print(f"\nEstimated disk usage: {mtsd_est_gb:.2f} GB")
    print(f"  (avg image size: {avg_bytes/1024:.0f} KB × {len(selected):,} images)")

    # --- storage budget ---
    total_est_gb = mtsd_est_gb + _FIXED_GB
    print(f"\nStorage budget ({_DATA_BUDGET_GB:.0f} GB available for data on AWS):")
    print(f"  MTSD 4-class subset:    {mtsd_est_gb:.2f} GB")
    print(f"  LISA 4-class images:    {_LISA_GB:.2f} GB")
    print(f"  Labels:                 {_LABELS_GB:.2f} GB")
    print(f"  Weights (yolov8m.pt):   {_WEIGHTS_GB:.2f} GB")
    print(f"  Runs + checkpoints:     {_RUNS_GB:.2f} GB")
    print(f"  {'-'*41}")
    print(f"  Total:                  {total_est_gb:.2f} GB / {_DATA_BUDGET_GB:.0f} GB available")

    if total_est_gb <= _DATA_BUDGET_GB:
        print(f"  OK — {_DATA_BUDGET_GB - total_est_gb:.2f} GB headroom")
    else:
        overhead = total_est_gb - _DATA_BUDGET_GB
        max_images = int((_DATA_BUDGET_GB - _FIXED_GB) * 1e9 / avg_bytes) if avg_bytes > 0 else args.n_images
        max_images = (max_images // 4) * 4
        print(f"  EXCEEDS BUDGET by {overhead:.2f} GB")
        print(f"  Recommended max: --n-images {max_images}")
        print(f"  Re-run: python scripts/subset_mtsd.py --4class-only --n-images {max_images}")

    print(f"\nConfig: {yaml_path}")
    print(f"Labels: {out_labels} ({len(selected):,} files)")
    print(f"Images: {out_images} -> {src_images}")


if __name__ == "__main__":
    main()
