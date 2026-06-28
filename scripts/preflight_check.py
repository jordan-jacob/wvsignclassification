#!/usr/bin/env python3
"""
Preflight validator for YOLO dataset YAMLs.

Exits 0 if all yamls pass, 1 if any check fails.
Uses only stdlib + pyyaml; safe to run before torch/ultralytics loads.

Usage:
    python scripts/preflight_check.py dataset_a.yaml [dataset_b.yaml ...]
"""
import argparse
import os
import random
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SAMPLE_LIMIT = 500


def _check_dir(path: Path) -> str | None:
    """Return an error string if path is missing/dangling, else None."""
    if path.is_dir():
        return None
    try:
        target = os.readlink(str(path))
        return f"dangling symlink/junction: {path} -> {target}"
    except OSError:
        pass
    return f"does not exist: {path}"


def _labels_dir(img_dir: Path) -> Path:
    if img_dir.name == "images":
        return img_dir.parent / "labels"
    return Path(str(img_dir).replace("images", "labels"))


def check_split(root: Path, split_key: str, nc: int) -> tuple[bool, dict]:
    split_path = Path(split_key)
    img_dir = split_path if split_path.is_absolute() else root / split_path
    lbl_dir = _labels_dir(img_dir)

    errors, warnings = [], []

    err = _check_dir(img_dir)
    if err:
        errors.append(f"images: {err}")
        return False, {"errors": errors, "warnings": warnings,
                       "img_dir": img_dir, "lbl_dir": lbl_dir}

    err = _check_dir(lbl_dir)
    if err:
        errors.append(f"labels: {err}")
        return False, {"errors": errors, "warnings": warnings,
                       "img_dir": img_dir, "lbl_dir": lbl_dir}

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

    if not images:
        errors.append(f"no image files (.png/.jpg/.jpeg) in {img_dir}")
    if not labels:
        errors.append(f"no label files (.txt) in {lbl_dir}")

    img_missing_lbl = len(set(images) - set(labels))
    lbl_missing_img = len(set(labels) - set(images))
    if img_missing_lbl:
        errors.append(f"{img_missing_lbl} image(s) have no label file")
    if lbl_missing_img:
        errors.append(f"{lbl_missing_img} label file(s) have no image")

    bg_labels = sum(1 for p in labels.values() if p.stat().st_size == 0)
    non_empty = len(labels) - bg_labels

    # Sample label files to check class indices against nc
    sample = random.sample(list(labels.values()), min(SAMPLE_LIMIT, len(labels)))
    max_cls = -1
    bad_files = 0
    for lbl_path in sample:
        text = lbl_path.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls > max_cls:
                max_cls = cls
            if cls >= nc:
                bad_files += 1
                break
    if bad_files:
        warnings.append(f"{bad_files} sampled label file(s) reference class index >= nc={nc}")
    if max_cls >= nc:
        errors.append(f"max class index in labels ({max_cls}) >= nc ({nc})")

    ok = not errors
    return ok, {
        "errors": errors,
        "warnings": warnings,
        "img_dir": img_dir,
        "lbl_dir": lbl_dir,
        "images": len(images),
        "labels": len(labels),
        "non_empty_labels": non_empty,
        "bg_labels": bg_labels,
        "img_missing_lbl": img_missing_lbl,
        "lbl_missing_img": lbl_missing_img,
    }


def check_yaml(yaml_path: Path) -> bool:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Checking: {yaml_path}")

    if not yaml_path.exists():
        print(f"  ERROR: file not found: {yaml_path}")
        return False

    data = yaml.safe_load(yaml_path.read_text())
    nc = data.get("nc", 0)

    root_raw = data.get("path", str(yaml_path.parent))
    root = Path(root_raw)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    print(f"  root: {root}")
    print(f"  nc:   {nc}")

    all_ok = True
    for split in ("train", "val"):
        split_key = data.get(split)
        if split_key is None:
            print(f"\n  [{split}] not present in yaml, skipping")
            continue

        ok, res = check_split(root, split_key, nc)
        if not ok:
            all_ok = False

        print(f"\n  [{split}]")
        print(f"    images dir : {res['img_dir']}")
        print(f"    labels dir : {res['lbl_dir']}")
        if "images" in res:
            print(f"    images     : {res['images']}")
            print(f"    labels     : {res['labels']}"
                  f"  (non-empty: {res['non_empty_labels']},"
                  f" background: {res['bg_labels']})")
            print(f"    mismatches : img_missing_lbl={res['img_missing_lbl']},"
                  f" lbl_missing_img={res['lbl_missing_img']}")
        for e in res["errors"]:
            print(f"    ERROR: {e}")
        for w in res["warnings"]:
            print(f"    WARN:  {w}")
        print(f"    --> {'PASS' if ok else 'FAIL'}")

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  {yaml_path.name}: {status}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Preflight check for YOLO dataset YAMLs")
    ap.add_argument("yamls", nargs="+", type=Path, metavar="YAML")
    args = ap.parse_args()

    results = [check_yaml(p) for p in args.yamls]

    print(f"\n{'=' * 60}")
    if all(results):
        print("All yamls PASS.")
        sys.exit(0)
    else:
        n_fail = sum(1 for r in results if not r)
        print(f"{n_fail}/{len(results)} yaml(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
