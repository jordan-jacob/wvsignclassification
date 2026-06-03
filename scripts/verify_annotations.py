#!/usr/bin/env python3
"""
Sanity-check the processed YOLO-format annotations in data/processed/mtsd
and data/processed/lisa.

Checks per dataset:
  1. Every image in images/ has a matching label .txt in labels/
  2. Every label .txt has a matching image
  3. No label file is empty (image has annotations)
  4. All YOLO coordinate values are in [0, 1]
  5. Class indices are within [0, num_classes)
  6. Counts match the expected totals from prepare_data.py

Prints a per-split summary and exits non-zero if any check fails.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

EXPECTED = {
    "mtsd": {
        "train": {"images": 36_589},
        "val":   {"images":  5_320},
    },
    "lisa": {
        "train": {"images":  6_618},
    },
}


def load_classes(dataset_dir: Path) -> list[str]:
    p = dataset_dir / "classes.txt"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


_IMG_GLOBS = ("*.jpg", "*.jpeg", "*.png")


def _iter_images(img_dir: Path):
    for pat in _IMG_GLOBS:
        yield from img_dir.glob(pat)


def check_split(dataset: str, split: str, split_dir: Path, num_classes: int) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists():
        return [f"{dataset}/{split}: images/ directory missing"], []
    if not lbl_dir.exists():
        return [f"{dataset}/{split}: labels/ directory missing"], []

    images = {p.stem: p for p in _iter_images(img_dir)}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

    # Check 1 & 2: 1-to-1 correspondence
    imgs_without_labels = set(images) - set(labels)
    labels_without_imgs = set(labels) - set(images)
    if imgs_without_labels:
        errors.append(f"{dataset}/{split}: {len(imgs_without_labels)} image(s) with no label file")
    if labels_without_imgs:
        errors.append(f"{dataset}/{split}: {len(labels_without_imgs)} label file(s) with no image")

    # Check 3, 4, 5: label file contents
    empty_labels = 0
    coord_errors = 0
    cls_errors = 0
    for stem, lbl_path in labels.items():
        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                coord_errors += 1
                continue
            ci, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if num_classes and not (0 <= ci < num_classes):
                cls_errors += 1
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                coord_errors += 1

    # Empty labels are valid (background images / fully-filtered annotations)
    if empty_labels:
        warnings.append(f"{dataset}/{split}: {empty_labels} label file(s) are empty (all annotations filtered)")
    if coord_errors:
        errors.append(f"{dataset}/{split}: {coord_errors} box(es) with out-of-range or malformed coordinates")
    if cls_errors:
        errors.append(f"{dataset}/{split}: {cls_errors} box(es) with class index out of range (num_classes={num_classes})")

    # Check 6: expected counts
    exp = EXPECTED.get(dataset, {}).get(split, {})
    exp_imgs = exp.get("images")
    if exp_imgs is not None and len(images) != exp_imgs:
        errors.append(
            f"{dataset}/{split}: expected {exp_imgs} images, found {len(images)}"
        )

    return errors, warnings


def main() -> None:
    all_errors: list[str] = []
    all_warnings: list[str] = []
    report_lines: list[str] = []

    for ds_dir in sorted(PROCESSED.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        classes = load_classes(ds_dir)
        num_classes = len(classes)

        for split_dir in sorted(ds_dir.iterdir()):
            if not split_dir.is_dir() or split_dir.name == "__pycache__":
                continue
            split = split_dir.name
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not img_dir.exists():
                continue

            errs, warns = check_split(dataset, split, split_dir, num_classes)
            n_imgs = sum(1 for _ in _iter_images(img_dir))
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            status = "FAIL" if errs else ("WARN" if warns else "PASS")
            report_lines.append(
                f"  [{status}] {dataset}/{split}: {n_imgs} images, {n_lbls} labels, {num_classes} classes"
            )
            all_errors.extend(errs)
            all_warnings.extend(warns)

    print("\nVerification report")
    print("=" * 54)
    for line in report_lines:
        print(line)

    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")

    if all_errors:
        print(f"\nFAILURES ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll checks passed." if not all_warnings else "\nPassed (with warnings).")


if __name__ == "__main__":
    main()
