#!/usr/bin/env python3
"""
Convert LISA and MTSD datasets to YOLOv8 format.

Output layout:
  data/processed/lisa/train/images/   (all LISA — no official split)
  data/processed/lisa/train/labels/
  data/processed/mtsd/train/images/
  data/processed/mtsd/train/labels/
  data/processed/mtsd/val/images/
  data/processed/mtsd/val/labels/

YOLOv8 label format: one .txt per image, each line is
  class_idx  x_center  y_center  width  height   (all normalized 0-1)
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LISA_DIR = ROOT / "data" / "lisa"
MTSD_DIR = ROOT / "data" / "mtsd"
OUT_DIR = ROOT / "data" / "processed"
CFG_DIR = ROOT / "configs"

_MTSD_BASE = MTSD_DIR / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated"
MTSD_ANN_DIR = _MTSD_BASE / "annotations"
MTSD_SPLITS_DIR = _MTSD_BASE / "splits"
MTSD_IMG_DIRS = [
    MTSD_DIR / f"mtsd_fully_annotated_images.train.{i}" for i in range(3)
] + [MTSD_DIR / "mtsd_fully_annotated_images.val"]


# ---------------------------------------------------------------------------
# Path auto-detection
# ---------------------------------------------------------------------------

def _detect_lisa_root(root: Path) -> Path:
    """Find allAnnotations.csv under root and return its parent as the effective LISA root."""
    matches = list(root.rglob("allAnnotations.csv"))
    if not matches:
        raise FileNotFoundError(f"allAnnotations.csv not found under {root}")
    if len(matches) > 1:
        print(f"  Warning: multiple allAnnotations.csv found; using {matches[0]}")
    return matches[0].parent


def _detect_mtsd_paths(root: Path) -> tuple[Path, Path, list[Path]]:
    """
    Scan root and return (ann_dir, splits_dir, img_dirs).

    ann_dir    — first directory containing 10+ .json files
    splits_dir — parent of train.txt; fully-annotated variant preferred
    img_dirs   — all directories containing 100+ .jpg files
    """
    # Splits: locate train.txt, prefer fully-annotated over partial
    train_txts = list(root.rglob("train.txt"))
    if not train_txts:
        raise FileNotFoundError(f"train.txt not found under {root}")
    fully = [p for p in train_txts if "fully" in str(p).lower()]
    splits_dir = (fully[0] if fully else train_txts[0]).parent

    # Annotations: first directory with 10+ .json files
    ann_dir: Path | None = None
    for f in root.rglob("*.json"):
        parent = f.parent
        if sum(1 for _ in parent.glob("*.json")) >= 10:
            ann_dir = parent
            break
    if ann_dir is None:
        raise FileNotFoundError(f"No directory with 10+ JSON files found under {root}")

    # Images: all directories with 100+ .jpg files
    seen: set[Path] = set()
    img_dirs: list[Path] = []
    for f in root.rglob("*.jpg"):
        p = f.parent
        if p in seen:
            continue
        seen.add(p)
        if sum(1 for _ in p.glob("*.jpg")) >= 100:
            img_dirs.append(p)
    if not img_dirs:
        raise FileNotFoundError(f"No image directories (100+ JPGs) found under {root}")

    return ann_dir, splits_dir, img_dirs


# ---------------------------------------------------------------------------
# LISA
# ---------------------------------------------------------------------------

def _lisa_classes_from_categories() -> list[str]:
    """Parse categories.txt into an ordered list of unique class names."""
    seen: set[str] = set()
    ordered: list[str] = []
    for line in (LISA_DIR / "categories.txt").read_text().splitlines():
        if ":" not in line:
            continue
        for part in line.split(":", 1)[1].split(","):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def prepare_lisa(dry_run: bool):
    categories_classes = _lisa_classes_from_categories()
    categories_set = set(categories_classes)

    # First pass: collect all annotation tags present in the CSV
    image_boxes: dict[str, list] = {}
    csv_tags: set[str] = set()
    with open(LISA_DIR / "allAnnotations.csv", newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if row[0] == "Filename":
                continue
            path, tag = row[0], row[1]
            csv_tags.add(tag)
            image_boxes.setdefault(path, []).append(
                (tag, int(row[2]), int(row[3]), int(row[4]), int(row[5]))
            )

    # Class list: categories.txt order first, then any extra CSV tags sorted
    extra = sorted(csv_tags - categories_set)
    classes = categories_classes + extra
    if extra:
        print(f"  LISA: {len(extra)} tags in CSV but not in categories.txt (appended): {extra}")
    missing_in_csv = categories_set - csv_tags
    if missing_in_csv:
        print(f"  LISA: {len(missing_in_csv)} categories.txt entries never seen in CSV: {sorted(missing_in_csv)}")

    cls_idx = {c: i for i, c in enumerate(classes)}

    out_img = OUT_DIR / "lisa" / "train" / "images"
    out_lbl = OUT_DIR / "lisa" / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    keys = list(image_boxes)[:100] if dry_run else list(image_boxes)
    n_imgs = n_boxes = 0
    counts: Counter = Counter()
    missing: list[str] = []

    for rel in keys:
        src = LISA_DIR / rel
        if not src.exists():
            missing.append(rel)
            continue

        with Image.open(src) as im:
            W, H = im.size

        # Flatten path to avoid collisions from different subdirectories
        slug = rel.replace("/", "__").replace("\\", "__")
        shutil.copy2(src, out_img / slug)

        lines = []
        for tag, x1, y1, x2, y2 in image_boxes[rel]:
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            ci = cls_idx[tag]
            lines.append(f"{ci} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[tag] += 1
            n_boxes += 1

        (out_lbl / (Path(slug).stem + ".txt")).write_text("\n".join(lines))
        n_imgs += 1

    return n_imgs, n_boxes, counts, classes, missing


# ---------------------------------------------------------------------------
# MTSD
# ---------------------------------------------------------------------------

def prepare_mtsd(dry_run: bool):
    # Index image key -> file path across all image archives
    key_to_img: dict[str, Path] = {}
    for d in MTSD_IMG_DIRS:
        for p in d.rglob("*.jpg"):
            key_to_img[p.stem] = p

    # Load official train/val split assignments
    split_of: dict[str, str] = {}
    for split in ("train", "val"):
        for line in (MTSD_SPLITS_DIR / f"{split}.txt").read_text().splitlines():
            if line.strip():
                split_of[line.strip()] = split

    ann_files = sorted(MTSD_ANN_DIR.glob("*.json"))
    if dry_run:
        ann_files = ann_files[:100]

    # Single pass: collect labels and build annotation records
    all_labels: set[str] = set()
    records = []  # (key, split, W, H, filtered_objects)
    for ann_file in ann_files:
        data = json.loads(ann_file.read_bytes())
        objs = [
            o for o in data.get("objects", [])
            if not any(
                o["properties"].get(k)
                for k in ("occluded", "ambiguous", "out-of-frame")
            )
        ]
        for o in objs:
            all_labels.add(o["label"])
        records.append(
            (ann_file.stem, split_of.get(ann_file.stem, "train"),
             data["width"], data["height"], objs)
        )

    classes = sorted(all_labels)
    cls_idx = {c: i for i, c in enumerate(classes)}

    n_imgs = n_boxes = 0
    counts: Counter = Counter()
    missing: list[str] = []

    for key, split, W, H, objs in records:
        if key not in key_to_img:
            missing.append(key)
            continue

        src = key_to_img[key]
        out_img = OUT_DIR / "mtsd" / split / "images"
        out_lbl = OUT_DIR / "mtsd" / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, out_img / src.name)

        lines = []
        for o in objs:
            bb = o["bbox"]
            w = (bb["xmax"] - bb["xmin"]) / W
            h = (bb["ymax"] - bb["ymin"]) / H
            if w <= 0 or h <= 0:  # inverted or degenerate bbox in source data
                continue
            xc = ((bb["xmin"] + bb["xmax"]) / 2) / W
            yc = ((bb["ymin"] + bb["ymax"]) / 2) / H
            ci = cls_idx[o["label"]]
            lines.append(f"{ci} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[o["label"]] += 1
            n_boxes += 1

        (out_lbl / (key + ".txt")).write_text("\n".join(lines))
        n_imgs += 1

    return n_imgs, n_boxes, counts, classes, missing


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def _names_block(classes: list[str]) -> str:
    return "names:\n" + "".join(f"  - {c}\n" for c in classes)


def write_configs(lisa_cls: list[str], mtsd_cls: list[str]) -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    (CFG_DIR / "lisa.yaml").write_text(
        f"path: {OUT_DIR / 'lisa'}\n"
        f"train: train/images\n"
        f"nc: {len(lisa_cls)}\n"
        + _names_block(lisa_cls)
    )

    (CFG_DIR / "mtsd.yaml").write_text(
        f"path: {OUT_DIR / 'mtsd'}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(mtsd_cls)}\n"
        + _names_block(mtsd_cls)
    )

    # LISA taxonomy used for the combined config.
    # MTSD label strings won't match LISA class indices — use configs/mtsd.yaml
    # for MTSD-only pre-training.
    (CFG_DIR / "data.yaml").write_text(
        f"# Combined paths; LISA class taxonomy.\n"
        f"# For MTSD pre-training use configs/mtsd.yaml — its labels differ.\n"
        f"path: {OUT_DIR}\n"
        f"train:\n"
        f"  - lisa/train/images\n"
        f"  - mtsd/train/images\n"
        f"val: mtsd/val/images\n"
        f"nc: {len(lisa_cls)}\n"
        + _names_block(lisa_cls)
    )

    # Plain class lists for the verify script
    (OUT_DIR / "lisa").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mtsd").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lisa" / "classes.txt").write_text("\n".join(lisa_cls))
    (OUT_DIR / "mtsd" / "classes.txt").write_text("\n".join(mtsd_cls))

    for p in [CFG_DIR / "lisa.yaml", CFG_DIR / "mtsd.yaml", CFG_DIR / "data.yaml"]:
        print(f"  wrote {p.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    name: str,
    n_imgs: int,
    n_boxes: int,
    counts: Counter,
    classes: list[str],
    missing: list[str],
) -> None:
    print(f"\n{'=' * 54}")
    print(f"  {name}")
    print(f"  Images : {n_imgs:,}")
    print(f"  Boxes  : {n_boxes:,}")
    print(f"  Classes: {len(classes)}")
    if counts:
        print(f"  Class distribution (top 20 by box count):")
        for cls, cnt in counts.most_common(20):
            print(f"    {cls:<40} {cnt:6,}")
        if len(counts) > 20:
            print(f"    ... ({len(counts) - 20} more classes)")
    if missing:
        print(f"\n  WARNING: {len(missing)} annotation(s) with no matching image:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... ({len(missing) - 10} more)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 100 images per dataset for a quick test.",
    )
    ap.add_argument(
        "--lisa-root",
        help="Path to raw LISA data (default: data/lisa/ under repo root).",
    )
    ap.add_argument(
        "--mtsd-root",
        help="Path to raw MTSD data (default: data/mtsd/ under repo root).",
    )
    ap.add_argument(
        "--output-root",
        help="Where to write processed data (default: data/processed/ under repo root).",
    )
    args = ap.parse_args()

    global LISA_DIR, MTSD_DIR, OUT_DIR, MTSD_ANN_DIR, MTSD_SPLITS_DIR, MTSD_IMG_DIRS
    if args.lisa_root:
        LISA_DIR = Path(args.lisa_root)
    if args.mtsd_root:
        MTSD_DIR = Path(args.mtsd_root)
    if args.output_root:
        OUT_DIR = Path(args.output_root)

    print("\nDetecting LISA paths ...")
    LISA_DIR = _detect_lisa_root(LISA_DIR)
    print(f"  allAnnotations.csv : {LISA_DIR / 'allAnnotations.csv'}")
    print(f"  categories.txt     : {LISA_DIR / 'categories.txt'}")
    print(f"  image base         : {LISA_DIR}")

    print("\nDetecting MTSD paths ...")
    MTSD_ANN_DIR, MTSD_SPLITS_DIR, MTSD_IMG_DIRS = _detect_mtsd_paths(MTSD_DIR)
    n_ann = sum(1 for _ in MTSD_ANN_DIR.glob("*.json"))
    print(f"  annotations  : {MTSD_ANN_DIR}  ({n_ann:,} JSONs)")
    print(f"  splits       : {MTSD_SPLITS_DIR}")
    for d in MTSD_IMG_DIRS:
        n_jpg = sum(1 for _ in d.glob("*.jpg"))
        print(f"  images       : {d}  ({n_jpg:,} JPGs)")

    if args.dry_run:
        print("\nDRY RUN: first 100 annotation entries per dataset")

    print("\nPreparing LISA ...")
    lisa_out = prepare_lisa(args.dry_run)
    print(f"  {lisa_out[0]:,} images, {lisa_out[1]:,} boxes")

    print("\nPreparing MTSD ...")
    mtsd_out = prepare_mtsd(args.dry_run)
    print(f"  {mtsd_out[0]:,} images, {mtsd_out[1]:,} boxes")

    print("\nWriting configs ...")
    write_configs(lisa_out[3], mtsd_out[3])

    _print_summary("LISA", *lisa_out)
    _print_summary("MTSD", *mtsd_out)


if __name__ == "__main__":
    main()
