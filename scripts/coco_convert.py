"""
Convert processed YOLO-format datasets to COCO JSON for Sparse R-CNN / detectron2.

Usage:
    python scripts/coco_convert.py --dataset lisa
    python scripts/coco_convert.py --dataset mtsd_coarse
    python scripts/coco_convert.py --dataset both
    python scripts/coco_convert.py --dataset lisa --smoke   # 100 images only

Output:
    data/coco/<dataset>/annotations/instances_train.json
    data/coco/<dataset>/annotations/instances_val.json  (mtsd_coarse only)

LISA uses lisa_4class/train/labels/ (already remapped to 4 coarse classes).
LISA has no val split; instances_train.json doubles as the smoke-test val.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
COCO_OUT = PROJECT_ROOT / "data" / "coco"

# Per-dataset config: which image/label dirs to use, and class names.
# Category IDs in output JSON are 1-indexed (YOLO class 0 → COCO category_id 1).
DATASETS = {
    "lisa": {
        "splits": {
            "train": {
                "images": PROCESSED / "lisa" / "train" / "images",
                # Use lisa_4class labels (already remapped from 47→4 classes).
                # Empty label files = image has no annotated signs in coarse scheme.
                "labels": PROCESSED / "lisa_4class" / "train" / "labels",
            }
            # No val split for LISA.
        },
        "classes": ["stop", "yield", "warning", "speed-limit"],
    },
    "mtsd_coarse": {
        "splits": {
            "train": {
                "images": PROCESSED / "mtsd_coarse" / "train" / "images",
                "labels": PROCESSED / "mtsd_coarse" / "train" / "labels",
            },
            "val": {
                "images": PROCESSED / "mtsd_coarse" / "val" / "images",
                "labels": PROCESSED / "mtsd_coarse" / "val" / "labels",
            },
        },
        "classes": ["stop", "yield", "warning", "speed-limit", "other-sign"],
    },
}


def build_categories(class_names: list[str]) -> list[dict]:
    return [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]


def convert_split(
    img_dir: Path,
    label_dir: Path,
    categories: list[dict],
    output_path: Path,
    limit: int | None = None,
    split_name: str = "",
) -> dict:
    """
    Convert one YOLO split to COCO JSON.
    Returns summary dict with counts and bad-bbox count.
    """
    img_paths = sorted(p for p in img_dir.iterdir() if p.is_file())
    if limit:
        img_paths = img_paths[:limit]

    images = []
    annotations = []
    ann_id = 1
    bad_bbox = 0
    missing_labels = 0

    for img_id, img_path in enumerate(
        tqdm(img_paths, desc=f"  {split_name}", unit="img"), start=1
    ):
        with Image.open(img_path) as im:
            W, H = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H,
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels += 1
            continue

        text = label_path.read_text().strip()
        if not text:
            continue

        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:])

            x_tl = (cx - nw / 2) * W
            y_tl = (cy - nh / 2) * H
            w_abs = nw * W
            h_abs = nh * H

            if w_abs <= 0 or h_abs <= 0:
                bad_bbox += 1
                continue

            # Clamp to image bounds (guard against float precision edge cases)
            x_tl = max(0.0, x_tl)
            y_tl = max(0.0, y_tl)
            w_abs = min(w_abs, W - x_tl)
            h_abs = min(h_abs, H - y_tl)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO is 1-indexed
                "bbox": [round(x_tl, 2), round(y_tl, 2),
                         round(w_abs, 2), round(h_abs, 2)],
                "area": round(w_abs * h_abs, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_json, indent=2))

    return {
        "images": len(images),
        "annotations": len(annotations),
        "bad_bbox": bad_bbox,
        "missing_labels": missing_labels,
    }


def try_create_image_symlink(link_path: Path, target_path: Path) -> None:
    """Create a symlink/junction for the image directory. Non-fatal on failure."""
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows without Developer Mode requires admin for symlinks.
        # Use junction instead.
        if sys.platform == "win32":
            import subprocess
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J",
                 str(link_path), str(target_path)],
                capture_output=True,
            )
            if result.returncode != 0:
                print(f"    NOTE: could not create image link at {link_path}.")
                print(f"    On AWS run: ln -sfn {target_path} {link_path}")
        else:
            print(f"    NOTE: could not create symlink {link_path} → {target_path}")


def validate_coco_json(json_path: Path) -> None:
    """Load output with pycocotools and print summary. Skips if not installed."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("    pycocotools not installed — skipping validation.")
        print("    Install with: pip install pycocotools")
        return

    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):  # suppress COCO's own print
        coco = COCO(str(json_path))

    n_cats = len(coco.getCatIds())
    n_imgs = len(coco.getImgIds())
    n_anns = len(coco.getAnnIds())
    print(f"    pycocotools OK: {n_cats} categories, {n_imgs} images, {n_anns} annotations")

    # Verify every annotation's category_id is in the categories list
    valid_cats = set(coco.getCatIds())
    bad = [a for a in coco.anns.values() if a["category_id"] not in valid_cats]
    if bad:
        print(f"    WARNING: {len(bad)} annotations have unknown category_id")
    else:
        print("    All annotation category_ids are valid.")


def process_dataset(name: str, cfg: dict, smoke: bool) -> None:
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    categories = build_categories(cfg["classes"])
    limit = 100 if smoke else None
    smoke_tag = " (smoke: 100 images)" if smoke else ""

    for split_name, split_cfg in cfg["splits"].items():
        img_dir = split_cfg["images"]
        label_dir = split_cfg["labels"]

        if not img_dir.exists():
            print(f"  SKIP {split_name}: image dir not found: {img_dir}")
            continue
        if not label_dir.exists():
            print(f"  SKIP {split_name}: label dir not found: {label_dir}")
            continue

        out_json = COCO_OUT / name / "annotations" / f"instances_{split_name}.json"
        print(f"\n  [{split_name}{smoke_tag}] -> {out_json.relative_to(PROJECT_ROOT)}")

        stats = convert_split(
            img_dir, label_dir, categories, out_json,
            limit=limit, split_name=split_name,
        )

        print(f"    images: {stats['images']:,}  |  "
              f"annotations: {stats['annotations']:,}  |  "
              f"missing labels: {stats['missing_labels']:,}")

        if stats["bad_bbox"] > 0:
            print(f"    WARNING: {stats['bad_bbox']} bbox(es) with w<=0 or h<=0 were skipped.")

        validate_coco_json(out_json)

        # Create image directory link so detectron2 can find images via COCO JSON.
        # register_coco_instances uses data/coco/<dataset>/<split>/ as image_root.
        link = COCO_OUT / name / split_name
        try_create_image_symlink(link, img_dir.resolve())
        if link.exists():
            print(f"    Image dir linked: {link.relative_to(PROJECT_ROOT)} -> {img_dir}")
        else:
            print(f"    Image dir NOT linked -- on AWS: "
                  f"ln -sfn {img_dir} {link.relative_to(PROJECT_ROOT)}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["lisa", "mtsd_coarse", "both"],
        help="Which dataset to convert",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Convert only first 100 images (for quick validation)",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "both" else [args.dataset]
    for name in targets:
        process_dataset(name, DATASETS[name], smoke=args.smoke)


if __name__ == "__main__":
    main()
