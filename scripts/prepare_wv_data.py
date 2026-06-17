"""
Converts a Label Studio YOLO export (data/wv_annotations/) into the standard
YOLO directory structure under data/processed/wv/ with an 80/20 train/val split.
Also writes configs/wv.yaml and validates image/label pairing.

Label filenames from Label Studio are URL-encoded paths of the form:
  {hash}__candidates%5C{stem}.txt   ->  annotation_frames/candidates/{stem}.jpg
  {hash}__background%5C{stem}.txt   ->  annotation_frames/background/{stem}.jpg

Class remapping applied at conversion time (see CLAUDE.md § WV Taxonomy):
  - missing_expected, occluded : dropped entirely (boxes removed)
  - damaged, laneEnds, intersection, schoolZone, ruralCrossing_other : merged → other

Usage:
    python scripts/prepare_wv_data.py
    python scripts/prepare_wv_data.py --images-dir /path/to/annotation_frames
"""

import argparse
import random
import shutil
import urllib.parse
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ANNO_DIR = ROOT / "data" / "wv_annotations"
OUT_DIR = ROOT / "data" / "processed" / "wv"
CONFIG_PATH = ROOT / "configs" / "wv.yaml"
LOW_SAMPLE_THRESHOLD = 10
SEED = 42

# Maps original Label Studio class index → new index, or None to drop the box.
# Original order (from data/wv_annotations/classes.txt):
#   0:chevron 1:curve 2:damaged 3:deerCrossing 4:guide 5:intersection
#   6:laneEnds 7:missing_expected 8:occluded 9:other 10:pedestrianCrossing
#   11:railroadCrossing 12:ruralCrossing_other 13:schoolZone 14:speedLimit
#   15:stop 16:warning 17:yield
CLASS_REMAP: dict[int, int | None] = {
    0: 0,    # chevron
    1: 1,    # curve
    2: 4,    # damaged        → other
    3: 2,    # deerCrossing
    4: 3,    # guide
    5: 4,    # intersection   → other
    6: 4,    # laneEnds       → other
    7: None, # missing_expected  DROPPED
    8: None, # occluded          DROPPED
    9: 4,    # other
    10: 5,   # pedestrianCrossing
    11: 6,   # railroadCrossing
    12: 4,   # ruralCrossing_other → other
    13: 4,   # schoolZone     → other
    14: 7,   # speedLimit
    15: 8,   # stop
    16: 9,   # warning
    17: 10,  # yield
}

NEW_CLASSES = [
    "chevron",            # 0
    "curve",              # 1
    "deerCrossing",       # 2
    "guide",              # 3
    "other",              # 4  (damaged, laneEnds, intersection, schoolZone, ruralCrossing_other merged here)
    "pedestrianCrossing", # 5
    "railroadCrossing",   # 6
    "speedLimit",         # 7
    "stop",               # 8
    "warning",            # 9
    "yield",              # 10
]


def parse_label_filename(label_name: str):
    """Return (subdir, image_stem) from a Label Studio label filename."""
    stem = urllib.parse.unquote(label_name.removesuffix(".txt"))
    if "__" not in stem:
        return None, None
    _, path_part = stem.split("__", 1)
    if "\\" in path_part:
        subdir, image_stem = path_part.split("\\", 1)
    elif "/" in path_part:
        subdir, image_stem = path_part.split("/", 1)
    else:
        return None, None
    return subdir, image_stem


def find_image(images_root: Path, subdir: str, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_root / subdir / (stem + ext)
        if p.exists():
            return p
    return None


def remap_label(text: str) -> str:
    """Apply CLASS_REMAP to a label file's text. Returns remapped lines (dropped boxes omitted)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        new_cls = CLASS_REMAP.get(int(parts[0]))
        if new_cls is not None:
            out.append(f"{new_cls} {' '.join(parts[1:])}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=Path.home() / "Desktop" / "annotation_frames",
        help="Root of the annotation_frames directory (contains candidates/ and background/)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    label_files = sorted((ANNO_DIR / "labels").glob("*.txt"))
    print(f"Label files found: {len(label_files)}")

    # --- pair labels with images ---
    pairs = []
    missing_images = []
    unparseable = []

    for lf in label_files:
        subdir, stem = parse_label_filename(lf.name)
        if subdir is None:
            unparseable.append(lf.name)
            continue
        img = find_image(args.images_dir, subdir, stem)
        if img is None:
            missing_images.append(lf.name)
        else:
            pairs.append((lf, img))

    if unparseable:
        print(f"WARN: {len(unparseable)} label files could not be parsed (skipped)")
    if missing_images:
        print(f"WARN: {len(missing_images)} labels have no matching image (skipped):")
        for name in missing_images[:5]:
            print(f"  {name}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")

    # --- check for images without labels ---
    labeled_stems = set()
    for lf, img in pairs:
        labeled_stems.add(img.stem)
    total_images = sum(
        len(list((args.images_dir / sub).glob("*.*")))
        for sub in ("candidates", "background")
        if (args.images_dir / sub).exists()
    )
    unlabeled_count = total_images - len(pairs)
    if unlabeled_count > 0:
        print(f"INFO: {unlabeled_count} images have no label (will not be included)")

    print(f"Usable pairs: {len(pairs)}")

    # --- 80/20 split ---
    random.seed(args.seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    splits = {"train": shuffled[:cut], "val": shuffled[cut:]}
    print(f"Train: {len(splits['train'])}   Val: {len(splits['val'])}")

    # --- wipe and recreate output dirs to avoid stale data ---
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # --- copy to output structure, applying class remapping ---
    for split_name, split_pairs in splits.items():
        img_out = OUT_DIR / split_name / "images"
        lbl_out = OUT_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for lf, img in split_pairs:
            shutil.copy2(img, img_out / img.name)
            remapped = remap_label(lf.read_text())
            (lbl_out / (img.stem + ".txt")).write_text(remapped)

    # --- class distribution (post-remap) ---
    class_counts: dict[int, int] = defaultdict(int)
    for lf, _ in pairs:
        for line in remap_label(lf.read_text()).splitlines():
            line = line.strip()
            if line:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    print("\nClass distribution after remapping (instances across all pairs):")
    total_instances = sum(class_counts.values())
    for i, cls in enumerate(NEW_CLASSES):
        count = class_counts.get(i, 0)
        warn = "  *** LOW-SAMPLE WARNING (<10)" if count < LOW_SAMPLE_THRESHOLD else ""
        print(f"  {i:2d}  {cls:<25}  {count:5d}{warn}")
    print(f"      {'TOTAL':<25}  {total_instances:5d}")

    # --- write configs/wv.yaml ---
    config = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"\nWrote {CONFIG_PATH}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
