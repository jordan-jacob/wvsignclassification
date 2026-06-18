"""
Detect temporal near-duplicates in WV dashcam annotations.

Dashcam footage at 1fps shows the same physical sign for several consecutive
seconds, producing many near-identical frames per sign. This script groups
annotated frames into temporal clusters (same video, frames within --gap-sec
of each other), selects representative frames per cluster, and writes a
cluster CSV that prepare_wv_data.py uses for cluster-safe train/val splits.

Frame filenames are {video_id}_{timestamp_ms}.jpg (from extract_wv_frames.py).

Usage:
    python scripts/dedupe_frames.py
    python scripts/dedupe_frames.py --gap-sec 5 --keep 1 --val-frac 0.2
"""

import argparse
import csv
import random
import urllib.parse
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

# Original Label Studio class IDs for each targeted rare class
# (from CLASS_REMAP in prepare_wv_data.py)
TARGETED_ORIG_IDS: dict[str, set[int]] = {
    "curve":              {1},
    "deerCrossing":       {3},
    "pedestrianCrossing": {10},
    "railroadCrossing":   {11},
}

# Flat reverse map: orig_id → class_name
ORIG_TO_TARGET: dict[int, str] = {
    orig_id: cls
    for cls, orig_ids in TARGETED_ORIG_IDS.items()
    for orig_id in orig_ids
}

# Image roots to search for sharpness scoring (tried in order)
IMAGE_ROOTS = [
    ROOT / "data" / "annotation_frames",
    ROOT / "data" / "annotation_frames_round2",
]


def parse_label_filename(name: str):
    """Return (subdir, image_stem) from a Label Studio label filename."""
    stem = urllib.parse.unquote(name.removesuffix(".txt"))
    if "__" not in stem:
        return None, None
    _, path_part = stem.split("__", 1)
    sep = "\\" if "\\" in path_part else ("/" if "/" in path_part else None)
    if sep is None:
        return None, None
    subdir, image_stem = path_part.split(sep, 1)
    return subdir, image_stem


def parse_stem(stem: str):
    """'13892_5000' → ('13892', 5000). Returns (video_id, None) on parse fail."""
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return stem, None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return parts[0], None


def read_orig_class_ids(label_path: Path) -> set[int]:
    ids: set[int] = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass
    return ids


def max_bbox_area(label_path: Path, orig_ids: set[int]) -> float:
    """Largest bbox area (w*h in normalized coords) among boxes of given classes."""
    best = 0.0
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            if int(parts[0]) in orig_ids:
                best = max(best, float(parts[3]) * float(parts[4]))
        except ValueError:
            pass
    return best


def find_image(stem: str, subdir: str) -> Path | None:
    for root in IMAGE_ROOTS:
        for ext in (".jpg", ".jpeg", ".png"):
            p = root / subdir / (stem + ext)
            if p.exists():
                return p
    return None


def laplacian_var(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else 0.0


def frame_score(fr: dict, orig_ids: set[int]) -> float:
    """Sharpness × sqrt(bbox_area). Falls back to bbox_area if image not found."""
    area = max_bbox_area(fr["label_path"], orig_ids)
    img_path = find_image(fr["stem"], fr["subdir"])
    if img_path is not None:
        return laplacian_var(img_path) * (area ** 0.5)
    return area


def temporal_clusters(records: list[dict], gap_ms: int) -> list[list[dict]]:
    """
    Records must be pre-sorted by ts_ms. New cluster starts when gap from
    the previous frame exceeds gap_ms (not from cluster start).
    """
    if not records:
        return []
    clusters: list[list[dict]] = []
    current = [records[0]]
    for rec in records[1:]:
        if rec["ts_ms"] - current[-1]["ts_ms"] > gap_ms:
            clusters.append(current)
            current = []
        current.append(rec)
    clusters.append(current)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", type=Path,
                    default=ROOT / "data" / "wv_annotations")
    ap.add_argument("--gap-sec", type=float, default=5.0,
                    help="Gap in seconds that defines a new sign instance")
    ap.add_argument("--keep", type=int, default=1,
                    help="Representative frames to keep per cluster per targeted class")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT / "data" / "wv_clusters.csv")
    args = ap.parse_args()

    gap_ms = int(args.gap_sec * 1000)
    label_dir = args.annotations_dir / "labels"
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"Label files: {len(label_files)}")

    # ── 1. Parse all label files ──────────────────────────────────────────────
    all_frames: list[dict] = []
    skipped = 0
    for lf in label_files:
        subdir, stem = parse_label_filename(lf.name)
        if stem is None:
            skipped += 1
            continue
        vid_id, ts_ms = parse_stem(stem)
        if ts_ms is None:
            skipped += 1
            continue
        orig_ids = read_orig_class_ids(lf)
        targeted = {ORIG_TO_TARGET[i] for i in orig_ids if i in ORIG_TO_TARGET}
        all_frames.append({
            "stem":       stem,
            "subdir":     subdir or "",
            "vid_id":     vid_id,
            "ts_ms":      ts_ms,
            "label_path": lf,
            "orig_ids":   orig_ids,
            "targeted":   targeted,
        })

    if skipped:
        print(f"WARN: {skipped} label files skipped (unparseable name or non-numeric timestamp)")
    print(f"Parsed frames: {len(all_frames)}")

    # ── 2. Global temporal clustering (for split-safety across all classes) ───
    by_video: dict[str, list[dict]] = defaultdict(list)
    for fr in all_frames:
        by_video[fr["vid_id"]].append(fr)

    global_cluster_map: dict[str, str] = {}
    gid = 0
    for vid_id in sorted(by_video):
        vid_frames = sorted(by_video[vid_id], key=lambda r: r["ts_ms"])
        for cluster in temporal_clusters(vid_frames, gap_ms):
            cid = f"{vid_id}_c{gid}"
            for fr in cluster:
                global_cluster_map[fr["stem"]] = cid
            gid += 1

    n_global_clusters = gid
    print(f"Global temporal clusters: {n_global_clusters} (gap = {args.gap_sec}s)")

    # ── 3. Assign clusters to splits at cluster level ─────────────────────────
    rng = random.Random(args.seed)
    all_cluster_ids = list({cid for cid in global_cluster_map.values()})
    rng.shuffle(all_cluster_ids)
    n_val = max(1, round(n_global_clusters * args.val_frac))
    val_clusters = set(all_cluster_ids[:n_val])
    cluster_split: dict[str, str] = {
        cid: ("val" if cid in val_clusters else "train")
        for cid in all_cluster_ids
    }

    # ── 4. Per-class deduplication for targeted rare classes ──────────────────
    keep_stems: set[str] = set()

    # Non-targeted frames are always kept
    for fr in all_frames:
        if not fr["targeted"]:
            keep_stems.add(fr["stem"])

    print(f"\n{'Class':<22} {'Frames':>7} {'Instances':>10} {'→ Keep':>7}  {'Frames/sign':>12}")
    print("-" * 64)

    for cls_name in ["curve", "deerCrossing", "pedestrianCrossing", "railroadCrossing"]:
        orig_ids = TARGETED_ORIG_IDS[cls_name]
        cls_frames = [fr for fr in all_frames if cls_name in fr["targeted"]]
        before = len(cls_frames)

        # Group this class's frames by their global cluster
        by_cluster: dict[str, list[dict]] = defaultdict(list)
        for fr in cls_frames:
            by_cluster[global_cluster_map[fr["stem"]]].append(fr)

        n_instances = len(by_cluster)
        after = 0
        for cluster_frames in by_cluster.values():
            for fr in cluster_frames:
                fr["_score"] = frame_score(fr, orig_ids)
            cluster_frames.sort(key=lambda r: -r["_score"])
            for rank, fr in enumerate(cluster_frames):
                if rank < args.keep:
                    keep_stems.add(fr["stem"])
                    after += 1

        avg = before / n_instances if n_instances else 0
        print(f"  {cls_name:<20} {before:>7} {n_instances:>10} {after:>7}  {avg:>12.1f}")

    print()

    # ── 5. Write cluster CSV ──────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fr in all_frames:
        cid = global_cluster_map[fr["stem"]]
        rows.append({
            "frame_stem":       fr["stem"],
            "cluster_id":       cid,
            "split":            cluster_split[cid],
            "keep":             str(fr["stem"] in keep_stems),
            "targeted_classes": "|".join(sorted(fr["targeted"])),
        })

    rows.sort(key=lambda r: r["frame_stem"])
    fieldnames = ["frame_stem", "cluster_id", "split", "keep", "targeted_classes"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_kept = sum(1 for r in rows if r["keep"] == "True")
    print(f"Total frames: {len(all_frames)}  kept: {n_kept}  dropped (deduped): {len(all_frames) - n_kept}")
    print(f"Cluster CSV → {args.out_csv}")
    print(f"\nNext: python scripts/prepare_wv_data.py --clusters-file {args.out_csv}")


if __name__ == "__main__":
    main()
