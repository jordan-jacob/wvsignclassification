"""
Find targeted annotation frames for rare WV sign classes.

Uses checkpoints/phase2_full_best.pt (LISA 47-class) at conf=0.15 on
unannotated frames in data/raw/wvdoh_frames/. Maps LISA classes to WV
target buckets and runs a yellow-diamond heuristic for deer crossing and
railroad crossing (no LISA equivalents exist for these).

Outputs top --top-k candidates per target class to data/annotation_frames_round2/.
One subfolder per class; manifest.csv covers all selected frames.

Usage:
    python scripts/find_targeted_frames.py
    python scripts/find_targeted_frames.py --top-k 60 --conf 0.12
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# LISA classes that map directly to WV target buckets
CURVE_CLASSES = frozenset({"curveRight", "curveLeft"})
PEDESTRIAN_CLASSES = frozenset({"pedestrianCrossing"})

# LISA warning-like classes that might be WV rural signs misclassified
# (diamond-shaped signs the model has never seen assigned to nearest LISA class)
RURAL_CANDIDATE_CLASSES = frozenset({
    "school", "roundabout", "intersection", "dip", "slow",
    "signalAhead", "addedLane",
})

# Output order and per-class top-k defaults
TARGET_CLASSES = ["curve", "pedestrianCrossing", "deerCrossing", "railroadCrossing", "possible_rural_crossing"]
DEFAULT_TOP_K: dict[str, int] = {
    "curve": 100,
    "pedestrianCrossing": 80,
    "deerCrossing": 80,
    "railroadCrossing": 80,
    "possible_rural_crossing": 40,
}


def load_video_meta(dl_csv: Path) -> dict:
    meta = {}
    with open(dl_csv, newline="") as f:
        for row in csv.DictReader(f):
            meta[row["video_id"]] = {
                "road_type": row.get("sign_system_label", ""),
                "county": row.get("primary_county", ""),
            }
    return meta


def parse_video_id(stem: str) -> str:
    """'13892_5000' → '13892'"""
    return stem.rsplit("_", 1)[0]


def yellow_diamond_score(frame: np.ndarray) -> float:
    """
    Score a frame for a yellow diamond warning sign in the upper 60%.
    Returns float in [0, 1]; 0 means no plausible candidate found.
    """
    h = frame.shape[0]
    roi = frame[: int(0.6 * h), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (18, 100, 120), (38, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 60_000:
            continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if not (0.55 < aspect < 1.8):  # diamond signs are roughly square
            continue
        # Polygon approx: diamond should compress to ~4 vertices
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 6):
            continue
        # Solidity: solid yellow fill (rules out road markings, foliage patches)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.55:
            continue
        area_score = min(area / 15_000, 1.0)
        aspect_score = 1.0 - abs(aspect - 1.0) / 1.25
        score = area_score * aspect_score * solidity
        if score > best:
            best = score

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/raw/wvdoh_frames/")
    ap.add_argument("--annotated-dir", default="data/annotation_frames/")
    ap.add_argument("--output-dir", default="data/annotation_frames_round2/")
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt")
    ap.add_argument("--dl-csv", default="configs/wv_download_list.csv")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--top-k", type=int, default=None,
                    help="Override top-k for all classes (default: per-class values in DEFAULT_TOP_K)")
    args = ap.parse_args()

    top_k = {cls: (args.top_k if args.top_k is not None else DEFAULT_TOP_K[cls])
             for cls in TARGET_CLASSES}

    from ultralytics import YOLO
    model = YOLO(args.checkpoint)

    frames_dir = Path(args.frames_dir)
    annotated_dir = Path(args.annotated_dir)
    output_dir = Path(args.output_dir)

    # Build exclusion set from already-annotated frames
    annotated = set()
    for subdir in ("candidates", "background"):
        d = annotated_dir / subdir
        if d.exists():
            annotated.update(p.name for p in d.iterdir())
    print(f"Excluding {len(annotated)} already-annotated frames")

    video_meta = load_video_meta(Path(args.dl_csv))

    all_frames = sorted(frames_dir.rglob("*.jpg"))
    frames = [f for f in all_frames if f.name not in annotated]
    print(f"Frames to scan: {len(frames)} (of {len(all_frames)} total)")

    # bucket → list of record dicts (include path for copying, exclude from CSV)
    buckets: dict[str, list] = defaultdict(list)

    print(f"Running inference at conf={args.conf} ...")
    for i, fp in enumerate(frames):
        if i % 500 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        vid_id = parse_video_id(fp.stem)
        county = video_meta.get(vid_id, {}).get("county", "")

        # Model inference
        results = model(frame, conf=args.conf, verbose=False)[0]
        boxes = results.boxes

        # Per-frame best hit per bucket (score, lisa_class)
        hits: dict[str, tuple[float, str]] = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = x2 - x1, y2 - y1
                aspect = bw / bh if bh > 0 else 0

                if cls_name in CURVE_CLASSES:
                    if conf > hits.get("curve", (0,))[0]:
                        hits["curve"] = (conf, cls_name)
                elif cls_name in PEDESTRIAN_CLASSES:
                    if conf > hits.get("pedestrianCrossing", (0,))[0]:
                        hits["pedestrianCrossing"] = (conf, cls_name)
                elif cls_name in RURAL_CANDIDATE_CLASSES and 0.5 < aspect < 2.0:
                    # Low-confidence detection with diamond-ish bbox → possible rural crossing
                    if conf > hits.get("possible_rural_crossing", (0,))[0]:
                        hits["possible_rural_crossing"] = (conf, cls_name)

        # Yellow-diamond heuristic covers deerCrossing and railroadCrossing
        # (LISA has no equivalent classes for these)
        yd = yellow_diamond_score(frame)
        if yd > 0.05:
            hits["deerCrossing"] = (yd, "yellow_diamond_heuristic")
            hits["railroadCrossing"] = (yd, "yellow_diamond_heuristic")

        for bucket, (score, lisa_cls) in hits.items():
            buckets[bucket].append({
                "path": fp,
                "filename": fp.name,
                "video_id": vid_id,
                "county": county,
                "predicted_class": lisa_cls,
                "confidence": round(score, 4),
            })

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    print("\n--- Candidates found per target class ---")
    for bucket in TARGET_CLASSES:
        records = sorted(buckets.get(bucket, []), key=lambda r: -r["confidence"])
        total = len(records)
        selected = records[: top_k[bucket]]

        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(exist_ok=True)

        for rec in selected:
            shutil.copy2(rec["path"], bucket_dir / rec["filename"])
            manifest_rows.append({
                "filename": rec["filename"],
                "target_class": bucket,
                "predicted_class": rec["predicted_class"],
                "confidence": rec["confidence"],
                "video_id": rec["video_id"],
                "county": rec["county"],
            })

        print(f"  {bucket:<25} {total:>5} candidates  (top {len(selected)} selected)")

    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["filename", "target_class", "predicted_class", "confidence", "video_id", "county"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path}")
    print(f"Output:   {output_dir}")


if __name__ == "__main__":
    main()
