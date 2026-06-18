"""
Find targeted annotation frames for rare WV sign classes.

Uses checkpoints/phase2_full_best.pt (LISA 47-class) at conf=0.15 on
unannotated frames in data/raw/wvdoh_frames/. Maps LISA classes to WV
target buckets. For classes with no LISA equivalent (deerCrossing,
railroadCrossing) and for curve signs that the LISA model misses on WV
footage, uses per-class diamond-interior heuristics to separate candidates
rather than the old approach of assigning every yellow diamond to both
deerCrossing and railroadCrossing simultaneously.

Diamond interior heuristics (run on yellow diamonds found in upper 60%):
  deerCrossing:     single irregular dark blob in lower 45% of diamond
                    (animal silhouette — low solidity, organic shape)
  railroadCrossing: diagonal edge pattern inside diamond
                    (RXR advance-warning symbol has crossing diagonals)
  curve:            model path first; heuristic fallback is geometric
                    directional blob (arrow — higher solidity than deer)

Frames that pass the yellow-diamond shape test but fail all interior
heuristics are bucketed as possible_yellow_diamond for manual annotation.

Usage:
    python scripts/find_targeted_frames.py
    python scripts/find_targeted_frames.py --top-k 60 --conf 0.12
    python scripts/find_targeted_frames.py --csv configs/wv_supplemental_download.csv
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# LISA classes that map directly to WV target buckets
CURVE_CLASSES = frozenset({"curveRight", "curveLeft", "turnRight", "turnLeft"})
PEDESTRIAN_CLASSES = frozenset({"pedestrianCrossing"})

# LISA warning-like classes that might be WV rural signs misclassified
RURAL_CANDIDATE_CLASSES = frozenset({
    "school", "roundabout", "intersection", "dip", "slow",
    "signalAhead", "addedLane",
})

TARGET_CLASSES = [
    "curve", "pedestrianCrossing", "deerCrossing", "railroadCrossing",
    "possible_yellow_diamond", "possible_rural_crossing",
]
DEFAULT_TOP_K: dict[str, int] = {
    "curve":                  100,
    "pedestrianCrossing":     80,
    "deerCrossing":           80,
    "railroadCrossing":       80,
    "possible_yellow_diamond": 60,
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


# ── Yellow-diamond heuristics ──────────────────────────────────────────────────

def find_best_yellow_diamond(frame: np.ndarray) -> tuple[float, tuple | None]:
    """
    Find the best yellow diamond warning sign in the upper 60% of the frame.
    Returns (score, (x, y, w, h)) in full-frame pixel coords, or (0.0, None).
    Score is in [0, 1].
    """
    fh = frame.shape[0]
    roi_h = int(0.6 * fh)
    roi = frame[:roi_h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (18, 100, 120), (38, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_bbox: tuple | None = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 60_000:
            continue
        rx, ry, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if not (0.55 < aspect < 1.8):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 6):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.55:
            continue
        score = min(area / 15_000, 1.0) * (1.0 - abs(aspect - 1.0) / 1.25) * solidity
        if score > best_score:
            best_score = score
            # ROI starts at y=0, so roi coords == frame coords for x/y
            best_bbox = (rx, ry, bw, bh)

    return best_score, best_bbox


def _diamond_interior(frame: np.ndarray, bbox: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the diamond region and return (crop, interior_mask).

    interior_mask is the eroded filled diamond shape — it covers the entire
    sign face (yellow background AND dark sign content), eroded inward to
    strip the border. This prevents:
      - Railroad false-positives from the diamond's own 4 diagonal edges
      - Deer false-positives from dark road background in bounding-rect corners

    We fill the diamond contour solid rather than keeping only yellow pixels,
    because dark sign content (deer pictogram, railroad lines, arrows) is
    explicitly non-yellow and would be self-masked out of a yellow-only mask.
    """
    x, y, w, h = bbox
    pad = max(4, int(0.06 * min(w, h)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    crop = frame[y0:y1, x0:x1]

    # Detect the yellow sign outline
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 70, 100), (42, 255, 255))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, close_k)

    # Fill the largest yellow contour solid so interior covers dark content too
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sign_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.fillPoly(sign_mask, [largest], 255)

    # Erode inward ~12% of the shorter dimension to strip the border
    k_size = max(5, int(0.12 * min(crop.shape[:2]))) | 1  # ensure odd
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    interior = cv2.erode(sign_mask, erode_k, iterations=1)

    return crop, interior


def deer_pictogram_score(frame: np.ndarray, bbox: tuple) -> float:
    """
    Score for deer crossing: a single irregular dark blob in the lower half
    of the sign interior. Deer silhouette has low solidity (concavities from
    legs/antlers) vs. geometric shapes (arrows, letters) which are more solid.

    Split point is the actual diamond vertical center (from bbox), not a
    fixed crop-height fraction, so the check is stable regardless of padding.
    Railroad crossing lines span the full sign and produce near-equal dark
    coverage above and below center — the symmetry check rejects them.
    Returns float in [0, 1]; 0 = no evidence.
    """
    crop, interior = _diamond_interior(frame, bbox)
    if crop.size == 0 or cv2.countNonZero(interior) == 0:
        return 0.0

    # Diamond center in crop coordinates (anchored to bbox, not crop height %)
    x, y, w, h = bbox
    pad = max(4, int(0.06 * min(w, h)))
    y0 = max(0, y - pad)
    center_y = (y + h // 2) - y0

    rh = crop.shape[0]
    lower_mask = interior.copy()
    lower_mask[:center_y, :] = 0   # keep only below diamond center

    if cv2.countNonZero(lower_mask) < 50:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, dark_all = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    dark_lower = cv2.bitwise_and(dark_all, lower_mask)

    contours, _ = cv2.findContours(dark_lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    interior_area = cv2.countNonZero(lower_mask)
    largest = max(contours, key=cv2.contourArea)
    blob_area = cv2.contourArea(largest)

    if blob_area < 0.04 * interior_area or blob_area > 0.65 * interior_area:
        return 0.0

    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area == 0:
        return 0.0
    solidity = blob_area / hull_area

    # Animal silhouette: irregular (solidity 0.35–0.78)
    # Geometric/arrow: > 0.80
    if solidity > 0.82:
        return 0.0

    # Symmetry check anchored to diamond center: deer silhouette is concentrated
    # below center; railroad X lines span both halves roughly equally.
    upper_mask = interior.copy()
    upper_mask[center_y:, :] = 0
    n_dark_upper = cv2.countNonZero(cv2.bitwise_and(dark_all, upper_mask))
    n_dark_lower = cv2.countNonZero(dark_lower)
    if n_dark_lower > 0 and n_dark_upper / n_dark_lower > 0.65:
        return 0.0   # roughly symmetric dark content → crossing pattern, not deer

    irregularity = 1.0 - solidity
    coverage = blob_area / interior_area
    return float(irregularity * min(coverage / 0.15, 1.0))


def railroad_pattern_score(frame: np.ndarray, bbox: tuple) -> float:
    """
    Score for railroad advance warning: diagonal edges inside the sign interior.
    The WV 'RXR' advance-warning diamond has two crossing diagonal strokes.

    Canny + HoughLinesP is applied only within the eroded interior mask to
    avoid counting the diamond's own 4 diagonal boundary edges.
    Returns float in [0, 1]; 0 = no evidence.
    """
    crop, interior = _diamond_interior(frame, bbox)
    if crop.size == 0 or cv2.countNonZero(interior) == 0:
        return 0.0

    rh, rw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    # Mask to interior only — this is the key fix vs. naively using boundingRect
    edges = cv2.bitwise_and(edges, interior)

    min_len = int(0.15 * min(rh, rw))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=10, minLineLength=min_len, maxLineGap=8,
    )
    if lines is None:
        return 0.0

    diagonal_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if 20 <= angle <= 70:   # wide range to tolerate sign tilt/perspective
            diagonal_count += 1

    if diagonal_count < 2:
        return 0.0
    return float(min(diagonal_count / 6.0, 1.0))


def curve_arrow_score(frame: np.ndarray, bbox: tuple) -> float:
    """
    Score for a curve/turn arrow inside a yellow diamond.
    Arrows: few blobs (1–4), moderate-to-high solidity (geometric, not organic),
    coverage 6–55% of the interior, AND dark content is NOT concentrated
    exclusively below diamond center (which would indicate a deer silhouette).
    Analysis masked to eroded interior.
    Returns float in [0, 1]; 0 = no evidence.
    """
    crop, interior = _diamond_interior(frame, bbox)
    if crop.size == 0 or cv2.countNonZero(interior) == 0:
        return 0.0

    # Diamond center in crop coordinates (same calc as deer_pictogram_score)
    x, y, w, h = bbox
    pad = max(4, int(0.06 * min(w, h)))
    y0 = max(0, y - pad)
    center_y = (y + h // 2) - y0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, dark_all = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark_all, interior)

    # Reject if dark content is entirely below center (deer, not arrow)
    upper_mask = interior.copy()
    upper_mask[center_y:, :] = 0
    n_dark_upper = cv2.countNonZero(cv2.bitwise_and(dark_all, upper_mask))
    n_dark_total = cv2.countNonZero(dark)
    if n_dark_total > 0 and n_dark_upper / n_dark_total < 0.10:
        return 0.0   # content confined to lower half → deer, not arrow

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    interior_area = cv2.countNonZero(interior)
    significant = [c for c in contours if cv2.contourArea(c) > 0.015 * interior_area]
    if not (1 <= len(significant) <= 4):
        return 0.0

    total_area = sum(cv2.contourArea(c) for c in significant)
    if total_area < 0.06 * interior_area or total_area > 0.55 * interior_area:
        return 0.0

    solidities = []
    for c in significant:
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0:
            solidities.append(cv2.contourArea(c) / hull)

    if not solidities:
        return 0.0
    mean_sol = sum(solidities) / len(solidities)

    # Arrow more geometric (> 0.60) than animal silhouette
    if mean_sol < 0.60:
        return 0.0

    coverage = min(total_area / (0.22 * interior_area), 1.0)
    return float(mean_sol * coverage)


def classify_yellow_diamond(frame: np.ndarray) -> dict[str, tuple[float, str]]:
    """
    Full yellow-diamond pipeline: find sign, classify interior, return per-class hits.
    Never assigns the same frame to both deerCrossing and railroadCrossing.
    Falls back to possible_yellow_diamond when interior is ambiguous.
    """
    yd_score, bbox = find_best_yellow_diamond(frame)
    if yd_score < 0.05 or bbox is None:
        return {}

    deer_s  = deer_pictogram_score(frame, bbox)
    rail_s  = railroad_pattern_score(frame, bbox)
    curve_s = curve_arrow_score(frame, bbox)

    DEER_THRESH  = 0.12
    RAIL_THRESH  = 0.25
    CURVE_THRESH = 0.22

    hits: dict[str, tuple[float, str]] = {}

    if deer_s >= DEER_THRESH:
        hits["deerCrossing"] = (yd_score * (0.5 + deer_s), "deer_pictogram_heuristic")
    if rail_s >= RAIL_THRESH:
        hits["railroadCrossing"] = (yd_score * (0.5 + rail_s), "railroad_pattern_heuristic")
    if curve_s >= CURVE_THRESH:
        hits["curve"] = (yd_score * curve_s, "curve_arrow_heuristic")

    if not hits:
        hits["possible_yellow_diamond"] = (yd_score, "yellow_diamond_heuristic")

    return hits


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/raw/wvdoh_frames/")
    ap.add_argument("--annotated-dir", default="data/annotation_frames/")
    ap.add_argument("--output-dir", default="data/annotation_frames_round2/")
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt")
    ap.add_argument("--dl-csv", default="configs/wv_download_list.csv")
    ap.add_argument("--csv", default=None,
                    help="restrict scan to video_ids listed in this CSV (video_id column); "
                         "scans data/raw/wvdoh_frames/{video_id}/ for each ID")
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

    # Collect frames: specific video_id subdirs (--csv) or full rglob
    if args.csv is not None:
        with open(args.csv, newline="") as f:
            target_ids = [row["video_id"] for row in csv.DictReader(f)]
        all_frames = []
        for vid_id in target_ids:
            vid_dir = frames_dir / vid_id
            if vid_dir.is_dir():
                all_frames.extend(sorted(vid_dir.glob("*.jpg")))
            else:
                all_frames.extend(sorted(frames_dir.glob(f"{vid_id}_*.jpg")))
        all_frames.sort()
        print(f"Scanning {len(target_ids)} video_ids from {args.csv}")
    else:
        all_frames = sorted(frames_dir.rglob("*.jpg"))

    frames = [f for f in all_frames if f.name not in annotated]
    print(f"Frames to scan: {len(frames)} (of {len(all_frames)} total)")

    # bucket → list of record dicts
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

        # Per-frame best hit per bucket: (score, predicted_class, detection_method)
        hits: dict[str, tuple[float, str, str]] = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_name = model.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw_box = x2 - x1
                bh_box = y2 - y1
                aspect = bw_box / bh_box if bh_box > 0 else 0

                if cls_name in CURVE_CLASSES:
                    if conf_val > hits.get("curve", (0,))[0]:
                        hits["curve"] = (conf_val, cls_name, "model")
                elif cls_name in PEDESTRIAN_CLASSES:
                    if conf_val > hits.get("pedestrianCrossing", (0,))[0]:
                        hits["pedestrianCrossing"] = (conf_val, cls_name, "model")
                elif cls_name in RURAL_CANDIDATE_CLASSES and 0.5 < aspect < 2.0:
                    if conf_val > hits.get("possible_rural_crossing", (0,))[0]:
                        hits["possible_rural_crossing"] = (conf_val, cls_name, "model")

        # Yellow-diamond heuristics: classify interior to separate deer/railroad/curve
        for cls_name, (score, method) in classify_yellow_diamond(frame).items():
            # Heuristic only overrides model if model score is lower
            if score > hits.get(cls_name, (0,))[0]:
                hits[cls_name] = (score, cls_name, method)

        for bucket, (score, pred_cls, method) in hits.items():
            buckets[bucket].append({
                "path":             fp,
                "filename":         fp.name,
                "video_id":         vid_id,
                "county":           county,
                "predicted_class":  pred_cls,
                "detection_method": method,
                "confidence":       round(score, 4),
            })

    # Write output — wipe first so consecutive runs don't accumulate
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest_rows = []

    print("\n--- Candidates found per target class ---")
    for bucket in TARGET_CLASSES:
        records = sorted(buckets.get(bucket, []), key=lambda r: -r["confidence"])
        total = len(records)
        selected = records[: top_k[bucket]]

        # Method breakdown for transparency
        method_counts: dict[str, int] = defaultdict(int)
        for r in records:
            method_counts[r["detection_method"]] += 1
        method_summary = "  ".join(f"{m}={n}" for m, n in sorted(method_counts.items()))

        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(exist_ok=True)

        for rec in selected:
            shutil.copy2(rec["path"], bucket_dir / rec["filename"])
            manifest_rows.append({
                "filename":         rec["filename"],
                "target_class":     bucket,
                "predicted_class":  rec["predicted_class"],
                "detection_method": rec["detection_method"],
                "confidence":       rec["confidence"],
                "video_id":         rec["video_id"],
                "county":           rec["county"],
            })

        print(f"  {bucket:<28} {total:>5} candidates  (top {len(selected)} selected)")
        if method_summary:
            print(f"    methods: {method_summary}")

    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "filename", "target_class", "predicted_class", "detection_method",
        "confidence", "video_id", "county",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path}")
    print(f"Output:   {output_dir}")


if __name__ == "__main__":
    main()
