"""
Two-stage annotation frame selection for WVDOH dashcam footage.

Stage 1: Flag candidate frames (sign-likely) via:
  - YOLOv8 phase2 model at conf=0.10
  - Color heuristics: red/yellow pixel ratio in upper third
  - High-contrast rectangular regions (speed limit signs)

Stage 2: Curate frames from candidates + background.
  Default (first pass): ~1,000 candidates + 150 background.
  Incremental (--target-total): selects only the additional frames needed
  to reach the target, excluding already-annotated frames, with
  County Route 1.5x weight and county-diversity weighting.

Outputs:
  data/annotation_frames/candidates/         (default / first pass)
  data/annotation_frames/background/
  data/annotation_frames/manifest.csv
  -- or, with --target-total --
  data/annotation_frames_round2_bulk/candidates/
  data/annotation_frames_round2_bulk/background/
  data/annotation_frames_round2_bulk/manifest.csv
"""

import argparse
import csv
import random
import shutil
import urllib.parse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


TOTAL_CANDIDATE_TARGET = 1000
BACKGROUND_TARGET = 150
BLUR_THRESHOLD = 50.0
NEAR_DUPLICATE_THRESHOLD = 0.05  # skip if < 5% pixels changed vs last accepted frame from same video
MIN_FRAMES_PER_VIDEO = 15
COUNTY_ROUTE_FRACTION = 0.40
CANDIDATE_FRACTION = 0.85  # used by incremental mode

RED_RATIO_THRESHOLD = 0.005
YELLOW_RATIO_THRESHOLD = 0.005


def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def color_candidate_reason(frame, upper_gray):
    """Return heuristic trigger string or None. Checks upper third of frame."""
    h = frame.shape[0]
    upper = frame[:h // 3, :]
    total = upper.shape[0] * upper.shape[1]

    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

    # Red: stop signs, wrong-way (two hue ranges)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)),
        cv2.inRange(hsv, (170, 80, 80), (180, 255, 255)),
    )
    if red_mask.sum() / 255 / total > RED_RATIO_THRESHOLD:
        return "red"

    # Yellow: warning signs
    yellow_mask = cv2.inRange(hsv, (20, 80, 120), (35, 255, 255))
    if yellow_mask.sum() / 255 / total > YELLOW_RATIO_THRESHOLD:
        return "yellow"

    # High-contrast rectangle: white rect with dark border (speed limit signs)
    _, thresh = cv2.threshold(upper_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 30000:
            continue
        _, _, w, bh = cv2.boundingRect(cnt)
        aspect = w / bh if bh > 0 else 0
        if 0.4 < aspect < 2.5:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) in (4, 5):
                return "contrast"

    return None


def load_video_meta(dl_csv):
    meta = {}
    for row in csv.DictReader(open(dl_csv)):
        meta[row["video_id"]] = {
            "road_type": row["sign_system_label"],
            "county": row["primary_county"],
        }
    return meta


def parse_video_id(stem):
    """Extract video_id from stem like '13892_5000' → '13892'."""
    return stem.rsplit("_", 1)[0]


def _decode_label_stem(label_filename):
    """Decode a Label Studio label filename to the image stem.

    Label files look like: {hash}__candidates%5C{video_id}_{frame}.txt
    Decoded:               {hash}__candidates\\{video_id}_{frame}
    Returns image stem (e.g. '13892_000140') or None if unparseable.
    """
    stem = urllib.parse.unquote(label_filename.removesuffix(".txt"))
    if "__" not in stem:
        return None
    _, path_part = stem.split("__", 1)
    sep = "\\" if "\\" in path_part else ("/" if "/" in path_part else None)
    if sep is None:
        return None
    _, image_stem = path_part.split(sep, 1)
    return image_stem


def load_existing_stems(annotations_dir):
    """Return set of image stems (without extension) for all Label Studio annotations."""
    labels_dir = Path(annotations_dir) / "labels"
    if not labels_dir.exists():
        return set()
    stems = set()
    for lf in labels_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        s = _decode_label_stem(lf.name)
        if s:
            stems.add(s)
    return stems


def compute_existing_county_counts(annotations_dir, video_meta):
    """Count existing annotated frames per county (for diversity weighting)."""
    county_counts = defaultdict(int)
    for stem in load_existing_stems(annotations_dir):
        vid_id = parse_video_id(stem)
        county = video_meta.get(vid_id, {}).get("county", "Unknown")
        county_counts[county] += 1
    return dict(county_counts)


def _county_weighted_sample(pool, n, existing_county_counts, rng):
    """
    Sample n records from pool.
    County Route frames get 1.5x weight. Counties with fewer existing
    annotated frames get proportionally higher weight (diversity).
    """
    if not pool or n <= 0:
        return []

    by_county = defaultdict(list)
    for r in pool:
        by_county[r["county"]].append(r)

    # County weight = mean road-type weight across frames / (1 + existing annotations)
    county_weight = {}
    for county, records in by_county.items():
        mean_rt_w = sum(1.5 if r["road_type"] == "County Route" else 1.0
                        for r in records) / len(records)
        existing = existing_county_counts.get(county, 0)
        county_weight[county] = mean_rt_w / (1 + existing)

    total_w = sum(county_weight.values())

    # Proportional budget per county, clamped to available frames
    county_budget = {}
    for county, w in county_weight.items():
        county_budget[county] = min(
            max(1, round(n * w / total_w)),
            len(by_county[county]),
        )

    # Top-up to reach n, in descending weight order
    allocated = sum(county_budget.values())
    if allocated < n:
        for county in sorted(county_weight, key=lambda c: -county_weight[c]):
            if allocated >= n:
                break
            headroom = len(by_county[county]) - county_budget[county]
            add = min(headroom, n - allocated)
            county_budget[county] += add
            allocated += add

    selected = []
    for county, budget in county_budget.items():
        selected.extend(rng.sample(by_county[county], budget))
    return selected


def select_incremental(candidates, non_candidates, n_new, existing_county_counts, rng):
    """Select n_new frames with 85/15 candidate/background split and diversity weighting."""
    n_cand = round(n_new * CANDIDATE_FRACTION)
    n_bg = n_new - n_cand
    selected_cands = _county_weighted_sample(candidates, n_cand, existing_county_counts, rng)
    selected_bg = _county_weighted_sample(non_candidates, n_bg, existing_county_counts, rng)
    return selected_cands, selected_bg


def run_stage1(frames, checkpoint, video_meta):
    """Scan all frames: YOLO at conf=0.10, color heuristics, blur check. Single read per frame."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)

    records = []
    last_frame_by_video = {}  # video_id -> last accepted frame array
    skipped_dupes = 0
    print(f"Stage 1: scanning {len(frames)} frames...")

    for i, fp in enumerate(frames):
        if i % 500 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        vid_id = parse_video_id(fp.stem)
        meta = video_meta.get(vid_id, {"road_type": "Unknown", "county": "Unknown"})

        # Near-duplicate check: skip if <5% pixels changed vs last accepted frame from this video
        prev = last_frame_by_video.get(vid_id)
        if prev is not None:
            changed = np.mean(
                np.any(np.abs(frame.astype(np.int16) - prev.astype(np.int16)) > 10, axis=2)
            )
            if changed < NEAR_DUPLICATE_THRESHOLD:
                skipped_dupes += 1
                continue
        last_frame_by_video[vid_id] = frame

        # Blur check (reuse gray for contrast heuristic too)
        h = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_sharp = laplacian_variance(gray) >= BLUR_THRESHOLD
        upper_gray = gray[:h // 3, :]

        # YOLO detection on numpy array (avoids second disk read)
        yolo_res = model(frame, conf=0.10, verbose=False)[0]
        det_count = len(yolo_res.boxes) if yolo_res.boxes is not None else 0

        # Color/contrast heuristics
        color_reason = color_candidate_reason(frame, upper_gray)

        det_reason = "model" if det_count > 0 else None
        is_candidate = det_reason is not None or color_reason is not None
        reason = det_reason or color_reason or "background"

        records.append({
            "filename": fp.name,
            "path": fp,
            "video_id": vid_id,
            "road_type": meta["road_type"],
            "county": meta["county"],
            "frame_number": int(fp.stem.rsplit("_", 1)[-1]),
            "detection_count_at_conf10": det_count,
            "candidate_reason": reason,
            "is_candidate": is_candidate,
            "is_sharp": is_sharp,
        })

    if skipped_dupes:
        print(f"  Near-duplicate frames skipped: {skipped_dupes}")
    return records


def compute_road_type_budget(video_meta, total=TOTAL_CANDIDATE_TARGET):
    """County Route = 40%; remainder split proportionally by video count."""
    county_budget = int(total * COUNTY_ROUTE_FRACTION)
    remaining = total - county_budget

    non_county = defaultdict(int)
    for m in video_meta.values():
        rt = m["road_type"]
        if rt != "County Route":
            non_county[rt] += 1

    total_non_county_videos = sum(non_county.values())
    budget = {"County Route": county_budget}
    for rt, count in non_county.items():
        budget[rt] = max(
            MIN_FRAMES_PER_VIDEO * count,
            round(remaining * count / total_non_county_videos),
        )
    return budget


def select_from_candidates(candidates, video_meta, rng):
    road_budget = compute_road_type_budget(video_meta)

    by_type = defaultdict(lambda: defaultdict(list))
    for r in candidates:
        by_type[r["road_type"]][r["video_id"]].append(r)

    selected = []
    for road_type, type_budget in road_budget.items():
        vid_groups = by_type.get(road_type, {})
        n_videos = len(vid_groups)
        if n_videos == 0:
            continue

        per_video_base = max(MIN_FRAMES_PER_VIDEO, type_budget // n_videos)
        chosen_by_vid = {}

        for vid_id, frames in vid_groups.items():
            take = min(per_video_base, len(frames))
            chosen_by_vid[vid_id] = rng.sample(frames, take)

        already_taken = sum(len(v) for v in chosen_by_vid.values())
        leftover = type_budget - already_taken

        # Distribute remaining budget to videos that still have frames
        if leftover > 0:
            eligible = sorted(
                [(vid_id, frames) for vid_id, frames in vid_groups.items()
                 if len(frames) > len(chosen_by_vid[vid_id])],
                key=lambda x: -len(x[1]),
            )
            for vid_id, frames in eligible:
                if leftover <= 0:
                    break
                already = {r["filename"] for r in chosen_by_vid[vid_id]}
                pool = [r for r in frames if r["filename"] not in already]
                add = min(leftover, len(pool))
                chosen_by_vid[vid_id].extend(rng.sample(pool, add))
                leftover -= add

        for frames in chosen_by_vid.values():
            selected.extend(frames)

    return selected


def select_background(non_candidates, rng, n=BACKGROUND_TARGET):
    """Sample n frames evenly across all videos."""
    by_video = defaultdict(list)
    for r in non_candidates:
        by_video[r["video_id"]].append(r)

    n_videos = len(by_video)
    if n_videos == 0:
        return []

    per_video = max(1, n // n_videos)
    selected = []
    for frames in by_video.values():
        take = min(per_video, len(frames))
        selected.extend(rng.sample(frames, take))

    # Top up if under target
    if len(selected) < n:
        picked = {r["filename"] for r in selected}
        pool = [r for r in non_candidates if r["filename"] not in picked]
        rng.shuffle(pool)
        selected.extend(pool[: n - len(selected)])

    return selected[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/raw/wvdoh_frames/")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory. Defaults to data/annotation_frames/ "
                         "or data/annotation_frames_round2_bulk/ when --target-total is set.")
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt")
    ap.add_argument("--dl-csv", default="configs/wv_download_list.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-existing", action="store_true",
                    help="Exclude frames already in --annotations-dir from the pool.")
    ap.add_argument("--annotations-dir", default="data/wv_annotations/",
                    help="Label Studio export directory (used by --exclude-existing "
                         "and --target-total for existing county counts).")
    ap.add_argument("--target-total", type=int, default=None,
                    help="Target total annotation set size. Script selects "
                         "(target - existing) new frames using county-diversity weighting.")
    args = ap.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = (
            "data/annotation_frames_bulk/" if args.target_total else "data/annotation_frames/"
        )

    rng = random.Random(args.seed)
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    annotations_dir = Path(args.annotations_dir)
    cand_dir = output_dir / "candidates"
    bg_dir = output_dir / "background"
    cand_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # Startup diagnostics — makes path issues immediately obvious
    children = list(frames_dir.iterdir()) if frames_dir.exists() else []
    all_jpgs = list(frames_dir.rglob("*.jpg")) if frames_dir.exists() else []
    print(f"frames_dir path: {frames_dir.resolve()}")
    print(f"path exists: {frames_dir.exists()}")
    print(f"direct children: {len(children)} items")
    print(f"total jpgs found via rglob: {len(all_jpgs)}")

    video_meta = load_video_meta(args.dl_csv)

    # Build exclusion set
    excluded_stems = set()
    if args.exclude_existing or args.target_total:
        excluded_stems = load_existing_stems(annotations_dir)
        print(f"Existing annotated frames: {len(excluded_stems)}")

    frames = sorted(frames_dir.rglob("*.jpg"))
    if excluded_stems:
        frames = [f for f in frames if f.stem not in excluded_stems]
        print(f"Frames after exclusion: {len(frames)} (of {len(all_jpgs)} total)")

    # Show existing county distribution before scanning (useful even when frames not local)
    existing_county_counts: dict[str, int] = {}
    if args.target_total:
        n_existing = len(excluded_stems)
        n_new = args.target_total - n_existing
        if n_new <= 0:
            print(f"Already at or above target ({n_existing} >= {args.target_total}). Nothing to do.")
            return
        print(f"Target: {args.target_total}  Existing: {n_existing}  Need: {n_new}")
        print(f"Split: {round(n_new * CANDIDATE_FRACTION)} candidates + "
              f"{n_new - round(n_new * CANDIDATE_FRACTION)} background")
        existing_county_counts = compute_existing_county_counts(annotations_dir, video_meta)
        print(f"Existing county coverage: {len(existing_county_counts)} counties")
        print("  Existing frames per county (top 10):")
        for county, cnt in sorted(existing_county_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {county:<25} {cnt:>5}")

    if not frames:
        print("No frames to scan. Exiting.")
        return

    all_records = run_stage1(frames, args.checkpoint, video_meta)

    # Split by candidacy and sharpness
    candidates = [r for r in all_records if r["is_candidate"] and r["is_sharp"]]
    non_candidates = [r for r in all_records if not r["is_candidate"] and r["is_sharp"]]
    candidates_raw = [r for r in all_records if r["is_candidate"]]

    print(f"Stage 2: {len(candidates)} sharp candidates, {len(non_candidates)} sharp non-candidates")

    if args.target_total:

        selected_candidates, selected_background = select_incremental(
            candidates, non_candidates, n_new, existing_county_counts, rng
        )
    else:
        selected_candidates = select_from_candidates(candidates, video_meta, rng)
        selected_background = select_background(non_candidates, rng)

    for r in selected_background:
        r["candidate_reason"] = "background"

    # Copy to output directories
    print("Copying candidates...")
    for r in selected_candidates:
        shutil.copy2(r["path"], cand_dir / r["filename"])

    print("Copying background...")
    for r in selected_background:
        shutil.copy2(r["path"], bg_dir / r["filename"])

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "filename", "video_id", "road_type", "county",
        "frame_number", "detection_count_at_conf10", "candidate_reason",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected_candidates + selected_background:
            writer.writerow({k: r[k] for k in fieldnames})

    # Summary
    total_scanned = len(all_records)
    n_cand_raw = len(candidates_raw)
    n_cand_sharp = len(candidates)
    n_final = len(selected_candidates)
    n_bg = len(selected_background)
    n_total_new = n_final + n_bg
    est_hours = n_total_new / 380

    print(f"\n{'=' * 50}")
    print(f"Total frames scanned:      {total_scanned:>6}")
    print(f"Candidates found:          {n_cand_raw:>6}  ({100*n_cand_raw/max(total_scanned,1):.1f}%)")
    print(f"After blur filter:         {n_cand_sharp:>6}")
    print(f"Candidates selected:       {n_final:>6}")
    print(f"Background selected:       {n_bg:>6}")
    print(f"Total new selection:       {n_total_new:>6}")
    print(f"Estimated annotation time: {est_hours:.1f} hrs  ({n_total_new} frames @ 380 frames/hr)")

    if args.target_total:
        all_selected = selected_candidates + selected_background

        road_type_counts = defaultdict(int)
        county_counts = defaultdict(int)
        for r in all_selected:
            road_type_counts[r["road_type"]] += 1
            county_counts[r["county"]] += 1

        print(f"\nBreakdown by road type:")
        for rt, cnt in sorted(road_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {rt:<30} {cnt:>5}")

        print(f"\nTop counties by new frames selected:")
        for county, cnt in sorted(county_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {county:<25} {cnt:>5}")

    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
