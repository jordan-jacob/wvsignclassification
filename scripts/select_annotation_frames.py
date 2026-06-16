"""
Two-stage annotation frame selection for WVDOH dashcam footage.

Stage 1: Flag candidate frames (sign-likely) via:
  - YOLOv8 phase2 model at conf=0.10
  - Color heuristics: red/yellow pixel ratio in upper third
  - High-contrast rectangular regions (speed limit signs)

Stage 2: Curate ~1,000 frames from candidates + 150 background frames.
  - Remove blurry frames (Laplacian variance < 50)
  - Min 15 frames per video
  - Road-type proportional budget (County Route = 40%)
  - Background sampled evenly across all videos

Outputs:
  data/annotation_frames/candidates/
  data/annotation_frames/background/
  data/annotation_frames/manifest.csv
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


TOTAL_CANDIDATE_TARGET = 1000
BACKGROUND_TARGET = 150
BLUR_THRESHOLD = 50.0
MIN_FRAMES_PER_VIDEO = 15
COUNTY_ROUTE_FRACTION = 0.40

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


def run_stage1(frames, checkpoint, video_meta):
    """Scan all frames: YOLO at conf=0.10, color heuristics, blur check. Single read per frame."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)

    records = []
    print(f"Stage 1: scanning {len(frames)} frames...")

    for i, fp in enumerate(frames):
        if i % 500 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        vid_id = parse_video_id(fp.stem)
        meta = video_meta.get(vid_id, {"road_type": "Unknown", "county": "Unknown"})

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
    ap.add_argument("--output-dir", default="data/annotation_frames/")
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt")
    ap.add_argument("--dl-csv", default="configs/wv_download_list.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    cand_dir = output_dir / "candidates"
    bg_dir = output_dir / "background"
    cand_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    video_meta = load_video_meta(args.dl_csv)
    frames = sorted(frames_dir.glob("*.jpg"))

    all_records = run_stage1(frames, args.checkpoint, video_meta)

    # Split by candidacy and sharpness
    candidates = [r for r in all_records if r["is_candidate"] and r["is_sharp"]]
    non_candidates = [r for r in all_records if not r["is_candidate"] and r["is_sharp"]]
    candidates_raw = [r for r in all_records if r["is_candidate"]]

    print(f"Stage 2: selecting from {len(candidates)} sharp candidates, "
          f"{len(non_candidates)} sharp non-candidates...")

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
    est_hours = (n_final * 2 + n_bg * 0.5) / 60

    print(f"\n{'=' * 50}")
    print(f"Total frames scanned:      {total_scanned:>6}")
    print(f"Candidates found:          {n_cand_raw:>6}  ({100*n_cand_raw/max(total_scanned,1):.1f}%)")
    print(f"After blur filter:         {n_cand_sharp:>6}")
    print(f"Final selection:           {n_final:>6}")
    print(f"Background frames:         {n_bg:>6}")
    print(f"Total annotation set:      {n_final + n_bg:>6}")
    print(f"Estimated annotation time: {est_hours:.1f} hrs")
    print(f"  (candidates ~2 min/frame, background ~0.5 min/frame)")
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
