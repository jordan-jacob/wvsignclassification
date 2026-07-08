"""
Collect close-range sign training examples from WVDOH dashcam videos.

Processes one video at a time: download → extract 1fps frames → run YOLO →
keep close-range target-class frames → delete everything else. Stays within
a 20GB disk budget in candidates\\

Usage:
  python scripts/collect_candidates_local.py [--smoke] [--conf FLOAT]

  --smoke   first 3 videos, 30 frames each (for testing)
  --conf    detection confidence threshold (default 0.25)
"""

import argparse
import csv
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import cv2
import requests
from ultralytics import YOLO


BASE_DIR = Path(r"C:\Users\jrj00048\Desktop\wv_collection")
MANIFEST = BASE_DIR / "wvu_sample_manifest.csv"
EXCLUDE_CSVS = [
    BASE_DIR / "wv_download_list.csv",
    BASE_DIR / "wv_supplemental_download.csv",
    BASE_DIR / "wv_supplemental_download_v2.csv",
    BASE_DIR / "wv_annotation_round3.csv",
]
CHECKPOINT = BASE_DIR / "phase3_wv_best.pt"
CANDIDATES_DIR = BASE_DIR / "candidates"
TEMP_DIR = BASE_DIR / "temp"
LOG_CSV = BASE_DIR / "collection_log.csv"
PROCESSED_TXT = BASE_DIR / "processed_videos.txt"

TARGET_CLASSES = {3, 4, 5, 6}
CLASS_NAMES = {3: "MileMarkers", 4: "Regulatory", 5: "SpeedLimits", 6: "StopSigns"}
CLOSE_RANGE_MIN_DIM = 0.08
BUDGET_GB = 20.0
DEFAULT_CONF = 0.25

# 0=exclude (interstate), 1=urban, 2=rural, 3=unknown fallback
ROAD_PRIORITY = {
    "Interstate": 0,
    "US Route": 1,
    "WV Route": 1,
    "Municipal Non-State": 1,
    "County Route": 2,
    "Federal Aid Non-State": 2,
}


def candidates_size_gb():
    if not CANDIDATES_DIR.exists():
        return 0.0
    return sum(f.stat().st_size for f in CANDIDATES_DIR.iterdir() if f.is_file()) / 1e9


def disk_free_gb():
    return shutil.disk_usage(BASE_DIR).free / 1e9


def load_exclude_ids():
    excluded = set()
    for csv_path in EXCLUDE_CSVS:
        if not csv_path.exists():
            continue
        try:
            for row in csv.DictReader(open(csv_path)):
                vid_id = row.get("video_id") or (list(row.values())[0] if row else None)
                if vid_id:
                    excluded.add(vid_id.strip())
        except Exception:
            pass
    return excluded


def load_processed():
    if not PROCESSED_TXT.exists():
        return set()
    return {ln.strip() for ln in PROCESSED_TXT.read_text().splitlines() if ln.strip()}


def download_video(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(unquote(url), stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest.stat().st_size / 1e6


def extract_frames(video_path, out_dir, stem, max_frames=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    frame_num = 0
    ms = 0.0
    while True:
        if max_frames is not None and frame_num >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ret, frame = cap.read()
        if not ret:
            break
        path = out_dir / f"{stem}_{frame_num:06d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
        frame_num += 1
        ms += 1000.0
    cap.release()
    return paths


def is_close_range(box):
    cls_id = int(box.cls[0])
    if cls_id not in TARGET_CLASSES:
        return False
    bw = float(box.xywhn[0][2])
    bh = float(box.xywhn[0][3])
    return bw > CLOSE_RANGE_MIN_DIM or bh > CLOSE_RANGE_MIN_DIM


def run_inference(model, frame_paths, conf):
    kept = []
    class_counts = {k: 0 for k in CLASS_NAMES}
    for path in frame_paths:
        results = model(str(path), conf=conf, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            continue
        close = [b for b in boxes if is_close_range(b)]
        if close:
            kept.append(path)
            for b in close:
                cls_id = int(b.cls[0])
                if cls_id in class_counts:
                    class_counts[cls_id] += 1
    return kept, class_counts


def prioritize_rows(rows):
    rows = [r for r in rows if ROAD_PRIORITY.get(r.get("sign_system_label", ""), 3) != 0]
    rows.sort(key=lambda r: ROAD_PRIORITY.get(r.get("sign_system_label", ""), 3))
    return rows


def append_log(row_data):
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_name", "frames_extracted", "frames_kept",
            "MileMarkers_detections", "Regulatory_detections",
            "SpeedLimits_detections", "StopSigns_detections",
            "disk_gb_remaining",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)


def print_summary(total_videos, total_kept, class_totals):
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print(f"  Videos processed   : {total_videos}")
    print(f"  Candidate frames   : {total_kept}")
    for cls_id, name in CLASS_NAMES.items():
        print(f"  {name:<20}: {class_totals[cls_id]}")
    print(f"  candidates/ usage  : {candidates_size_gb():.2f} GB")
    print(f"  Disk free          : {disk_free_gb():.1f} GB")
    print(f"  Log                : {LOG_CSV}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="first 3 videos, 30 frames each")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = ap.parse_args()

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(CHECKPOINT))
    excluded = load_exclude_ids()
    processed = load_processed()

    rows = prioritize_rows(list(csv.DictReader(open(MANIFEST))))
    if args.smoke:
        rows = rows[:3]

    total_videos = 0
    total_kept = 0
    class_totals = {k: 0 for k in CLASS_NAMES}

    for row in rows:
        video_id = row.get("video_id", "").strip()
        url = row.get("video_link", "").strip()
        if not video_id or not url:
            continue
        if video_id in processed or video_id in excluded:
            continue

        if candidates_size_gb() >= BUDGET_GB:
            print(f"\nDisk budget reached ({BUDGET_GB:.0f} GB). Stopping.")
            break

        filename = urlparse(unquote(url)).path.split("/")[-1]
        if not filename:
            print(f"  SKIP {video_id}: could not parse filename from URL")
            continue

        stem = Path(filename).stem
        temp_path = TEMP_DIR / filename
        frame_dir = TEMP_DIR / stem

        label = row.get("sign_system_label", "")
        county = row.get("primary_county", "")
        print(f"\n[{video_id}] {label} | {county}")

        # Download
        try:
            t0 = time.time()
            mb = download_video(url, temp_path)
            print(f"  Downloaded {mb:.1f} MB in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  FAIL download: {e}")
            append_log({
                "video_name": video_id, "frames_extracted": 0, "frames_kept": 0,
                "MileMarkers_detections": 0, "Regulatory_detections": 0,
                "SpeedLimits_detections": 0, "StopSigns_detections": 0,
                "disk_gb_remaining": f"{disk_free_gb():.2f}",
            })
            continue

        # Extract frames; always delete video immediately after
        try:
            max_frames = 30 if args.smoke else None
            frame_paths = extract_frames(temp_path, frame_dir, stem, max_frames)
        finally:
            temp_path.unlink(missing_ok=True)

        n_extracted = len(frame_paths)
        print(f"  Extracted {n_extracted} frames")

        # Inference — keep close-range target frames, delete the rest immediately
        kept, class_counts = run_inference(model, frame_paths, args.conf)
        kept_set = set(kept)

        for path in frame_paths:
            if path in kept_set:
                shutil.copy2(path, CANDIDATES_DIR / path.name)
            path.unlink(missing_ok=True)
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        n_kept = len(kept)
        total_videos += 1
        total_kept += n_kept
        for cls_id in CLASS_NAMES:
            class_totals[cls_id] += class_counts[cls_id]

        free_gb = disk_free_gb()
        append_log({
            "video_name": video_id,
            "frames_extracted": n_extracted,
            "frames_kept": n_kept,
            "MileMarkers_detections": class_counts[3],
            "Regulatory_detections": class_counts[4],
            "SpeedLimits_detections": class_counts[5],
            "StopSigns_detections": class_counts[6],
            "disk_gb_remaining": f"{free_gb:.2f}",
        })
        with open(PROCESSED_TXT, "a") as f:
            f.write(video_id + "\n")

        print(
            f"  kept={n_kept}/{n_extracted}  "
            f"MM={class_counts[3]} Reg={class_counts[4]} "
            f"SL={class_counts[5]} Stop={class_counts[6]}  "
            f"free={free_gb:.1f} GB"
        )

    print_summary(total_videos, total_kept, class_totals)


if __name__ == "__main__":
    main()
