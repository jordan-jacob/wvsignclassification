"""
End-to-end WVDOH video processing: download → extract frames → delete video.
Optionally runs pre-annotation on all extracted frames.

Usage:
  python scripts/process_wv_videos.py [--video-only] [--preannotate-only]
                                       [--video-id ID] [--smoke] [--conf FLOAT]
"""

import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2

DOWNLOAD_LIST = "configs/wv_download_list.csv"
TEMP_DIR = Path("data/raw/wvdoh/temp")
FRAMES_ROOT = Path("data/raw/wvdoh_frames")
PREANNO_DIR = Path("data/wvdoh_preannotations")
CHECKPOINT = "checkpoints/phase2_full_best.pt"
LOW_DISK_GB = 5.0
JPEG_QUALITY = 85


def disk_free_gb():
    return shutil.disk_usage(".").free / 1e9


def download_video(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "--continue", "--show-progress", "-O", str(dest), url],
        check=True,
    )
    return dest.stat().st_size / 1e6


def extract_frames(video_path, out_dir, video_id, max_seconds=None):
    """Extract at 1fps, naming {video_id}_{frame_number:06d}.jpg at JPEG 85."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(src_fps)))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_seconds is not None and frame_idx >= int(max_seconds * src_fps):
            break
        if frame_idx % interval == 0:
            cv2.imwrite(
                str(out_dir / f"{video_id}_{count:06d}.jpg"),
                frame,
                encode_params,
            )
            count += 1
        frame_idx += 1

    cap.release()
    return count


def process_video(row, smoke):
    video_id = row["video_id"]
    county = row["primary_county"]
    road_type = row["sign_system_label"]
    url = row["video_link"]
    filename = row["filename"]

    out_dir = FRAMES_ROOT / video_id

    # Skip if frames already exist for this video_id
    existing = list(out_dir.glob(f"{video_id}_*.jpg")) if out_dir.exists() else []
    if existing:
        frames_mb = sum(f.stat().st_size for f in existing) / 1e6
        print(
            f"SKIP {video_id}: {len(existing)} frames already exist "
            f"({frames_mb:.1f} MB) | disk_free={disk_free_gb():.1f} GB"
        )
        return

    free = disk_free_gb()
    if free < LOW_DISK_GB:
        print(f"WARNING: only {free:.1f} GB free — stopping before downloading {video_id}")
        raise SystemExit(1)

    dest = TEMP_DIR / filename
    print(f"\n[{video_id}] {county} / {road_type}")
    print(f"  Downloading {filename} ...")
    video_mb = download_video(url, dest)

    free = disk_free_gb()
    if free < LOW_DISK_GB:
        print(f"WARNING: only {free:.1f} GB free after download — stopping")
        dest.unlink(missing_ok=True)
        raise SystemExit(1)

    max_seconds = 30 if smoke else None
    t0 = time.time()
    n_frames = extract_frames(dest, out_dir, video_id, max_seconds)
    elapsed = time.time() - t0

    dest.unlink()  # delete video immediately after extraction

    frames_mb = sum(f.stat().st_size for f in out_dir.glob("*.jpg")) / 1e6
    free = disk_free_gb()
    rate = n_frames / elapsed if elapsed > 0 else float("inf")

    print(
        f"  video_id={video_id}  county={county}  road_type={road_type}\n"
        f"  frames_extracted={n_frames}  video_size={video_mb:.1f} MB  "
        f"frames_size={frames_mb:.1f} MB  disk_free={free:.1f} GB  "
        f"rate={rate:.1f} frames/sec"
    )

    if free < LOW_DISK_GB:
        print(f"WARNING: only {free:.1f} GB free after processing {video_id} — stopping")
        raise SystemExit(1)


def preannotate(conf):
    from ultralytics import YOLO

    model = YOLO(CHECKPOINT)

    frames = sorted(FRAMES_ROOT.glob("**/*.jpg"))
    if not frames:
        print(f"No frames found in {FRAMES_ROOT}")
        return

    PREANNO_DIR.mkdir(parents=True, exist_ok=True)
    yolo_dir = PREANNO_DIR / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)
    no_detection = []
    ls_records = []

    for frame_path in frames:
        results = model(frame_path, conf=conf, verbose=False)[0]
        boxes = results.boxes
        h, w = results.orig_shape

        ls_result = []
        yolo_lines = []

        if boxes is None or len(boxes) == 0:
            no_detection.append(frame_path.name)
        else:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                ls_result.append({
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": w,
                    "original_height": h,
                    "value": {
                        "x": x1 / w * 100,
                        "y": y1 / h * 100,
                        "width": (x2 - x1) / w * 100,
                        "height": (y2 - y1) / h * 100,
                        "rectanglelabels": [cls_name],
                    },
                    "score": conf_val,
                })
                class_counts[cls_name] += 1

        (yolo_dir / (frame_path.stem + ".txt")).write_text("\n".join(yolo_lines))
        ls_records.append({
            "data": {"image": frame_path.name},
            "predictions": [{"result": ls_result}],
        })

    ls_path = PREANNO_DIR / "ls_annotations.json"
    ls_path.write_text(json.dumps(ls_records, indent=2))
    nd_path = PREANNO_DIR / "no_detection_frames.txt"
    nd_path.write_text("\n".join(no_detection))

    total = len(frames)
    with_det = total - len(no_detection)
    print(f"\nTotal frames: {total}")
    print(f"Frames with detections: {with_det} ({100 * with_det / max(total, 1):.1f}%)")
    print(f"Frames with NO detections: {len(no_detection)} → {nd_path}")
    print("Per-class counts:")
    for cls_name in sorted(class_counts):
        print(f"  {cls_name:<30} {class_counts[cls_name]}")
    print(f"\nLabel Studio JSON : {ls_path}")
    print(f"YOLO labels       : {yolo_dir}/")


def load_rows(video_id_filter=None):
    rows = list(csv.DictReader(open(DOWNLOAD_LIST)))
    if video_id_filter:
        rows = [r for r in rows if r["video_id"] == video_id_filter]
        if not rows:
            raise SystemExit(f"video_id {video_id_filter!r} not found in {DOWNLOAD_LIST}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-only", action="store_true",
                    help="download and extract only, skip pre-annotation")
    ap.add_argument("--preannotate-only", action="store_true",
                    help="run inference on existing frames, skip download/extract")
    ap.add_argument("--video-id", help="process a single video by ID (for testing)")
    ap.add_argument("--smoke", action="store_true",
                    help="first 3 videos only, first 30 seconds each")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence threshold (default 0.25)")
    args = ap.parse_args()

    if not args.preannotate_only:
        rows = load_rows(args.video_id)
        if args.smoke:
            rows = rows[:3]
        for row in rows:
            process_video(row, smoke=args.smoke)

    if not args.video_only:
        preannotate(conf=args.conf)


if __name__ == "__main__":
    main()
