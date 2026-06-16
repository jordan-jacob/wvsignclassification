"""
Extract frames from downloaded WVDOH videos.

Usage:
  python scripts/extract_wv_frames.py [--video-dir DIR] [--output-dir DIR]
                                       [--fps N] [--every-n-seconds N]

  --fps              output frames per second (default 1); ignored if
                     --every-n-seconds is set
  --every-n-seconds  seconds between extracted frames (overrides --fps)
"""

import argparse
import csv
from pathlib import Path

import cv2


def extract(video_path, output_dir, video_id, every_n_sec):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(fps * every_n_sec)))
    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.imwrite(str(output_dir / f"{video_id}_{ts_ms}.jpg"), frame)
            count += 1
        frame_idx += 1
    cap.release()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="data/raw/wvdoh/")
    ap.add_argument("--output-dir", default="data/raw/wvdoh_frames/")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="output frames per second (default 1)")
    ap.add_argument("--every-n-seconds", type=float, default=None,
                    help="seconds between frames; overrides --fps if set")
    args = ap.parse_args()

    every_n = (args.every_n_seconds if args.every_n_seconds is not None
               else 1.0 / args.fps)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map filename -> video_id from download list
    id_map = {}
    dl_csv = Path("data/wv_download_list.csv")
    if dl_csv.exists():
        for row in csv.DictReader(open(dl_csv)):
            id_map[row["filename"]] = row["video_id"]

    videos = sorted(video_dir.glob("*.MP4")) + sorted(video_dir.glob("*.mp4"))
    total_frames = 0
    processed = 0

    for vp in videos:
        video_id = id_map.get(vp.name, vp.stem)
        existing = list(output_dir.glob(f"{video_id}_*.jpg"))
        if existing:
            print(f"SKIP: {vp.name} ({len(existing)} frames already exist)")
            total_frames += len(existing)
            processed += 1
            continue

        n = extract(vp, output_dir, video_id, every_n)
        print(f"{vp.name}: {n} frames")
        total_frames += n
        processed += 1

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.jpg"))
    print(f"\nVideos processed: {processed}, frames extracted: {total_frames}, "
          f"total size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
