"""
Run YOLOv8 inference on extracted WVDOH frames and save pre-annotations.

Outputs:
  data/wvdoh_preannotations/preannotations.json  — Label Studio import format
  data/wvdoh_preannotations/yolo/                — YOLO txt labels
  data/wvdoh_preannotations/no_detection_frames.txt

Usage:
  python scripts/preannotate_wv.py [--frames-dir DIR] [--output-dir DIR]
                                    [--checkpoint PATH] [--conf FLOAT]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/raw/wvdoh_frames/")
    ap.add_argument("--output-dir", default="data/wvdoh_preannotations/")
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.checkpoint)

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    yolo_dir = output_dir / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    class_counts = defaultdict(int)
    class_conf_sum = defaultdict(float)
    no_detection = []
    ls_records = []

    for frame_path in frames:
        results = model(frame_path, conf=args.conf, verbose=False)[0]
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
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # YOLO format (normalized cx cy w h)
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Label Studio format (percentage coords from top-left)
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
                    "score": conf,
                })

                class_counts[cls_name] += 1
                class_conf_sum[cls_name] += conf

        (yolo_dir / (frame_path.stem + ".txt")).write_text("\n".join(yolo_lines))
        ls_records.append({
            "data": {"image": frame_path.name},
            "predictions": [{"result": ls_result}],
        })

    # Write outputs
    ls_path = output_dir / "preannotations.json"
    ls_path.write_text(json.dumps(ls_records, indent=2))

    nd_path = output_dir / "no_detection_frames.txt"
    nd_path.write_text("\n".join(no_detection))

    # Summary
    print(f"\nFrames processed: {len(frames)}")
    print(f"Frames with no detections: {len(no_detection)} "
          f"({100 * len(no_detection) / max(len(frames), 1):.1f}%)")
    print("\nPer-class detection counts and average confidence:")
    for cls_name in sorted(class_counts):
        n = class_counts[cls_name]
        print(f"  {cls_name:<30} {n:>6} detections  "
              f"avg conf: {class_conf_sum[cls_name] / n:.3f}")
    print(f"\nLabel Studio JSON : {ls_path}")
    print(f"YOLO labels       : {yolo_dir}/")
    print(f"No-detection list : {nd_path}")


if __name__ == "__main__":
    main()
