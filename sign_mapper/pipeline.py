"""
Core pipeline: frame extraction at 2fps, GPS interpolation, YOLO inference,
DBSCAN spatial clustering, and per-cluster aggregation.

CLI:
    python pipeline.py --video V --model M --output out.geojson [--sidecar S] [--conf 0.5]
                       [--inventory I.geojson] [--annotations-dir /path/to/labels/]

--sidecar is optional: when omitted, GPS is extracted from the video's embedded
metadata with ExifTool (see gps_extractor.py).
"""
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from gps_parser import parse_sidecar, interpolate_gps
from geojson_utils import haversine_m

EXTRACT_FPS = 2          # frames per second to extract
# Clustering works in meters via an equirectangular projection (Item 2) — a
# latitude-consistent, physically meaningful replacement for the old degree eps.
DBSCAN_EPS_M = 15.0      # meters; DBSCAN neighborhood radius
CLUSTER_MERGE_M = 20.0   # post-DBSCAN: merge same-class clusters within this distance
TOP_N_POSITION = 5       # highest-confidence detections used for the position median
MIN_CONF_DEFAULT = 0.5


def run_pipeline(model_path, video_path, sidecar_path=None, output_dir='.',
                 conf_threshold=MIN_CONF_DEFAULT, progress=None,
                 annotations_dir=None, extract_fps=EXTRACT_FPS):
    """
    Returns list of cluster dicts.  progress(str) receives status messages.
    If annotations_dir is provided, also writes qa_report.json to output_dir.

    GPS source: a sidecar (.srt/.csv) when sidecar_path is given, otherwise the
    video's embedded GPS metadata via ExifTool (gps_extractor).
    """
    if progress is None:
        progress = print

    output_dir = Path(output_dir)
    (output_dir / 'frames').mkdir(parents=True, exist_ok=True)

    if sidecar_path is not None:
        progress("Loading GPS sidecar...")
        fixes = parse_sidecar(sidecar_path)
        progress(f"GPS: {len(fixes)} fixes loaded from {Path(sidecar_path).name}")
    else:
        from gps_extractor import EXIFTOOL_AVAILABLE, extract_gps_from_video
        if not EXIFTOOL_AVAILABLE:
            raise ValueError(
                "No GPS sidecar provided and ExifTool is not installed. "
                "Either provide a .srt or .csv sidecar, or install ExifTool "
                "from https://exiftool.org/install.html")
        fixes = extract_gps_from_video(Path(video_path), progress=progress)

    progress("Loading YOLO model...")
    model = YOLO(str(model_path))
    class_names = model.names  # {0: 'Curves', ...}
    progress(f"Model loaded: {len(class_names)} classes")

    progress("Extracting frames and running inference...")
    detections = _run_inference(model, class_names, Path(video_path),
                                fixes, conf_threshold, output_dir, progress,
                                save_frames=annotations_dir is not None,
                                extract_fps=extract_fps)
    progress(f"Inference complete: {len(detections)} detections")

    if not detections:
        progress("No detections found — check confidence threshold or GPS alignment.")
        return []

    progress(f"Clustering {len(detections)} detections (DBSCAN eps={DBSCAN_EPS_M:.0f}m)...")
    clusters = _cluster(detections, Path(video_path).name, output_dir)
    progress(f"Clustering complete: {len(clusters)} clusters")

    if annotations_dir:
        ann_dir = Path(annotations_dir)
        progress(f"Running QA comparison against annotations in {ann_dir.name}/...")
        qa_items = _qa_compare(detections, ann_dir, class_names)
        progress(f"QA complete: {len(qa_items)} mismatches found")
        qa_path = Path(output_dir) / 'qa_report.json'
        _write_qa_report(qa_items, Path(video_path).name, qa_path)
        progress(f"QA report written: {qa_path}")

    return clusters


def _run_inference(model, class_names, video_path, fixes, conf_threshold,
                   output_dir, progress, save_frames=False, extract_fps=EXTRACT_FPS):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = (total_frames / raw_fps) * 1000 if total_frames and raw_fps else 0
    interval_ms = 1000.0 / extract_fps

    detections = []
    ts_ms = 0.0
    frame_num = 0
    skipped_gps = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()
        if not ret:
            break

        actual_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if duration_ms and actual_ts > duration_ms + interval_ms:
            break

        gps = interpolate_gps(fixes, actual_ts)
        if gps is None:
            skipped_gps += 1
            ts_ms += interval_ms
            continue

        frame_num += 1
        if frame_num % 50 == 0:
            progress(f"Running inference: frame {frame_num} at "
                     f"{actual_ts/1000:.1f}s — {len(detections)} detections so far")

        # When QA mode is active, save full frames for Label Studio annotation
        if save_frames:
            frame_jpg = output_dir / 'frames' / f'frame_{int(actual_ts)}.jpg'
            cv2.imwrite(str(frame_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        results = model(frame, conf=conf_threshold, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                det_idx = len(detections)
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                cls_name = class_names.get(cls_idx, str(cls_idx))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h_px, w_px = frame.shape[:2]

                thumb_path = output_dir / 'frames' / f'det_{det_idx}.jpg'
                _save_thumbs(frame, x1, y1, x2, y2, cls_name, conf, thumb_path)

                detections.append({
                    'idx': det_idx,
                    'frame_ts': int(actual_ts),
                    'lat': gps['lat'],
                    'lon': gps['lon'],
                    'class_idx': cls_idx,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox_norm': [x1/w_px, y1/h_px, x2/w_px, y2/h_px],
                    'frame_w': w_px,
                    'frame_h': h_px,
                    'thumb': str(thumb_path),
                })

        ts_ms += interval_ms

    cap.release()

    if skipped_gps:
        progress(f"Warning: {skipped_gps} frames skipped — no GPS fix in range")

    return detections


# Per-class colors for the YOLO bbox overlay (BGR for OpenCV — Item 8). There is
# no per-class Leaflet marker color to match (markers are colored by review
# status), so this is the canonical class palette for thumbnails.
CLASS_COLORS = {
    'Curves':        (214, 158, 46),    # amber
    'HighwaySymbols': (99, 179, 237),   # blue
    'Informational': (128, 90, 213),    # purple
    'Regulatory':    (128, 90, 213),    # purple (collapses with Informational)
    'MileMarkers':   (56, 178, 172),    # teal
    'SpeedLimits':   (56, 161, 105),    # green
    'StopSigns':     (62, 62, 229),     # red
    'TrafficLights': (129, 199, 132),   # light green
    'Warnings':      (0, 165, 255),     # orange
}
_DEFAULT_COLOR = (128, 128, 128)


def _save_thumbs(frame, x1, y1, x2, y2, cls_name, conf, base_path: Path):
    """Save two thumbnails (Item 8): a clean padded crop (`<base>_clean.jpg`) and a
    copy with the YOLO box + class/confidence label drawn (`<base>_bbox.jpg`)."""
    h, w = frame.shape[:2]
    # 40% padding on each side (relative to box size) for context around the sign
    pad_x = 0.4 * (x2 - x1)
    pad_y = 0.4 * (y2 - y1)
    cx0, cy0 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx1, cy1 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    crop = frame[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        crop, cx0, cy0 = frame, 0, 0
    scale = min(320 / crop.shape[1], 240 / crop.shape[0], 1.0)
    clean = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale))) if scale < 1.0 else crop

    base = str(base_path)
    cv2.imwrite(base.replace('.jpg', '_clean.jpg'), clean, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # bbox overlay: box coordinates relative to the (scaled) crop origin
    bbox_img = clean.copy()
    color = CLASS_COLORS.get(cls_name, _DEFAULT_COLOR)
    p1 = (int((x1 - cx0) * scale), int((y1 - cy0) * scale))
    p2 = (int((x2 - cx0) * scale), int((y2 - cy0) * scale))
    cv2.rectangle(bbox_img, p1, p2, color, 2)
    cv2.putText(bbox_img, f"{cls_name} {conf:.2f}", (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(base.replace('.jpg', '_bbox.jpg'), bbox_img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def degrees_to_meters(lats, lons):
    """Fast equirectangular approximation. Good enough for <1km clusters."""
    lat_center = np.mean(lats)
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    x = lons * meters_per_deg_lon
    y = lats * meters_per_deg_lat
    return np.column_stack([x, y])


def _cluster_center(dets):
    """Median position from the top-N highest-confidence detections (Item 2b):
    low-confidence fringe sightings at the edge of a sign's range bias the
    center and are a common cause of split clusters."""
    top = sorted(dets, key=lambda d: d['confidence'], reverse=True)[:TOP_N_POSITION]
    return (float(np.median([d['lat'] for d in top])),
            float(np.median([d['lon'] for d in top])))


def _merge_close_clusters(groups):
    """One pass (Item 2c): union same-class DBSCAN groups whose top-N centers are
    within CLUSTER_MERGE_M, catching signs split at a cluster boundary."""
    info = []  # (top_class, center_lat, center_lon) per group
    for dets in groups:
        top_class = Counter(d['class_name'] for d in dets).most_common(1)[0][0]
        lat, lon = _cluster_center(dets)
        info.append((top_class, lat, lon))

    n = len(groups)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if info[i][0] != info[j][0]:
                continue
            if haversine_m(info[i][1], info[i][2], info[j][1], info[j][2]) <= CLUSTER_MERGE_M:
                parent[find(j)] = find(i)

    merged = {}
    for i in range(n):
        merged.setdefault(find(i), []).extend(groups[i])
    return list(merged.values())


def _cluster(detections, video_name, output_dir):
    lats = np.array([d['lat'] for d in detections])
    lons = np.array([d['lon'] for d in detections])
    xy = degrees_to_meters(lats, lons)
    labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=1).fit_predict(xy)

    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(detections[i])
    groups = _merge_close_clusters(list(groups.values()))

    frames_dir = output_dir / 'frames'
    clusters = []

    for cluster_id, dets in enumerate(groups):
        votes = Counter(d['class_name'] for d in dets)
        total_votes = sum(votes.values())
        top_class, top_count = votes.most_common(1)[0]
        vote_pct = top_count / total_votes

        by_conf = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        mean_conf = float(np.mean([d['confidence'] for d in by_conf[:3]]))
        best_det = by_conf[0]
        sighting_count = len(dets)

        reasons = []
        if sighting_count == 1:
            reasons.append("Single sighting")
        if mean_conf < 0.5:
            reasons.append(f"Low confidence ({mean_conf:.2f})")
        if vote_pct < 0.70 and len(votes) > 1:
            second_cls, second_cnt = votes.most_common(2)[1]
            reasons.append(
                f"Split class vote ({top_class} {vote_pct:.0%} / "
                f"{second_cls} {second_cnt/total_votes:.0%})"
            )

        # Promote best detection thumbnail to cluster-level file (clean + bbox, Item 8)
        for suffix in ('_clean', '_bbox'):
            src = Path(best_det['thumb'].replace('.jpg', f'{suffix}.jpg'))
            if src.exists():
                shutil.copy2(src, frames_dir / f'cluster_{cluster_id}{suffix}.jpg')

        lat_c, lon_c = _cluster_center(dets)

        clusters.append({
            'cluster_id': cluster_id,
            'sign_class': top_class,
            'confidence': round(mean_conf, 4),
            'sighting_count': sighting_count,
            'needs_review': bool(reasons),
            'review_reason': '; '.join(reasons) if reasons else None,
            'class_vote_pct': round(vote_pct, 4),
            'video_source': video_name,
            'first_seen_ts': min(d['frame_ts'] for d in dets),
            'best_frame_ts': best_det['frame_ts'],
            'lat_median': round(lat_c, 8),
            'lon_median': round(lon_c, 8),
            'lat_std': round(float(np.std([d['lat'] for d in dets])), 8),
            'lon_std': round(float(np.std([d['lon'] for d in dets])), 8),
            'inventory_match': None,
            'inventory_distance_m': None,
            'discrepancy_type': None,
            # Resolves only while the local Flask app is serving /frame/<id>;
            # 404s gracefully on GitHub Pages (handled client-side).
            'thumbnail_url': f'/frame/{cluster_id}',
        })

    # Clean up per-detection thumbnails (both clean and bbox variants)
    for d in detections:
        for suffix in ('_clean', '_bbox'):
            try:
                Path(d['thumb'].replace('.jpg', f'{suffix}.jpg')).unlink(missing_ok=True)
            except OSError:
                pass

    return clusters


# ---------------------------------------------------------------------------
# QA comparison (inference-vs-annotation mismatch detection)
# ---------------------------------------------------------------------------

def _iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] normalized boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _qa_compare(detections, ann_dir: Path, class_names: dict, iou_thresh=0.5) -> list:
    """
    Match YOLO-format annotation .txt files (named frame_{ts_ms}.txt) to
    detections by frame timestamp.  Returns list of mismatch dicts.

    Annotation files are exported from Label Studio in YOLO format:
        class_idx  x_center  y_center  width  height   (all normalized)
    File name convention: frame_{ts_ms}.txt where ts_ms is the video
    timestamp in milliseconds, matching the 'frame_ts' field in detections.
    """
    # Build lookup: ts_ms → list of detections
    det_by_ts = {}
    for d in detections:
        det_by_ts.setdefault(d['frame_ts'], []).append(d)

    items = []
    qa_id = 1

    for ann_file in sorted(ann_dir.glob('frame_*.txt')):
        stem = ann_file.stem  # e.g. "frame_14231500"
        try:
            ts_ms = int(stem.split('_', 1)[1])
        except (IndexError, ValueError):
            continue

        # Allow ±250ms (half the 2fps interval) for timestamp matching
        frame_dets = det_by_ts.get(ts_ms) or det_by_ts.get(ts_ms - 1) or det_by_ts.get(ts_ms + 1)
        if not frame_dets:
            # Try nearest within 300ms
            near = [k for k in det_by_ts if abs(k - ts_ms) <= 300]
            if near:
                frame_dets = det_by_ts[min(near, key=lambda k: abs(k - ts_ms))]
        if not frame_dets:
            continue

        lines = ann_file.read_text(encoding='utf-8').strip().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            ann_cls_idx = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            ann_box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            ann_class = class_names.get(ann_cls_idx, str(ann_cls_idx))

            # Find the detection with the best IoU against this annotation box
            best_iou, best_det = 0.0, None
            for det in frame_dets:
                score = _iou(ann_box, det['bbox_norm'])
                if score > best_iou:
                    best_iou, best_det = score, det

            if best_det is None or best_iou < iou_thresh:
                continue
            if best_det['class_name'] == ann_class:
                continue  # classes match — not a mismatch

            thumb = str(Path(best_det['thumb']).name)
            items.append({
                'id': f'qa_{qa_id:03d}',
                'frame_ts_ms': ts_ms,
                'lat': best_det['lat'],
                'lon': best_det['lon'],
                'annotated_class': ann_class,
                'predicted_class': best_det['class_name'],
                'confidence': round(best_det['confidence'], 4),
                'iou': round(best_iou, 4),
                'frame_thumbnail': f'outputs/frames/{thumb}',
                'status': 'pending',
                'resolution': None,
                'reviewer_note': None,
            })
            qa_id += 1

    return items


def _write_qa_report(items: list, video_source: str, path: Path):
    report = {
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'video_source': video_source,
        'total_frames_with_annotations': None,  # populated by caller if known
        'total_mismatches': len(items),
        'items': items,
    }
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='WV sign detection pipeline')
    ap.add_argument('--video', required=True)
    ap.add_argument('--sidecar', help='Optional GPS sidecar (.srt/.csv); '
                                       'omit to extract GPS from the video via ExifTool')
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True, help='Output GeoJSON path')
    ap.add_argument('--inventory', help='Optional single WVDOH inventory GeoJSON')
    ap.add_argument('--inventory-dir', dest='inventory_dir',
                    help='Directory of WVDOH inventory GeoJSONs (merged before comparison)')
    ap.add_argument('--annotations-dir', dest='annotations_dir',
                    help='Directory of YOLO-format .txt labels (frame_<ts_ms>.txt)')
    ap.add_argument('--conf', '--confidence', dest='conf', type=float, default=0.5,
                    help='Confidence threshold')
    ap.add_argument('--fps', type=float, default=EXTRACT_FPS,
                    help='Frames per second to extract for inference')
    args = ap.parse_args()

    out_path = Path(args.output)
    out_dir = out_path.parent / 'pipeline_frames'

    clusters = run_pipeline(
        model_path=args.model,
        video_path=args.video,
        sidecar_path=args.sidecar,
        output_dir=out_dir,
        conf_threshold=args.conf,
        annotations_dir=args.annotations_dir,
        extract_fps=args.fps,
    )

    if args.inventory_dir:
        from geojson_utils import load_inventory_dir, compare_with_inventory
        print(f"Loading inventory directory: {args.inventory_dir}")
        inv = load_inventory_dir(args.inventory_dir)
        print(f"Merged inventory: {len(inv['features'])} features")
        clusters = compare_with_inventory(clusters, inv)
    elif args.inventory:
        from geojson_utils import load_geojson, normalize_inventory, compare_with_inventory
        inv = normalize_inventory(load_geojson(args.inventory), Path(args.inventory).name)
        clusters = compare_with_inventory(clusters, inv)

    from geojson_utils import save_geojson
    save_geojson(clusters, out_path, Path(args.video).name)

    # Summary table
    class_counts = Counter(c['sign_class'] for c in clusters)
    print(f"\n{'Class':<22} {'Clusters':>8}")
    print('-' * 32)
    for cls, cnt in sorted(class_counts.items()):
        print(f"{cls:<22} {cnt:>8}")
    print(f"\nTotal clusters : {len(clusters)}")
    print(f"Need review    : {sum(1 for c in clusters if c['needs_review'])}")
    print(f"Output         : {out_path}")
