#!/usr/bin/env python3
"""
Compare multiple YOLO checkpoints on the same validation split.

Usage:
  python scripts/compare_models.py \\
    --models yolov8:checkpoints/phase2_full_best.pt \\
             yolo11:checkpoints/phase2_yolo11_full_best.pt \\
    --data configs/lisa.yaml \\
    --split val

  # With FND overhead measurement:
  python scripts/compare_models.py \\
    --models yolov8:checkpoints/phase2_full_best.pt \\
    --data configs/lisa.yaml --split val --with-fnd

Output: runs/comparison/results.md  (also printed to stdout)
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "configs"))

COARSE_NAMES = ["stop", "yield", "warning", "speed-limit"]

# Maps LISA fine class name → coarse index, matching train_phase2_lisa.py
LISA_4CLASS: dict[str, int | None] = {
    "stop": 0, "yield": 1,
    "addedLane": 2, "curveRight": 2, "curveLeft": 2, "dip": 2,
    "intersection": 2, "laneEnds": 2, "merge": 2, "pedestrianCrossing": 2,
    "signalAhead": 2, "slow": 2, "stopAhead": 2, "thruMergeLeft": 2,
    "thruMergeRight": 2, "thruTrafficMergeLeft": 2, "yieldAhead": 2,
    "school": 2, "zoneAhead25": 2, "zoneAhead45": 2,
    "speedLimit15": 3, "speedLimit25": 3, "speedLimit30": 3, "speedLimit35": 3,
    "speedLimit40": 3, "speedLimit45": 3, "speedLimit50": 3, "speedLimit55": 3,
    "speedLimit65": 3, "truckSpeedLimit55": 3, "speedLimitUrdbl": 3,
    "rampSpeedAdvisory20": 3, "rampSpeedAdvisory35": 3, "rampSpeedAdvisory40": 3,
    "rampSpeedAdvisory45": 3, "rampSpeedAdvisory50": 3, "rampSpeedAdvisoryUrdbl": 3,
    "schoolSpeedLimit25": 3,
    "doNotPass": None, "keepRight": None, "rightLaneMustTurn": None,
    "doNotEnter": None, "noLeftTurn": None, "noRightTurn": None,
    "roundabout": None, "turnLeft": None, "turnRight": None,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _param_count(model) -> float:
    """Total parameters in millions."""
    return sum(p.numel() for p in model.model.parameters()) / 1e6


def _measure_inference_ms(model, img_paths: list, imgsz: int = 640) -> float:
    """Average per-image inference time in ms over the first 50 images."""
    sample = img_paths[:50]
    t0 = time.perf_counter()
    for p in sample:
        model.predict(str(p), imgsz=imgsz, verbose=False)
    elapsed = (time.perf_counter() - t0) / len(sample) * 1000
    return elapsed


def _measure_inference_fnd_ms(fnd, img_paths: list, imgsz: int = 640) -> tuple[float, int]:
    """
    Returns (avg_ms_per_image, total_fnd_recovered_boxes).
    FND-recovered boxes (cls=-1) are counted but excluded from mAP.
    """
    sample = img_paths[:50]
    total_recovered = 0
    t0 = time.perf_counter()
    for p in sample:
        results = fnd(p, imgsz=imgsz)
        total_recovered += sum(1 for d in results if d.recovery)
    elapsed = (time.perf_counter() - t0) / len(sample) * 1000
    return elapsed, total_recovered


def _collect_image_paths(data_yaml: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path", data_yaml.parent))
    key = cfg.get(split) or cfg.get("val") or cfg.get("train")
    if key is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")
    img_dir = Path(key) if Path(key).is_absolute() else base / key
    return sorted(img_dir.glob("*.*"))


def _ensure_split_yaml(data_yaml: Path, split: str) -> Path:
    """
    If the source YAML lacks the requested split key, write a sibling YAML
    that maps the split to the train images dir. Ultralytics model.val() needs
    the split key to exist in the YAML it reads.
    Returns a path to a YAML guaranteed to have the split key.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    if cfg.get(split):
        return data_yaml  # already has the key — use as-is

    # Fall back to train images for the requested split
    train_key = cfg.get("train", "train/images")
    patched = dict(cfg)
    patched[split] = train_key
    out = data_yaml.parent / f"_compare_{split}.yaml"
    out.write_text(yaml.dump(patched, default_flow_style=False, allow_unicode=True))
    return out


def _coarse_ap50(metrics, class_names: list) -> dict[int, float]:
    """Average AP50 per coarse group from a 47-class model."""
    from collections import defaultdict
    ap_idx = metrics.box.ap_class_index
    ap50 = metrics.box.ap50
    by_name = {class_names[int(i)]: float(v) for i, v in zip(ap_idx, ap50)}
    groups: dict[int, list] = defaultdict(list)
    for name, coarse in LISA_4CLASS.items():
        if coarse is not None and name in by_name:
            groups[coarse].append(by_name[name])
    return {i: sum(v) / len(v) for i, v in groups.items()}


def _direct_ap50(metrics, nc: int) -> dict[int, float]:
    ap_idx = metrics.box.ap_class_index
    ap50 = metrics.box.ap50
    return {int(i): float(v) for i, v in zip(ap_idx, ap50)}


# ── markdown table builder ────────────────────────────────────────────────────

def _md_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
        for row in rows
    )
    return "\n".join([header, sep, body])


# ── per-class AP table ────────────────────────────────────────────────────────

def _per_class_rows(results: list[dict]) -> list[dict]:
    rows = []
    for cls_idx, cls_name in enumerate(COARSE_NAMES):
        row = {"Class": cls_name}
        for r in results:
            ap = r.get("per_class_ap50", {}).get(cls_idx, float("nan"))
            row[r["label"]] = f"{ap:.4f}"
        rows.append(row)
    return rows


# ── Sparse R-CNN evaluator ────────────────────────────────────────────────────

class SparseRCNNEvaluator:
    """
    Wraps detectron2's COCOEvaluator for Sparse R-CNN checkpoints.

    Spec format: sparse_rcnn:<config_path>:<checkpoint_path>
    e.g.  sparse_rcnn:configs/sparse_rcnn_lisa.yaml:runs/sparse_rcnn/model_final.pth

    Returns a result dict with keys matching the YOLO summary row:
        mAP50, mAP50-95, Precision (N/A), Recall (N/A), Params(M), Inference(ms)
    """

    def __init__(self, config_rel: str, ckpt_rel: str):
        self.config_path = ROOT / config_rel
        self.ckpt_path = ROOT / ckpt_rel
        self.label = f"sparse_rcnn:{self.ckpt_path.stem}"

    @staticmethod
    def available() -> bool:
        try:
            import detectron2  # noqa: F401
            return True
        except ImportError:
            return False

    def evaluate(self) -> dict | None:
        if not self.available():
            print("  sparse_rcnn requires detectron2 — skipping.")
            print("  Install: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
            return None
        if not self.ckpt_path.exists():
            print(f"  WARNING: checkpoint not found: {self.ckpt_path}", file=sys.stderr)
            return None

        import sparse_rcnn_register  # noqa: F401 — registers datasets
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.MODEL.WEIGHTS = str(self.ckpt_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.freeze()

        predictor = DefaultPredictor(cfg)
        dataset_name = cfg.DATASETS.TEST[0]

        evaluator = COCOEvaluator(dataset_name, output_dir=str(ROOT / "runs" / "comparison"))
        loader = build_detection_test_loader(cfg, dataset_name)
        results = inference_on_dataset(predictor.model, loader, evaluator)

        bbox = results.get("bbox", {})
        ap50_95 = bbox.get("AP", float("nan")) / 100   # detectron2 returns 0-100 scale
        ap50    = bbox.get("AP50", float("nan")) / 100
        per_cls: dict[int, float] = {}
        for cls_idx, cls_name in enumerate(COARSE_NAMES):
            key = f"AP-{cls_name}"
            if key in bbox:
                per_cls[cls_idx] = bbox[key] / 100

        # Inference time: measure on first 50 images via raw predictor
        import time
        from detectron2.data import DatasetCatalog
        samples = DatasetCatalog.get(dataset_name)[:50]
        t0 = time.perf_counter()
        for s in samples:
            import cv2
            img = cv2.imread(s["file_name"])
            predictor(img)
        infer_ms = (time.perf_counter() - t0) / len(samples) * 1000

        n_params = sum(p.numel() for p in predictor.model.parameters()) / 1e6

        return {
            "Model": self.label,
            "mAP50": f"{ap50:.4f}",
            "mAP50-95": f"{ap50_95:.4f}",
            "Precision": "N/A",
            "Recall": "N/A",
            "Params(M)": f"{n_params:.1f}",
            "Inference(ms)": f"{infer_ms:.1f}",
            "per_class_ap50": per_cls,
        }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", required=True,
        metavar="TYPE:PATH",
        help="One or more model specs, e.g. yolov8:checkpoints/phase2_full_best.pt",
    )
    ap.add_argument("--data", required=True, help="Dataset YAML (e.g. configs/lisa.yaml)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--with-fnd", action="store_true",
                    help="Also measure FND overhead and recovered box counts. "
                         "FND-recovered boxes (cls=-1) are excluded from mAP.")
    ap.add_argument("--fnd-ckpt", default="checkpoints/fnd_classifier.pt",
                    help="FND classifier checkpoint (used with --with-fnd)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    data_yaml = ROOT / args.data
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    # Guarantee the YAML has the requested split key (LISA only ships with train)
    val_yaml = _ensure_split_yaml(data_yaml, args.split)
    if val_yaml != data_yaml:
        print(f"NOTE: '{args.split}' key absent in {data_yaml.name} — "
              f"using train images for validation (wrote {val_yaml.name})")

    img_paths = _collect_image_paths(data_yaml, args.split)
    if not img_paths:
        raise RuntimeError(f"No images found for split '{args.split}' in {data_yaml}")

    data_cfg = yaml.safe_load(data_yaml.read_text())
    class_names: list[str] = data_cfg.get("names", [])
    nc = data_cfg.get("nc", len(class_names))
    is_coarse = (nc == 4)

    summary_rows = []
    per_class_results = []

    for spec in args.models:
        if ":" not in spec:
            print(f"WARNING: skipping '{spec}' — expected TYPE:PATH format", file=sys.stderr)
            continue

        parts = spec.split(":", 2)
        model_type = parts[0]

        print(f"\n{'='*60}")

        # ── Sparse R-CNN ──────────────────────────────────────────────────────
        if model_type == "sparse_rcnn":
            if len(parts) != 3:
                print(
                    f"WARNING: sparse_rcnn spec needs 3 parts: "
                    f"sparse_rcnn:<config>:<checkpoint>  (got: {spec})",
                    file=sys.stderr,
                )
                continue
            evaluator = SparseRCNNEvaluator(config_rel=parts[1], ckpt_rel=parts[2])
            print(f"Evaluating {evaluator.label} ...")
            row = evaluator.evaluate()
            if row is None:
                continue
            cls_ap50 = row.pop("per_class_ap50", {})
            summary_rows.append(row)
            per_class_results.append({"label": row["Model"], "per_class_ap50": cls_ap50})
            continue

        # ── YOLO (yolov8 / yolo11) ────────────────────────────────────────────
        ckpt_rel = parts[1] if len(parts) > 1 else ""
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found, skipping: {ckpt_path}", file=sys.stderr)
            continue

        label = f"{model_type}:{ckpt_path.stem}"
        print(f"Evaluating {label} ...")

        from ultralytics import YOLO
        model = YOLO(str(ckpt_path))
        model_nc = model.model.nc
        if model_nc != nc:
            print(
                f"WARNING: skipping {label} — model nc={model_nc} does not match "
                f"dataset nc={nc} in {data_yaml.name}. "
                f"Use a checkpoint trained on this dataset.",
                file=sys.stderr,
            )
            continue
        params_m = _param_count(model)

        # val metrics — use val_yaml which guarantees the split key exists
        metrics = model.val(data=str(val_yaml), split=args.split,
                             imgsz=args.imgsz, verbose=False)

        # per-class AP
        if is_coarse:
            cls_ap50 = _direct_ap50(metrics, nc)
        else:
            cls_ap50 = _coarse_ap50(metrics, class_names)

        # base inference time
        infer_ms = _measure_inference_ms(model, img_paths, args.imgsz)

        row = {
            "Model": label,
            "mAP50": f"{metrics.box.map50:.4f}",
            "mAP50-95": f"{metrics.box.map:.4f}",
            "Precision": f"{metrics.box.mp:.4f}",
            "Recall": f"{metrics.box.mr:.4f}",
            "Params(M)": f"{params_m:.1f}",
            "Inference(ms)": f"{infer_ms:.1f}",
        }

        if args.with_fnd:
            from fnd import FND, FNDClassifier  # noqa: F401
            fnd_ckpt = ROOT / args.fnd_ckpt
            fnd = FND(model, fnd_ckpt=fnd_ckpt if fnd_ckpt.exists() else None)
            fnd_ms, fnd_recovered = _measure_inference_fnd_ms(fnd, img_paths, args.imgsz)
            row["Inference+FND(ms)"] = f"{fnd_ms:.1f}"
            # FND_recovered is count over the 50-image sample; scale to per-image
            row["FND_recovered/img"] = f"{fnd_recovered / min(50, len(img_paths)):.2f}"

        summary_rows.append(row)
        per_class_results.append({"label": label, "per_class_ap50": cls_ap50})

    # ── build output ──────────────────────────────────────────────────────────
    summary_cols = [
        "Model", "mAP50", "mAP50-95", "Precision", "Recall", "Params(M)", "Inference(ms)"
    ]
    if args.with_fnd:
        summary_cols += ["Inference+FND(ms)", "FND_recovered/img"]

    per_class_cols = ["Class"] + [r["label"] for r in per_class_results]
    per_class_rows_data = _per_class_rows(per_class_results)

    out_lines = [
        "# Model Comparison",
        f"\nDataset: `{args.data}`  |  Split: `{args.split}`  |  imgsz: {args.imgsz}",
        "\n## Overall Metrics\n",
        _md_table(summary_rows, summary_cols),
        "\n## Per-Class AP@50 (coarse: stop / yield / warning / speed-limit)\n",
        _md_table(per_class_rows_data, per_class_cols),
    ]
    if args.with_fnd:
        out_lines.append(
            "\n> **FND note**: `FND_recovered/img` counts boxes added by the False "
            "Negative Detector (cls=-1, class unknown). These are excluded from mAP "
            "calculations. `Inference+FND(ms)` includes both the base model and FND "
            "classifier forward passes."
        )
    out_lines.append("")

    md = "\n".join(out_lines)
    print("\n" + md)

    out_dir = ROOT / "runs" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.md"
    out_file.write_text(md)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
