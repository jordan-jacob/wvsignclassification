#!/usr/bin/env python3
"""
Evaluate phase-2 models on the LISA dataset and print a comparison table.

Both models are validated against the same split (train/images — LISA has no
official val split). The full 47-class model's per-class AP is aggregated to
the 4 coarse classes for an apples-to-apples comparison.
"""
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"

# Must match train_phase2_lisa.py exactly
LISA_4CLASS: dict[str, int | None] = {
    "stop": 0,
    "yield": 1,
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

COARSE_NAMES = ["stop", "yield", "warning", "speed-limit"]


def _write_eval_yaml(data_dir: Path, nc: int, names: list) -> Path:
    data = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "train/images",
        "nc": nc,
        "names": names,
    }
    out = data_dir / "_eval.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _coarse_ap50_from_full(metrics, class_names: list) -> dict[int, float]:
    """Average AP50 within each coarse group for the 47-class model."""
    ap_idx = metrics.box.ap_class_index
    ap50 = metrics.box.ap50
    ap_by_name = {class_names[int(i)]: float(v) for i, v in zip(ap_idx, ap50)}

    groups: dict[int, list[float]] = defaultdict(list)
    for name, coarse in LISA_4CLASS.items():
        if coarse is not None and name in ap_by_name:
            groups[coarse].append(ap_by_name[name])

    return {i: (sum(v) / len(v)) for i, v in groups.items()}


def _direct_ap50(metrics, nc: int) -> dict[int, float]:
    """AP50 per class index for the 4-class model."""
    ap_idx = metrics.box.ap_class_index
    ap50 = metrics.box.ap50
    return {int(i): float(v) for i, v in zip(ap_idx, ap50)}


def _row(label: str, m50: float, m: float, p: float, r: float) -> str:
    return f"  {label:<22} {m50:>8.4f}   {m:>9.4f}   {p:>9.4f}   {r:>6.4f}"


def main():
    from ultralytics import YOLO

    ckpt_dir = ROOT / "checkpoints"
    full_ckpt = ckpt_dir / "phase2_full_best.pt"
    four_ckpt = ckpt_dir / "phase2_4class_best.pt"

    for p in (full_ckpt, four_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    lisa_names = yaml.safe_load((CONFIGS / "lisa.yaml").read_text())["names"]
    lisa_dir = PROCESSED / "lisa"
    lisa_4class_dir = PROCESSED / "lisa_4class"

    full_yaml = _write_eval_yaml(lisa_dir, len(lisa_names), lisa_names)
    four_yaml = _write_eval_yaml(lisa_4class_dir, len(COARSE_NAMES), COARSE_NAMES)

    print("Evaluating full 47-class model ...")
    full_model = YOLO(str(full_ckpt))
    full_m = full_model.val(data=str(full_yaml), split="val", verbose=False)

    print("Evaluating 4-class model ...")
    four_model = YOLO(str(four_ckpt))
    four_m = four_model.val(data=str(four_yaml), split="val", verbose=False)

    header = (
        f"\n  {'Model':<22} {'mAP@50':>8}   {'mAP@50-95':>9}"
        f"   {'Precision':>9}   {'Recall':>6}"
    )
    sep = "  " + "-" * 58

    print("\n" + "=" * 62)
    print("  Overall metrics (LISA train split, used as val)")
    print("=" * 62)
    print(header)
    print(sep)
    print(_row("phase2_full (47-cls)", full_m.box.map50, full_m.box.map,
               full_m.box.mp, full_m.box.mr))
    print(_row("phase2_4class", four_m.box.map50, four_m.box.map,
               four_m.box.mp, four_m.box.mr))

    full_coarse = _coarse_ap50_from_full(full_m, lisa_names)
    four_direct = _direct_ap50(four_m, len(COARSE_NAMES))

    print("\n" + "=" * 62)
    print("  Per-class AP@50 (coarse, averaged over fine classes for full model)")
    print("=" * 62)
    print(f"  {'Class':<22} {'phase2_full':>12}   {'phase2_4class':>13}")
    print(sep)
    for idx, name in enumerate(COARSE_NAMES):
        ap_full = full_coarse.get(idx, float("nan"))
        ap_four = four_direct.get(idx, float("nan"))
        print(f"  {name:<22} {ap_full:>12.4f}   {ap_four:>13.4f}")

    print()


if __name__ == "__main__":
    main()
