#!/usr/bin/env python3
"""
Phase 2: Fine-tune on LISA from phase-1 checkpoint.

Trains two models in sequence:
  a) Full 47-class model   → checkpoints/phase2_full_best.pt
  b) 4-class coarse model  → checkpoints/phase2_4class_best.pt

LISA has no official val split; training images are reused for validation.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed"

# Maps LISA class name → coarse index (0-3), or None to drop the box.
# warning (2): all warning/advisory signs
# speed-limit (3): all speed-limit and ramp-advisory signs
# skip (None): regulatory signs not in the 4-class taxonomy
LISA_4CLASS: dict[str, int | None] = {
    "stop": 0,
    "yield": 1,
    # warning
    "addedLane": 2, "curveRight": 2, "curveLeft": 2, "dip": 2,
    "intersection": 2, "laneEnds": 2, "merge": 2, "pedestrianCrossing": 2,
    "signalAhead": 2, "slow": 2, "stopAhead": 2, "thruMergeLeft": 2,
    "thruMergeRight": 2, "thruTrafficMergeLeft": 2, "yieldAhead": 2,
    "school": 2, "zoneAhead25": 2, "zoneAhead45": 2,
    # speed-limit
    "speedLimit15": 3, "speedLimit25": 3, "speedLimit30": 3, "speedLimit35": 3,
    "speedLimit40": 3, "speedLimit45": 3, "speedLimit50": 3, "speedLimit55": 3,
    "speedLimit65": 3, "truckSpeedLimit55": 3, "speedLimitUrdbl": 3,
    "rampSpeedAdvisory20": 3, "rampSpeedAdvisory35": 3, "rampSpeedAdvisory40": 3,
    "rampSpeedAdvisory45": 3, "rampSpeedAdvisory50": 3, "rampSpeedAdvisoryUrdbl": 3,
    "schoolSpeedLimit25": 3,
    # skip: regulatory signs not in any coarse class
    "doNotPass": None, "keepRight": None, "rightLaneMustTurn": None,
    "doNotEnter": None, "noLeftTurn": None, "noRightTurn": None,
    "roundabout": None, "turnLeft": None, "turnRight": None,
}

COARSE_NAMES = ["stop", "yield", "warning", "speed-limit"]


def _junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())],
        check=True,
    )


def ensure_lisa_4class() -> Path:
    lisa = PROCESSED / "lisa"
    out_dir = PROCESSED / "lisa_4class"

    lisa_names = yaml.safe_load((CONFIGS / "lisa.yaml").read_text())["names"]
    # fine index → coarse index (or None)
    idx_map = {i: LISA_4CLASS[name] for i, name in enumerate(lisa_names)}

    src_lbl = lisa / "train" / "labels"
    dst_lbl = out_dir / "train" / "labels"
    dst_img = out_dir / "train" / "images"

    _junction(dst_img, lisa / "train" / "images")
    dst_lbl.mkdir(parents=True, exist_ok=True)

    src_files = list(src_lbl.glob("*.txt"))
    existing = {p.name for p in dst_lbl.glob("*.txt")}
    todo = [f for f in src_files if f.name not in existing]

    if not todo:
        print(f"  4-class labels already present ({len(src_files)} files)")
    else:
        print(f"  Remapping {len(todo)} LISA label files to 4 classes ...")
        for src_file in todo:
            lines = []
            for line in src_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                ci = idx_map[int(parts[0])]
                if ci is not None:
                    lines.append(f"{ci} {' '.join(parts[1:])}")
            (dst_lbl / src_file.name).write_text("\n".join(lines))

    return out_dir


def _write_yaml(data_dir: Path, nc: int, names: list, suffix: str) -> Path:
    data = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "train/images",  # no official LISA val split
        "nc": nc,
        "names": names,
    }
    out = data_dir / f"dataset_{suffix}.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _train(variant: str, data_yaml: Path, init_weights: Path, batch: int) -> Path:
    from ultralytics import YOLO

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    model = YOLO(str(init_weights))
    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=batch,
        project=str(ROOT / "runs"),
        name=f"phase2_{variant}",
        exist_ok=True,
    )

    best = ROOT / "runs" / f"phase2_{variant}" / "weights" / "best.pt"
    dest = ckpt_dir / f"phase2_{variant}_best.pt"
    shutil.copy(best, dest)
    print(f"Saved: {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    phase1_ckpt = ROOT / "checkpoints" / "phase1_mtsd_best.pt"
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_ckpt}")

    lisa_dir = PROCESSED / "lisa"
    lisa_names = yaml.safe_load((CONFIGS / "lisa.yaml").read_text())["names"]

    # --- variant a: full 47-class ---
    print("\n=== Phase 2a: full 47-class LISA ===")
    full_yaml = _write_yaml(lisa_dir, len(lisa_names), lisa_names, "full")
    _train("full", full_yaml, phase1_ckpt, args.batch)

    # --- variant b: 4-class ---
    print("\n=== Phase 2b: 4-class LISA ===")
    lisa_4class_dir = ensure_lisa_4class()
    four_yaml = _write_yaml(lisa_4class_dir, len(COARSE_NAMES), COARSE_NAMES, "4class")
    _train("4class", four_yaml, phase1_ckpt, args.batch)


if __name__ == "__main__":
    main()
