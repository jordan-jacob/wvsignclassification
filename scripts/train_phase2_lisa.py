#!/usr/bin/env python3
"""
Phase 2: Fine-tune on LISA from phase-1 checkpoint.

Trains two models in sequence:
  a) Full 47-class model   → checkpoints/phase2_full_best.pt
  b) 4-class coarse model  → checkpoints/phase2_4class_best.pt

Flags
-----
--production : 300 epochs, patience=50, batch=16, imgsz=640
--smoke      : 2 epochs, 200 images, batch=4, imgsz=416
"""
import argparse
import shutil
import subprocess
import sys
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


def _push_checkpoint(path: Path) -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "save_checkpoint_to_github.sh"),
         str(path), f"Add {path.name} — training complete"],
        cwd=str(ROOT), check=False,
    )


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

    for split in ("train", "val"):
        src_lbl = lisa / split / "labels"
        dst_lbl = out_dir / split / "labels"
        dst_img = out_dir / split / "images"
        _junction(dst_img, lisa / split / "images")
        dst_lbl.mkdir(parents=True, exist_ok=True)
        src_files = list(src_lbl.glob("*.txt"))
        existing = {p.name for p in dst_lbl.glob("*.txt")}
        todo = [f for f in src_files if f.name not in existing]
        if not todo:
            print(f"  4-class {split} labels already present ({len(src_files)} files)")
            continue
        print(f"  Remapping {len(todo)} LISA {split} label files to 4 classes ...")
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
        "val": "val/images",
        "nc": nc,
        "names": names,
    }
    out = data_dir / f"dataset_{suffix}.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _make_smoke_txt(images_dir: Path, n: int, out_path: Path) -> Path:
    """Write a .txt file listing n image paths for YOLO's txt-file dataset format.

    Use images_dir / p.name (not p.resolve()) so junction paths are preserved —
    YOLO's images->labels substitution must resolve to the correct labels dir.
    """
    imgs = sorted(images_dir.glob("*.*"))[:n]
    out_path.write_text("\n".join(str(images_dir / p.name) for p in imgs))
    print(f"  Smoke subset: {len(imgs)} images -> {out_path.name}")
    return out_path


def _write_yaml_from_txt(txt_path: Path, nc: int, names: list, suffix: str) -> Path:
    """Write a dataset YAML whose train/val keys point to a .txt image list."""
    data = {
        "train": str(txt_path.resolve()),
        "val": str(txt_path.resolve()),
        "nc": nc,
        "names": names,
    }
    out = txt_path.parent / f"dataset_{suffix}.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _add_early_stop_callback(model, total_epochs: int, ckpt_name: str) -> None:
    def on_train_end(trainer):
        actual = trainer.epoch + 1
        best = trainer.best_fitness
        dest = f"checkpoints/{ckpt_name}"
        if actual < total_epochs:
            print(f"\nEarly stop: epoch {actual}/{total_epochs} -- "
                  f"best mAP50={best:.4f}. Saved: {dest}")
        else:
            print(f"\nTraining complete: {total_epochs} epochs. "
                  f"Best mAP50={best:.4f}. Saved: {dest}")
    model.add_callback("on_train_end", on_train_end)


def _preflight(yaml_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight_check.py"), str(yaml_path)],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Preflight check failed for {yaml_path}; aborting.")


def _train(variant: str, data_yaml: Path, init_weights: Path,
           batch: int, epochs: int = 30, imgsz: int = 640,
           patience: int | None = None) -> Path:
    from ultralytics import YOLO

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_name = f"phase2_{variant}_best.pt"
    model = YOLO(str(init_weights))
    _add_early_stop_callback(model, epochs, ckpt_name)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs"),
        name=f"phase2_{variant}",
        exist_ok=True,
    )
    if patience is not None:
        train_kwargs["patience"] = patience

    model.train(**train_kwargs)

    best = ROOT / "runs" / f"phase2_{variant}" / "weights" / "best.pt"
    dest = ckpt_dir / ckpt_name
    shutil.copy(best, dest)
    _push_checkpoint(dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick validation: 2 epochs, 200 images, batch=4, imgsz=416")
    ap.add_argument("--production", action="store_true",
                    help="Production: 300 epochs, patience=50, batch=16, imgsz=640")
    ap.add_argument("--4class-only", dest="four_class_only", action="store_true",
                    help="Skip the 47-class variant; train only the 4-class coarse model")
    args = ap.parse_args()

    phase1_ckpt = ROOT / "checkpoints" / "phase1_mtsd_best.pt"
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_ckpt}")

    lisa_dir = PROCESSED / "lisa"
    lisa_names = yaml.safe_load((CONFIGS / "lisa.yaml").read_text())["names"]

    if args.smoke:
        epochs, imgsz, batch, patience, n = 2, 416, 4, None, 200
        print(f"\n*** SMOKE MODE: {epochs} epochs, {n} images, batch={batch}, imgsz={imgsz} ***")

        # variant a: full 47-class smoke
        print("\n=== Phase 2a (smoke): full 47-class LISA ===")
        smoke_full_txt = _make_smoke_txt(
            lisa_dir / "train" / "images", n,
            PROCESSED / "lisa_smoke_full.txt",
        )
        full_yaml = _write_yaml_from_txt(smoke_full_txt, len(lisa_names), lisa_names, "smoke_full")
        _train("smoke_full", full_yaml, phase1_ckpt, batch, epochs=epochs, imgsz=imgsz)

        # variant b: 4-class smoke
        # ensure_lisa_4class creates remapped labels under lisa_4class/train/labels/;
        # images are a junction to lisa/train/images so YOLO resolves labels correctly.
        print("\n=== Phase 2b (smoke): 4-class LISA ===")
        lisa_4class_dir = ensure_lisa_4class()
        smoke_4class_txt = _make_smoke_txt(
            lisa_4class_dir / "train" / "images", n,
            PROCESSED / "lisa_smoke_4class.txt",
        )
        four_yaml = _write_yaml_from_txt(smoke_4class_txt, len(COARSE_NAMES), COARSE_NAMES, "smoke_4class")
        _train("smoke_4class", four_yaml, phase1_ckpt, batch, epochs=epochs, imgsz=imgsz)

    else:
        epochs  = 300 if args.production else 30
        imgsz   = 640
        batch   = args.batch
        patience = 50 if args.production else None

        if args.production:
            print(f"\n*** PRODUCTION MODE: {epochs} epochs, patience={patience}, "
                  f"batch={batch}, imgsz={imgsz} ***")

        if not args.four_class_only:
            # --- variant a: full 47-class ---
            print("\n=== Phase 2a: full 47-class LISA ===")
            full_yaml = _write_yaml(lisa_dir, len(lisa_names), lisa_names, "full")
            _preflight(full_yaml)
            _train("full", full_yaml, phase1_ckpt, batch, epochs=epochs,
                   imgsz=imgsz, patience=patience)
        else:
            print("\n=== Skipping Phase 2a (--4class-only) ===")

        # --- variant b: 4-class ---
        print("\n=== Phase 2b: 4-class LISA ===")
        lisa_4class_dir = ensure_lisa_4class()
        four_yaml = _write_yaml(lisa_4class_dir, len(COARSE_NAMES), COARSE_NAMES, "4class")
        _preflight(four_yaml)
        _train("4class", four_yaml, phase1_ckpt, batch, epochs=epochs,
               imgsz=imgsz, patience=patience)


if __name__ == "__main__":
    main()
