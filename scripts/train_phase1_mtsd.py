#!/usr/bin/env python3
"""
Phase 1: Pre-train YOLOv8 on MTSD with 5 coarse classes.

Flags
-----
--smoke      : 5-batch sanity check       (yolov8n, batch=4, imgsz=416, 1 epoch, 20 images)
--overnight  : 10 000-image subset run    (yolov8n, batch=4, imgsz=416, 6 epochs)
--full       : all 36 589 MTSD images     (init from overnight ckpt, batch=4, imgsz=416, 8 epochs, patience=20)
--production : all 36 589 MTSD images     (yolov8m.pt ImageNet init, batch=16, imgsz=640, 200 epochs, patience=50)
default      : original full run          (yolov8m, batch=<--batch>, imgsz=640, 50 epochs)
"""
import argparse
import random
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed"


def _push_checkpoint(path: Path) -> None:
    subprocess.run(["git", "add", str(path)], cwd=ROOT, check=False)
    subprocess.run(["git", "commit", "-m", f"Add checkpoint {path.name}"], cwd=ROOT, check=False)
    subprocess.run(["git", "push"], cwd=ROOT, check=False)


def _load_coarse_cfg():
    cfg = yaml.safe_load((CONFIGS / "mtsd_to_coarse.yaml").read_text())
    return cfg["rules"], cfg["names"]


def _build_fine_to_coarse(fine_classes, rules):
    default = 4  # other-sign
    mapping = {}
    for i, label in enumerate(fine_classes):
        coarse = default
        for rule in rules:
            if label.startswith(rule["prefix"]):
                coarse = rule["class"]
                break
        mapping[i] = coarse
    return mapping


def ensure_coarse_dataset() -> Path:
    """
    Write 5-class coarse labels directly to mtsd/{split}/labels/, backing up
    the original 401-class labels to mtsd/{split}/labels_fine/.

    Ultralytics always resolves label paths as images_dir/../labels/, so the
    coarse labels must live in the canonical location alongside the images.
    No directory junctions are used.
    """
    mtsd = PROCESSED / "mtsd"
    fine_classes = yaml.safe_load((CONFIGS / "mtsd.yaml").read_text())["names"]
    rules, _ = _load_coarse_cfg()
    fine_to_coarse = _build_fine_to_coarse(fine_classes, rules)

    for split in ("train", "val"):
        labels_dir = mtsd / split / "labels"        # where ultralytics looks
        fine_dir   = mtsd / split / "labels_fine"   # backup of originals

        if fine_dir.exists():
            n = sum(1 for _ in labels_dir.glob("*.txt"))
            print(f"  {split}: coarse labels already in place ({n} files)")
            continue

        # Backup original labels, then write coarse labels in their place
        print(f"  {split}: backing up original labels to labels_fine/ ...")
        labels_dir.rename(fine_dir)
        labels_dir.mkdir()

        src_files = list(fine_dir.glob("*.txt"))
        print(f"  {split}: remapping {len(src_files)} label files ...")
        for src_file in src_files:
            lines = []
            for line in src_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                ci = fine_to_coarse[int(parts[0])]
                lines.append(f"{ci} {' '.join(parts[1:])}")
            (labels_dir / src_file.name).write_text("\n".join(lines))

        # Remove any stale label cache so ultralytics rescans cleanly
        stale = mtsd / split / "labels.cache"
        if stale.exists():
            stale.unlink()

    return mtsd


def _write_dataset_yaml(mtsd_dir: Path, names: list) -> Path:
    data = {
        "path": str(mtsd_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    out = mtsd_dir / "coarse_dataset.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _write_subset_yaml(mtsd_dir: Path, names: list, n_train: int, n_val: int,
                       tag: str, seed: int = 42) -> Path:
    """
    Sample random subsets from the real mtsd/ image dirs and write a dataset
    yaml using txt image-path lists. Labels resolve directly to
    mtsd/{split}/labels/ (which now contains the coarse 5-class labels).
    """
    rng = random.Random(seed)

    def _sample(img_dir: Path, n: int):
        imgs = sorted(img_dir.glob("*.jpg"))
        return rng.sample(imgs, min(n, len(imgs))) if n < len(imgs) else imgs

    train_imgs = _sample(mtsd_dir / "train" / "images", n_train)
    val_imgs   = _sample(mtsd_dir / "val"   / "images", n_val)

    subset_dir = PROCESSED / "subsets"
    subset_dir.mkdir(exist_ok=True)

    train_txt = subset_dir / f"{tag}_train.txt"
    val_txt   = subset_dir / f"{tag}_val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_imgs))
    val_txt.write_text(  "\n".join(str(p) for p in val_imgs))

    data = {
        "train": str(train_txt),
        "val":   str(val_txt),
        "nc":    len(names),
        "names": names,
    }
    out = subset_dir / f"dataset_{tag}.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _add_early_stop_callback(model, total_epochs: int, ckpt_name: str) -> None:
    """Print a clear summary when Ultralytics training ends (early stop or completion)."""
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


def _add_eta_callback(model, total_epochs: int) -> None:
    """After epoch 1 finishes, print actual duration and projected finish time."""
    t0 = [None]

    def on_epoch_start(trainer):
        if trainer.epoch == 0:
            t0[0] = time.time()

    def on_epoch_end(trainer):
        if trainer.epoch == 0 and t0[0] is not None:
            epoch1_sec = time.time() - t0[0]
            remaining_sec = epoch1_sec * (total_epochs - 1)
            h, rem = divmod(int(remaining_sec), 3600)
            m = rem // 60
            finish = time.strftime("%H:%M", time.localtime(time.time() + remaining_sec))
            print(
                f"\n  *** Epoch 1 took {epoch1_sec / 60:.1f} min  →  "
                f"~{h}h {m:02d}m remaining for {total_epochs - 1} more epochs  "
                f"(est. finish {finish}) ***\n"
            )

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end",   on_epoch_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch",     type=int,  default=16)
    ap.add_argument("--smoke",     action="store_true",
                    help="5-batch sanity check (yolov8n, batch=4, imgsz=416, 1 epoch)")
    ap.add_argument("--overnight", action="store_true",
                    help="10 000-image subset (yolov8n, batch=4, imgsz=416, 6 epochs)")
    ap.add_argument("--full",      action="store_true",
                    help="Full MTSD run init from overnight ckpt (batch=4, imgsz=416, 8 epochs, patience=20)")
    ap.add_argument("--production", action="store_true",
                    help="Production: yolov8m.pt ImageNet init, 200 epochs, patience=50, batch=16, imgsz=640")
    ap.add_argument("--4class-only", dest="four_class_only", action="store_true",
                    help="Accepted for command-line consistency; Phase 1 has only one production variant")
    args = ap.parse_args()

    from ultralytics import YOLO

    # --production --4class-only uses the pre-built subset (subset_mtsd.py).
    # Every other mode needs the full coarse MTSD dataset.
    if not (args.production and args.four_class_only):
        print("Preparing coarse MTSD dataset ...")
        mtsd_dir = ensure_coarse_dataset()
        _, names = _load_coarse_cfg()

    patience = None  # ultralytics default (100); overridden for --full

    if args.smoke:
        dataset_yaml  = _write_subset_yaml(mtsd_dir, names, n_train=20, n_val=20, tag="smoke")
        model_weights = "yolov8n.pt"
        epochs, batch, imgsz, run_name = 1, 4, 416, "phase1_smoke"

    elif args.overnight:
        dataset_yaml  = _write_subset_yaml(mtsd_dir, names, n_train=10000, n_val=400, tag="overnight")
        model_weights = "yolov8n.pt"
        epochs, batch, imgsz, run_name = 6, 4, 416, "phase1_overnight"

    elif args.full:
        init_ckpt = ROOT / "checkpoints" / "phase1_overnight_best.pt"
        if not init_ckpt.exists():
            raise FileNotFoundError(
                f"Overnight checkpoint not found: {init_ckpt}\n"
                "Run --overnight first."
            )
        dataset_yaml  = _write_dataset_yaml(mtsd_dir, names)
        model_weights = str(init_ckpt)
        epochs, batch, imgsz, run_name = 8, 4, 416, "phase1_full"
        patience = 20

    elif args.production:
        if args.four_class_only:
            dataset_yaml = CONFIGS / "mtsd_4class_subset.yaml"
            if not dataset_yaml.exists():
                raise FileNotFoundError(
                    f"4-class subset config not found: {dataset_yaml}\n"
                    "Run first: python scripts/subset_mtsd.py --4class-only"
                )
            # ~45-60 sec/epoch on T4 with 8K images at 640px; ~200 ep = ~3-4 hrs,
            # early stop likely around ep 80
        else:
            dataset_yaml = _write_dataset_yaml(mtsd_dir, names)
        model_weights = "yolov8m.pt"  # ImageNet init; NOT the overnight nano ckpt (different arch)
        epochs, batch, imgsz, run_name = 200, args.batch, 640, "phase1_mtsd"
        patience = 50

    else:
        dataset_yaml  = _write_dataset_yaml(mtsd_dir, names)
        model_weights = "yolov8m.pt"
        epochs, batch, imgsz, run_name = 50, args.batch, 640, "phase1_mtsd"

    mode = ("smoke" if args.smoke else "overnight" if args.overnight
            else "full" if args.full else "production" if args.production else "default")
    print(f"Mode    : {mode}")
    print(f"Model   : {model_weights}")
    print(f"Config  : epochs={epochs}  batch={batch}  imgsz={imgsz}"
          + (f"  patience={patience}" if patience is not None else ""))
    print(f"Dataset : {dataset_yaml}")

    if args.full:
        # Scale-based time estimate: full run has 36,589/10,000 = 3.66x images per epoch.
        # Epoch-ratio for 8 vs 6 adds another 8/6 = 1.33x on top.
        n_full, n_over = 36589, 10000
        img_scale  = n_full / n_over          # 3.66 — more images per epoch
        ep_scale_6 = img_scale                # 6-epoch full vs 6-epoch overnight
        ep_scale_8 = img_scale * (8 / 6)     # 8-epoch ceiling
        print()
        print(f"  Init weights : {init_ckpt.name}  (NOT yolov8n.pt)")
        print(f"  Train images : {n_full:,}  ({img_scale:.1f}x overnight per epoch)")
        print(f"  Early-stop   : patience={patience} — won't trigger within 8 epochs;")
        print(f"                 effectively runs all 8 unless you Ctrl+C")
        print()
        print("  Time estimates (based on overnight run duration):")
        for oh_h, label in [(5, "5 h overnight"), (8, "8 h overnight"), (11, "11 h overnight")]:
            h6, m6 = divmod(int(oh_h * ep_scale_6 * 3600), 3600)
            m6 //= 60
            h8, m8 = divmod(int(oh_h * ep_scale_8 * 3600), 3600)
            m8 //= 60
            print(f"    {label}: 6-ep likely ~{h6}h {m6:02d}m  |  8-ep ceiling ~{h8}h {m8:02d}m")
        print("  Exact per-epoch ETA printed after epoch 1 completes.")
        print()

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    model = YOLO(model_weights)
    if args.overnight or args.full:
        _add_eta_callback(model, epochs)
    if not args.smoke and not args.overnight:
        _add_early_stop_callback(model, epochs, f"{run_name}_best.pt")

    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )
    if patience is not None:
        train_kwargs["patience"] = patience

    model.train(**train_kwargs)

    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    ckpt_name = (
        "phase1_full_best.pt"      if args.full      else
        "phase1_overnight_best.pt" if args.overnight  else
        "phase1_smoke_best.pt"     if args.smoke      else
        "phase1_mtsd_best.pt"
    )
    dest = ckpt_dir / ckpt_name
    shutil.copy(best, dest)
    print(f"Saved: {dest}")
    _push_checkpoint(dest)
    if args.overnight or args.full:
        print()
        print("To use this checkpoint for phase 2, run:")
        print(f"  copy {dest} {ckpt_dir / 'phase1_mtsd_best.pt'}")


if __name__ == "__main__":
    main()
