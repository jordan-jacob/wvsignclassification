#!/usr/bin/env python3
"""
Phase 1: Pre-train YOLOv8 on MTSD with 5 coarse classes.

Flags
-----
--smoke     : 5-batch sanity check  (yolov8n, batch=4, imgsz=416, 1 epoch, 20 images)
--overnight : 10 000-image subset run (yolov8n, batch=4, imgsz=416, 6 epochs)
default     : full run               (yolov8m, batch=<--batch>, imgsz=640, 50 epochs)
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


def _junction(link: Path, target: Path) -> None:
    """Create a Windows directory junction (no admin required)."""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())],
        check=True,
    )


def ensure_coarse_dataset() -> Path:
    mtsd = PROCESSED / "mtsd"
    coarse = PROCESSED / "mtsd_coarse"

    fine_classes = yaml.safe_load((CONFIGS / "mtsd.yaml").read_text())["names"]
    rules, _ = _load_coarse_cfg()
    fine_to_coarse = _build_fine_to_coarse(fine_classes, rules)

    for split in ("train", "val"):
        src_lbl = mtsd / split / "labels"
        dst_lbl = coarse / split / "labels"
        dst_img = coarse / split / "images"

        _junction(dst_img, mtsd / split / "images")
        dst_lbl.mkdir(parents=True, exist_ok=True)

        existing = {p.name for p in dst_lbl.glob("*.txt")}
        src_files = list(src_lbl.glob("*.txt"))
        todo = [f for f in src_files if f.name not in existing]
        if not todo:
            print(f"  {split}: coarse labels already present ({len(src_files)} files)")
            continue

        print(f"  {split}: remapping {len(todo)} label files ...")
        for src_file in todo:
            lines = []
            for line in src_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                ci = fine_to_coarse[int(parts[0])]
                lines.append(f"{ci} {' '.join(parts[1:])}")
            (dst_lbl / src_file.name).write_text("\n".join(lines))

    return coarse


def _write_dataset_yaml(coarse_dir: Path, names: list) -> Path:
    data = {
        "path": str(coarse_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    out = coarse_dir / "dataset.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


def _write_subset_yaml(coarse_dir: Path, names: list, n_train: int, n_val: int,
                       tag: str, seed: int = 42) -> Path:
    """
    Sample random subsets and write a dataset yaml using txt image-path lists.
    Paths keep the junction dir so ultralytics resolves labels to
    mtsd_coarse/{split}/labels/ correctly (not the original mtsd/ labels).
    """
    rng = random.Random(seed)

    def _sample(img_dir: Path, n: int):
        imgs = sorted(img_dir.glob("*.jpg"))
        return rng.sample(imgs, min(n, len(imgs))) if n < len(imgs) else imgs

    train_imgs = _sample(coarse_dir / "train" / "images", n_train)
    val_imgs   = _sample(coarse_dir / "val"   / "images", n_val)

    train_txt = coarse_dir / f"_{tag}_train.txt"
    val_txt   = coarse_dir / f"_{tag}_val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_imgs))
    val_txt.write_text(  "\n".join(str(p) for p in val_imgs))

    data = {
        "train": str(train_txt),
        "val":   str(val_txt),
        "nc":    len(names),
        "names": names,
    }
    out = coarse_dir / f"_dataset_{tag}.yaml"
    out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    return out


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
    args = ap.parse_args()

    from ultralytics import YOLO

    print("Preparing coarse MTSD dataset ...")
    coarse_dir = ensure_coarse_dataset()

    _, names = _load_coarse_cfg()

    if args.smoke:
        # 20 images ÷ batch=4 = exactly 5 batches
        dataset_yaml  = _write_subset_yaml(coarse_dir, names, n_train=20, n_val=20, tag="smoke")
        model_weights = "yolov8n.pt"
        epochs, batch, imgsz, run_name = 1, 4, 416, "phase1_smoke"
    elif args.overnight:
        dataset_yaml  = _write_subset_yaml(coarse_dir, names, n_train=10000, n_val=400, tag="overnight")
        model_weights = "yolov8n.pt"
        epochs, batch, imgsz, run_name = 6, 4, 416, "phase1_overnight"
    else:
        dataset_yaml  = _write_dataset_yaml(coarse_dir, names)
        model_weights = "yolov8m.pt"
        epochs, batch, imgsz, run_name = 50, args.batch, 640, "phase1_mtsd"

    print(f"Mode    : {'smoke' if args.smoke else 'overnight' if args.overnight else 'full'}")
    print(f"Model   : {model_weights}  epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print(f"Dataset : {dataset_yaml}")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    model = YOLO(model_weights)
    if args.overnight:
        _add_eta_callback(model, epochs)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )

    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    ckpt_name = (
        "phase1_overnight_best.pt" if args.overnight else
        "phase1_smoke_best.pt"     if args.smoke     else
        "phase1_mtsd_best.pt"
    )
    dest = ckpt_dir / ckpt_name
    shutil.copy(best, dest)
    print(f"Saved: {dest}")
    if args.overnight:
        print()
        print("To use this checkpoint for phase 2, run:")
        print(f"  copy {dest} {ckpt_dir / 'phase1_mtsd_best.pt'}")


if __name__ == "__main__":
    main()
