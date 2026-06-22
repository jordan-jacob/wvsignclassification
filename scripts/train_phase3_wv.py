"""
Phase 3: Fine-tune on WVDOH dashcam annotations.

Starts from checkpoints/phase2_full_best.pt and trains on data/processed/wv/
as configured in configs/wv.yaml.

Flags
-----
--smoke      : 10 epochs, batch=8
--production : 150 epochs, patience=30, batch=32
--augmented  : 150 epochs, patience=30, batch=32 + aggressive color augmentation
               targeting shadow occlusion and artificial lighting failure modes
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "phase2_full_best.pt"
CONFIG = ROOT / "configs" / "wv.yaml"
OUT_CKPT = ROOT / "checkpoints" / "phase3_wv_best.pt"

# Augmentation targeting identified failure modes:
#   hsv_h/hsv_s: artificial lighting (yellow streetlight tint → warning confusion)
#   hsv_v:       shadow occlusion (white signs go grey, lose contrast)
#   scale:       distant sign size variation
#   degrees:     angled/tilted sign appearance
AUGMENTED_KWARGS = dict(
    hsv_h=0.05,
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=10,
    scale=0.6,
)


def _add_early_stop_callback(model, total_epochs: int, out_ckpt: Path) -> None:
    def on_train_end(trainer):
        actual = trainer.epoch + 1
        best = trainer.best_fitness
        if actual < total_epochs:
            print(f"\nEarly stop: epoch {actual}/{total_epochs}  best mAP50={best:.4f}")
        else:
            print(f"\nTraining complete: {total_epochs} epochs  best mAP50={best:.4f}")
        print(f"Saved: {out_ckpt}")

    model.add_callback("on_train_end", on_train_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="10 epochs, batch=8")
    ap.add_argument("--production", action="store_true",
                    help="150 epochs, patience=30, batch=32")
    ap.add_argument("--augmented", action="store_true",
                    help="150 epochs, patience=30, batch=32 + color/scale augmentation")
    args = ap.parse_args()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {CHECKPOINT}")
    if not CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {CONFIG}\n"
            "Run scripts/prepare_wv_data.py first."
        )

    if args.smoke:
        epochs, batch, patience = 10, 8, None
        run_name = "phase3_wv"
        out_ckpt = ROOT / "checkpoints" / "phase3_wv_best.pt"
        print(f"*** SMOKE MODE: {epochs} epochs, batch={batch} ***")
    elif args.augmented:
        epochs, batch, patience = 150, 32, 30
        run_name = "phase3_wv_augmented"
        out_ckpt = ROOT / "checkpoints" / "phase3_wv_augmented_best.pt"
        print(f"*** AUGMENTED MODE: {epochs} epochs, patience={patience}, batch={batch} ***")
        print(f"    Augmentation: {AUGMENTED_KWARGS}")
    elif args.production:
        epochs, batch, patience = 150, 32, 30
        run_name = "phase3_wv"
        out_ckpt = ROOT / "checkpoints" / "phase3_wv_best.pt"
        print(f"*** PRODUCTION MODE: {epochs} epochs, patience={patience}, batch={batch} ***")
    else:
        epochs, batch, patience = 10, 8, None
        run_name = "phase3_wv"
        out_ckpt = ROOT / "checkpoints" / "phase3_wv_best.pt"
        print(f"*** DEFAULT (smoke): {epochs} epochs, batch={batch} ***")

    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    _add_early_stop_callback(model, epochs, out_ckpt)

    train_kwargs = dict(
        data=str(CONFIG),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )
    if patience is not None:
        train_kwargs["patience"] = patience
    if args.augmented:
        train_kwargs.update(AUGMENTED_KWARGS)

    model.train(**train_kwargs)

    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    out_ckpt.parent.mkdir(exist_ok=True)
    shutil.copy(best, out_ckpt)


if __name__ == "__main__":
    main()
