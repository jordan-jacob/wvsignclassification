"""
Phase 3: Fine-tune on WVDOH dashcam annotations.

Starts from checkpoints/phase2_full_best.pt and trains on data/processed/wv/
as configured in configs/wv.yaml.

Flags
-----
--smoke      : 10 epochs, batch=8
--production : 150 epochs, patience=30, batch=32
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "phase2_full_best.pt"
CONFIG = ROOT / "configs" / "wv.yaml"
OUT_CKPT = ROOT / "checkpoints" / "phase3_wv_best.pt"


def _add_early_stop_callback(model, total_epochs: int) -> None:
    def on_train_end(trainer):
        actual = trainer.epoch + 1
        best = trainer.best_fitness
        if actual < total_epochs:
            print(f"\nEarly stop: epoch {actual}/{total_epochs}  best mAP50={best:.4f}")
        else:
            print(f"\nTraining complete: {total_epochs} epochs  best mAP50={best:.4f}")
        print(f"Saved: {OUT_CKPT}")

    model.add_callback("on_train_end", on_train_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="10 epochs, batch=8")
    ap.add_argument("--production", action="store_true",
                    help="150 epochs, patience=30, batch=32")
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
        print(f"*** SMOKE MODE: {epochs} epochs, batch={batch} ***")
    elif args.production:
        epochs, batch, patience = 150, 32, 30
        print(f"*** PRODUCTION MODE: {epochs} epochs, patience={patience}, batch={batch} ***")
    else:
        epochs, batch, patience = 10, 8, None
        print(f"*** DEFAULT (smoke): {epochs} epochs, batch={batch} ***")

    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    _add_early_stop_callback(model, epochs)

    train_kwargs = dict(
        data=str(CONFIG),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=str(ROOT / "runs"),
        name="phase3_wv",
        exist_ok=True,
    )
    if patience is not None:
        train_kwargs["patience"] = patience

    model.train(**train_kwargs)

    best = ROOT / "runs" / "phase3_wv" / "weights" / "best.pt"
    OUT_CKPT.parent.mkdir(exist_ok=True)
    shutil.copy(best, OUT_CKPT)


if __name__ == "__main__":
    main()
