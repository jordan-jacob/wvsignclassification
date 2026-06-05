#!/usr/bin/env python3
"""
Train the FND classifier from a YOLO checkpoint + LISA dataset.

Positive examples:  GT boxes the detector missed (max IoU with any pred < 0.5)
Negative examples:  grid crops confirmed background (max IoU with all GT < 0.1)

Usage:
  python scripts/train_fnd.py --checkpoint checkpoints/phase2_full_best.pt
  python scripts/train_fnd.py --checkpoint checkpoints/phase2_full_best.pt --smoke
  python scripts/train_fnd.py --checkpoint checkpoints/phase2_4class_best.pt --production

Flags
-----
--production : epochs=50, batch=32, neg-ratio=5, patience=10
--smoke      : 50 images, 2 epochs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd import FNDClassifier  # noqa: E402


# ── data helpers ─────────────────────────────────────────────────────────────

def _load_gt(label_path: Path, W: int, H: int) -> np.ndarray:
    """YOLO-format label file → (N, 4) pixel boxes [x1, y1, x2, y2]."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split()[:5])
        boxes.append([
            (cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H,
        ])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) → (N,M) IoU matrix."""
    x1 = np.maximum(a[:, 0:1], b[:, 0])
    y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2])
    y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def _crop_resize(img: Image.Image, box, size: int = 64):
    x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), \
                      int(min(img.width, box[2])), int(min(img.height, box[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = img.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
    return np.array(patch, dtype=np.float32) / 255.0


def generate_training_data(
    yolo_model,
    images_dir: Path,
    labels_dir: Path,
    imgsz: int = 640,
    max_images: int | None = None,
    neg_ratio: int = 3,
) -> tuple[list, list]:
    """
    Returns (pos_crops, neg_crops) as lists of float32 (H,W,C) numpy arrays.

    Uses a fixed P4 stride (16px) grid for negative sampling regardless of
    model architecture. neg_ratio negatives are sampled per positive crop.
    """
    pos_crops, neg_crops = [], []
    img_paths = sorted(images_dir.glob("*.*"))
    if max_images:
        img_paths = img_paths[:max_images]

    print(f"  Scanning {len(img_paths)} images for FND training data ...")
    stride = 16  # P4 equivalent; background crops at this scale

    for img_path in img_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_boxes = _load_gt(label_path, W, H)
        if len(gt_boxes) == 0:
            continue

        results = yolo_model.predict(str(img_path), imgsz=imgsz, conf=0.1, verbose=False)
        pred_np = (results[0].boxes.xyxy.cpu().numpy()
                   if len(results[0].boxes) else np.empty((0, 4), dtype=np.float32))

        # positives: GT boxes the detector missed
        if len(pred_np) == 0:
            missed = np.ones(len(gt_boxes), dtype=bool)
        else:
            missed = _pairwise_iou(gt_boxes, pred_np).max(axis=1) < 0.5

        for box in gt_boxes[missed]:
            crop = _crop_resize(img, box)
            if crop is not None:
                pos_crops.append(crop)

        n_pos = int(missed.sum())
        if n_pos == 0:
            continue

        # negatives: background grid cells (P4 stride, 3x context window)
        gxv, gyv = np.meshgrid(
            np.arange(0, W, stride, dtype=np.float32),
            np.arange(0, H, stride, dtype=np.float32),
        )
        cells = np.stack([
            gxv, gyv,
            np.clip(gxv + stride, 0, W),
            np.clip(gyv + stride, 0, H),
        ], axis=-1).reshape(-1, 4)

        bg_mask = _pairwise_iou(cells, gt_boxes).max(axis=1) < 0.1
        bg_indices = np.where(bg_mask)[0]
        chosen = np.random.choice(
            bg_indices,
            min(len(bg_indices), n_pos * neg_ratio),
            replace=False,
        )
        half = stride * 1.5
        for idx in chosen:
            cx = (cells[idx, 0] + cells[idx, 2]) / 2
            cy = (cells[idx, 1] + cells[idx, 3]) / 2
            crop = _crop_resize(img, [cx - half, cy - half, cx + half, cy + half])
            if crop is not None:
                neg_crops.append(crop)

    print(f"  Positives (missed signs): {len(pos_crops)}")
    print(f"  Negatives (background):   {len(neg_crops)}")
    return pos_crops, neg_crops


# ── dataset + training loop ───────────────────────────────────────────────────

class FNDDataset(Dataset):
    def __init__(self, pos_crops: list, neg_crops: list):
        self.crops = pos_crops + neg_crops
        self.labels = [1] * len(pos_crops) + [0] * len(neg_crops)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        return torch.from_numpy(self.crops[i]).permute(2, 0, 1), self.labels[i]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/phase2_full_best.pt",
                    help="Trained YOLO checkpoint to mine false negatives from")
    ap.add_argument("--data", default=None,
                    help="Dataset YAML (e.g. configs/lisa.yaml); derives images/labels dirs. "
                         "Defaults to data/processed/lisa/train/ if omitted.")
    ap.add_argument("--smoke", action="store_true",
                    help="50 images, 2 epochs")
    ap.add_argument("--production", action="store_true",
                    help="Production: epochs=50, batch=32, neg-ratio=5, patience=10")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate training data and print class balance, then exit without training")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Real dashcam footage has a much higher background:sign ratio than LISA/MTSD.
    # Increase --neg-ratio when fine-tuning on WVDOH data to match that distribution.
    ap.add_argument("--neg-ratio", type=int, default=3,
                    help="Negative crops sampled per positive crop (default 3; "
                         "increase for dashcam data where background dominates)")
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt_path}")

    # Derive images/labels dirs from --data YAML or fall back to default LISA location
    if args.data:
        import yaml as _yaml
        data_cfg = _yaml.safe_load((ROOT / args.data).read_text())
        data_base = Path(data_cfg.get("path", (ROOT / args.data).parent))
        train_key = data_cfg.get("train", "train/images")
        images_dir = Path(train_key) if Path(train_key).is_absolute() else data_base / train_key
        labels_dir = images_dir.parent.parent / "train" / "labels"
        if not labels_dir.exists():
            labels_dir = images_dir.parent / ".." / "labels"
            labels_dir = labels_dir.resolve()
    else:
        processed = ROOT / "data" / "processed" / "lisa"
        images_dir = processed / "train" / "images"
        labels_dir = processed / "train" / "labels"

    if args.smoke:
        epochs, batch, neg_ratio, max_images = 2, args.batch, args.neg_ratio, 50
        print(f"*** SMOKE MODE: {max_images} images, {epochs} epochs ***\n")
    elif args.production:
        epochs, batch, neg_ratio, max_images = 50, 32, 5, None
        print(f"*** PRODUCTION MODE: {epochs} epochs, batch={batch}, neg-ratio={neg_ratio} ***\n")
    else:
        epochs, batch, neg_ratio, max_images = args.epochs, args.batch, args.neg_ratio, None
    if args.dry_run:
        print("*** DRY-RUN MODE: generating training data only, no training ***\n")

    yolo = YOLO(str(ckpt_path))

    pos_crops, neg_crops = generate_training_data(
        yolo,
        images_dir=images_dir,
        labels_dir=labels_dir,
        imgsz=640,
        max_images=max_images,
        neg_ratio=neg_ratio,
    )

    total = len(pos_crops) + len(neg_crops)
    print(f"\n  Class balance:")
    print(f"    sign (positive): {len(pos_crops):>6}  ({100*len(pos_crops)/max(total,1):.1f}%)")
    print(f"    background:      {len(neg_crops):>6}  ({100*len(neg_crops)/max(total,1):.1f}%)")
    print(f"    total:           {total:>6}")

    if args.dry_run:
        print("\nDry-run complete — exiting without training.")
        return

    if not pos_crops:
        raise RuntimeError(
            "No positive crops generated — the detector may already catch everything, "
            "or the checkpoint and data paths may not match."
        )

    dataset = FNDDataset(pos_crops, neg_crops)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

    model = FNDClassifier().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    out = ROOT / "checkpoints" / "fnd_classifier.pt"
    out.parent.mkdir(exist_ok=True)

    fnd_patience = 10
    best_loss = float("inf")
    no_improve = 0

    print(f"\nTraining FND classifier ({len(dataset)} samples, {epochs} epochs, "
          f"patience={fnd_patience}) ...")
    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(model, loader, optimizer, criterion, args.device)
        print(f"  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")
        if loss < best_loss - 1e-4:
            best_loss = loss
            no_improve = 0
            torch.save(model.state_dict(), out)
        else:
            no_improve += 1
            if no_improve >= fnd_patience:
                print(f"\nEarly stop: epoch {epoch}/{epochs} -- "
                      f"best loss={best_loss:.4f}. Saved: {out}")
                break
    else:
        print(f"\nTraining complete: {epochs} epochs. "
              f"Best loss={best_loss:.4f}. Saved: {out}")


if __name__ == "__main__":
    main()
