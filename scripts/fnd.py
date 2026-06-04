#!/usr/bin/env python3
"""
False Negative Detector (FND) — Rahman et al., arXiv:1903.06391.

Hooks into the YOLOv8/YOLOv11 Detect layer (P3/P4/P5 inputs) to find grid
regions not covered by any detection, then runs a lightweight CNN on image
patches at those locations to recover missed signs.

Inference note: P4+P5 only (LEVELS = (1, 2)) to avoid the 6 400-cell flood
from P3. Change LEVELS to include 0 if you want P3 too.
"""
import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclasses.dataclass
class Detection:
    box: list           # [x1, y1, x2, y2] pixel coords
    conf: float
    cls: int            # -1 for FND-recovered boxes (class unknown)
    recovery: bool = False


class FNDClassifier(nn.Module):
    """3-conv / 2-FC binary classifier. Input: (B, 3, 64, 64). Output: (B, 2)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # →32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # →16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),  # →4
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 16, 256), nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class FND:
    """
    Wraps a YOLO detection model with a false-negative recovery pass.

    Compatible with YOLOv8 and YOLOv11 (Ultralytics ≥ 8.1.0) — detects the
    Detect layer by class name rather than by layer index.
    """

    PATCH_SIZE = 64
    IOU_THRESH = 0.1      # cells above this are considered covered
    FND_THRESH = 0.5      # p(sign) threshold for emitting a recovery box
    LEVELS = (1, 2)       # which FPN levels to scan: 0=P3, 1=P4, 2=P5
    CONTEXT_SCALE = 1.5   # patch half-width = stride * CONTEXT_SCALE (pixels)

    def __init__(
        self,
        yolo_model,
        fnd_ckpt: Path | None = None,
        device: str = "cpu",
    ):
        self.yolo = yolo_model
        self.device = device
        self._fmaps: list[torch.Tensor] = []

        self.classifier = FNDClassifier().to(device)
        if fnd_ckpt and Path(fnd_ckpt).exists():
            self.classifier.load_state_dict(
                torch.load(fnd_ckpt, map_location=device, weights_only=True)
            )
        self.classifier.eval()

        self._find_detect_layer().register_forward_hook(self._capture)

    def _find_detect_layer(self):
        for m in self.yolo.model.modules():
            if type(m).__name__ == "Detect":
                return m
        raise RuntimeError(
            "No Detect layer found — expected a YOLO detection model"
        )

    def _capture(self, module, inputs, output):
        # inputs[0] is the list [P3, P4, P5] passed to Detect.forward
        self._fmaps = [f.detach().cpu() for f in inputs[0]]

    @staticmethod
    def _batch_max_iou(cells: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Max IoU of each cell (N,4) against all detections (M,4). Returns (N,)."""
        x1 = np.maximum(cells[:, 0:1], dets[:, 0])
        y1 = np.maximum(cells[:, 1:2], dets[:, 1])
        x2 = np.minimum(cells[:, 2:3], dets[:, 2])
        y2 = np.minimum(cells[:, 3:4], dets[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_c = (cells[:, 2] - cells[:, 0]) * (cells[:, 3] - cells[:, 1])
        area_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        union = area_c[:, None] + area_d[None, :] - inter
        return (inter / (union + 1e-6)).max(axis=1)

    def _collect_candidates(
        self, img: Image.Image, det_boxes: list, imgsz: int
    ) -> tuple[torch.Tensor, list]:
        """
        Returns (patches, boxes).
          patches: (N, 3, PATCH_SIZE, PATCH_SIZE) float32 in [0, 1]
          boxes:   N x [x1, y1, x2, y2] pixel coords
        """
        W, H = img.size
        patches, boxes = [], []

        for level in self.LEVELS:
            if level >= len(self._fmaps):
                continue
            fm = self._fmaps[level]
            fh, fw = fm.shape[-2], fm.shape[-1]
            stride = imgsz / fw
            scale_x, scale_y = W / imgsz, H / imgsz

            gxv, gyv = np.meshgrid(
                np.arange(fw, dtype=np.float32),
                np.arange(fh, dtype=np.float32),
            )
            cells = np.stack([
                gxv * stride * scale_x,
                gyv * stride * scale_y,
                (gxv + 1) * stride * scale_x,
                (gyv + 1) * stride * scale_y,
            ], axis=-1).reshape(-1, 4)

            if det_boxes:
                max_iou = self._batch_max_iou(cells, np.array(det_boxes, dtype=np.float32))
            else:
                max_iou = np.zeros(len(cells), dtype=np.float32)

            half = stride * self.CONTEXT_SCALE * scale_x
            for idx in np.where(max_iou < self.IOU_THRESH)[0]:
                cx = (cells[idx, 0] + cells[idx, 2]) / 2
                cy = (cells[idx, 1] + cells[idx, 3]) / 2
                px1, py1 = int(max(0, cx - half)), int(max(0, cy - half))
                px2, py2 = int(min(W, cx + half)), int(min(H, cy + half))
                if px2 - px1 < 4 or py2 - py1 < 4:
                    continue
                patch = img.crop((px1, py1, px2, py2)).resize(
                    (self.PATCH_SIZE, self.PATCH_SIZE), Image.BILINEAR
                )
                arr = np.array(patch, dtype=np.float32) / 255.0
                patches.append(torch.from_numpy(arr).permute(2, 0, 1))
                boxes.append([px1, py1, px2, py2])

        if not patches:
            return torch.empty(0), []
        return torch.stack(patches), boxes

    def __call__(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        batch_size: int = 64,
    ) -> list[Detection]:
        img = Image.open(image_path).convert("RGB")
        self._fmaps = []

        results = self.yolo.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]

        base, det_boxes = [], []
        for i in range(len(r.boxes)):
            xyxy = r.boxes.xyxy[i].tolist()
            det_boxes.append(xyxy)
            base.append(Detection(
                box=xyxy, conf=float(r.boxes.conf[i]), cls=int(r.boxes.cls[i])
            ))

        if not self._fmaps:
            return base

        patches, cand_boxes = self._collect_candidates(img, det_boxes, imgsz)
        if patches.numel() == 0:
            return base

        recovered = []
        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch = patches[start : start + batch_size].to(self.device)
                probs = torch.softmax(self.classifier(batch), dim=1)[:, 1]
                for j, p in enumerate(probs.tolist()):
                    if p > self.FND_THRESH:
                        recovered.append(Detection(
                            box=cand_boxes[start + j], conf=p, cls=-1, recovery=True
                        ))

        return base + recovered


# ── smoke-test CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="FND smoke test")
    ap.add_argument("--smoke", action="store_true", required=True)
    ap.add_argument("--checkpoint", required=True, help="YOLO model checkpoint")
    ap.add_argument("--images", required=True, help="Directory of test images")
    ap.add_argument("--n", type=int, default=10, help="Number of images to test")
    sa = ap.parse_args()

    from ultralytics import YOLO

    img_dir = Path(sa.images)
    if not img_dir.exists():
        # LISA has no val split — fall back to train/images automatically
        fallback = img_dir.parent.parent / "train" / "images"
        if fallback.exists():
            print(f"NOTE: {img_dir} not found — using {fallback} (LISA has no val split)")
            img_dir = fallback
        else:
            print(f"ERROR: image directory not found: {img_dir}", file=sys.stderr)
            sys.exit(1)

    imgs = sorted(img_dir.glob("*.*"))[: sa.n]
    if not imgs:
        print(f"ERROR: no images in {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {sa.checkpoint}")
    model = YOLO(sa.checkpoint)
    print("Creating FND (untrained — weights random, recoveries will be noise)")
    fnd = FND(model, fnd_ckpt=None, device="cpu")

    print(f"\nRunning FND on {len(imgs)} images ...")
    total_base, total_recovered, last_fmaps = 0, 0, []
    for img_path in imgs:
        results = fnd(img_path)
        total_base += sum(1 for d in results if not d.recovery)
        total_recovered += sum(1 for d in results if d.recovery)
        last_fmaps = fnd._fmaps  # capture after last image

    print("\n--- Smoke test results ---")
    print(f"  Hook fired:          {'YES' if last_fmaps else 'NO  ← FAIL'}")
    print(f"  Feature map levels:  {len(last_fmaps)}")
    if last_fmaps:
        for i, fm in enumerate(last_fmaps):
            print(f"    Level {i}: {tuple(fm.shape)}")
    print(f"  Base detections:     {total_base} across {len(imgs)} images")
    print(f"  FND recovered:       {total_recovered} (random weights — noise expected)")
    if not last_fmaps:
        print("\nFAIL — hook did not fire")
        sys.exit(1)
    print("\nPASS")
