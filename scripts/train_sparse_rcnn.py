# Requires: pip install detectron2
#   (see https://detectron2.readthedocs.io/tutorials/install.html)
# and: git clone https://github.com/PeizeSun/SparseR-CNN
# This script assumes SparseR-CNN is cloned adjacent to
# this repo or on PYTHONPATH.
#
# Before running, convert data:
#   python scripts/coco_convert.py --dataset lisa
#   python scripts/coco_convert.py --dataset mtsd_coarse  (if needed)

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import sparse_rcnn_register  # noqa: F401 — registers lisa_train/val, mtsd_coarse_train/val

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Sparse R-CNN on sign data")
    parser.add_argument(
        "--config", default="configs/sparse_rcnn_lisa.yaml",
        help="Path to detectron2 config YAML (relative to repo root)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 100 iters, batch=2 — confirms data loads and model initializes",
    )
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / args.config))

    if args.smoke:
        cfg.SOLVER.MAX_ITER = 100
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.TEST.EVAL_PERIOD = 0  # skip eval during smoke

    cfg.OUTPUT_DIR = str(ROOT / cfg.OUTPUT_DIR)
    cfg.freeze()

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
