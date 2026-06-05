"""
Register our COCO-format datasets with detectron2's DatasetCatalog.

Import this module before any detectron2 training or evaluation call:
    import configs.sparse_rcnn_register  # noqa
or via sys.path manipulation in train_sparse_rcnn.py.

Run coco_convert.py first to generate the JSON files.
"""

from pathlib import Path

from detectron2.data.datasets import register_coco_instances

ROOT = Path(__file__).parent.parent
COCO = ROOT / "data" / "coco"


def _reg(name: str, json_path: Path, img_dir: Path) -> None:
    if not json_path.exists():
        raise FileNotFoundError(
            f"COCO JSON not found: {json_path}\n"
            "Run: python scripts/coco_convert.py --dataset lisa"
        )
    register_coco_instances(name, {}, str(json_path), str(img_dir))


# LISA (4 classes: stop / yield / warning / speed-limit)
# No val split for LISA; lisa_val reuses train JSON for smoke evaluation.
_reg("lisa_train",
     COCO / "lisa" / "annotations" / "instances_train.json",
     COCO / "lisa" / "train")
_reg("lisa_val",
     COCO / "lisa" / "annotations" / "instances_train.json",
     COCO / "lisa" / "train")

# MTSD coarse (5 classes: stop / yield / warning / speed-limit / other-sign)
_reg("mtsd_coarse_train",
     COCO / "mtsd_coarse" / "annotations" / "instances_train.json",
     COCO / "mtsd_coarse" / "train")
_reg("mtsd_coarse_val",
     COCO / "mtsd_coarse" / "annotations" / "instances_val.json",
     COCO / "mtsd_coarse" / "val")
