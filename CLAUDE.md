# CLAUDE.md

# WV Roadway Sign Detection — Project Context

## Goal
Detect missing, obscured, and damaged roadway signs on West Virginia roads
using computer vision and image segmentation on dashcam video from WVDOH.

## Pipeline (planned)
1. Pre-train backbone on MTSD (global, binary detection)
2. Fine-tune classification head on MTSD North America + LISA (47 MUTCD classes)
3. Fine-tune on WVDOH dashcam footage (pending data delivery)
4. Spatial post-processing layer over WV road network graph (STGAN-style)
   to flag segments where expected signs are absent or undetected

## Data locations (local, not in repo)
- LISA (Kaggle mirror, 47 classes, CSV annotations): data/lisa/
- MTSD (fully annotated, ~52K images, per-image JSON): data/mtsd/
- WVDOH dashcam footage: not yet received

## Key design decisions
- Backbone: ResNet-50, ImageNet init
- Detector: YOLOv8 (target) — pipeline validated with LISA first
- No ordinal contrastive loss (sign classes are discrete/unordered)
- Standard SupCon may be used for occlusion-robust embeddings
- Geographic filtering of MTSD: NOT applied at pre-training;
  US label remapping applied at classification head training only

## WV Taxonomy (Phase 3, 11 classes — configs/wv.yaml)
chevron, curve, deerCrossing, guide, other, pedestrianCrossing,
railroadCrossing, speedLimit, stop, warning, yield

**Dropped classes — `missing_expected`, `occluded`:**
Absence and occlusion detection cannot be resolved from a single image frame.
Both require georeferencing against WVDOT's sign inventory (a planned separate
component) to determine what sign *should* be present and whether it is blocked.
Annotators confirmed these categories are not reliably identifiable visually
per-frame. Boxes for these classes are discarded at data prep time.

**Merged into `other`:** damaged, laneEnds, intersection, schoolZone,
ruralCrossing_other — too few examples to train a reliable head, and the
distinctions are lower priority for the initial Phase 3 model.

Remapping is applied in `scripts/prepare_wv_data.py` (CLASS_REMAP dict).

## Repo structure
scripts/        training and eval scripts
notebooks/      EDA and visualization
configs/        YAML configs for data paths and hyperparameters
data/           symlinks to local data (gitignored)
checkpoints/    model weights (gitignored)

## Commands
# Install deps
pip install -r requirements.txt

# Explore LISA annotations
python scripts/explore_lisa.py

# Pre-train on MTSD (Phase 1)
python scripts/pretrain_mtsd.py --config configs/pretrain.yaml

### general rules

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## In Progress

### Sparse R-CNN

Scripts are written and smoke-tested. **One remaining blocker before training can run:**
detectron2 environment setup on AWS (~45 min, build from source). Do NOT install
detectron2 into the existing Ultralytics venv — they have conflicting torch version
requirements in some configurations.

**What's done:**
- `scripts/coco_convert.py` — converts YOLO-format processed data to COCO JSON.
  Uses `lisa_4class/train/labels/` (4 coarse classes) for LISA; `mtsd_coarse/` (5 classes).
  Validates output with pycocotools. Smoke-tested: 4 categories, 100 images, 60 annotations.
- `configs/sparse_rcnn_lisa.yaml` — detectron2 CfgNode config (4-class LISA, 3x schedule).
- `configs/sparse_rcnn_register.py` — registers COCO datasets with detectron2's DatasetCatalog.
- `scripts/train_sparse_rcnn.py` — DefaultTrainer launch script, `--smoke` flag.
- `scripts/compare_models.py` — `SparseRCNNEvaluator` added; handles missing detectron2 gracefully.

**AWS setup (TODO — do not modify aws_setup.sh tonight):**
A separate detectron2 install block needs to be added to `aws_setup.sh`:
```bash
conda create -n sparsercnn python=3.9
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone https://github.com/PeizeSun/SparseR-CNN  # clone adjacent to this repo
cd SparseR-CNN && pip install -r requirements.txt
```
Then run `coco_convert.py --dataset lisa` (and `--dataset mtsd_coarse` if needed)
before the first training run.

See `configs/sparse_rcnn.yaml` for full data format notes and AWS time estimates.

## Deferred
