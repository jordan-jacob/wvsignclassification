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
