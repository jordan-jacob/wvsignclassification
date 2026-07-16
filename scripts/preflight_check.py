#!/usr/bin/env python3
"""
preflight_check.py — fail fast before YOLO training on a broken dataset.

Validates one or more Ultralytics dataset YAMLs and exits NONZERO if anything
that would waste GPU hours (or silently corrupt a run) is wrong. Designed to run
in <5s on CPU using only the standard library + pyyaml (no torch/ultralytics),
so it can be the first step of a training chain:

    python scripts/preflight_check.py configs/wv.yaml || exit 1

What it checks, per split (train/val), for each yaml:
  - images dir exists, resolves through symlinks to a real dir, is non-empty
  - labels dir exists (images path with /images -> /labels) and has .txt files
  - NO dangling symlinks (the exact NVMe-wipe / missing-junction signature)
  - every label line has exactly 5 fields  (class cx cy w h)
  - every class index is an integer in [0, nc)   <-- taxonomy-contamination guard
  - bbox coords are in [0, 1]
  - image/label stems pair up (reports orphans both directions)

Empty (0-byte) label files are VALID verified-background negatives: counted,
never treated as errors.

Exit code 0 only if ALL yamls pass every hard check. Warnings (e.g. orphan
images) do not fail the run unless --strict is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FATAL: pyyaml not installed (pip install pyyaml --break-system-packages)")
    sys.exit(2)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SAMPLE_LABELS_FOR_INDEX_CHECK = 100000  # effectively all; cheap enough


class SplitReport:
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.n_images = 0
        self.n_labels = 0
        self.n_empty_labels = 0
        self.n_boxes = 0


def resolve_split_paths(yaml_path: Path, data: dict, split: str):
    """Return (images_dir, labels_dir) Paths the way ultralytics resolves them.

    `path` is the dataset root; `train`/`val` are relative to it unless absolute.
    labels dir is the images dir with the final '/images' replaced by '/labels'.
    """
    root = data.get("path")
    if root is None:
        # No explicit root: paths are relative to the yaml's own directory.
        root = yaml_path.parent
    root = Path(root)

    rel = data.get(split)
    if rel is None:
        return None, None
    rel = Path(rel)
    images_dir = rel if rel.is_absolute() else (root / rel)

    # derive labels dir: replace a trailing 'images' component with 'labels'
    parts = list(images_dir.parts)
    if "images" in parts:
        # replace the LAST occurrence of 'images'
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                break
        labels_dir = Path(*parts)
    else:
        labels_dir = images_dir.parent / "labels"

    return images_dir, labels_dir


def check_dir(path: Path, kind: str, rpt: SplitReport) -> bool:
    """Validate a dir exists, isn't a dangling symlink, and is a real directory."""
    if path is None:
        rpt.errors.append(f"{kind} path is not defined in yaml")
        return False
    if path.is_symlink() and not path.exists():
        target = "?"
        try:
            target = str(path.readlink())
        except OSError:
            pass
        rpt.errors.append(
            f"{kind} dir is a DANGLING SYMLINK: {path} -> {target} "
            f"(NVMe wipe / missing junction signature)"
        )
        return False
    if not path.exists():
        rpt.errors.append(f"{kind} dir missing: {path}")
        return False
    if not path.is_dir():
        rpt.errors.append(f"{kind} path is not a directory: {path}")
        return False
    return True


def validate_split(yaml_path: Path, data: dict, split: str, nc: int) -> SplitReport:
    rpt = SplitReport(split)
    images_dir, labels_dir = resolve_split_paths(yaml_path, data, split)

    if not check_dir(images_dir, f"{split} images", rpt):
        return rpt
    if not check_dir(labels_dir, f"{split} labels", rpt):
        return rpt

    images = [p for p in images_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMG_EXTS]
    labels = [p for p in labels_dir.glob("*.txt")]
    rpt.n_images = len(images)
    rpt.n_labels = len(labels)

    if rpt.n_images == 0:
        rpt.errors.append(f"{split} images dir is EMPTY: {images_dir}")
    if rpt.n_labels == 0:
        rpt.errors.append(f"{split} labels dir has NO .txt files: {labels_dir}")
    if rpt.errors:
        return rpt

    # ---- validate label contents ----
    bad_field_count = 0
    bad_index = 0
    bad_coord = 0
    worst_index = -1
    examples: list[str] = []

    for lf in labels:
        text = lf.read_text()
        if not text.strip():
            rpt.n_empty_labels += 1
            continue
        for ln, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_field_count += 1
                if len(examples) < 5:
                    examples.append(f"{lf.name}:{ln} -> {len(parts)} fields: {line[:60]}")
                continue
            rpt.n_boxes += 1
            try:
                idx = int(parts[0])
            except ValueError:
                bad_index += 1
                if len(examples) < 5:
                    examples.append(f"{lf.name}:{ln} -> non-int class: {parts[0]}")
                continue
            if idx < 0 or idx >= nc:
                bad_index += 1
                worst_index = max(worst_index, idx)
                if len(examples) < 5:
                    examples.append(f"{lf.name}:{ln} -> class {idx} (nc={nc})")
            try:
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                bad_coord += 1
                continue
            if any(c < 0.0 or c > 1.0 for c in coords):
                bad_coord += 1
                if len(examples) < 5:
                    examples.append(f"{lf.name}:{ln} -> coords out of [0,1]: {coords}")

    if bad_field_count:
        rpt.errors.append(f"{bad_field_count} label line(s) with != 5 fields")
    if bad_index:
        rpt.errors.append(
            f"{bad_index} box(es) with class index out of range [0,{nc}) "
            f"(worst seen: {worst_index}) — TAXONOMY CONTAMINATION"
        )
    if bad_coord:
        rpt.errors.append(f"{bad_coord} box(es) with coords outside [0,1]")
    for ex in examples:
        rpt.errors.append(f"    e.g. {ex}")

    # ---- pairing (warnings, not hard fails by default) ----
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    imgs_no_label = img_stems - lbl_stems
    lbls_no_image = lbl_stems - img_stems
    if imgs_no_label:
        rpt.warnings.append(
            f"{len(imgs_no_label)} image(s) have NO label file "
            f"(will be SKIPPED by run_merged; create empty .txt to include as background)"
        )
    if lbls_no_image:
        rpt.warnings.append(f"{len(lbls_no_image)} label(s) have no matching image")

    return rpt


def validate_yaml(yaml_path: Path, strict: bool) -> bool:
    print(f"\n=== {yaml_path} ===")
    if not yaml_path.exists():
        print(f"  FAIL: yaml does not exist")
        return False
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as e:
        print(f"  FAIL: yaml could not be parsed: {e}")
        return False

    nc = data.get("nc")
    names = data.get("names")
    if nc is None and names is not None:
        nc = len(names)
    if nc is None:
        print("  FAIL: yaml has neither 'nc' nor 'names'")
        return False
    if names is not None and len(names) != nc:
        print(f"  FAIL: nc={nc} but names has {len(names)} entries")
        return False
    print(f"  nc={nc}  names={'present' if names else 'MISSING'}")

    ok = True
    for split in ("train", "val"):
        if split not in data:
            print(f"  [{split}] not defined in yaml — skipping")
            continue
        rpt = validate_split(yaml_path, data, split, nc)
        status = "PASS" if not rpt.errors else "FAIL"
        print(
            f"  [{split}] {status}  "
            f"images={rpt.n_images} labels={rpt.n_labels} "
            f"(empty/background={rpt.n_empty_labels}) boxes={rpt.n_boxes}"
        )
        for w in rpt.warnings:
            print(f"      WARN: {w}")
            if strict:
                ok = False
        for e in rpt.errors:
            print(f"      ERROR: {e}")
        if rpt.errors:
            ok = False

    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yamls", nargs="+", type=Path,
                    help="One or more Ultralytics dataset YAML files")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings (orphan images/labels) as failures")
    args = ap.parse_args()

    all_ok = True
    for y in args.yamls:
        if not validate_yaml(y, args.strict):
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("PREFLIGHT PASSED — safe to train")
        sys.exit(0)
    else:
        print("PREFLIGHT FAILED — fix the above before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
