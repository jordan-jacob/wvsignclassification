"""
Audit actual disk usage for processed training data.
Reports sizes, junction/symlink detection, and minimum viable upload estimate.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def dir_size_and_count(path: Path, follow_junctions: bool = True) -> tuple[int, int, int]:
    """Returns (bytes, n_real_files, n_junction_targets)."""
    total_bytes = 0
    n_real = 0
    n_junctions = 0
    for entry in path.rglob("*"):
        if entry.is_file(follow_symlinks=follow_junctions):
            try:
                stat = entry.stat(follow_symlinks=False)
                # On Windows, reparse points show as 0 bytes; follow to get real size
                if stat.st_size == 0 and entry.is_symlink():
                    stat = entry.stat(follow_symlinks=True)
                total_bytes += stat.st_size
                if entry.is_symlink():
                    n_junctions += 1
                else:
                    n_real += 1
            except OSError:
                pass
    return total_bytes, n_real, n_junctions


def is_junction(path: Path) -> bool:
    try:
        return path.is_symlink() or (
            os.path.islink(str(path)) or
            (hasattr(os, 'readlink') and os.readlink(str(path)) != str(path))
        )
    except OSError:
        return False


def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def fmt_count(n: int) -> str:
    return f"{n:,}"


def report_dir(label: str, path: Path, note: str = ""):
    if not path.exists():
        print(f"  {label}: MISSING  {path}")
        return 0, 0
    junction_marker = " [JUNCTION → real data elsewhere]" if is_junction(path) else ""
    print(f"\n  {label}{junction_marker}{(' — ' + note) if note else ''}")
    print(f"    Path: {path}")

    # Count subdirs
    subdirs = [d for d in path.iterdir() if d.is_dir()] if path.is_dir() else []

    total = 0
    total_files = 0
    for sd in sorted(subdirs):
        j = is_junction(sd)
        j_label = " [junction]" if j else ""
        bytes_, n_real, n_junc = dir_size_and_count(sd, follow_junctions=True)
        total += bytes_
        total_files += n_real + n_junc
        print(f"    {sd.name}/{j_label}  {fmt_gb(bytes_)}  ({fmt_count(n_real + n_junc)} files)")

    # If no subdirs, measure directly
    if not subdirs:
        bytes_, n_real, n_junc = dir_size_and_count(path, follow_junctions=True)
        total = bytes_
        total_files = n_real + n_junc

    bytes_top, n_real_top, _ = dir_size_and_count(path, follow_junctions=True)
    print(f"    TOTAL: {fmt_gb(bytes_top)}  ({fmt_count(n_real_top)} files)")
    return bytes_top, n_real_top


def main():
    print("=" * 60)
    print("STORAGE AUDIT — data/processed/")
    print("=" * 60)

    sizes = {}

    # --- LISA ---
    print("\n[1] LISA processed")
    lisa_images = PROCESSED / "lisa" / "train" / "images"
    lisa_labels = PROCESSED / "lisa" / "train" / "labels"

    img_bytes, img_n = report_dir("train/images", lisa_images,
                                   "real directory — also used by lisa_4class via junction")
    lbl_bytes, lbl_n = report_dir("train/labels", lisa_labels)
    sizes["lisa"] = img_bytes + lbl_bytes
    print(f"\n  LISA upload total (images + labels): {fmt_gb(sizes['lisa'])}")

    # --- MTSD images (shared by mtsd and mtsd_coarse via junctions) ---
    print("\n[2] MTSD processed (mtsd_coarse images are junctions to these)")
    mtsd_train_img = PROCESSED / "mtsd" / "train" / "images"
    mtsd_val_img   = PROCESSED / "mtsd" / "val"   / "images"
    mtsd_train_lbl = PROCESSED / "mtsd" / "train" / "labels"
    mtsd_val_lbl   = PROCESSED / "mtsd" / "val"   / "labels"

    ti_bytes, _ = report_dir("mtsd/train/images", mtsd_train_img)
    vi_bytes, _ = report_dir("mtsd/val/images",   mtsd_val_img)
    tl_bytes, _ = report_dir("mtsd/train/labels", mtsd_train_lbl)
    vl_bytes, _ = report_dir("mtsd/val/labels",   mtsd_val_lbl)
    sizes["mtsd_images"] = ti_bytes + vi_bytes
    sizes["mtsd_labels"] = tl_bytes + vl_bytes
    sizes["mtsd"] = sizes["mtsd_images"] + sizes["mtsd_labels"]
    print(f"\n  MTSD images (train+val): {fmt_gb(sizes['mtsd_images'])}")
    print(f"  MTSD labels (train+val): {fmt_gb(sizes['mtsd_labels'])}")
    print(f"  MTSD upload total:       {fmt_gb(sizes['mtsd'])}")

    # --- mtsd_coarse labels (separate from mtsd labels) ---
    print("\n[3] mtsd_coarse labels (coarse-remapped, upload separately from mtsd labels)")
    mc_train_lbl = PROCESSED / "mtsd_coarse" / "train" / "labels"
    mc_val_lbl   = PROCESSED / "mtsd_coarse" / "val"   / "labels"
    mc_tl_bytes, _ = report_dir("mtsd_coarse/train/labels", mc_train_lbl)
    mc_vl_bytes, _ = report_dir("mtsd_coarse/val/labels",   mc_val_lbl)
    sizes["mtsd_coarse_labels"] = mc_tl_bytes + mc_vl_bytes
    print(f"\n  mtsd_coarse labels total: {fmt_gb(sizes['mtsd_coarse_labels'])}")

    # --- Checkpoints ---
    print("\n[4] checkpoints/")
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if ckpt_dir.exists():
        ckpt_bytes, ckpt_n = report_dir("checkpoints", ckpt_dir)
        sizes["checkpoints"] = ckpt_bytes
    else:
        print("  checkpoints/: MISSING")
        sizes["checkpoints"] = 0

    # --- Summary ---
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)

    naive_total = sum(sizes.values())
    mvp_total = (
        sizes["lisa"] +
        sizes["mtsd_images"] +          # images shared by mtsd and mtsd_coarse
        sizes["mtsd_coarse_labels"] +   # coarse labels used by training
        sizes["checkpoints"]
    )

    print(f"\nNaive rsync of everything:    {fmt_gb(naive_total)}")
    print(f"  (WARNING: mtsd/train/images uploaded once but counted once — junctions avoided)")
    print(f"\nMinimum viable upload:")
    print(f"  data/processed/lisa/train/images: {fmt_gb(sizes['lisa'])}")
    print(f"  data/processed/mtsd/train/images + val/images: {fmt_gb(sizes['mtsd_images'])}")
    print(f"  data/processed/mtsd_coarse/train/labels + val/labels: {fmt_gb(sizes['mtsd_coarse_labels'])}")
    print(f"  checkpoints/: {fmt_gb(sizes['checkpoints'])}")
    print(f"  TOTAL MVP:    {fmt_gb(mvp_total)}")

    print(f"\nItems NOT needed on AWS:")
    print(f"  data/processed/mtsd/labels (fine-grained, unused):  {fmt_gb(sizes['mtsd_labels'])}")
    print(f"  data/processed/mtsd_coarse/{{train,val}}/images: [junctions, recreate on AWS with ln -s]")
    print(f"  data/processed/lisa_4class/train/images: [junction, recreate on AWS with ln -s]")
    print(f"  data/lisa/, data/mtsd/: [raw data, not needed if processed/ is uploaded]")
    print(f"  *.cache files: YOLO rebuilds these automatically")

    print(f"\nEstimated S3 cost: ${mvp_total / 1e9 * 0.023:.2f}/month")

    # Flag if MTSD exceeds 40 GB threshold
    if sizes["mtsd"] > 40e9:
        print(f"\n*** MTSD processed data ({fmt_gb(sizes['mtsd'])}) exceeds 40 GB threshold.")
        print("    Consider running scripts/subset_mtsd.py --n-images 15000")
    else:
        print(f"\nMTSD processed ({fmt_gb(sizes['mtsd'])}) is under 40 GB — subset not required.")


if __name__ == "__main__":
    main()
