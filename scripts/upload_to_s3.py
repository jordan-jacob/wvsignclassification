"""
Upload processed training data to S3 for AWS instance restore.

Usage:
    python scripts/upload_to_s3.py --bucket my-bucket --upload-lisa --upload-mtsd
    python scripts/upload_to_s3.py --bucket my-bucket --upload-all

Junctions (mtsd_coarse/*/images, lisa_4class/train/images) are NOT uploaded —
aws_setup.sh recreates them with ln -s on the instance.

Files already in S3 with matching MD5/ETag are skipped (resumable).
"""

import argparse
import hashlib
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Directories that are Windows junctions — skip them, recreated by aws_setup.sh
JUNCTION_DIRS = {
    PROCESSED / "lisa_4class" / "train" / "images",
    PROCESSED / "mtsd_coarse" / "train" / "images",
    PROCESSED / "mtsd_coarse" / "val" / "images",
    PROCESSED / "mtsd_4class_subset" / "train" / "images",  # junction → mtsd/train/images
}

# Files YOLO regenerates automatically — never upload
SKIP_EXTENSIONS = {".cache"}


def is_junction(path: Path) -> bool:
    return path.is_symlink() or (
        os.path.islink(str(path))
    )


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_etag(s3_client, bucket: str, key: str) -> str | None:
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        return resp["ETag"].strip('"')
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise


def collect_files(root: Path, s3_prefix: str) -> list[tuple[Path, str]]:
    """Return list of (local_path, s3_key) for all real files under root."""
    result = []
    if not root.exists():
        return result
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix in SKIP_EXTENSIONS:
            continue
        # Skip if this file lives inside a junction directory
        if any(str(p).startswith(str(j)) for j in JUNCTION_DIRS):
            continue
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        key = f"{s3_prefix}/{rel}" if s3_prefix else rel
        result.append((p, key))
    return result


def upload_batch(s3_client, bucket: str, files: list[tuple[Path, str]],
                 label: str) -> tuple[int, int]:
    """Upload files, skipping those already in S3. Returns (uploaded_bytes, skipped_count)."""
    uploaded_bytes = 0
    skipped = 0

    bar = tqdm(files, desc=label, unit="file")
    for local_path, s3_key in bar:
        local_md5 = md5_of_file(local_path)
        remote_etag = s3_etag(s3_client, bucket, s3_key)
        if remote_etag == local_md5:
            skipped += 1
            bar.set_postfix(skipped=skipped)
            continue
        size = local_path.stat().st_size
        with tqdm(total=size, unit="B", unit_scale=True,
                  desc=local_path.name, leave=False) as pbar:
            s3_client.upload_file(
                str(local_path), bucket, s3_key,
                Callback=lambda n: pbar.update(n),
            )
        uploaded_bytes += size

    return uploaded_bytes, skipped


def collect_mtsd_subset_files() -> list[tuple[Path, str]]:
    """
    Collect files for --upload-mtsd-subset:
    - Label files from mtsd_4class_subset/train/labels/ (real files)
    - The 8K selected images from mtsd/train/images/ uploaded under
      data/processed/mtsd_4class_subset/train/images/ keys in S3.

    The local images/ directory is a junction, so we resolve actual image
    paths by matching label file stems against the source images directory.
    """
    subset_dir = PROCESSED / "mtsd_4class_subset"
    labels_dir = subset_dir / "train" / "labels"
    src_images = PROCESSED / "mtsd" / "train" / "images"

    result: list[tuple[Path, str]] = []

    for lf in sorted(labels_dir.glob("*.txt")):
        # label file
        rel = lf.relative_to(PROJECT_ROOT).as_posix()
        result.append((lf, rel))
        # corresponding image (always .jpg in MTSD)
        img = src_images / (lf.stem + ".jpg")
        if img.exists():
            img_key = f"data/processed/mtsd_4class_subset/train/images/{lf.stem}.jpg"
            result.append((img, img_key))

    return result


def main():
    parser = argparse.ArgumentParser(description="Upload processed data to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--upload-lisa", action="store_true",
                        help="Upload data/processed/lisa/")
    parser.add_argument("--upload-mtsd", action="store_true",
                        help="Upload data/processed/mtsd/ and mtsd_coarse/ labels")
    parser.add_argument("--upload-checkpoints", action="store_true",
                        help="Upload checkpoints/")
    parser.add_argument("--upload-configs", action="store_true",
                        help="Upload configs/")
    parser.add_argument("--upload-mtsd-subset", action="store_true",
                        help="Upload data/processed/mtsd_4class_subset/ (labels + 8K images)")
    parser.add_argument("--upload-all", action="store_true",
                        help="Upload everything (all flags above)")
    args = parser.parse_args()

    if args.upload_all:
        args.upload_lisa = args.upload_mtsd = args.upload_checkpoints = args.upload_configs = True
        args.upload_mtsd_subset = True

    if not any([args.upload_lisa, args.upload_mtsd, args.upload_checkpoints,
                args.upload_configs, args.upload_mtsd_subset]):
        parser.error("Specify at least one --upload-* flag or --upload-all")

    s3 = boto3.client("s3")
    total_uploaded = 0
    total_skipped = 0

    batches = []

    if args.upload_lisa:
        # Upload lisa/train/images and lisa/train/labels (not lisa_4class junction)
        batches.append((PROCESSED / "lisa", "LISA (~5.4 GB)"))

    if args.upload_mtsd:
        # mtsd/ = actual images + fine labels; junctions excluded by collect_files
        batches.append((PROCESSED / "mtsd", "MTSD images+labels (~35.7 GB)"))
        # mtsd_coarse/ = coarse labels + yamls; image subdirs are junctions, skipped
        batches.append((PROCESSED / "mtsd_coarse", "mtsd_coarse labels+yamls"))

    if args.upload_checkpoints:
        batches.append((PROJECT_ROOT / "checkpoints", "checkpoints (~40 MB)"))

    if args.upload_configs:
        batches.append((PROJECT_ROOT / "configs", "configs"))

    # Deduplicate roots (e.g., mtsd_coarse added twice would double-count yamls)
    seen_roots = set()
    deduped = []
    for root, label in batches:
        if root not in seen_roots:
            seen_roots.add(root)
            deduped.append((root, label))

    for root, label in deduped:
        if isinstance(root, Path) and root.is_dir():
            files = collect_files(root, s3_prefix="")
        else:
            continue
        if not files:
            print(f"  {label}: no files found, skipping")
            continue
        up, sk = upload_batch(s3, args.bucket, files, label)
        total_uploaded += up
        total_skipped += sk

    if args.upload_mtsd_subset:
        files = collect_mtsd_subset_files()
        n_labels = sum(1 for _, k in files if k.endswith(".txt"))
        n_images = sum(1 for _, k in files if k.endswith(".jpg"))
        label = f"MTSD 4-class subset ({n_images:,} images + {n_labels:,} labels, ~6.7 GB)"
        if not files:
            print(f"  {label}: no files found — run subset_mtsd.py first")
        else:
            up, sk = upload_batch(s3, args.bucket, files, label)
            total_uploaded += up
            total_skipped += sk

    print(f"\n{'='*50}")
    print(f"Total uploaded: {total_uploaded / 1e9:.2f} GB")
    print(f"Files skipped (already in S3): {total_skipped:,}")
    print(f"Estimated S3 cost: ${total_uploaded / 1e9 * 0.023:.4f}/month (new data only)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
