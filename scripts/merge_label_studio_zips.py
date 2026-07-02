#!/usr/bin/env python3
"""
Merge 5 Label Studio export zip pairs into a single clean dataset.

Output layout:
    wv_merged_new/
        labels/      — one .txt per frame (empty = background)
        images_src/  — matching image files (.jpeg normalised to .jpg)

Run from anywhere:
    python scripts/merge_label_studio_zips.py
"""
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

import winreg as _winreg
_key     = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER,
               r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
_DESKTOP = Path(_winreg.QueryValueEx(_key, "Desktop")[0])
_winreg.CloseKey(_key)

DATA_DIR = _DESKTOP / "data"
OUT_DIR  = _DESKTOP / "wv_merged_new"

# Ordered: first-wins deduplication across rounds
ZIP_PAIRS = [
    ("annotation_round1.zip",       "annotation_round1_annotations.zip"),
    ("annotation_round2.zip",       "annotation_round2_annotations.zip"),
    ("annotation_round2_bulk.zip",  "annotation_round2_bulk_annotations.zip"),
    ("annotation_round2_final.zip", "annotation_round2_final_annotations.zip"),
    ("annotation_frames_bulk.zip",  "annotation_frames_bulk_annotations.zip"),
]

HASH_RE    = re.compile(r"^[0-9a-f]{8}__")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SKIP_NAMES = {"classes.txt", "notes.json"}
CLASS_RE   = re.compile(r"^\s*(\S+)")


# ── helpers ──────────────────────────────────────────────────────────────────

def decode_stem(raw_filename: str) -> str:
    """
    Label Studio export filename → clean stem.

    010d0e11__foo%5Ccandidates%5C30055_000114.txt  →  30055_000114
    """
    stem = Path(raw_filename).stem  # drop .txt
    stem = unquote(stem)            # %5C → backslash, etc.
    stem = HASH_RE.sub("", stem)    # strip 8-hex-char hash prefix
    # split on both / and \, keep last part
    return re.split(r"[/\\]", stem)[-1]


def normalise_ext(original_name: str) -> str:
    e = Path(original_name).suffix.lower()
    return ".jpg" if e == ".jpeg" else e


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    labels_dir = OUT_DIR / "labels"
    images_dir = OUT_DIR / "images_src"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    seen: set = set()   # stems already written (first-wins)
    rows = []           # per-round summary dicts

    # ── load seen_stems (frames imported into Label Studio for annotation_round2_bulk) ──
    seen_stems_path = _DESKTOP / "seen_stems.txt"
    if not seen_stems_path.exists():
        print(f"ERROR: seen_stems.txt not found at {seen_stems_path}")
        return 1
    seen_stems: set = {
        line.strip() for line in seen_stems_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"Loaded seen_stems.txt: {len(seen_stems)} stems")

    for img_zip_name, ann_zip_name in ZIP_PAIRS:
        round_name = ann_zip_name.replace("_annotations.zip", "")
        img_zip    = DATA_DIR / img_zip_name
        ann_zip    = DATA_DIR / ann_zip_name

        missing = [n for n, p in ((img_zip_name, img_zip), (ann_zip_name, ann_zip))
                   if not p.exists()]
        if missing:
            print(f"\n[SKIP] {round_name}: zip(s) not found: {missing}")
            rows.append({"round": round_name, "missing": True})
            continue

        print(f"\n=== {round_name} ===")

        # ── load annotation labels ────────────────────────────────────────
        ann: dict = {}
        with zipfile.ZipFile(ann_zip) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fname = Path(info.filename).name
                if fname in SKIP_NAMES or not fname.endswith(".txt"):
                    continue
                stem = decode_stem(fname)
                if stem in ann:
                    print(f"  NOTE: duplicate ann stem in zip, keeping first: {stem}")
                    continue
                ann[stem] = zf.read(info.filename)

        print(f"  Annotation zip: {len(ann)} label files decoded")

        # ── load image index from image zip ───────────────────────────────
        # stem → internal zip path  (first occurrence wins within same zip)
        img_members: dict = {}
        with zipfile.ZipFile(img_zip) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                p = Path(info.filename)
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                if p.stem not in img_members:
                    img_members[p.stem] = info.filename

        n_imgs_in_zip = len(img_members)
        print(f"  Image zip: {n_imgs_in_zip} images found (recursive)")

        # ── seen_stems diagnostic (annotation_round2_bulk only) ─────────
        if img_zip_name == "annotation_round2_bulk.zip":
            bulk_stems     = set(img_members.keys())
            in_seen        = bulk_stems & seen_stems
            not_in_seen    = bulk_stems - seen_stems
            ann_not_in_seen = set(ann.keys()) - seen_stems
            print()
            print("  [DIAGNOSTIC] annotation_round2_bulk vs seen_stems.txt")
            print(f"    seen_stems.txt total:                    {len(seen_stems)}")
            print(f"    bulk image stems in seen_stems:          {len(in_seen)}")
            print(f"    bulk image stems NOT in seen_stems:      {len(not_in_seen)}  ← will be skipped")
            print(f"    annotated stems NOT in seen_stems:       {len(ann_not_in_seen)}"
                  + ("  *** seen_stems.txt may be incomplete ***" if ann_not_in_seen else "  (OK)"))
            print()

        # ── write annotated frames ────────────────────────────────────────
        use_seen_filter = img_zip_name == "annotation_round2_bulk.zip"
        n_ann = n_bg = n_dup = n_no_img = n_unreviewed = 0

        with zipfile.ZipFile(img_zip) as zf:
            for stem, content in ann.items():
                if stem in seen:
                    n_dup += 1
                    continue
                if stem not in img_members:
                    print(f"  WARNING: label has no matching image: {stem}")
                    n_no_img += 1
                    continue
                (labels_dir / f"{stem}.txt").write_bytes(content)
                ext = normalise_ext(img_members[stem])
                (images_dir / f"{stem}{ext}").write_bytes(zf.read(img_members[stem]))
                seen.add(stem)
                n_ann += 1

            # ── write background frames (images with no label) ────────────
            for stem, member in img_members.items():
                if stem in ann:
                    continue   # already handled above
                if stem in seen:
                    n_dup += 1
                    continue
                if use_seen_filter and stem not in seen_stems:
                    n_unreviewed += 1
                    continue   # never imported into Label Studio — not a verified background
                (labels_dir / f"{stem}.txt").write_bytes(b"")   # intentionally empty
                ext = normalise_ext(member)
                (images_dir / f"{stem}{ext}").write_bytes(zf.read(member))
                seen.add(stem)
                n_bg += 1

        rows.append({
            "round":        round_name,
            "ann_in_zip":   len(ann),
            "img_in_zip":   n_imgs_in_zip,
            "written_ann":  n_ann,
            "written_bg":   n_bg,
            "skipped_dup":  n_dup,
            "no_img":       n_no_img,
            "unreviewed":   n_unreviewed,
            "missing":      False,
        })
        unrev_note = f", {n_unreviewed} unreviewed (skipped)" if n_unreviewed else ""
        print(f"  Written: {n_ann} annotated, {n_bg} background  |  "
              f"skipped: {n_dup} dup, {n_no_img} label-no-image{unrev_note}")

    # ── validation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("VALIDATION")
    print("=" * 68)

    all_labels = sorted(labels_dir.glob("*.txt"))
    all_images = sorted(images_dir.iterdir())

    lbl_stems  = {p.stem for p in all_labels}
    img_stems  = {p.stem for p in all_images}

    n_empty    = sum(1 for p in all_labels if p.stat().st_size == 0)
    n_nonempty = len(all_labels) - n_empty

    print(f"Total labels:  {len(all_labels):>6}  "
          f"(annotated: {n_nonempty}, background: {n_empty})")
    print(f"Total images:  {len(all_images):>6}")

    errors = []

    orphan_lbl = lbl_stems - img_stems
    orphan_img = img_stems - lbl_stems
    if orphan_lbl:
        sample = sorted(orphan_lbl)[:5]
        errors.append(
            f"LABELS WITHOUT MATCHING IMAGE ({len(orphan_lbl)}): "
            + ", ".join(sample) + ("..." if len(orphan_lbl) > 5 else "")
        )
    if orphan_img:
        sample = sorted(orphan_img)[:5]
        errors.append(
            f"IMAGES WITHOUT MATCHING LABEL ({len(orphan_img)}): "
            + ", ".join(sample) + ("..." if len(orphan_img) > 5 else "")
        )

    # Class index range check — must be integers in [0, 21]
    bad_lines = []
    for p in all_labels:
        if p.stat().st_size == 0:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"CANNOT READ {p.name}: {exc}")
            continue
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            m = CLASS_RE.match(line)
            if not m:
                bad_lines.append((p.name, lineno, "<no token>"))
                continue
            token = m.group(1)
            try:
                cls = int(token)
            except ValueError:
                bad_lines.append((p.name, lineno, f"non-integer '{token}'"))
                continue
            if not (0 <= cls <= 21):
                bad_lines.append((p.name, lineno, f"cls={cls} out of [0,21]"))

    if bad_lines:
        errors.append(
            f"OUT-OF-RANGE / CORRUPT CLASS INDICES ({len(bad_lines)} lines):"
        )
        for fname, lineno, msg in bad_lines[:30]:
            errors.append(f"    {fname}  line {lineno}: {msg}")
        if len(bad_lines) > 30:
            errors.append(f"    ... and {len(bad_lines)-30} more")

    # ── per-round summary table ───────────────────────────────────────────────
    col = "{:<38} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>8}"
    hdr = col.format("Round", "Ann/zip", "Img/zip", "WrAnn", "WrBG", "Dup", "NoImg", "Unrev")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("missing"):
            print(f"{r['round']:<38}  MISSING ZIPS")
            continue
        print(col.format(
            r["round"], r["ann_in_zip"], r["img_in_zip"],
            r["written_ann"], r["written_bg"], r["skipped_dup"],
            r["no_img"], r["unreviewed"],
        ))

    # ── result ───────────────────────────────────────────────────────────────
    print()
    if errors:
        print("ERRORS — DO NOT PROCEED UNTIL RESOLVED:")
        for e in errors:
            print(f"  {e}")
        return 1

    print("All validation checks passed.")
    print(f"\nOutput written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
