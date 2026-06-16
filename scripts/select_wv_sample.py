"""
Reads data/wvu_sample_manifest.csv, prints a summary, selects a stratified
28-video sample, writes data/wv_download_list.csv and
scripts/download_wv_sample.sh.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

MANIFEST = "data/wvu_sample_manifest.csv"
OUT_CSV = "configs/wv_download_list.csv"
OUT_SH = "scripts/download_wv_sample.sh"

QUOTA = {
    "Interstate": 3,
    "US Route": 4,
    "WV Route": 4,
    "County Route": 10,
    "Municipal Non-State": 4,
    "Federal Aid Non-State": 3,
}


def filename_from_url(url):
    return urlparse(url).path.split("/")[-1]


def get_suffix(url):
    fname = filename_from_url(url)
    if "_FH." in fname:
        return "FH"
    if "_FL." in fname:
        return "FL"
    return ""


def main():
    rows = list(csv.DictReader(open(MANIFEST)))

    # ── Summary ──────────────────────────────────────────────────────────────
    label_counts = defaultdict(int)
    county_counts = defaultdict(int)
    fl = fh = 0
    for r in rows:
        label_counts[r["sign_system_label"]] += 1
        county_counts[r["primary_county"]] += 1
        s = get_suffix(r["video_link"])
        if s == "FL":
            fl += 1
        elif s == "FH":
            fh += 1

    print("=== Manifest Summary ===")
    print("\nCount per sign_system_label:")
    for label in sorted(label_counts):
        print(f"  {label:<30} {label_counts[label]}")

    print("\nCount per primary_county (top 20):")
    for county, n in sorted(county_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {county:<25} {n}")

    print(f"\nTotal unique counties: {len(county_counts)}")
    print(f"Filename suffix patterns: _FL={fl}, _FH={fh}")

    # ── Stratified selection ──────────────────────────────────────────────────
    # Group by (label, county); pick one best candidate per county (FH preferred,
    # deterministic by video_id within county); shuffle with seed=42 then take quota.
    by_label_county = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label_county[r["sign_system_label"]][r["primary_county"]].append(r)

    rng = random.Random(42)
    selected = []

    for label, quota in QUOTA.items():
        county_groups = by_label_county[label]
        candidates = []
        for county in sorted(county_groups):
            county_rows = sorted(county_groups[county], key=lambda r: r["video_id"])
            fh_rows = [r for r in county_rows if get_suffix(r["video_link"]) == "FH"]
            candidates.append(fh_rows[0] if fh_rows else county_rows[0])

        rng.shuffle(candidates)
        picked = candidates[:quota]

        # Fallback: fill remaining slots if fewer unique counties than quota
        if len(picked) < quota:
            picked_ids = {r["video_id"] for r in picked}
            remaining = sorted(
                [r for r in rows if r["sign_system_label"] == label
                 and r["video_id"] not in picked_ids],
                key=lambda r: r["video_id"],
            )
            rng.shuffle(remaining)
            picked += remaining[: quota - len(picked)]

        selected.extend(picked)

    # ── Print selection table ─────────────────────────────────────────────────
    print("\n=== Selected 28 Videos ===")
    print(f"{'video_id':<10} {'label':<25} {'county':<22} {'sfx':<5} filename")
    print("-" * 105)
    for label in QUOTA:
        for r in [r for r in selected if r["sign_system_label"] == label]:
            fname = filename_from_url(r["video_link"])
            sfx = get_suffix(r["video_link"])
            print(f"{r['video_id']:<10} {label:<25} {r['primary_county']:<22} {sfx:<5} {fname}")

    # ── Write download CSV ────────────────────────────────────────────────────
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "video_link", "sign_system_label",
                        "primary_county", "filename"],
        )
        writer.writeheader()
        for r in selected:
            writer.writerow({**r, "filename": filename_from_url(r["video_link"])})
    print(f"\nWrote {OUT_CSV} ({len(selected)} rows)")

    # ── Write download shell script ───────────────────────────────────────────
    sh_lines = [
        "#!/bin/bash",
        "# Downloads selected WV videos to data/raw/wvdoh/",
        "# Skips files already downloaded",
        "# Shows progress per file",
        "# Usage: bash scripts/download_wv_sample.sh",
        "",
        "mkdir -p data/raw/wvdoh",
        "",
    ]
    for r in selected:
        fname = filename_from_url(r["video_link"])
        url = r["video_link"]
        sh_lines += [
            f'if [ ! -f "data/raw/wvdoh/{fname}" ]; then',
            f'  wget -c --show-progress -O "data/raw/wvdoh/{fname}" "{url}"',
            "else",
            f'  echo "SKIP: {fname} already exists"',
            "fi",
            "",
        ]
    with open(OUT_SH, "w", newline="\n") as f:
        f.write("\n".join(sh_lines))
    print(f"Wrote {OUT_SH}")


if __name__ == "__main__":
    main()
