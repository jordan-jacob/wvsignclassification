"""
Select supplemental videos to address thin rare classes after deduplication.

Scores each undownloaded video by a weighted combination of:
  - Per-class road-type affinity (e.g., Municipal Non-State for pedestrianCrossing)
  - Per-class county-type affinity (rural/forested for deerCrossing, urban for pedestrianCrossing)
  - Per-class instance-deficit weight (fewer instances → higher urgency)

Enforces county diversity (cap per county) and excludes already-downloaded videos.

Usage:
    python scripts/select_supplemental_videos.py
    python scripts/select_supplemental_videos.py --n 20 --seed 42
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Class urgency weights (based on unique instance counts after dedup) ────────
# Lower instance count → higher urgency multiplier.
INSTANCE_DEFICIT = {
    "deerCrossing":       4.0,   # 1 unique instance
    "pedestrianCrossing": 3.5,   # 3 unique instances
    "railroadCrossing":   3.0,   # 4 unique instances
    "curve":              1.5,   # 12 unique instances (less urgent)
}

# ── Road-type affinity per class ────────────────────────────────────────────────
ROAD_TYPE_AFFINITY: dict[str, dict[str, float]] = {
    "deerCrossing": {
        "County Route":          3.0,
        "WV Route":              2.5,
        "Federal Aid Non-State": 2.0,
        "US Route":              1.5,
        "Municipal Non-State":   0.3,
        "Interstate":            0.4,
    },
    "pedestrianCrossing": {
        "Municipal Non-State":   4.0,
        "US Route":              1.5,
        "Federal Aid Non-State": 1.0,
        "WV Route":              1.0,
        "County Route":          0.5,
        "Interstate":            0.2,
    },
    "railroadCrossing": {
        # Railroad crossings appear across all road types in WV;
        # slightly prefer non-Interstate routes where at-grade crossings exist.
        "County Route":          2.0,
        "WV Route":              2.0,
        "US Route":              2.0,
        "Federal Aid Non-State": 1.5,
        "Municipal Non-State":   1.5,
        "Interstate":            0.4,
    },
    "curve": {
        "County Route":          3.0,
        "WV Route":              2.5,
        "Federal Aid Non-State": 2.0,
        "US Route":              1.5,
        "Municipal Non-State":   0.5,
        "Interstate":            0.3,
    },
}

# ── County-type affinity per class ─────────────────────────────────────────────
# WV's most rural/forested counties (Appalachian highlands, low population density).
# These have abundant deer habitat and winding mountain roads.
RURAL_FORESTED_COUNTIES = {
    "Pocahontas", "Webster", "Nicholas", "Randolph", "Tucker", "Pendleton",
    "Grant", "Hardy", "Hampshire", "Morgan", "Mineral", "Upshur",
    "Braxton", "Clay", "Gilmer", "Calhoun", "Wirt", "Roane", "Jackson",
    "McDowell", "Wyoming", "Logan", "Lincoln", "Mingo",
}

# Counties with larger towns/cities (more likely to have pedestrian crossings).
URBAN_COUNTIES = {
    "Kanawha",    # Charleston
    "Cabell",     # Huntington
    "Ohio",       # Wheeling
    "Monongalia", # Morgantown
    "Wood",       # Parkersburg
    "Berkeley",   # Martinsburg
    "Jefferson",  # Charles Town
    "Marion",     # Fairmont
    "Harrison",   # Clarksburg
    "Raleigh",    # Beckley
    "Mercer",     # Princeton/Bluefield
}

COUNTY_AFFINITY: dict[str, dict[str, float]] = {
    "deerCrossing":       {"rural": 1.6, "urban": 0.6,  "other": 1.0},
    "pedestrianCrossing": {"rural": 0.6, "urban": 1.6,  "other": 1.0},
    "railroadCrossing":   {"rural": 1.1, "urban": 1.1,  "other": 1.0},
    "curve":              {"rural": 1.5, "urban": 0.5,  "other": 1.0},
}


def county_type(county: str) -> str:
    if county in RURAL_FORESTED_COUNTIES:
        return "rural"
    if county in URBAN_COUNTIES:
        return "urban"
    return "other"


def class_score(row: dict, cls: str) -> float:
    """Score a candidate video for a specific target class."""
    rt_w = ROAD_TYPE_AFFINITY[cls].get(row["sign_system_label"], 1.0)
    ct_w = COUNTY_AFFINITY[cls][county_type(row["primary_county"])]
    return rt_w * ct_w


# Per-class video budgets. Higher urgency (fewer existing instances) = more videos.
# Total should equal --n; these are starting proportions that scale with --n.
CLASS_BUDGET_FRACTIONS: dict[str, float] = {
    "deerCrossing":       0.28,   # 1 instance — most urgent
    "pedestrianCrossing": 0.28,   # 3 instances
    "railroadCrossing":   0.28,   # 4 instances
    "curve":              0.16,   # 12 instances — least urgent
}


def select_for_class(
    cls: str,
    candidates: list[dict],
    budget: int,
    county_counts: Counter,
    max_per_county: int,
    already_selected: set[str],
    rng: random.Random,
) -> list[dict]:
    """Greedily pick `budget` videos for `cls`, respecting county cap and dedup."""
    ranked = sorted(candidates, key=lambda r: -class_score(r, cls))
    picked = []
    for r in ranked:
        if len(picked) >= budget:
            break
        vid_id = r["video_id"]
        county = r["primary_county"]
        if vid_id in already_selected:
            continue
        if county_counts[county] >= max_per_county:
            continue
        picked.append(r)
        county_counts[county] += 1
        already_selected.add(vid_id)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "wvu_sample_manifest.csv")
    ap.add_argument("--download-list", type=Path, default=ROOT / "configs" / "wv_download_list.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "wv_supplemental_download.csv")
    ap.add_argument("--n", type=int, default=18,
                    help="Number of supplemental videos to select")
    ap.add_argument("--max-per-county", type=int, default=2,
                    help="Max videos per county in the supplemental batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load manifest
    with open(args.manifest, newline="") as f:
        manifest = list(csv.DictReader(f))
    print(f"Manifest: {len(manifest)} videos")

    # Load already-downloaded video_ids
    with open(args.download_list, newline="") as f:
        existing_ids = {row["video_id"] for row in csv.DictReader(f)}
    print(f"Already downloaded: {len(existing_ids)} videos")

    candidates = [r for r in manifest if r["video_id"] not in existing_ids]
    print(f"Candidates: {len(candidates)} videos\n")

    for r in candidates:
        r["_ctype"] = county_type(r["primary_county"])

    # Compute per-class budgets (integer, sum to --n)
    raw = {cls: args.n * frac for cls, frac in CLASS_BUDGET_FRACTIONS.items()}
    budgets = {cls: max(1, round(v)) for cls, v in raw.items()}
    # Adjust to exactly args.n by modifying the largest class
    diff = args.n - sum(budgets.values())
    largest = max(budgets, key=lambda c: budgets[c])
    budgets[largest] += diff

    print("Per-class video budgets:")
    for cls, b in budgets.items():
        print(f"  {cls:<22}  {b} videos")
    print()

    rng = random.Random(args.seed)
    county_counts: Counter = Counter()
    already_selected: set[str] = set()
    selected_by_class: dict[str, list[dict]] = {}

    # Select in urgency order (most urgent first gets first pick of counties)
    for cls in sorted(budgets, key=lambda c: -INSTANCE_DEFICIT[c]):
        picked = select_for_class(
            cls, candidates, budgets[cls],
            county_counts, args.max_per_county, already_selected, rng,
        )
        selected_by_class[cls] = picked

    all_selected = [r for picks in selected_by_class.values() for r in picks]
    print(f"Selected {len(all_selected)} supplemental videos\n")

    # Print breakdown
    print("-- Per-class selection --")
    for cls, picks in selected_by_class.items():
        print(f"\n  {cls}  ({len(picks)} videos)")
        for r in picks:
            sc = class_score(r, cls)
            print(f"    {r['video_id']:>8}  {r['sign_system_label']:<25}"
                  f" {r['primary_county']:<18} {r['_ctype']:<7} score={sc:.2f}")

    rt_counts = Counter(r["sign_system_label"] for r in all_selected)
    ct_counts = Counter(r["_ctype"] for r in all_selected)
    county_sel = Counter(r["primary_county"] for r in all_selected)

    print("\n-- Road type breakdown --")
    for rt, n in rt_counts.most_common():
        print(f"  {rt:<28} {n}")

    print("\n-- County type breakdown --")
    for ct, n in ct_counts.most_common():
        print(f"  {ct:<28} {n}")

    print("\n-- Counties selected --")
    for county, n in county_sel.most_common():
        print(f"  {county:<28} {n}")

    print("\n-- Per-class avg affinity of selected batch --")
    for cls in ["deerCrossing", "pedestrianCrossing", "railroadCrossing", "curve"]:
        avg = (sum(class_score(r, cls) for r in all_selected) / len(all_selected)
               if all_selected else 0)
        print(f"  {cls:<22}  avg affinity: {avg:.2f}")

    # Write supplemental CSV (same format as wv_download_list.csv)
    fieldnames = ["video_id", "video_link", "sign_system_label", "primary_county", "filename"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_selected:
            writer.writerow({
                "video_id":          r["video_id"],
                "video_link":        r["video_link"],
                "sign_system_label": r["sign_system_label"],
                "primary_county":    r["primary_county"],
                "filename":          r["video_link"].split("/")[-1],
            })

    print(f"\nSupplemental list written to {args.out}  ({len(all_selected)} videos)")
    print(f"Append to download list:  Get-Content {args.out} | Select-Object -Skip 1 | Add-Content {args.download_list}")


if __name__ == "__main__":
    main()
