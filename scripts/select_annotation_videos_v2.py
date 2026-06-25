"""
Select 60 new annotation videos from the WV manifest for round 3.

Budget:
  Municipal Non-State  20
  US Route             20
  WV Route             12
  County Route          8
  (Interstate and Federal Aid Non-State skipped)

Selection rules:
  - Exclude any video_id already in the three exclusion CSVs
  - No county > 3 times across all 60 selected videos
  - Maximize counties not already in the existing annotation set
  - For MNS and US Route: priority counties sorted first (weight 2x)

Output: configs/wv_annotation_round3.csv
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

MANIFEST = "data/wvu_sample_manifest.csv"
EXCLUSION_CSVS = [
    "data/wv_download_list.csv",
    "configs/wv_supplemental_download.csv",
    "configs/wv_supplemental_download_v2.csv",
]
OUTPUT = "configs/wv_annotation_round3.csv"

BUDGET = {
    "Municipal Non-State": 20,
    "US Route": 20,
    "WV Route": 12,
    "County Route": 8,
}
PRIORITY_TYPES = {"Municipal Non-State", "US Route"}
PRIORITY_COUNTIES = {
    "Kanawha", "Monongalia", "Cabell", "Berkeley",
    "Wood", "Harrison", "Marion", "Raleigh",
}
MAX_PER_COUNTY = 3

random.seed(42)


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def filename_from_url(url):
    return url.rsplit("/", 1)[-1]


def main():
    # Collect excluded video_ids and counties from all prior annotation rounds
    excluded_ids = set()
    existing_counties = set()
    for csv_path in EXCLUSION_CSVS:
        for row in load_csv(csv_path):
            excluded_ids.add(row["video_id"])
            existing_counties.add(row["primary_county"])

    # Load manifest, drop already-annotated videos
    manifest = load_csv(MANIFEST)
    eligible = [r for r in manifest if r["video_id"] not in excluded_ids]

    # Group by road type
    by_type = defaultdict(list)
    for row in eligible:
        by_type[row["sign_system_label"]].append(row)

    county_counts = defaultdict(int)  # global cap across all 60
    selected = []

    for road_type, budget in BUDGET.items():
        pool = list(by_type[road_type])
        random.shuffle(pool)  # randomize within priority tiers

        if road_type in PRIORITY_TYPES:
            # Lower sort key = higher priority: priority county first, new county second
            pool.sort(key=lambda r: (
                0 if r["primary_county"] in PRIORITY_COUNTIES else 1,
                0 if r["primary_county"] not in existing_counties else 1,
            ))
        else:
            pool.sort(key=lambda r: 0 if r["primary_county"] not in existing_counties else 1)

        chosen = []
        for row in pool:
            if len(chosen) >= budget:
                break
            if county_counts[row["primary_county"]] < MAX_PER_COUNTY:
                chosen.append(row)
                county_counts[row["primary_county"]] += 1

        selected.extend(chosen)

        county_dist = defaultdict(int)
        for r in chosen:
            county_dist[r["primary_county"]] += 1

        print(f"\n{road_type}: {len(chosen)}/{budget} selected")
        for county, count in sorted(county_dist.items()):
            tag = " [priority]" if road_type in PRIORITY_TYPES and county in PRIORITY_COUNTIES else ""
            print(f"  {county}: {count}{tag}")
        if len(chosen) < budget:
            print(f"  WARNING: only {len(chosen)} eligible videos, needed {budget}")

    # Report priority counties with no eligible MNS/US Route candidates
    priority_pool_counties = set()
    for road_type in PRIORITY_TYPES:
        for row in by_type[road_type]:
            priority_pool_counties.add(row["primary_county"])
    missed = PRIORITY_COUNTIES - priority_pool_counties
    if missed:
        print(f"\nHigh-priority counties with no eligible videos in MNS/US Route: {sorted(missed)}")

    # Overall county distribution
    print(f"\n--- County distribution across all {len(selected)} selected videos ---")
    all_county_dist = defaultdict(int)
    for r in selected:
        all_county_dist[r["primary_county"]] += 1
    for county, count in sorted(all_county_dist.items()):
        print(f"  {county}: {count}")

    # Write output
    out_path = Path(OUTPUT)
    fieldnames = ["video_id", "video_link", "sign_system_label", "primary_county", "filename"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "video_id": row["video_id"],
                "video_link": row["video_link"],
                "sign_system_label": row["sign_system_label"],
                "primary_county": row["primary_county"],
                "filename": filename_from_url(row["video_link"]),
            })

    print(f"\nWrote {len(selected)} rows to {out_path}")


if __name__ == "__main__":
    main()
