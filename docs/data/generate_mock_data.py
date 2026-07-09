"""
Generate realistic mock sign detection data along US-19 near Summersville, WV.
Produces demo_detections.geojson and demo_discrepancies.json in this directory.
Run: python generate_mock_data.py
"""
import json
import math
import random
from pathlib import Path

random.seed(42)

CLASSES = {
    'Warnings':       0.25,
    'Curves':         0.20,
    'StopSigns':      0.15,
    'SpeedLimits':    0.15,
    'Informational':  0.10,
    'HighwaySymbols': 0.07,
    'Regulatory':     0.04,
    'TrafficLights':  0.02,
    'MileMarkers':    0.02,
}

# Approximate waypoints along US-19, south-to-north through Summersville, WV
ROUTE = [
    (38.160, -80.890),
    (38.185, -80.882),
    (38.210, -80.875),
    (38.235, -80.868),
    (38.258, -80.860),
    (38.278, -80.855),
    (38.298, -80.848),
    (38.318, -80.838),
    (38.340, -80.825),
    (38.365, -80.812),
    (38.390, -80.800),
]


def _interpolate(waypoints, t):
    segs = []
    total = 0.0
    for i in range(len(waypoints) - 1):
        la1, lo1 = waypoints[i]
        la2, lo2 = waypoints[i + 1]
        d = math.hypot(la2 - la1, lo2 - lo1)
        segs.append(d)
        total += d
    target = t * total
    acc = 0.0
    for i, seg in enumerate(segs):
        if acc + seg >= target or i == len(segs) - 1:
            f = (target - acc) / seg if seg else 0.0
            la1, lo1 = waypoints[i]
            la2, lo2 = waypoints[i + 1]
            return la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
        acc += seg
    return waypoints[-1]


def _jitter(v, scale):
    return v + random.gauss(0, scale)


def main():
    out_dir = Path(__file__).parent
    n = 150

    classes = random.choices(list(CLASSES), weights=list(CLASSES.values()), k=n)

    # Indices that will be flagged as detected-not-in-inventory
    not_in_inv_idx = set(random.sample(range(n), 7))
    # Indices that will get a class_mismatch with inventory
    mismatch_idx = set(random.sample([i for i in range(n) if i not in not_in_inv_idx], 8))

    features = []
    discrepancies = []
    disc_num = 1

    for i, cls in enumerate(classes):
        t = i / (n - 1)
        lat_c, lon_c = _interpolate(ROUTE, t)
        lat = round(_jitter(lat_c, 0.00018), 6)
        lon = round(_jitter(lon_c, 0.00025), 6)

        conf = round(random.uniform(0.30, 0.52), 3) if random.random() < 0.14 \
               else round(random.uniform(0.68, 0.97), 3)
        sightings = random.randint(1, 2) if random.random() < 0.10 \
                    else random.randint(3, 12)

        reasons = []
        if sightings <= 1:
            reasons.append("Single sighting")
        if conf < 0.55:
            reasons.append(f"Low confidence ({conf:.2f})")
        if random.random() < 0.05:
            reasons.append("Split class vote")

        disc_type = None
        inv_match = None
        inv_dist = None
        inv_cls = None

        if i in not_in_inv_idx:
            disc_type = "detected_not_in_inventory"
            reasons.append("Not in WVDOH inventory")
            discrepancies.append({
                "id": f"disc_{disc_num:03d}",
                "type": "detected_not_in_inventory",
                "cluster_id": f"cluster_{i:03d}",
                "sign_class": cls,
                "confidence": conf,
                "lat": lat,
                "lon": lon,
                "sighting_count": sightings,
                "review_reason": "Not in WVDOH inventory",
                "status": random.choice(["pending", "pending", "pending", "confirmed", "dismissed"]),
                "reviewer_note": None,
            })
            disc_num += 1
        elif i in mismatch_idx:
            inv_cls_pool = [c for c in CLASSES if c != cls]
            inv_cls = random.choice(inv_cls_pool)
            disc_type = "class_mismatch"
            reasons.append(f"Class mismatch (detected {cls}, inventory {inv_cls})")
            inv_match = f"INV-{1000 + i}"
            inv_dist = round(random.uniform(1.5, 22.0), 1)
            discrepancies.append({
                "id": f"disc_{disc_num:03d}",
                "type": "class_mismatch",
                "cluster_id": f"cluster_{i:03d}",
                "sign_class": cls,
                "inventory_class": inv_cls,
                "confidence": conf,
                "lat": lat,
                "lon": lon,
                "sighting_count": sightings,
                "review_reason": f"Class mismatch (detected {cls}, inventory {inv_cls})",
                "status": random.choice(["pending", "pending", "needs_annotation", "confirmed"]),
                "reviewer_note": None,
            })
            disc_num += 1
        else:
            inv_match = f"INV-{1000 + i}"
            inv_dist = round(random.uniform(1.5, 22.0), 1)

        needs_review = bool(reasons) or disc_type is not None
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "cluster_id": i,
                "sign_class": cls,
                "confidence": conf,
                "sighting_count": sightings,
                "needs_review": needs_review,
                "review_reason": "; ".join(reasons) if reasons else None,
                "class_vote_pct": round(random.uniform(0.65, 1.0), 3),
                "video_source": "demo_us19_segment.mp4",
                "first_seen_ts": 1000 * (i * 12 + random.randint(0, 5)),
                "best_frame_ts": 1000 * (i * 12 + random.randint(3, 8)),
                "lat_std": round(random.uniform(5e-6, 3e-5), 8),
                "lon_std": round(random.uniform(5e-6, 3e-5), 8),
                "inventory_match": inv_match,
                "inventory_distance_m": inv_dist,
                "discrepancy_type": disc_type,
            },
        })

    # Synthetic in-inventory-not-detected features (7 more)
    for j in range(7):
        t = random.uniform(0.04, 0.96)
        lat_c, lon_c = _interpolate(ROUTE, t)
        lat = round(_jitter(lat_c, 0.00022), 6)
        lon = round(_jitter(lon_c, 0.00030), 6)
        inv_cls = random.choice(list(CLASSES))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "cluster_id": n + j,
                "sign_class": inv_cls,
                "confidence": 0.0,
                "sighting_count": 0,
                "needs_review": True,
                "review_reason": "Inventory sign not detected within 25m",
                "class_vote_pct": 0.0,
                "video_source": None,
                "first_seen_ts": None,
                "best_frame_ts": None,
                "lat_std": 0.0,
                "lon_std": 0.0,
                "inventory_match": f"INV-{2000 + j}",
                "inventory_distance_m": None,
                "discrepancy_type": "in_inventory_not_detected",
            },
        })
        discrepancies.append({
            "id": f"disc_{disc_num:03d}",
            "type": "in_inventory_not_detected",
            "cluster_id": f"cluster_{n + j:03d}",
            "sign_class": inv_cls,
            "confidence": 0.0,
            "lat": lat,
            "lon": lon,
            "sighting_count": 0,
            "review_reason": "Inventory sign not detected within 25m",
            "status": random.choice(["pending", "pending", "dismissed", "needs_annotation"]),
            "reviewer_note": None,
        })
        disc_num += 1

    fc = {"type": "FeatureCollection", "features": features}
    geojson_path = out_dir / "demo_detections.geojson"
    geojson_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")

    n_review = sum(1 for f in features if f["properties"]["needs_review"])
    report = {
        "video_source": "demo_us19_segment.mp4",
        "processed_at": "2026-07-01T14:23:00Z",
        "total_detections": 178,
        "total_clusters": len(features),
        "needs_review": n_review,
        "discrepancies": discrepancies,
    }
    disc_path = out_dir / "demo_discrepancies.json"
    disc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(features)} features to {geojson_path.name}")
    print(f"Wrote {len(discrepancies)} discrepancies to {disc_path.name}")
    print(f"Needs review: {n_review} of {len(features)}")


if __name__ == "__main__":
    main()
