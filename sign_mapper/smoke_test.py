"""Quick smoke test for gps_parser and geojson_utils — no video or model needed."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gps_parser
import geojson_utils

# --- SRT parsing ---
SRT = """1
00:00:01,000 --> 00:00:01,033
GPS(38.123456, -81.234567, 245)

2
00:00:02,000 --> 00:00:02,033
GPS(38.123478, -81.234589, 245)

3
00:00:03,000 --> 00:00:03,033
GPS(38.123500, -81.234611, 245)
"""

with tempfile.NamedTemporaryFile(suffix='.srt', mode='w', delete=False, encoding='utf-8') as f:
    f.write(SRT)
    srt_path = f.name

fixes = gps_parser.parse_srt(Path(srt_path))
os.unlink(srt_path)
assert len(fixes) == 3, f"Expected 3 fixes, got {len(fixes)}"
print(f"SRT parse: {len(fixes)} fixes  first={fixes[0]}")

# Interpolation at 1500ms
gps = gps_parser.interpolate_gps(fixes, 1500)
assert gps is not None
assert abs(gps['lat'] - 38.123467) < 1e-5, f"lat interp wrong: {gps['lat']}"
print(f"Interp at 1.5s: lat={gps['lat']:.6f} lon={gps['lon']:.6f}")

# Out-of-range (>2s beyond last fix at 3000ms)
oob = gps_parser.interpolate_gps(fixes, 6000)
assert oob is None, f"Expected None for OOB, got {oob}"
print("Out-of-range returns None: OK")

# --- CSV parsing ---
CSV = "timestamp,latitude,longitude,altitude\n0,38.1,81.0,200\n1000,38.2,-81.1,201\n"
with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False, encoding='utf-8') as f:
    f.write(CSV)
    csv_path = f.name

fixes_csv = gps_parser.parse_csv(Path(csv_path))
os.unlink(csv_path)
assert len(fixes_csv) == 2
print(f"CSV parse: {len(fixes_csv)} fixes  first={fixes_csv[0]}")

# --- GeoJSON round-trip ---
clusters = [{
    'cluster_id': 0, 'sign_class': 'StopSigns', 'confidence': 0.85,
    'sighting_count': 5, 'needs_review': False, 'review_reason': None,
    'class_vote_pct': 1.0, 'video_source': 'test.mp4',
    'first_seen_ts': 1000, 'best_frame_ts': 1500,
    'lat_median': 38.123456, 'lon_median': -81.234567,
    'lat_std': 0.0, 'lon_std': 0.0,
    'inventory_match': None, 'inventory_distance_m': None, 'discrepancy_type': None,
}]
with tempfile.NamedTemporaryFile(suffix='.geojson', mode='w', delete=False, encoding='utf-8') as f:
    gj_path = f.name
geojson_utils.save_geojson(clusters, gj_path)
fc = json.loads(Path(gj_path).read_text())
os.unlink(gj_path)
feat = fc['features'][0]
coords = feat['geometry']['coordinates']
assert coords == [-81.234567, 38.123456], f"Wrong coord order: {coords}"
print(f"GeoJSON coord order [lon, lat]: {coords}  OK")

# --- Haversine sanity ---
d = geojson_utils.haversine_m(38.0, -81.0, 39.0, -81.0)
assert 110000 < d < 112000, f"Haversine wrong: {d}"
print(f"Haversine 1-deg lat: {d:.0f} m  OK")

# --- Inventory comparison ---
inv = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [-81.234567, 38.123456]},
            'properties': {'OBJECTID': 'INV-001', 'sign_class': 'StopSigns'},
        },
        {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [-81.9, 38.9]},
            'properties': {'OBJECTID': 'INV-002', 'sign_class': 'Warnings'},
        },
    ],
}
clusters2 = [
    {'cluster_id': 0, 'sign_class': 'StopSigns', 'confidence': 0.9,
     'sighting_count': 4, 'needs_review': False, 'review_reason': None,
     'class_vote_pct': 1.0, 'video_source': 'v.mp4', 'first_seen_ts': 0,
     'best_frame_ts': 0, 'lat_median': 38.123456, 'lon_median': -81.234567,
     'lat_std': 0.0, 'lon_std': 0.0,
     'inventory_match': None, 'inventory_distance_m': None, 'discrepancy_type': None},
]
result = geojson_utils.compare_with_inventory(clusters2, inv, match_radius_m=25.0)
matched = result[0]
assert matched['inventory_match'] == 'INV-001', f"Wrong match: {matched['inventory_match']}"
assert matched['discrepancy_type'] is None, "Should be matched, no discrepancy"
# INV-002 should appear as synthetic 'in_inventory_not_detected'
synthetic = [c for c in result if c.get('discrepancy_type') == 'in_inventory_not_detected']
assert len(synthetic) == 1
print(f"Inventory compare: match={matched['inventory_match']} dist={matched['inventory_distance_m']}m")
print(f"Synthetic unmatched: {synthetic[0]['sign_class']}  OK")

print()
print("All smoke tests passed.")
