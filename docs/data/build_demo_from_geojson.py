"""
Build the GitHub Pages demo data from a real pipeline output GeoJSON.

Unlike generate_mock_data.py (synthetic), this converts an actual
sign_mapper/pipeline.py run (ideally with --inventory-dir) into the four files
the docs/ site loads. Regenerate whenever a new inference run is available.

Usage:
    python build_demo_from_geojson.py <detections.geojson> [video_name]

Writes (in this directory):
    demo_detections.geojson   — detections FeatureCollection (pretty)
    demo_detections.js        — embedded copy for file:// fallback
    demo_discrepancies.json   — inventory-vs-detection discrepancy report
    demo_discrepancies.js     — embedded copy for file:// fallback
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent


def write_embed(filename, var, obj, comment):
    body = f"// {comment}\n{var} = {json.dumps(obj, separators=(',', ':'))};\n"
    (HERE / filename).write_text(body, encoding='utf-8')


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    fc = json.loads(src.read_text(encoding='utf-8'))
    feats = fc.get('features', [])
    video = sys.argv[2] if len(sys.argv) > 2 else next(
        (f['properties'].get('video_source') for f in feats
         if f['properties'].get('video_source')), None)

    # 1. Detections GeoJSON (+ embed)
    (HERE / 'demo_detections.geojson').write_text(json.dumps(fc, indent=2), encoding='utf-8')
    write_embed('demo_detections.js', 'window.__DEMO_GEOJSON__', fc,
                'Auto-generated embedded copy of demo_detections.geojson so the '
                'map works under file:// (fetch is blocked). Regenerate on change.')

    # 2. Discrepancy report (+ embed)
    discrepancies = []
    n = 1
    for f in feats:
        p = f['properties']
        dt = p.get('discrepancy_type')
        if not dt:
            continue
        lon, lat = f['geometry']['coordinates'][:2]
        item = {
            'id': f'disc_{n:03d}',
            'type': dt,
            'cluster_id': f"cluster_{p.get('cluster_id')}",
            'sign_class': p.get('sign_class'),
            'confidence': p.get('confidence'),
            'lat': lat,
            'lon': lon,
            'sighting_count': p.get('sighting_count'),
            'review_reason': p.get('review_reason'),
            'status': 'pending',
            'reviewer_note': None,
        }
        if dt == 'class_mismatch':
            inv = p.get('inventory_class')
            if not inv:  # older runs stored it only in the review_reason text
                m = re.search(r'inventory ([A-Za-z]+)\)', p.get('review_reason') or '')
                inv = m.group(1) if m else None
            item['inventory_class'] = inv
        discrepancies.append(item)
        n += 1

    report = {
        'video_source': video,
        'processed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_detections': sum((f['properties'].get('sighting_count') or 0) for f in feats),
        'total_clusters': len(feats),
        'needs_review': sum(1 for f in feats if f['properties'].get('needs_review')),
        'discrepancies': discrepancies,
    }
    (HERE / 'demo_discrepancies.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_embed('demo_discrepancies.js', 'window.__DEMO_DISCREPANCIES__', report,
                'Auto-generated embedded copy of demo_discrepancies.json so the QA '
                'demo button works under file:// (fetch is blocked). Regenerate on change.')

    print(f"Detections : {len(feats)} features -> demo_detections.geojson/.js")
    print(f"Discrepancy: {len(discrepancies)} items -> demo_discrepancies.json/.js")
    print(f"Video      : {video}")


if __name__ == '__main__':
    main()
