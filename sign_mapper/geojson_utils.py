"""
GeoJSON I/O and inventory comparison.
Coordinate order is [longitude, latitude] per GeoJSON spec (RFC 7946).
"""
import json
import math
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_geojson(path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_geojson(clusters: list, path, video_source=None):
    """Write clusters to a GeoJSON FeatureCollection file."""
    features = []
    for c in clusters:
        if c.get('lat_median') is None or c.get('lon_median') is None:
            continue
        props = {k: v for k, v in c.items() if k not in ('lat_median', 'lon_median', 'cluster_id')}
        if video_source and 'video_source' not in props:
            props['video_source'] = video_source
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [c['lon_median'], c['lat_median']],  # [lon, lat] — GeoJSON spec
            },
            'properties': props,
        })

    fc = {'type': 'FeatureCollection', 'features': features}
    Path(path).write_text(json.dumps(fc, indent=2), encoding='utf-8')
    return fc


def compare_with_inventory(clusters: list, inventory: dict, match_radius_m=25.0) -> list:
    """
    Annotates clusters with inventory comparison fields (mutates in place).
    Appends synthetic 'in_inventory_not_detected' clusters for unmatched inventory signs.
    Returns the (extended) cluster list.
    """
    inv_features = inventory.get('features', [])
    inv_matched = set()

    for cluster in clusters:
        lat, lon = cluster['lat_median'], cluster['lon_median']
        best_dist, best_idx = float('inf'), None

        for i, feat in enumerate(inv_features):
            try:
                ilon, ilat = feat['geometry']['coordinates'][:2]
            except (KeyError, TypeError, IndexError):
                continue
            d = haversine_m(lat, lon, ilat, ilon)
            if d < best_dist:
                best_dist, best_idx = d, i

        if best_idx is not None and best_dist <= match_radius_m:
            inv_feat = inv_features[best_idx]
            inv_matched.add(best_idx)
            props = inv_feat.get('properties') or {}
            cluster['inventory_match'] = (
                inv_feat.get('id') or props.get('OBJECTID') or props.get('id')
            )
            cluster['inventory_distance_m'] = round(best_dist, 2)

            inv_class = (props.get('sign_class') or props.get('SignClass')
                         or props.get('Category') or props.get('TYPE'))
            if inv_class and inv_class != cluster['sign_class']:
                cluster['discrepancy_type'] = 'class_mismatch'
                cluster['needs_review'] = True
                extra = (f"Class mismatch (detected {cluster['sign_class']}, "
                         f"inventory {inv_class})")
                cluster['review_reason'] = (
                    '; '.join(filter(None, [cluster.get('review_reason'), extra]))
                )
        else:
            cluster['discrepancy_type'] = 'detected_not_in_inventory'
            cluster['needs_review'] = True
            cluster['review_reason'] = '; '.join(filter(None, [
                cluster.get('review_reason'), 'Not in WVDOH inventory'
            ]))

    # Synthetic clusters for inventory signs with no nearby detection
    next_id = max((c['cluster_id'] for c in clusters), default=-1) + 1
    for i, feat in enumerate(inv_features):
        if i in inv_matched:
            continue
        try:
            ilon, ilat = feat['geometry']['coordinates'][:2]
        except (KeyError, TypeError, IndexError):
            continue
        props = feat.get('properties') or {}
        clusters.append({
            'cluster_id': next_id,
            'sign_class': (props.get('sign_class') or props.get('SignClass')
                           or props.get('Category') or 'Unknown'),
            'confidence': 0.0,
            'sighting_count': 0,
            'needs_review': True,
            'review_reason': 'Inventory sign not detected within 25m',
            'class_vote_pct': 0.0,
            'video_source': None,
            'first_seen_ts': None,
            'best_frame_ts': None,
            'lat_median': ilat,
            'lon_median': ilon,
            'lat_std': 0.0,
            'lon_std': 0.0,
            'inventory_match': feat.get('id') or props.get('OBJECTID'),
            'inventory_distance_m': None,
            'discrepancy_type': 'in_inventory_not_detected',
        })
        next_id += 1

    return clusters
