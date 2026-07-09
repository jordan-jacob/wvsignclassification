"""
GeoJSON I/O and inventory comparison.
Coordinate order is [longitude, latitude] per GeoJSON spec (RFC 7946).

-----------------------------------------------------------------------------
WVDOH open-access sign inventory schema (Desktop/wvdoh_signs/, inspected Jul 2026)
-----------------------------------------------------------------------------
All 10 files share one schema.  Feature counts and the classifying fields:

  Curve_Signs .............. 17,265   MUTCDCAT WARNING;  MUTCDCODE W1-* (curves)
  Highway_Shields .......... 32,226   MUTCDCAT GUIDE-ROUTE SIGN; MUTCDCODE M*
  Informational_Signs ...... 79,883   MUTCDCODE OTR/OTI/D1*, GUIDE-DESTINATION
  Intersections ............  2,945   MUTCDCAT REGULATORY/WARNING; R3-9B, W2-*
  Mile_Markers .............  2,515   MUTCDCODE D10-* (reference location)
  Speed_Limit_Signs ........ 30,360   MUTCDCODE R2-* (speed) + W13-* (advisory)
  Stop_Signs ...............  8,743   MUTCDCODE R1-1 (+ W3-1 stop-ahead warnings)
  Traffic_Light_Signs ......    845   MUTCDCODE R10-*, W3-3
  other_regulatory_signs ... 44,300   MUTCDCAT REGULATORY; R*, OM-* markers
  yields ...................    534   MUTCDCODE R1-2 (YIELD)

  geometry: Point, coordinates [lon, lat]  (also duplicated in GPSLONGS/GPSLATS)

  First feature properties (Curve_Signs), for the full field list:
    {'OBJECTID', 'FKEY', 'ROUTE_ID', 'ELEMENT_ID', 'CO_CODE', 'ROUTE',
     'SUBROUTE', 'SYS', 'SUPP_CODE', 'SUPP_DESCR', 'DIRECTION', 'BEG_MP',
     'OFFSET', 'LOC', 'NO_OF_SIGN', 'MUTCDNAME', 'MUTCDCODE', 'MUTCDCAT',
     'TEXT_', 'CONDITION', 'COMMENT', 'GPSLONGS', 'GPSLATS', 'GPSELEVS',
     'FILENAME', 'DIST', 'COUNTY', 'SIGN_SYS', 'SUPP_NUM', 'CO_DESC',
     'RTE_NUM', 'VIDEO_STAR'}

Classification uses MUTCDCODE (authoritative sign id) with a MUTCDCAT fallback;
see normalize_wvdoh_class() below.  There is no dedicated general-Warning feature
class in the open-access set, hence INVENTORY_CLASSES_MISSING.
-----------------------------------------------------------------------------
"""
import json
import math
from pathlib import Path

# WVDOH open-access inventory is missing Warning signs (as of July 2026).
# Expand once WVDOH provides the full sign feature class.
# For now: suppress "in_inventory_not_detected" for Warnings only.
INVENTORY_CLASSES_MISSING = {"Warnings"}

# MUTCDCAT (WVDOH category field) -> model 9-class taxonomy — the coarse
# fallback, used when no MUTCDCODE prefix rule matches below.
WVDOH_FIELD_MAP = {
    'REGULATORY': 'Regulatory',
    'WARNING': 'Warnings',
    'SCHOOL': 'Warnings',
    'OBJECT & END OF ROAD MARKERS': 'Regulatory',
    # GUIDE-* categories are resolved by prefix rules / substring below.
}

# MUTCDCODE prefixes -> model taxonomy, applied *before* the category fallback
# because the sign code identifies the class more precisely than the category
# (e.g. a W1 curve-warning lives in a file whose category is just "WARNING").
# Order matters: first matching prefix wins.
_MUTCD_CODE_RULES = [
    ('R1-1', 'StopSigns'),
    ('R2',   'SpeedLimits'),
    ('R10',  'TrafficLights'),
    ('W1',   'Curves'),          # W1-* turn/curve warning series = the Curves class
    ('D10',  'MileMarkers'),
    ('M',    'HighwaySymbols'),  # route shields / auxiliaries
    ('D1',   'Informational'),   # destination/distance guide signs
    ('OT',   'Informational'),   # OTR/OTI/OTDD/OTW "other" informational
]


def normalize_wvdoh_class(props: dict) -> str:
    """Map a WVDOH inventory feature's properties to the model's 9-class taxonomy."""
    code = (props.get('MUTCDCODE') or '').strip().upper()
    cat = (props.get('MUTCDCAT') or '').strip().upper()
    for prefix, cls in _MUTCD_CODE_RULES:
        if code.startswith(prefix):
            return cls
    if cat.startswith('GUIDE'):
        return 'HighwaySymbols' if 'ROUTE SIGN' in cat else 'Informational'
    return WVDOH_FIELD_MAP.get(cat, 'Regulatory')


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


def load_inventory_dir(directory) -> dict:
    """
    Load and merge every .geojson/.json file in a directory into one
    FeatureCollection.  Each feature gets two normalized properties:
      - 'sign_class'       : model-taxonomy class (normalize_wvdoh_class)
      - 'inventory_source' : the source filename it came from
    A single-file inventory can be loaded with load_geojson() instead; pass it
    through normalize_inventory() to get the same normalized properties.
    """
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in ('.geojson', '.json'))
    merged = []
    for path in files:
        fc = load_geojson(path)
        for feat in fc.get('features', []):
            props = feat.setdefault('properties', {})
            props['sign_class'] = normalize_wvdoh_class(props)
            props['inventory_source'] = path.name
            merged.append(feat)
    return {'type': 'FeatureCollection', 'features': merged}


def normalize_inventory(inventory: dict, source_name=None) -> dict:
    """Add normalized 'sign_class'/'inventory_source' to a single-file inventory."""
    for feat in inventory.get('features', []):
        props = feat.setdefault('properties', {})
        if 'sign_class' not in props:
            props['sign_class'] = normalize_wvdoh_class(props)
        props.setdefault('inventory_source', source_name)
    return inventory


def save_geojson(clusters: list, path, video_source=None):
    """Write clusters to a GeoJSON FeatureCollection file."""
    features = []
    for c in clusters:
        if c.get('lat_median') is None or c.get('lon_median') is None:
            continue
        # Keep cluster_id in properties — the map keys markers / grader reviews by it.
        props = {k: v for k, v in c.items() if k not in ('lat_median', 'lon_median')}
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


def _inv_class(props: dict) -> str:
    """Class of an inventory feature — prefer a pre-normalized value, else derive."""
    return (props.get('sign_class') or normalize_wvdoh_class(props)
            or props.get('SignClass') or props.get('Category') or 'Unknown')


def _route_corridor(clusters: list, buffer_m: float):
    """
    Bounding box (with a metric buffer) around the detected clusters, i.e. the
    driven route.  Statewide inventories hold ~250k signs across all of WV; only
    those inside the corridor are candidates for matching / "missing" flagging.
    Returns (min_lat, min_lon, max_lat, max_lon) or None if no located clusters.
    """
    pts = [(c['lat_median'], c['lon_median']) for c in clusters
           if c.get('lat_median') is not None and c.get('lon_median') is not None]
    if not pts:
        return None
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    mean_lat = sum(lats) / len(lats)
    dlat = buffer_m / 111_000.0
    dlon = buffer_m / (111_000.0 * max(math.cos(math.radians(mean_lat)), 1e-6))
    return (min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon)


def compare_with_inventory(clusters: list, inventory: dict, match_radius_m=25.0,
                           corridor_buffer_m=60.0) -> list:
    """
    Annotates clusters with inventory comparison fields (mutates in place) and
    appends synthetic 'in_inventory_not_detected' clusters for inventory signs on
    the route corridor that were never detected.  Returns the (extended) list.

    Only inventory features inside the route corridor (bbox of the detections +
    corridor_buffer_m) are considered — otherwise a statewide inventory would emit
    a "missing" cluster for every sign in WV.  Warnings are exempt from the
    "missing" direction (see INVENTORY_CLASSES_MISSING) because the open-access
    inventory has no comprehensive Warning feature class.
    """
    all_features = inventory.get('features', [])
    corridor = _route_corridor(clusters, corridor_buffer_m)

    # Real detection points (used for the "missing" nearest-detection test below).
    det_pts = [(c['lat_median'], c['lon_median']) for c in clusters
               if c.get('lat_median') is not None and c.get('lon_median') is not None]

    # Restrict to the route corridor bbox and pre-extract coordinates once.
    inv = []  # list of (lat, lon, feat)
    for feat in all_features:
        try:
            ilon, ilat = feat['geometry']['coordinates'][:2]
        except (KeyError, TypeError, IndexError):
            continue
        if corridor and not (corridor[0] <= ilat <= corridor[2]
                             and corridor[1] <= ilon <= corridor[3]):
            continue
        inv.append((ilat, ilon, feat))

    inv_matched = set()

    for cluster in clusters:
        lat, lon = cluster['lat_median'], cluster['lon_median']
        best_dist, best_idx = float('inf'), None
        for i, (ilat, ilon, _feat) in enumerate(inv):
            d = haversine_m(lat, lon, ilat, ilon)
            if d < best_dist:
                best_dist, best_idx = d, i

        if best_idx is not None and best_dist <= match_radius_m:
            inv_feat = inv[best_idx][2]
            inv_matched.add(best_idx)
            props = inv_feat.get('properties') or {}
            cluster['inventory_match'] = (
                inv_feat.get('id') or props.get('OBJECTID') or props.get('id')
            )
            cluster['inventory_distance_m'] = round(best_dist, 2)
            cluster['inventory_source'] = props.get('inventory_source')

            inv_class = _inv_class(props)
            cluster['inventory_class'] = inv_class
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

    # Synthetic clusters for corridor inventory signs with no nearby detection.
    next_id = max((c['cluster_id'] for c in clusters), default=-1) + 1
    for i, (ilat, ilon, feat) in enumerate(inv):
        if i in inv_matched:
            continue
        props = feat.get('properties') or {}
        inv_class = _inv_class(props)
        # WVDOH open-access inventory has no comprehensive Warning feature class,
        # so don't flag "Warning in inventory not detected" (one-directional only).
        if inv_class in INVENTORY_CLASSES_MISSING:
            continue
        # Only signs actually on the driven route are "expected": require a real
        # detection within corridor_buffer_m.  Without this a statewide inventory
        # would flag every sign inside the (possibly large) route bbox as missing.
        if det_pts:
            nearest = min(haversine_m(ilat, ilon, dlat, dlon) for dlat, dlon in det_pts)
            if nearest > corridor_buffer_m:
                continue
        clusters.append({
            'cluster_id': next_id,
            'sign_class': inv_class,
            'confidence': 0.0,
            'sighting_count': 0,
            'needs_review': True,
            'review_reason': f'Inventory sign not detected within {match_radius_m:.0f}m',
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
            'inventory_source': props.get('inventory_source'),
            'discrepancy_type': 'in_inventory_not_detected',
        })
        next_id += 1

    return clusters
