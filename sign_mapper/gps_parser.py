"""
GPS sidecar parsers for SRT (dashcam) and CSV formats.
Auto-detects format by extension. Returns sorted list of
{ts_ms, lat, lon, alt} dicts; interpolate_gps() does linear
interpolation for a given video timestamp (ms from start).
"""
import csv
import re
from datetime import datetime
from pathlib import Path

# Multiple SRT GPS patterns — dashcam manufacturers vary the format
_SRT_GPS = [
    re.compile(r'GPS\s*\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*(?:,\s*(-?\d+\.?\d*))?\s*\)'),
    re.compile(r'GPS\s*:\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)'),
    re.compile(r'[Ll]at\s*[=:]\s*(-?\d+\.\d+).*?[Ll]on\s*[=:]\s*(-?\d+\.\d+)', re.DOTALL),
]
_TC = re.compile(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->')


def parse_srt(path: Path) -> list:
    fixes = []
    text = path.read_text(encoding='utf-8', errors='replace')
    for block in re.split(r'\n\s*\n', text.strip()):
        tc = _TC.search(block)
        if not tc:
            continue
        h, m, s, ms = tc.group(1, 2, 3, 4)
        ts_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
        gm = None
        for pat in _SRT_GPS:
            gm = pat.search(block)
            if gm:
                break
        if not gm:
            continue
        try:
            lat, lon = float(gm.group(1)), float(gm.group(2))
        except (ValueError, IndexError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            continue
        alt = None
        try:
            if gm.lastindex and gm.lastindex >= 3 and gm.group(3):
                alt = float(gm.group(3))
        except (ValueError, IndexError):
            pass
        fixes.append({'ts_ms': ts_ms, 'lat': lat, 'lon': lon, 'alt': alt})
    return sorted(fixes, key=lambda x: x['ts_ms'])


_LAT = {'lat', 'latitude'}
_LON = {'lon', 'lng', 'longitude', 'long'}
_TS = {'timestamp', 'time', 'ts', 'datetime'}
_ALT = {'altitude', 'alt', 'elevation'}


def _col(headers, names):
    for h in headers:
        if h.strip().lower() in names:
            return h
    return None


def _parse_ts_ms(val: str) -> float:
    val = val.strip()
    try:
        v = float(val)
        # Epoch seconds vs milliseconds heuristic
        return v * 1000 if v < 2e10 else v
    except ValueError:
        pass
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(val, fmt).timestamp() * 1000
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {val!r}")


def parse_csv(path: Path) -> list:
    fixes = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return fixes
        headers = list(reader.fieldnames)
        lat_col = _col(headers, _LAT)
        lon_col = _col(headers, _LON)
        ts_col = _col(headers, _TS)
        alt_col = _col(headers, _ALT)
        if not all([lat_col, lon_col, ts_col]):
            raise ValueError(f"CSV missing required columns. Found: {headers}")
        for row in reader:
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                ts_ms = _parse_ts_ms(row[ts_col])
            except (ValueError, KeyError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
                continue
            alt = None
            if alt_col and row.get(alt_col, '').strip():
                try:
                    alt = float(row[alt_col])
                except ValueError:
                    pass
            fixes.append({'ts_ms': ts_ms, 'lat': lat, 'lon': lon, 'alt': alt})

    fixes = sorted(fixes, key=lambda x: x['ts_ms'])

    # If timestamps look absolute (> 1 day in ms), normalize to video-relative
    # by subtracting the first timestamp so the series starts at ~0.
    if fixes and fixes[0]['ts_ms'] > 86_400_000:
        offset = fixes[0]['ts_ms']
        for f in fixes:
            f['ts_ms'] -= offset

    return fixes


def parse_sidecar(path) -> list:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.srt':
        fixes = parse_srt(path)
    elif ext == '.csv':
        fixes = parse_csv(path)
    else:
        raise ValueError(f"Unsupported GPS format {ext!r}. Expected .srt or .csv")
    if not fixes:
        raise ValueError(f"No valid GPS fixes found in {path.name}")
    return fixes


def interpolate_gps(fixes: list, ts_ms: float):
    """
    Linearly interpolate lat/lon for ts_ms (milliseconds from video start).
    Returns {lat, lon, alt} or None if ts_ms is > 2s outside the GPS range.
    """
    if not fixes:
        return None
    if ts_ms < fixes[0]['ts_ms'] - 2000 or ts_ms > fixes[-1]['ts_ms'] + 2000:
        return None
    if ts_ms <= fixes[0]['ts_ms']:
        return {k: fixes[0][k] for k in ('lat', 'lon', 'alt')}
    if ts_ms >= fixes[-1]['ts_ms']:
        return {k: fixes[-1][k] for k in ('lat', 'lon', 'alt')}

    lo, hi = 0, len(fixes) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fixes[mid]['ts_ms'] <= ts_ms:
            lo = mid
        else:
            hi = mid

    t0, t1 = fixes[lo]['ts_ms'], fixes[hi]['ts_ms']
    if t1 == t0:
        return {k: fixes[lo][k] for k in ('lat', 'lon', 'alt')}

    frac = (ts_ms - t0) / (t1 - t0)
    lat = fixes[lo]['lat'] + frac * (fixes[hi]['lat'] - fixes[lo]['lat'])
    lon = fixes[lo]['lon'] + frac * (fixes[hi]['lon'] - fixes[lo]['lon'])
    alt = None
    if fixes[lo]['alt'] is not None and fixes[hi]['alt'] is not None:
        alt = fixes[lo]['alt'] + frac * (fixes[hi]['alt'] - fixes[lo]['alt'])
    return {'lat': lat, 'lon': lon, 'alt': alt}
