"""
Extract GPS track from a dashcam video using ExifTool.
Supports two extraction strategies:
  1. Embedded GPS stream (most dashcams): exiftool -ee -p gpx.fmt
  2. Per-frame metadata (some formats): exiftool -GPSLatitude -GPSLongitude
     -GPSDateTime -csv

Strategy 1 is tried first; strategy 2 is the fallback. Both return a sorted list
of {ts_ms, lat, lon, alt} dicts (video-relative milliseconds), matching the
shape produced by gps_parser.parse_sidecar(), so the rest of the pipeline is
oblivious to how the GPS came in.
"""
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from gps_parser import parse_exiftool

# Bundled GPX print-format template (see gpx.fmt). Under PyInstaller the data
# files live in sys._MEIPASS, not next to this module.
if getattr(sys, 'frozen', False):
    _FMT_FILE = Path(sys._MEIPASS) / 'gpx.fmt'
else:
    _FMT_FILE = Path(__file__).resolve().parent / 'gpx.fmt'


def exiftool_info():
    """Return (available: bool, version: str|None) by running `exiftool -ver`."""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False, None
    if result.returncode == 0:
        return True, (result.stdout.strip() or None)
    return False, None


EXIFTOOL_AVAILABLE, EXIFTOOL_VERSION = exiftool_info()


def _run_exiftool(args, timeout=300):
    """Run exiftool, returning stdout (str). Raises CalledProcessError on failure."""
    return subprocess.run(
        ["exiftool", *args],
        capture_output=True, text=True, timeout=timeout, check=True,
    ).stdout


def _parse_iso_ms(text):
    """Parse an ISO / ExifTool datetime string to epoch milliseconds, or None."""
    text = (text or '').strip().rstrip('Z')
    if not text:
        return None
    # ExifTool date form "2024:03:15 14:23:01" -> ISO "2024-03-15 14:23:01"
    if len(text) >= 10 and text[4] == ':' and text[7] == ':':
        text = text[:4] + '-' + text[5:7] + '-' + text[8:]
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp() * 1000.0
        except ValueError:
            continue
    return None


def _relativize(raw):
    """[(abs_ms|None, lat, lon, alt)] -> sorted video-relative fix dicts.

    Times are absolute wall-clock; subtract the first so the series starts at ~0,
    matching gps_parser's handling of absolute sidecar timestamps.
    """
    timed = [r for r in raw if r[0] is not None]
    if len(timed) < 2:
        return []
    timed.sort(key=lambda r: r[0])
    base = timed[0][0]
    return [{'ts_ms': ms - base, 'lat': lat, 'lon': lon, 'alt': alt}
            for ms, lat, lon, alt in timed]


def _extract_via_gpx_stream(video_path):
    """Strategy 1: embedded GPS stream -> GPX -> fixes. Returns [] on any failure."""
    if not _FMT_FILE.exists():
        return []
    try:
        gpx = _run_exiftool(['-ee', '-p', str(_FMT_FILE), str(video_path)])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    if not gpx.strip():
        return []
    try:
        root = ET.fromstring(gpx)
    except ET.ParseError:
        return []

    raw = []  # (abs_ms|None, lat, lon, alt)
    for pt in root.iter():
        if not pt.tag.endswith('trkpt'):
            continue
        try:
            lat = float(pt.get('lat'))
            lon = float(pt.get('lon'))
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            continue
        abs_ms, alt = None, None
        for child in pt:
            if child.tag.endswith('time') and child.text:
                abs_ms = _parse_iso_ms(child.text)
            elif child.tag.endswith('ele') and child.text:
                try:
                    alt = float(child.text)
                except ValueError:
                    pass
        raw.append((abs_ms, lat, lon, alt))

    return _relativize(raw)


def _extract_via_frame_metadata(video_path):
    """Strategy 2: per-frame GPS CSV -> existing parse_exiftool(). [] on failure."""
    try:
        csv_text = _run_exiftool([
            '-ee', '-GPSLatitude', '-GPSLongitude', '-GPSAltitude',
            '-GPSDateTime', '-GPSSpeed', '-csv', str(video_path),
        ])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    if not csv_text.strip():
        return []
    # parse_exiftool() reads a file; hand it the CSV via a temp file.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False,
                                         encoding='utf-8', newline='') as tf:
            tf.write(csv_text)
            tmp = Path(tf.name)
        return parse_exiftool(tmp)
    except ValueError:
        return []
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _log_summary(fixes, strategy, log):
    lats = [f['lat'] for f in fixes]
    span_s = (fixes[-1]['ts_ms'] - fixes[0]['ts_ms']) / 1000.0
    interval = span_s / max(len(fixes) - 1, 1)
    log(f"{len(fixes)} GPS fixes extracted (ExifTool strategy {strategy}, "
        f"{interval:.1f}s interval, {min(lats):.2f}°N–{max(lats):.2f}°N)")


def extract_gps_from_video(video_path, progress=None):
    """Extract a GPS track from video_path. Raises ValueError if neither strategy
    yields at least two fixes. progress(str) receives status messages."""
    video_path = Path(video_path)
    log = progress or (lambda *_: None)

    log("Extracting GPS from video metadata (ExifTool strategy 1: GPS stream)...")
    fixes = _extract_via_gpx_stream(video_path)
    if len(fixes) >= 2:
        _log_summary(fixes, 1, log)
        return fixes

    log("Strategy 1 found no GPS stream — trying ExifTool strategy 2 (frame metadata)...")
    fixes = _extract_via_frame_metadata(video_path)
    if len(fixes) >= 2:
        _log_summary(fixes, 2, log)
        return fixes

    raise ValueError(
        f"Could not extract GPS from {video_path.name}. "
        "Ensure the video has embedded GPS data, or provide a sidecar file."
    )
