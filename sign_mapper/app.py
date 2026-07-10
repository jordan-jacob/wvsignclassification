"""
WV Sign Mapper — Flask web application.
Run: python app.py  (browser opens automatically)
"""
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path
from queue import Empty, Queue

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

# When frozen by PyInstaller the executable and its data land in different places:
#   sys.executable      → .../dist/WVSignMapper/WVSignMapper.exe  (writable, for outputs)
#   sys._MEIPASS        → same dist folder in --onedir mode        (read-only bundle data)
# In normal Python both resolve to the sign_mapper/ source directory.
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
    _STATIC_DIR = Path(sys._MEIPASS) / 'static'
else:
    BASE_DIR = Path(__file__).resolve().parent   # resolve() guarantees absolute even when __file__ is relative
    _STATIC_DIR = BASE_DIR / 'static'

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB

# ---------------------------------------------------------------------------
# Shared processing state (single-user local app — no session isolation needed)
# ---------------------------------------------------------------------------
_state = {
    'status': 'idle',   # idle | running | done | error
    'log': [],
    'clusters': [],
    'error': None,
    'subscribers': [],
}
_lock = threading.Lock()


def _emit(msg: str):
    with _lock:
        _state['log'].append(msg)
        for q in _state['subscribers']:
            q.put({'type': 'log', 'message': msg})


def _broadcast(event: dict):
    with _lock:
        for q in _state['subscribers']:
            q.put(event)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/process', methods=['POST'])
def start_processing():
    with _lock:
        if _state['status'] == 'running':
            return jsonify({'error': 'Already processing'}), 409

    model_file = request.files.get('model')
    video_file = request.files.get('video')
    sidecar_file = request.files.get('sidecar')
    inventory_files = request.files.getlist('inventory')  # 0, 1, or many
    # Server-side directory of WVDOH GeoJSONs (e.g. Desktop/wvdoh_signs), the
    # web-app equivalent of the CLI's --inventory-dir flag.
    inventory_dir = (request.form.get('inventory_dir') or '').strip()

    if not all([model_file, video_file, sidecar_file]):
        return jsonify({'error': 'model, video, and sidecar are required'}), 400

    conf_threshold = float(request.form.get('conf_threshold', 0.5))
    match_radius = float(request.form.get('match_radius', 25.0))
    extract_fps = float(request.form.get('fps', 2.0))

    upload_dir = OUTPUTS_DIR / 'uploads'
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    model_path = upload_dir / (model_file.filename or 'model.pt')
    video_path = upload_dir / (video_file.filename or 'video.mp4')
    sidecar_path = upload_dir / (sidecar_file.filename or 'sidecar.srt')
    model_file.save(str(model_path))
    video_file.save(str(video_path))
    sidecar_file.save(str(sidecar_path))

    inventory_paths = []
    for inv_file in inventory_files:
        if inv_file and inv_file.filename:
            p = upload_dir / inv_file.filename
            inv_file.save(str(p))
            inventory_paths.append(p)

    # Clear frames from previous run
    frames_dir = OUTPUTS_DIR / 'frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    with _lock:
        _state['status'] = 'running'
        _state['log'] = []
        _state['clusters'] = []
        _state['error'] = None

    video_filename = video_file.filename

    def _run():
        try:
            # Import here so the module search path (sign_mapper/) is already set
            from pipeline import run_pipeline
            from geojson_utils import (compare_with_inventory, load_geojson,
                                       load_inventory_dir, normalize_inventory,
                                       save_geojson)

            clusters = run_pipeline(
                model_path=model_path,
                video_path=video_path,
                sidecar_path=sidecar_path,
                output_dir=OUTPUTS_DIR,
                conf_threshold=conf_threshold,
                progress=_emit,
                extract_fps=extract_fps,
            )

            inv = None
            if inventory_dir:
                _emit(f"Loading inventory directory: {inventory_dir}")
                inv = load_inventory_dir(inventory_dir)
            elif len(inventory_paths) > 1:
                _emit(f"Merging {len(inventory_paths)} inventory files...")
                merged = {'type': 'FeatureCollection', 'features': []}
                for p in inventory_paths:
                    merged['features'].extend(
                        normalize_inventory(load_geojson(p), p.name)['features'])
                inv = merged
            elif inventory_paths:
                inv = normalize_inventory(load_geojson(inventory_paths[0]),
                                          inventory_paths[0].name)

            if inv is not None:
                _emit(f"Comparing with WVDOH inventory ({len(inv['features'])} features)...")
                clusters = compare_with_inventory(clusters, inv, match_radius_m=match_radius)
                n_disc = sum(1 for c in clusters if c.get('discrepancy_type'))
                _emit(f"Inventory comparison complete — {n_disc} discrepancies flagged")

            geojson_path = OUTPUTS_DIR / 'detections.geojson'
            save_geojson(clusters, geojson_path, video_filename)
            _emit(f"GeoJSON saved: {len(clusters)} features → detections.geojson")

            with _lock:
                _state['clusters'] = clusters
                _state['status'] = 'done'

            _broadcast({'type': 'complete', 'count': len(clusters)})

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _emit(f"ERROR: {exc}")
            print(tb, file=sys.stderr)
            with _lock:
                _state['status'] = 'error'
                _state['error'] = str(exc)
            _broadcast({'type': 'error', 'message': str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/api/stream')
def stream():
    def generate():
        # Atomically subscribe, replaying buffered log if processing is mid-run
        with _lock:
            if _state['status'] == 'done':
                yield f'data: {json.dumps({"type": "complete", "count": len(_state["clusters"])})}\n\n'
                return
            if _state['status'] == 'error':
                yield f'data: {json.dumps({"type": "error", "message": _state["error"]})}\n\n'
                return
            q = Queue()
            for msg in _state['log']:
                q.put({'type': 'log', 'message': msg})
            _state['subscribers'].append(q)

        try:
            while True:
                try:
                    event = q.get(timeout=20)
                    yield f'data: {json.dumps(event)}\n\n'
                    if event.get('type') in ('complete', 'error'):
                        break
                except Empty:
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            with _lock:
                try:
                    _state['subscribers'].remove(q)
                except ValueError:
                    pass

    return Response(
        generate(),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/results')
def results():
    with _lock:
        return jsonify({
            'status': _state['status'],
            'clusters': _state['clusters'],
            'error': _state['error'],
        })


@app.route('/frame/<int:cluster_id>')
def serve_frame(cluster_id):
    thumb = OUTPUTS_DIR / 'frames' / f'cluster_{cluster_id}_clean.jpg'
    if thumb.exists():
        return send_file(str(thumb), mimetype='image/jpeg')
    return '', 404


@app.route('/frame/<int:cluster_id>/bbox')
def serve_frame_bbox(cluster_id):
    """Thumbnail with the YOLO detection box drawn (Item 8)."""
    thumb = OUTPUTS_DIR / 'frames' / f'cluster_{cluster_id}_bbox.jpg'
    if thumb.exists():
        return send_file(str(thumb), mimetype='image/jpeg')
    return '', 404


@app.route('/api/mark-reviewed/<int:cluster_id>', methods=['POST'])
def mark_reviewed(cluster_id):
    with _lock:
        for c in _state['clusters']:
            if c['cluster_id'] == cluster_id:
                c['needs_review'] = False
                c['review_reason'] = None
                return jsonify({'ok': True})
    return jsonify({'error': 'Cluster not found'}), 404


@app.route('/api/qa-report')
def qa_report():
    """Return the qa_report.json produced by --annotations-dir, if it exists."""
    qa_path = OUTPUTS_DIR / 'qa_report.json'
    if not qa_path.exists():
        return jsonify({'error': 'No QA report found — run with --annotations-dir to generate one'}), 404
    return send_file(str(qa_path), mimetype='application/json')


@app.route('/api/export')
def export_geojson():
    geojson_path = OUTPUTS_DIR / 'detections.geojson'
    if not geojson_path.exists():
        return jsonify({'error': 'No results yet — run processing first'}), 404

    # Re-save with any in-session review updates
    from geojson_utils import save_geojson
    with _lock:
        clusters = list(_state['clusters'])
    save_geojson(clusters, geojson_path)

    return send_file(
        str(geojson_path),
        as_attachment=True,
        download_name='detections.geojson',
        mimetype='application/geo+json',
    )


if __name__ == '__main__':
    import socket
    port = 5000
    url = f'http://localhost:{port}'

    # Detect port conflict before Flask tries to bind, so the error is readable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            _s.bind(('0.0.0.0', port))
        except OSError:
            print(f'\n  ERROR: port {port} is already in use.')
            print(f'  Close the other WV Sign Mapper window and try again.\n')
            raise SystemExit(1)

    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f'\n  WV Sign Mapper')
    print(f'  Running at {url}')
    print(f'  Close this window to stop the server.\n')
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
