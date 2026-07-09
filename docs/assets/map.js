// WV Sign Detection — Leaflet map logic for index.html

const DEMO_GEOJSON = 'data/demo_detections.geojson';

// Human-readable label for the demo route (real WVDOH dashcam run).
// Updated by Task 2 to match the selected inference video.
const ROUTE_LABEL = 'Wetzel County, WV — US Route (Apr 2026)';

// ── Color helpers ──────────────────────────────────────────────────────────
function markerColor(props) {
  if (props.discrepancy_type === 'in_inventory_not_detected') return '#718096';
  if (props.discrepancy_type)                                 return '#e53e3e';
  if (!props.needs_review)                                    return '#38a169';
  return '#d69e2e';
}

function markerRadius(props) {
  return Math.min(7 + (props.sighting_count || 1) * 1.2, 18);
}

function popupHtml(props) {
  const rows = [
    ['Confidence',   props.confidence != null ? (props.confidence * 100).toFixed(1) + '%' : '—'],
    ['Sightings',    props.sighting_count != null ? props.sighting_count : '—'],
    ['Needs review', props.needs_review ? '⚠ Yes' : '✓ No'],
  ];
  if (props.review_reason)         rows.push(['Reason',       `<em>${props.review_reason}</em>`]);
  if (props.discrepancy_type)      rows.push(['Discrepancy',  props.discrepancy_type]);
  if (props.inventory_distance_m != null) rows.push(['Inv. dist.', props.inventory_distance_m + ' m']);
  if (props.video_source)          rows.push(['Source',       props.video_source]);

  const tableRows = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  return `<div style="min-width:190px">
    <strong style="font-size:13px">${props.sign_class || 'Unknown'}</strong>
    <table style="width:100%;border-collapse:collapse;margin-top:5px;font-size:11px">
      ${tableRows}
    </table>
  </div>`;
}

// ── Map init ───────────────────────────────────────────────────────────────
const map = L.map('map').setView([38.28, -80.85], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}).addTo(map);

let layerGroup = L.layerGroup().addTo(map);

// ── Info box (bottom-left) ─────────────────────────────────────────────────
const infoControl = L.control({ position: 'bottomleft' });
infoControl.onAdd = function () {
  this._div = L.DomUtil.create('div', 'map-info-box');
  this._div.innerHTML = 'Loading detections…';
  return this._div;
};
infoControl.update = function (count) {
  this._div.innerHTML =
    `<strong>${count} sign detections</strong> from ${ROUTE_LABEL}.<br>` +
    `Powered by YOLOv8m trained on WVDOH dashcam footage.`;
};
infoControl.addTo(map);

// ── Render GeoJSON features ────────────────────────────────────────────────
function renderFeatures(fc) {
  layerGroup.clearLayers();
  const features = fc.features || [];
  const bounds = [];

  features.forEach(feat => {
    if (!feat.geometry || feat.geometry.type !== 'Point') return;
    const [lon, lat] = feat.geometry.coordinates;
    const props = feat.properties || {};

    const m = L.circleMarker([lat, lon], {
      radius: markerRadius(props),
      color: markerColor(props),
      fillColor: markerColor(props),
      fillOpacity: 0.75,
      weight: 2,
    });
    m.bindPopup(popupHtml(props), { maxWidth: 240 });
    m.addTo(layerGroup);
    bounds.push([lat, lon]);
  });

  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30] });
  infoControl.update(features.length);
  const subtitle = document.getElementById('route-subtitle');
  if (subtitle) subtitle.textContent = `${ROUTE_LABEL} · ${features.length} detections`;
  updateStats(features);
}

// ── Stats panel ────────────────────────────────────────────────────────────
function updateStats(features) {
  const total = features.length;
  const needsReview = features.filter(f => f.properties && f.properties.needs_review).length;
  const discrepancies = features.filter(f => f.properties && f.properties.discrepancy_type).length;
  const classCounts = {};
  features.forEach(f => {
    const cls = (f.properties && f.properties.sign_class) || 'Unknown';
    classCounts[cls] = (classCounts[cls] || 0) + 1;
  });
  const topClass = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0];

  document.getElementById('stat-total').textContent     = total;
  document.getElementById('stat-review').textContent    = needsReview;
  document.getElementById('stat-disc').textContent      = discrepancies;
  document.getElementById('stat-top').textContent       = topClass ? `${topClass[0]} (${topClass[1]})` : '—';
  document.getElementById('stats-panel').style.display  = '';
}

// ── Load demo data ─────────────────────────────────────────────────────────
fetch(DEMO_GEOJSON)
  .then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(fc => {
    renderFeatures(fc);
    document.getElementById('load-status').textContent =
      `Loaded ${(fc.features || []).length} detections from demo dataset`;
    document.getElementById('load-status').style.color = '#38a169';
  })
  .catch(err => {
    // fetch() is blocked under file:// — fall back to the embedded copy.
    if (window.__DEMO_GEOJSON__) {
      renderFeatures(window.__DEMO_GEOJSON__);
      document.getElementById('load-status').textContent =
        `Loaded ${(window.__DEMO_GEOJSON__.features || []).length} detections from demo dataset`;
      document.getElementById('load-status').style.color = '#38a169';
      return;
    }
    document.getElementById('load-status').textContent = 'Demo data unavailable — upload a GeoJSON file';
    document.getElementById('load-status').style.color = '#718096';
  });

// ── File upload / drag-drop ────────────────────────────────────────────────
const dropZone = document.getElementById('upload-drop-zone');
const fileInput = document.getElementById('geojson-file-input');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const fc = JSON.parse(e.target.result);
      if (fc.type !== 'FeatureCollection') throw new Error('Not a FeatureCollection');
      renderFeatures(fc);
      document.getElementById('load-status').textContent =
        `Loaded ${(fc.features || []).length} detections from ${file.name}`;
      document.getElementById('load-status').style.color = '#38a169';
    } catch (err) {
      document.getElementById('load-status').textContent = 'Invalid GeoJSON: ' + err.message;
      document.getElementById('load-status').style.color = '#e53e3e';
    }
  };
  reader.readAsText(file);
}
