// WV Sign Detection — Leaflet map logic for index.html

const DEMO_GEOJSON = 'data/demo_detections.geojson';

// Human-readable label for the demo route (real WVDOH dashcam run).
// Updated by Task 2 to match the selected inference video.
const ROUTE_LABEL = 'Wetzel County, WV — US Route (Apr 2026)';

// ── Color helpers ──────────────────────────────────────────────────────────
function markerColor(props) {
  const dec = SignGrader.decisionColor(props.cluster_id);   // grader decision wins
  if (dec) return dec;
  if (props.discrepancy_type === 'in_inventory_not_detected') return '#718096';
  if (props.discrepancy_type)                                 return '#e53e3e';
  if (!props.needs_review)                                    return '#38a169';
  return '#d69e2e';
}

function markerRadius(props) {
  return Math.min(7 + (props.sighting_count || 1) * 1.2, 18);
}

function graderMeta(props, lat, lon) {
  return {
    sign_class: props.sign_class, lat, lon,
    sighting_count: props.sighting_count,
    thumbnail_url: props.thumbnail_url || `/frame/${props.cluster_id}`,
  };
}

function popupHtml(props, lat, lon) {
  const rows = [
    ['Confidence',   props.confidence != null ? (props.confidence * 100).toFixed(1) + '%' : '—'],
    ['Sightings',    props.sighting_count != null ? props.sighting_count : '—'],
    ['Needs review', props.needs_review ? '⚠ Yes' : '✓ No'],
  ];
  if (props.review_reason)         rows.push(['Reason',       `<em>${props.review_reason}</em>`]);
  if (props.discrepancy_type)      rows.push(['Discrepancy',  props.discrepancy_type]);
  if (props.inventory_distance_m != null) rows.push(['Inv. dist.', props.inventory_distance_m + ' m']);
  if (props.inventory_source)      rows.push(['Inv. source',  props.inventory_source]);
  if (props.video_source)          rows.push(['Source',       props.video_source]);

  const tableRows = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  return `<div class="grader-popup" style="min-width:210px">
    <strong style="font-size:13px">${props.sign_class || 'Unknown'}</strong>
    <table style="width:100%;border-collapse:collapse;margin-top:5px;font-size:11px">
      ${tableRows}
    </table>
    ${SignGrader.graderSectionHtml(props.cluster_id, graderMeta(props, lat, lon))}
  </div>`;
}

// ── Map init ───────────────────────────────────────────────────────────────
const map = L.map('map').setView([38.28, -80.85], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}).addTo(map);

let layerGroup = L.layerGroup().addTo(map);
let markerMap = {};        // cluster_id → { marker, props, lat, lon }
let allFeatures = [];      // current feature list (for advance-to-next)
let currentPopup = null;   // { id, marker } of the open grader popup

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
  markerMap = {};
  const features = fc.features || [];
  allFeatures = features;
  const bounds = [];

  // Video source for review exports (first feature that names one).
  const withVideo = features.find(f => f.properties && f.properties.video_source);
  if (withVideo) SignGrader.setVideoSource(withVideo.properties.video_source);

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
    m.bindPopup(popupHtml(props, lat, lon), { maxWidth: 340 });
    m.on('popupopen', (e) => wireMarkerPopup(e.popup, m, props, lat, lon));
    m.on('popupclose', () => { if (currentPopup && currentPopup.id === props.cluster_id) currentPopup = null; });
    m.addTo(layerGroup);
    markerMap[props.cluster_id] = { marker: m, props, lat, lon };
    bounds.push([lat, lon]);
  });

  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30] });
  infoControl.update(features.length);
  const subtitle = document.getElementById('route-subtitle');
  if (subtitle) subtitle.textContent = `${ROUTE_LABEL} · ${features.length} detections`;
  updateStats(features);
  updateTally();
}

// ── Grader popup wiring ──────────────────────────────────────────────────────
function refreshMarker(props) {
  const entry = markerMap[props.cluster_id];
  if (entry) entry.marker.setStyle({ color: markerColor(props), fillColor: markerColor(props) });
  updateTally();
}

function wireMarkerPopup(popup, marker, props, lat, lon) {
  currentPopup = { id: props.cluster_id, marker };
  const root = popup.getElement();
  if (!root) return;
  SignGrader.attachHandlers(root, props.cluster_id, graderMeta(props, lat, lon),
    () => refreshMarker(props),
    () => { marker.closePopup(); advanceToNextReview(props, lat, lon); });
}

function advanceToNextReview(fromProps, fromLat, fromLon) {
  const candidates = allFeatures.filter(f => {
    const p = f.properties || {};
    return p.needs_review && !SignGrader.getReview(p.cluster_id) && markerMap[p.cluster_id];
  });
  if (!candidates.length) return;
  candidates.sort((a, b) => {
    const ea = markerMap[a.properties.cluster_id], eb = markerMap[b.properties.cluster_id];
    const da = (ea.lat - fromLat) ** 2 + (ea.lon - fromLon) ** 2;
    const db = (eb.lat - fromLat) ** 2 + (eb.lon - fromLon) ** 2;
    return da - db;
  });
  const entry = markerMap[candidates[0].properties.cluster_id];
  map.setView(entry.marker.getLatLng(), Math.max(map.getZoom(), 16));
  entry.marker.openPopup();
}

function refreshOpenPopup() {
  if (!currentPopup) return;
  const entry = markerMap[currentPopup.id];
  if (!entry) return;
  const popup = entry.marker.getPopup();
  popup.setContent(popupHtml(entry.props, entry.lat, entry.lon));
  if (popup.getElement()) wireMarkerPopup(popup, entry.marker, entry.props, entry.lat, entry.lon);
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

// ── Grader controls: tally, thumbnail toggle, export/import, about panel ─────
let tallyDiv = null;

function updateTally() {
  if (!tallyDiv) return;
  const t = SignGrader.tally();
  const total = allFeatures.filter(f => {
    const p = f.properties || {};
    return (p.sighting_count || 0) > 0 || p.discrepancy_type;
  }).length;
  tallyDiv.innerHTML =
    `<strong>Reviewed: ${t.reviewed} / ${total}</strong>` +
    `<div class="tally-sub">${t.correct} correct · ${t.wrong} wrong · ${t.unclear} unclear</div>`;
}

(function buildGraderControls() {
  const tally = L.control({ position: 'topright' });
  tally.onAdd = function () {
    const div = L.DomUtil.create('div', 'grader-tally');
    L.DomEvent.disableClickPropagation(div);
    tallyDiv = div;
    div.innerHTML = 'Reviewed: 0 / 0';
    return div;
  };
  tally.addTo(map);

  const panel = L.control({ position: 'topright' });
  panel.onAdd = function () {
    const div = L.DomUtil.create('div', 'grader-tally grader-controls');
    L.DomEvent.disableClickPropagation(div);
    const on = SignGrader.thumbnailsOn();
    div.innerHTML = `
      <label class="grader-toggle">
        <input type="checkbox" id="thumb-toggle" ${on ? 'checked' : ''}/> Show detection thumbnails (local app only)
      </label>
      <div class="grader-toggle-note" id="thumb-note" style="${on ? 'display:none' : ''}">
        Thumbnails require the local app (sign_mapper/) to be running.
      </div>
      <div class="grader-io-row">
        <button id="export-reviews">Export Reviews</button>
        <button id="import-reviews">Import Reviews</button>
      </div>
      <input type="file" id="import-file" accept=".json" style="display:none"/>`;
    return div;
  };
  panel.addTo(map);

  // About-this-data panel (bottom-right) — honest note on inventory coverage.
  const about = L.control({ position: 'bottomright' });
  about.onAdd = function () {
    const div = L.DomUtil.create('div', 'about-data-panel');
    L.DomEvent.disableClickPropagation(div);
    div.innerHTML =
      `<strong>About this data</strong><br>` +
      `Sign inventory provided by WVDOH open-access GIS data (July 2026). ` +
      `Warning signs are not included in the current inventory snapshot — comparison ` +
      `for the Warning class is one-directional (detections only). ` +
      `Full sign feature class pending from WVDOH.`;
    return div;
  };
  about.addTo(map);

  document.getElementById('thumb-toggle').addEventListener('change', function () {
    SignGrader.toggleThumbnails(this.checked);
    document.getElementById('thumb-note').style.display = this.checked ? 'none' : '';
    refreshOpenPopup();
  });
  document.getElementById('export-reviews').addEventListener('click', () => {
    SignGrader.downloadJson(SignGrader.exportReviews(), 'reviews.json');
  });
  const importFile = document.getElementById('import-file');
  document.getElementById('import-reviews').addEventListener('click', () => importFile.click());
  importFile.addEventListener('change', () => {
    const f = importFile.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = e => {
      try {
        SignGrader.importReviews(JSON.parse(e.target.result));
        renderFeatures({ features: allFeatures });   // recolor everything
      } catch (err) { alert('Invalid reviews JSON: ' + err.message); }
    };
    reader.readAsText(f);
    importFile.value = '';
  });
})();

// Keyboard shortcuts while a grader popup is open
document.addEventListener('keydown', e => {
  if (!currentPopup) return;
  const inField = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
  const key = e.key.toLowerCase();
  if (inField && key !== 'escape' && key !== 'enter') return;

  const entry = markerMap[currentPopup.id];
  if (!entry) return;
  const { props, lat, lon } = entry;
  const decMap = { c: 'correct', w: 'wrong_class', u: 'unclear' };

  if (decMap[key]) {
    SignGrader.decide(props.cluster_id, decMap[key], graderMeta(props, lat, lon));
    const root = entry.marker.getPopup().getElement();
    if (root) root.querySelectorAll('.grader-btn')
      .forEach(b => b.classList.toggle('sel', b.dataset.decision === decMap[key]));
    refreshMarker(props);
  } else if (key === 'enter') {
    e.preventDefault();
    const root = entry.marker.getPopup().getElement();
    const note = root && root.querySelector('.grader-note');
    SignGrader.setNote(props.cluster_id, note ? note.value : '', graderMeta(props, lat, lon));
    refreshMarker(props);
    entry.marker.closePopup();
    advanceToNextReview(props, lat, lon);
  } else if (key === 'escape') {
    entry.marker.closePopup();
  } else if (key === 't') {
    const state = SignGrader.toggleThumbnails();
    const cb = document.getElementById('thumb-toggle'); if (cb) cb.checked = state;
    const note = document.getElementById('thumb-note'); if (note) note.style.display = state ? 'none' : '';
    refreshOpenPopup();
  }
});

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
