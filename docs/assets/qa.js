// WV Sign Detection — QA review interface logic for qa.html

// ── State ──────────────────────────────────────────────────────────────────
let report = null;        // normalized report object
let items = [];           // normalized item array
let currentIdx = 0;       // index into items[]
let rawData = null;       // original parsed JSON (mutated on resolution)

// ── Normalize different JSON formats ──────────────────────────────────────
function normalizeReport(data) {
  rawData = data;
  if (data.items) {
    // qa_report format: inference-vs-annotation mismatch
    return {
      format: 'qa',
      meta: {
        video_source: data.video_source,
        generated_at: data.generated_at,
        total_frames: data.total_frames_with_annotations,
        total_mismatches: data.total_mismatches,
      },
      items: data.items.map(item => ({
        id: item.id,
        label_a: 'Annotated as',
        class_a: item.annotated_class,
        badge_a: 'annotated',
        label_b: 'Model predicted',
        class_b: item.predicted_class,
        badge_b: 'predicted',
        confidence: item.confidence,
        iou: item.iou,
        lat: item.lat,
        lon: item.lon,
        thumbnail: item.frame_thumbnail || null,
        reason: `IoU ${item.iou != null ? item.iou.toFixed(2) : '?'}, conf ${item.confidence != null ? (item.confidence * 100).toFixed(0) + '%' : '?'}`,
        status: item.status || 'pending',
        resolution: item.resolution || null,
        reviewer_note: item.reviewer_note || null,
        _raw: item,
      })),
    };
  } else if (data.discrepancies) {
    // discrepancy report format: inventory-vs-detection
    return {
      format: 'discrepancy',
      meta: {
        video_source: data.video_source,
        processed_at: data.processed_at,
        total_clusters: data.total_clusters,
        needs_review: data.needs_review,
      },
      items: data.discrepancies.map(item => {
        let label_a, class_a, badge_a, label_b, class_b, badge_b;
        if (item.type === 'class_mismatch') {
          label_a = 'Detected as'; class_a = item.sign_class; badge_a = 'detected';
          label_b = 'Inventory says'; class_b = item.inventory_class || '?'; badge_b = 'inventory';
        } else if (item.type === 'detected_not_in_inventory') {
          label_a = 'Detected'; class_a = item.sign_class; badge_a = 'detected';
          label_b = 'Inventory'; class_b = 'Not found'; badge_b = 'inventory';
        } else {
          label_a = 'Inventory'; class_a = item.sign_class; badge_a = 'inventory';
          label_b = 'Detected'; class_b = 'Not found'; badge_b = 'detected';
        }
        return {
          id: item.id,
          type: item.type,
          label_a, class_a, badge_a,
          label_b, class_b, badge_b,
          confidence: item.confidence,
          iou: null,
          lat: item.lat,
          lon: item.lon,
          thumbnail: null,
          reason: item.review_reason || item.type,
          status: item.status || 'pending',
          resolution: null,
          reviewer_note: item.reviewer_note || null,
          _raw: item,
        };
      }),
    };
  } else {
    throw new Error('Unrecognized format — expected "items" or "discrepancies" array.');
  }
}

// ── Load dispatcher ──────────────────────────────────────────────────────────
// Review-export files (from the map's "Export Reviews") have a top-level
// `reviews` object; qa_report/discrepancy files don't. Structure wins over the
// radio hint so a mislabeled load still renders correctly.
function handleLoadedData(data) {
  if (data && data.reviews && !data.items && !data.discrepancies) {
    loadReviewExport(data);
  } else {
    loadReport(data);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ── Review-export view (human grader decisions from the map) ─────────────────
function loadReviewExport(data) {
  const reviews = data.reviews || {};
  const rows = Object.entries(reviews).map(([id, r]) => Object.assign({ cluster_id: id }, r));
  const decided = rows.filter(r => r.decision);
  const total = decided.length;
  const correct = decided.filter(r => r.decision === 'correct').length;
  const wrong = decided.filter(r => r.decision === 'wrong_class');
  const unclear = decided.filter(r => r.decision === 'unclear').length;

  document.getElementById('qa-drop-screen').style.display = 'none';
  document.getElementById('qa-main').style.display = 'none';
  document.getElementById('qa-review-export').style.display = 'flex';

  const pct = n => total ? ` ${((n / total) * 100).toFixed(0)}%` : '';
  document.getElementById('rx-stats').innerHTML = `
    <div class="rx-stat"><span class="rx-num">${total}</span><span class="rx-lbl">Reviewed</span></div>
    <div class="rx-stat correct"><span class="rx-num">${correct}</span><span class="rx-lbl">Correct${pct(correct)}</span></div>
    <div class="rx-stat wrong"><span class="rx-num">${wrong.length}</span><span class="rx-lbl">Wrong${pct(wrong.length)}</span></div>
    <div class="rx-stat unclear"><span class="rx-num">${unclear}</span><span class="rx-lbl">Unclear${pct(unclear)}</span></div>`;

  // Prefer the explicit wrong_class_summary when present, else derive it.
  const wrongItems = (data.wrong_class_summary && data.wrong_class_summary.length)
    ? data.wrong_class_summary.map(w => ({
        cluster_id: w.cluster_id, sign_class: w.predicted, lat: w.lat, lon: w.lon, note: w.note }))
    : wrong.map(r => ({
        cluster_id: r.cluster_id, sign_class: r.sign_class, lat: r.lat, lon: r.lon, note: r.note }));

  renderWrongTable(wrongItems, '');
  document.getElementById('rx-filter').value = '';
  document.getElementById('rx-filter').oninput = e => renderWrongTable(wrongItems, e.target.value);
  document.getElementById('rx-csv-btn').onclick = () => exportReannotationCsv(wrongItems, data.video_source);
}

function renderWrongTable(items, filter) {
  const f = (filter || '').toLowerCase();
  const shown = items.filter(it =>
    !f || (it.sign_class || '').toLowerCase().includes(f) || (it.note || '').toLowerCase().includes(f));
  const body = shown.map(it => `
    <tr>
      <td class="rx-mono">${it.cluster_id}</td>
      <td>${it.sign_class || '—'}</td>
      <td class="rx-mono">${it.lat != null ? Number(it.lat).toFixed(5) : '—'}, ${it.lon != null ? Number(it.lon).toFixed(5) : '—'}</td>
      <td>${it.note ? escapeHtml(it.note) : '<span style="color:#a0aec0">—</span>'}</td>
    </tr>`).join('');
  document.getElementById('rx-table').innerHTML =
    '<thead><tr><th>Cluster</th><th>Predicted class</th><th>Lat, Lon</th><th>Note</th></tr></thead>' +
    `<tbody>${body || '<tr><td colspan="4" style="color:#a0aec0;padding:16px">No wrong-class detections.</td></tr>'}</tbody>`;
}

function exportReannotationCsv(items, videoSource) {
  const esc = v => {
    const s = v == null ? '' : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const header = ['cluster_id', 'predicted_class', 'lat', 'lon', 'note', 'video_source'];
  const lines = [header.join(',')];
  items.forEach(it => lines.push(
    [esc(it.cluster_id), esc(it.sign_class), esc(it.lat), esc(it.lon), esc(it.note), esc(videoSource)].join(',')));
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'reannotation_list.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Load report ────────────────────────────────────────────────────────────
function loadReport(data) {
  report = normalizeReport(data);
  items = report.items;
  // Sort by confidence ascending (lowest first — most likely wrong)
  items.sort((a, b) => (a.confidence ?? 1) - (b.confidence ?? 1));
  currentIdx = 0;

  document.getElementById('qa-drop-screen').style.display = 'none';
  document.getElementById('qa-main').style.display = 'flex';
  document.getElementById('qa-export-btn').style.display = '';

  renderQueue();
  renderDetail(currentIdx);
  updateProgress();
}

// ── Queue ──────────────────────────────────────────────────────────────────
function renderQueue() {
  const list = document.getElementById('qa-queue-list');
  list.innerHTML = items.map((item, i) => {
    const isPending = !item.resolution && item.status === 'pending';
    const badgeClass = item.type === 'in_inventory_not_detected' ? 'badge-grey'
      : item.type === 'class_mismatch' ? 'badge-blue'
      : item.type === 'detected_not_in_inventory' ? 'badge-red'
      : 'badge-yellow';
    const badgeLabel = item.type === 'in_inventory_not_detected' ? 'Missing'
      : item.type === 'class_mismatch' ? 'Mismatch'
      : item.type === 'detected_not_in_inventory' ? 'Not in inv.'
      : 'Review';
    const resolved = item.resolution || (item.status && item.status !== 'pending');

    return `<div class="qa-queue-item${i === currentIdx ? ' active' : ''}${resolved ? ' resolved' : ''}"
               data-idx="${i}" title="${item.reason || ''}">
      <div class="qi-header">
        <span class="qi-class">${item.class_a}</span>
        <span class="badge ${badgeClass}">${badgeLabel}</span>
      </div>
      <div class="qi-reason">${item.reason || ''}</div>
    </div>`;
  }).join('');

  list.querySelectorAll('.qa-queue-item').forEach(el => {
    el.addEventListener('click', () => {
      currentIdx = parseInt(el.dataset.idx);
      renderQueue();
      renderDetail(currentIdx);
    });
  });

  // Scroll active item into view
  const activeEl = list.querySelector('.qa-queue-item.active');
  if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
}

// ── Detail view ────────────────────────────────────────────────────────────
function renderDetail(idx) {
  const el = document.getElementById('qa-detail');
  if (!items.length) {
    el.innerHTML = '<p class="qa-empty">No items to review.</p>';
    return;
  }

  const item = items[idx];
  const conf = item.confidence != null ? (item.confidence * 100).toFixed(1) + '%' : '—';
  const iou  = item.iou != null ? item.iou.toFixed(3) : '—';
  const isQa = report.format === 'qa';

  // Thumbnail
  const thumbHtml = item.thumbnail
    ? `<div class="qa-thumbnail"><img src="${item.thumbnail}" alt="Frame thumbnail" /></div>`
    : `<div class="qa-thumbnail"><div class="qa-thumbnail-placeholder">
         No thumbnail available<br><small>(frame thumbnails are served by the local app)</small>
       </div></div>`;

  // Class comparison
  const classHtml = `
    <div class="qa-classes-row">
      <div>
        <div style="font-size:11px;font-weight:600;color:#718096;margin-bottom:4px">${item.label_a}</div>
        <span class="qa-class-badge ${item.badge_a}">${item.class_a}</span>
      </div>
      <div class="qa-vs">vs</div>
      <div>
        <div style="font-size:11px;font-weight:600;color:#718096;margin-bottom:4px">${item.label_b}</div>
        <span class="qa-class-badge ${item.badge_b}">${item.class_b}</span>
      </div>
    </div>`;

  // Resolution buttons
  const resOptions = isQa
    ? [
        { val: 'annotation_correct',  label: '1 — Annotation Correct',  title: 'Model was wrong; keep annotation' },
        { val: 'prediction_correct',  label: '2 — Prediction Correct',  title: 'Annotation was wrong; needs re-label' },
        { val: 'both_wrong',          label: '3 — Both Wrong',          title: 'Needs fresh annotation' },
        { val: 'ambiguous',           label: '4 — Ambiguous',           title: 'Flag for discussion' },
      ]
    : [
        { val: 'confirmed',           label: '1 — Confirm',             title: 'Correct as-is' },
        { val: 'dismissed',           label: '2 — Dismiss',             title: 'False detection / data error' },
        { val: 'needs_annotation',    label: '3 — Needs Annotation',    title: 'Requires fresh annotation' },
        { val: 'ambiguous',           label: '4 — Ambiguous',           title: 'Flag for discussion' },
      ];

  const curResolution = item.resolution || (item.status !== 'pending' ? item.status : null);
  const buttonsHtml = resOptions.map(opt =>
    `<button class="qa-btn${curResolution === opt.val ? ' selected' : ''}"
             title="${opt.title}"
             data-val="${opt.val}">${opt.label}</button>`
  ).join('');

  el.innerHTML = `
    <div class="qa-detail-class">${item.class_a} <span style="font-weight:400;color:#718096;font-size:14px">#${idx + 1} of ${items.length}</span></div>
    <div class="qa-meta-row">
      <span>Confidence: <strong>${conf}</strong></span>
      ${iou !== '—' ? `<span>IoU: <strong>${iou}</strong></span>` : ''}
      <span>Status: <strong>${item.status || 'pending'}</strong></span>
      ${item.lat != null ? `<span>GPS: <strong>${item.lat.toFixed(5)}, ${item.lon.toFixed(5)}</strong></span>` : ''}
    </div>
    ${classHtml}
    ${thumbHtml}
    <div class="qa-resolution-label">Resolution</div>
    <div class="qa-buttons" id="res-buttons">${buttonsHtml}</div>
    <label class="qa-note-label" for="qa-note-field">Reviewer note (optional)</label>
    <textarea class="qa-note" id="qa-note-field" placeholder="Notes about this item...">${item.reviewer_note || ''}</textarea>
    <div class="qa-nav-row">
      <button class="btn btn-secondary" id="prev-btn" ${idx === 0 ? 'disabled' : ''}>&#8592; Prev</button>
      <button class="btn btn-primary" id="next-btn" ${idx >= items.length - 1 ? 'disabled' : ''}>Next &#8594;</button>
    </div>
    <div class="keyboard-hint">
      <kbd>1</kbd>–<kbd>4</kbd> resolve &nbsp;
      <kbd>&#8592;</kbd><kbd>&#8594;</kbd> navigate
    </div>`;

  // Resolution button clicks
  el.querySelectorAll('#res-buttons .qa-btn').forEach(btn => {
    btn.addEventListener('click', () => setResolution(idx, btn.dataset.val));
  });

  // Note changes
  el.querySelector('#qa-note-field').addEventListener('input', e => {
    items[idx].reviewer_note = e.target.value || null;
    items[idx]._raw.reviewer_note = e.target.value || null;
  });

  el.querySelector('#prev-btn').addEventListener('click', () => navigate(-1));
  el.querySelector('#next-btn').addEventListener('click', () => navigate(1));
}

function setResolution(idx, val) {
  const item = items[idx];
  if (report.format === 'qa') {
    item.resolution = val;
    item._raw.resolution = val;
    item.status = 'resolved';
    item._raw.status = 'resolved';
  } else {
    item.status = val;
    item._raw.status = val;
  }
  updateProgress();
  renderQueue();
  renderDetail(idx);
  // Auto-advance to next pending item
  const nextPending = items.findIndex((it, i) => i > idx && it.status === 'pending' && !it.resolution);
  if (nextPending !== -1) {
    currentIdx = nextPending;
    setTimeout(() => { renderQueue(); renderDetail(currentIdx); }, 120);
  }
}

function navigate(dir) {
  currentIdx = Math.max(0, Math.min(items.length - 1, currentIdx + dir));
  renderQueue();
  renderDetail(currentIdx);
}

// ── Progress bar ───────────────────────────────────────────────────────────
function updateProgress() {
  const total = items.length;
  const resolved = items.filter(it =>
    it.resolution || (it.status && it.status !== 'pending')
  ).length;
  const pct = total ? (resolved / total) * 100 : 0;
  document.getElementById('qa-progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('qa-progress-text').textContent = `${resolved} of ${total} reviewed`;
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (!report || document.activeElement.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    navigate(e.key === 'ArrowRight' ? 1 : -1);
  }
  if (['1', '2', '3', '4'].includes(e.key)) {
    const isQa = report.format === 'qa';
    const vals = isQa
      ? ['annotation_correct', 'prediction_correct', 'both_wrong', 'ambiguous']
      : ['confirmed', 'dismissed', 'needs_annotation', 'ambiguous'];
    setResolution(currentIdx, vals[parseInt(e.key) - 1]);
  }
});

// ── Export ─────────────────────────────────────────────────────────────────
document.getElementById('qa-export-btn').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(rawData, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = report.format === 'qa' ? 'qa_report_reviewed.json' : 'discrepancies_reviewed.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

// ── File upload / drag-drop ────────────────────────────────────────────────
const dropZone = document.getElementById('qa-drop-zone');
const fileInput = document.getElementById('qa-file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) loadFileIntoQA(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFileIntoQA(fileInput.files[0]); });

function loadFileIntoQA(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      handleLoadedData(JSON.parse(e.target.result));
    } catch (err) {
      alert('Could not load file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

// ── Demo load button ───────────────────────────────────────────────────────
document.getElementById('load-demo-btn').addEventListener('click', () => {
  fetch('data/demo_discrepancies.json')
    .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(data => handleLoadedData(data))
    .catch(err => {
      // fetch() is blocked under file:// — fall back to the embedded copy.
      if (window.__DEMO_DISCREPANCIES__) { handleLoadedData(window.__DEMO_DISCREPANCIES__); return; }
      alert('Could not load demo data: ' + err.message);
    });
});
