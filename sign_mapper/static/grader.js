// SignGrader — shared in-place grader workflow for the detection maps.
//
// Identical copy lives in sign_mapper/static/grader.js (Flask app) and
// docs/assets/grader.js (GitHub Pages). Keep the two in sync.
//
// Framework-agnostic: holds review state keyed by cluster_id, builds the
// grader popup section, and exposes export/import + a running tally. Each map
// wires the map-specific glue (markers, keyboard "advance to next", controls).
window.SignGrader = (function () {
  const reviews = {};        // cluster_id -> {decision, note, reviewed_at, sign_class, lat, lon}
  let videoSource = null;

  const LOCAL_HOSTS = ['localhost', '127.0.0.1'];
  function isLocal() { return LOCAL_HOSTS.includes(window.location.hostname); }

  // Thumbnails default ON when served from the local Flask app, OFF elsewhere.
  let thumbnailsEnabled = isLocal();

  const DECISION_COLORS = { correct: '#38a169', wrong_class: '#e53e3e', unclear: '#805ad5' };
  const DECISION_LABELS = { correct: '✓ Correct', wrong_class: '✗ Wrong class', unclear: '? Unclear' };

  function setVideoSource(v) { if (v) videoSource = v; }
  function getReview(id) { return reviews[String(id)]; }

  function _ensure(id, meta) {
    id = String(id);
    const r = reviews[id] || {};
    if (meta) {
      if (meta.sign_class != null) r.sign_class = meta.sign_class;
      if (meta.lat != null) r.lat = meta.lat;
      if (meta.lon != null) r.lon = meta.lon;
    }
    reviews[id] = r;
    return r;
  }

  function decide(id, decision, meta) {
    const r = _ensure(id, meta);
    r.decision = decision;
    r.reviewed_at = new Date().toISOString();
    return r;
  }

  function setNote(id, note, meta) {
    const r = _ensure(id, meta);
    r.note = note || null;
    if (!r.reviewed_at) r.reviewed_at = new Date().toISOString();
    return r;
  }

  function clear(id) { delete reviews[String(id)]; }

  function decisionColor(id) {
    const r = reviews[String(id)];
    return r && r.decision ? DECISION_COLORS[r.decision] : null;
  }

  function tally() {
    let correct = 0, wrong = 0, unclear = 0;
    for (const id in reviews) {
      const d = reviews[id].decision;
      if (d === 'correct') correct++;
      else if (d === 'wrong_class') wrong++;
      else if (d === 'unclear') unclear++;
    }
    return { reviewed: correct + wrong + unclear, correct, wrong, unclear };
  }

  function thumbnailsOn() { return thumbnailsEnabled; }
  function toggleThumbnails(v) {
    thumbnailsEnabled = (v === undefined) ? !thumbnailsEnabled : !!v;
    return thumbnailsEnabled;
  }

  // ── Export / import ────────────────────────────────────────────────────────
  function exportReviews() {
    const out = {};
    const wrong_class_summary = [];
    for (const id in reviews) {
      const r = reviews[id];
      out[id] = {
        sign_class: r.sign_class ?? null,
        lat: r.lat ?? null,
        lon: r.lon ?? null,
        decision: r.decision ?? null,
        note: r.note ?? null,
        reviewed_at: r.reviewed_at ?? null,
      };
      if (r.decision === 'wrong_class') {
        wrong_class_summary.push({
          cluster_id: id, predicted: r.sign_class ?? null,
          lat: r.lat ?? null, lon: r.lon ?? null, note: r.note ?? null,
        });
      }
    }
    const payload = {
      exported_at: new Date().toISOString(),
      video_source: videoSource,
      reviews: out,
    };
    if (wrong_class_summary.length) payload.wrong_class_summary = wrong_class_summary;
    return payload;
  }

  function importReviews(payload) {
    Object.keys(reviews).forEach(k => delete reviews[k]);
    const src = (payload && payload.reviews) || {};
    for (const id in src) reviews[id] = Object.assign({}, src[id]);
    if (payload && payload.video_source) videoSource = payload.video_source;
    return reviews;
  }

  function downloadJson(obj, filename) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ── Popup grader section ────────────────────────────────────────────────────
  // meta: {sign_class, lat, lon, sighting_count, thumbnail_url}
  function graderSectionHtml(id, meta) {
    id = String(id);
    const r = reviews[id] || {};
    const showThumb = thumbnailsEnabled && meta.thumbnail_url && (meta.sighting_count || 0) > 0;
    const thumb = showThumb
      ? `<img class="grader-thumb" src="${meta.thumbnail_url}" alt="detection frame"
             onerror="this.style.display='none'" />`
      : '';
    const btn = (val) =>
      `<button class="grader-btn${r.decision === val ? ' sel' : ''}" data-decision="${val}" data-id="${id}">${DECISION_LABELS[val]}</button>`;
    const noteVal = (r.note || '').replace(/"/g, '&quot;');
    return `
      ${thumb}
      <div class="grader-decision-label">Review decision:</div>
      <div class="grader-btns">${btn('correct')}${btn('wrong_class')}${btn('unclear')}</div>
      <div class="grader-note-row">
        <input class="grader-note" data-id="${id}" placeholder="Note…" value="${noteVal}" />
        <button class="grader-save" data-id="${id}">Save</button>
      </div>`;
  }

  // Wire the buttons/note/save inside a freshly-opened popup DOM element.
  // onChange(review) fires after any decision or save so the map can recolor
  // the marker + refresh the tally. onSave(review) fires on Save specifically
  // (the map uses it to advance to the next needs-review marker).
  function attachHandlers(rootEl, id, meta, onChange, onSave) {
    id = String(id);
    const buttons = rootEl.querySelectorAll('.grader-btn');
    buttons.forEach(b => {
      b.onclick = () => {
        decide(id, b.dataset.decision, meta);
        buttons.forEach(x => x.classList.toggle('sel', x.dataset.decision === b.dataset.decision));
        onChange && onChange(getReview(id));
      };
    });
    const note = rootEl.querySelector('.grader-note');
    const save = rootEl.querySelector('.grader-save');
    if (save) {
      save.onclick = () => {
        setNote(id, note ? note.value : '', meta);
        onChange && onChange(getReview(id));
        onSave && onSave(getReview(id));
      };
    }
    return { note, save };
  }

  return {
    isLocal, setVideoSource, getReview, decide, setNote, clear,
    decisionColor, DECISION_COLORS, DECISION_LABELS, tally,
    thumbnailsOn, toggleThumbnails, exportReviews, importReviews, downloadJson,
    graderSectionHtml, attachHandlers,
  };
})();
