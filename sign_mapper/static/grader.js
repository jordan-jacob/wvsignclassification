// SignGrader — shared in-place grader workflow for the detection maps.
//
// Identical copy lives in sign_mapper/static/grader.js (Flask app) and
// docs/assets/grader.js (GitHub Pages). Keep the two in sync.
//
// Framework-agnostic: holds review state keyed by cluster_id, builds the
// grader popup section, and exposes export/import + a running tally. Each map
// wires the map-specific glue (markers, keyboard "advance to next", controls)
// and calls wireDelegation() once so popup clicks work via event delegation.
//
// Decisions: approved | wrong_class | unclear | no_sign
//   wrong_class carries corrected_class + original_class (see Item 4).
//   no_sign = false positive; excluded from the reviewed GeoJSON (Item 6).
window.SignGrader = (function () {
  const reviews = {};        // cluster_id -> {decision, corrected_class, original_class, note, reviewed_at, sign_class, lat, lon}
  let videoSource = null;

  const LOCAL_HOSTS = ['localhost', '127.0.0.1'];
  function isLocal() { return LOCAL_HOSTS.includes(window.location.hostname); }

  // Thumbnails default ON when served from the local Flask app, OFF elsewhere.
  let thumbnailsEnabled = isLocal();

  const DECISION_COLORS = {
    approved: '#38a169', wrong_class: '#e53e3e', unclear: '#805ad5', no_sign: '#a0aec0',
  };
  // Top-level decision buttons (No Sign lives inside the wrong-class dropdown).
  const DECISION_LABELS = { approved: '✓ Approve', wrong_class: '✗ Wrong class', unclear: '? Unclear' };

  // The 8 model classes offered as correction targets (Regulatory + Informational
  // collapsed per Item 1), plus the No-Sign option appended in the dropdown.
  const WV_CLASSES = [
    'Curves', 'HighwaySymbols', 'Informational/Regulatory', 'MileMarkers',
    'SpeedLimits', 'StopSigns', 'TrafficLights', 'Warnings',
  ];
  const NO_SIGN = '__no_sign__';

  // Display-layer merge of Regulatory + Informational (mirror of the backend
  // COLLAPSE_CLASSES in geojson_utils.py — Item 1).
  const COLLAPSE = { Regulatory: 'Informational/Regulatory', Informational: 'Informational/Regulatory' };
  function collapseClass(name) { return COLLAPSE[name] || name; }

  function setVideoSource(v) { if (v) videoSource = v; }
  function getVideoSource() { return videoSource; }
  function getReview(id) { return reviews[String(id)]; }
  function decisionOf(id) { const r = reviews[String(id)]; return r ? (r.decision || null) : null; }

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
    if (decision !== 'wrong_class') { delete r.corrected_class; delete r.original_class; }
    r.reviewed_at = new Date().toISOString();
    return r;
  }

  function setCorrectedClass(id, correctedClass, originalClass, meta) {
    const r = _ensure(id, meta);
    r.decision = 'wrong_class';
    r.corrected_class = correctedClass;
    r.original_class = collapseClass(originalClass != null ? originalClass : (meta && meta.sign_class));
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
    let approved = 0, wrong = 0, unclear = 0, no_sign = 0;
    for (const id in reviews) {
      const d = reviews[id].decision;
      if (d === 'approved') approved++;
      else if (d === 'wrong_class') wrong++;
      else if (d === 'unclear') unclear++;
      else if (d === 'no_sign') no_sign++;
    }
    return { reviewed: approved + wrong + unclear + no_sign, approved, wrong, unclear, no_sign };
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
    const false_positives = [];
    for (const id in reviews) {
      const r = reviews[id];
      out[id] = {
        sign_class: r.sign_class ?? null,
        lat: r.lat ?? null,
        lon: r.lon ?? null,
        decision: r.decision ?? null,
        corrected_class: r.corrected_class ?? null,
        original_class: r.original_class ?? null,
        note: r.note ?? null,
        reviewed_at: r.reviewed_at ?? null,
      };
      if (r.decision === 'wrong_class') {
        wrong_class_summary.push({
          cluster_id: id, predicted: r.original_class ?? r.sign_class ?? null,
          corrected: r.corrected_class ?? null,
          lat: r.lat ?? null, lon: r.lon ?? null, note: r.note ?? null,
        });
      } else if (r.decision === 'no_sign') {
        // Hard-negative candidates: model fired where there is no sign.
        false_positives.push({
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
    if (false_positives.length) payload.false_positives = false_positives;
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

  // Bake in-session review decisions into a GeoJSON FeatureCollection (Item 6).
  // approved -> needs_review:false, review_status:approved
  // wrong_class -> sign_class:=corrected, original_class added, review_status:corrected
  // no_sign -> excluded entirely (false positives don't belong in an inventory)
  // no decision -> passed through unchanged
  function buildReviewedGeoJson(features) {
    const out = [];
    (features || []).forEach(feat => {
      const props = Object.assign({}, feat.properties || {});
      const id = props.cluster_id;
      const r = (id != null) ? reviews[String(id)] : null;
      const decision = r && r.decision;
      if (decision === 'no_sign') return;
      if (decision === 'approved') {
        props.needs_review = false;
        props.review_status = 'approved';
      } else if (decision === 'wrong_class' && r.corrected_class) {
        props.original_class = r.original_class ?? props.sign_class;
        props.sign_class = r.corrected_class;
        props.review_status = 'corrected';
        props.needs_review = false;
      }
      out.push({ type: 'Feature', geometry: feat.geometry, properties: props });
    });
    return { type: 'FeatureCollection', features: out };
  }

  function reviewedFilename() {
    let stem = 'detections';
    if (videoSource) {
      stem = String(videoSource).replace(/\.[^./\\]+$/, '').replace(/[\\/]/g, '_') || 'detections';
    }
    return stem + '_reviewed.geojson';
  }

  // ── Popup grader section ────────────────────────────────────────────────────
  // meta: {sign_class, lat, lon, sighting_count, thumbnail_url}
  function graderSectionHtml(id, meta) {
    id = String(id);
    const r = reviews[id] || {};
    const cleanUrl = meta.thumbnail_url;
    const showThumb = thumbnailsEnabled && cleanUrl && (meta.sighting_count || 0) > 0;
    const bboxUrl = cleanUrl ? cleanUrl.replace(/\/$/, '') + '/bbox' : '';
    // Two thumbnail versions: clean crop (default) and YOLO bbox overlay (Item 8).
    const thumb = showThumb ? `
      <div class="grader-thumb-wrap">
        <img class="grader-thumb" src="${cleanUrl}" alt="detection frame"
             data-clean="${cleanUrl}" data-bbox="${bboxUrl}"
             onerror="this.style.display='none'" />
        <button class="grader-thumb-toggle" data-action="toggle-bbox"
                data-cluster-id="${id}" data-showing="clean">+ Box</button>
      </div>` : '';

    const decBtn = (val, action) =>
      `<button class="grader-btn${r.decision === val ? ' sel' : ''}" data-action="${action}" data-decision="${val}" data-cluster-id="${id}">${DECISION_LABELS[val]}</button>`;

    const predicted = collapseClass(meta.sign_class);
    const dropdownOpen = (r.decision === 'wrong_class' || r.decision === 'no_sign');
    const opts = WV_CLASSES.map(cls => {
      const disabled = cls === predicted;   // can't "correct" to the same class
      const checked = r.decision === 'wrong_class' && r.corrected_class === cls;
      return `<label class="grader-class-opt${disabled ? ' disabled' : ''}">
        <input type="radio" class="grader-class-radio" name="wc-${id}" data-cluster-id="${id}" data-class="${cls}"${disabled ? ' disabled' : ''}${checked ? ' checked' : ''}/>
        <span>${cls}</span></label>`;
    }).join('');
    const noSignOpt = `<label class="grader-class-opt no-sign-opt">
        <input type="radio" class="grader-class-radio" name="wc-${id}" data-cluster-id="${id}" data-class="${NO_SIGN}"${r.decision === 'no_sign' ? ' checked' : ''}/>
        <span>No Sign (false positive)</span></label>`;
    const dropdown = `
      <div class="grader-class-dropdown${dropdownOpen ? ' open' : ''}">
        <div class="grader-dropdown-label">Select correct class:</div>
        ${opts}${noSignOpt}
      </div>`;

    const noteVal = (r.note || '').replace(/"/g, '&quot;');
    return `
      ${thumb}
      <div class="grader-decision-label">Review decision:</div>
      <div class="grader-btns">${decBtn('approved', 'approve')}${decBtn('wrong_class', 'wrong_class')}${decBtn('unclear', 'unclear')}</div>
      ${dropdown}
      <div class="grader-note-row">
        <input class="grader-note" data-cluster-id="${id}" placeholder="Note…" value="${noteVal}" />
        <button class="grader-save" data-action="save" data-cluster-id="${id}">Save</button>
      </div>`;
  }

  // ── Popup interaction (event delegation) ─────────────────────────────────────
  // Re-sync a popup's button/dropdown visual state from the stored review.
  function refreshScope(scope, id) {
    if (!scope) return;
    const r = reviews[String(id)] || {};
    scope.querySelectorAll('.grader-btn').forEach(b => {
      b.classList.toggle('sel', b.dataset.decision === r.decision);
    });
    const dd = scope.querySelector('.grader-class-dropdown');
    if (dd) dd.classList.toggle('open', r.decision === 'wrong_class' || r.decision === 'no_sign');
  }

  function _dropdown(scope) { return scope && scope.querySelector('.grader-class-dropdown'); }
  function toggleDropdown(scope) {
    const dd = _dropdown(scope);
    if (!dd) return;
    const open = !dd.classList.contains('open');
    dd.classList.toggle('open', open);
    if (open) openWrongClass(scope);
  }
  function openWrongClass(scope) {
    const dd = _dropdown(scope);
    if (!dd) return;
    dd.classList.add('open');
    const target = dd.querySelector('.grader-class-radio:checked:not([disabled])')
                || dd.querySelector('.grader-class-radio:not([disabled])');
    if (target) target.focus();
  }
  function closeDropdown(scope) {
    const dd = _dropdown(scope);
    if (dd) dd.classList.remove('open');
  }

  // Wire ONE click/change listener on the map container. resolveMeta(id) returns
  // the grader meta for a cluster; hooks.onChange(id)/onSave(id) let the map
  // recolor markers, update the tally, and advance to the next review.
  function wireDelegation(container, resolveMeta, hooks) {
    hooks = hooks || {};
    container.addEventListener('click', (e) => {
      const actionEl = e.target.closest('[data-action]');
      const scope = e.target.closest('.grader-popup');
      if (!actionEl || !scope) return;
      const id = actionEl.dataset.clusterId;
      if (id == null) return;
      const meta = resolveMeta(id) || {};
      switch (actionEl.dataset.action) {
        case 'approve':
          decide(id, 'approved', meta); refreshScope(scope, id); hooks.onChange && hooks.onChange(id); break;
        case 'unclear':
          decide(id, 'unclear', meta); refreshScope(scope, id); hooks.onChange && hooks.onChange(id); break;
        case 'wrong_class':
          toggleDropdown(scope); break;
        case 'save': {
          const note = scope.querySelector('.grader-note');
          setNote(id, note ? note.value : '', meta);
          hooks.onChange && hooks.onChange(id);
          hooks.onSave && hooks.onSave(id);
          break;
        }
        case 'toggle-bbox': {
          const img = scope.querySelector('.grader-thumb');
          if (img) {
            const toBbox = actionEl.dataset.showing !== 'bbox';
            img.src = toBbox ? img.dataset.bbox : img.dataset.clean;
            img.style.display = '';
            actionEl.dataset.showing = toBbox ? 'bbox' : 'clean';
            actionEl.textContent = toBbox ? '- Box' : '+ Box';
          }
          break;
        }
      }
    });

    container.addEventListener('change', (e) => {
      const radio = e.target.closest('.grader-class-radio');
      if (!radio) return;
      const scope = e.target.closest('.grader-popup');
      const id = radio.dataset.clusterId;
      const meta = resolveMeta(id) || {};
      if (radio.dataset.class === NO_SIGN) decide(id, 'no_sign', meta);
      else setCorrectedClass(id, radio.dataset.class, meta.sign_class, meta);
      refreshScope(scope, id);
      hooks.onChange && hooks.onChange(id);
    });
  }

  return {
    isLocal, setVideoSource, getVideoSource, getReview, decisionOf,
    decide, setCorrectedClass, setNote, clear,
    decisionColor, DECISION_COLORS, DECISION_LABELS, WV_CLASSES, collapseClass, tally,
    thumbnailsOn, toggleThumbnails,
    exportReviews, importReviews, downloadJson, buildReviewedGeoJson, reviewedFilename,
    graderSectionHtml, wireDelegation, refreshScope, openWrongClass, closeDropdown,
  };
})();
