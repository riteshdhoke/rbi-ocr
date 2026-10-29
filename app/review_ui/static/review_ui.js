/* Shared vanilla JavaScript for the RBI complaint OCR reviewer pages (WS-G).
 *
 * Loaded by BOTH test_page.html and review.html as a *classic* script, not a
 * module: browsers block ES-module imports over file://, and one of the two
 * ways these pages must work is "double-click the file on a machine that has
 * no server". Everything hangs off the single global `RUI`.
 *
 * Air-gapped by construction: the only network call any function here makes is
 * to the API base URL the operator typed into the page. There is no CDN, no
 * font, no telemetry, no `eval`, and no `innerHTML` - every piece of envelope
 * data reaches the DOM through textContent, so a complaint containing "<script>"
 * is displayed, never executed.
 *
 * ES5 syntax on purpose (var, function, string concatenation). There is no
 * transpiler in this project and there never will be one.
 */
(function (global) {
  "use strict";

  /* The one absolute URL in this project's front end: a loopback default the
   * operator overwrites in the "API base" box. Never contacted unless the page
   * is opened from disk and the operator leaves it as-is. */
  var DEFAULT_API_BASE = "http://127.0.0.1:8000"; /* API-BASE-PLACEHOLDER */

  var STASH_KEY = "rbi-ocr-review-stash";
  var STASH_VERSION = 1;

  /* ------------------------------------------------------------------ DOM */

  function q(selector, root) {
    return (root || document).querySelector(selector);
  }

  function qa(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function el(tag, options) {
    var node = document.createElement(tag);
    var opts = options || {};
    if (opts.className) { node.className = opts.className; }
    if (opts.id) { node.id = opts.id; }
    if (opts.text !== undefined && opts.text !== null) {
      node.textContent = String(opts.text);
    }
    if (opts.title) { node.title = String(opts.title); }
    if (opts.attrs) {
      Object.keys(opts.attrs).forEach(function (name) {
        node.setAttribute(name, String(opts.attrs[name]));
      });
    }
    if (opts.children) {
      opts.children.forEach(function (child) {
        if (child) { node.appendChild(child); }
      });
    }
    return node;
  }

  function clear(node) {
    while (node && node.firstChild) { node.removeChild(node.firstChild); }
    return node;
  }

  function chip(label, className) {
    return el("li", { className: "chip" + (className ? " " + className : ""), text: label });
  }

  function chipList(labels, className) {
    var list = el("ul", { className: "chips" });
    (labels || []).forEach(function (label) {
      list.appendChild(chip(label, className));
    });
    return list;
  }

  function kvList(pairs) {
    var dl = el("dl", { className: "kv" });
    pairs.forEach(function (pair) {
      if (pair[1] === undefined) { return; }
      dl.appendChild(el("dt", { text: pair[0] }));
      dl.appendChild(el("dd", { text: pair[1] === null ? "—" : String(pair[1]) }));
    });
    return dl;
  }

  function setStatus(node, message, kind) {
    if (!node) { return; }
    node.className = "status" + (kind ? " " + kind : "");
    node.textContent = message || "";
  }

  /* -------------------------------------------------------------- format */

  function pct(value) {
    if (value === null || value === undefined || isNaN(value)) { return "—"; }
    return (Number(value) * 100).toFixed(1) + "%";
  }

  function tri(value) {
    if (value === true) { return "pass"; }
    if (value === false) { return "FAIL"; }
    return "—";
  }

  function orDash(value) {
    if (value === null || value === undefined || value === "") { return "—"; }
    return String(value);
  }

  function bboxText(bbox) {
    if (!bbox || bbox.length !== 8) { return "—"; }
    var b = bboxBounds(bbox);
    return "x" + Math.round(b.x) + " y" + Math.round(b.y) +
      " w" + Math.round(b.w) + " h" + Math.round(b.h);
  }

  /* ----------------------------------------------------------------- API */

  function normaliseBase(value) {
    var base = String(value || "").trim();
    if (!base) { base = defaultApiBase(); }
    return base.replace(/\/+$/, "");
  }

  function defaultApiBase() {
    /* Served by the API: talk to the origin that served this page. Opened from
     * disk (file:), there is no origin, so fall back to the loopback default. */
    if (global.location && /^https?:$/.test(global.location.protocol)) {
      return global.location.origin;
    }
    return DEFAULT_API_BASE;
  }

  function apiUrl(base, path) {
    return normaliseBase(base) + path;
  }

  function describeError(error, base) {
    var message = (error && error.message) ? error.message : String(error);
    var fromDisk = global.location && global.location.protocol === "file:";
    if (fromDisk && /fetch|network|failed/i.test(message)) {
      message += " — this page was opened from disk (file://), so the " +
        "browser sends Origin: null. Either serve this page from the API " +
        "itself, or allow the null origin in the API's CORS settings. " +
        "Checked base: " + normaliseBase(base);
    }
    return message;
  }

  function readResponse(response) {
    return response.text().then(function (body) {
      var payload = null;
      var parseError = null;
      if (body) {
        try { payload = JSON.parse(body); } catch (err) { parseError = err; }
      }
      if (!response.ok) {
        var detail = payload && (payload.detail || payload.message || payload.code);
        throw new Error("HTTP " + response.status + " " + response.statusText +
          (detail ? " — " + JSON.stringify(detail) : (body ? " — " + body.slice(0, 300) : "")));
      }
      if (parseError) {
        throw new Error("response was not JSON: " + body.slice(0, 200));
      }
      return payload;
    });
  }

  // `synthetic` is the provenance declaration the laptop profile requires
  // (PLAN.md invariant 6). It is sent on every upload, including as "false",
  // so the server sees an explicit statement rather than a missing field —
  // omitting it entirely is what made every upload through this page fail
  // under the default profile with a residency violation.
  function postDocument(base, file, fieldName, synthetic, requestedFields) {
    var form = new FormData();
    form.append(fieldName || "file", file, file.name);
    form.append("synthetic", synthetic ? "true" : "false");
    form.append("requested_fields", requestedFields || "");
    return global.fetch(apiUrl(base, "/v1/documents"), {
      method: "POST",
      body: form
    }).then(readResponse);
  }

  function fetchJob(base, jobId) {
    return global.fetch(apiUrl(base, "/v1/jobs/" + encodeURIComponent(jobId)))
      .then(readResponse);
  }

  function fetchHealth(base) {
    return global.fetch(apiUrl(base, "/healthz")).then(readResponse);
  }

  /* ------------------------------------------------------------ envelope */

  function looksLikeEnvelope(value) {
    return !!(value && typeof value === "object" &&
      value.bounce_decision && value.schema_version);
  }

  /* The API may hand back the envelope itself or wrap it; accept both rather
   * than guessing one and rendering an empty page against the other. */
  function unwrapEnvelope(payload, depth) {
    var level = depth || 0;
    if (looksLikeEnvelope(payload)) { return payload; }
    if (!payload || typeof payload !== "object" || level > 3) { return null; }
    var keys = ["envelope", "result", "result_envelope", "data", "document"];
    for (var i = 0; i < keys.length; i += 1) {
      if (payload[keys[i]]) {
        var found = unwrapEnvelope(payload[keys[i]], level + 1);
        if (found) { return found; }
      }
    }
    return null;
  }

  function isBounced(envelope) {
    return !!(envelope && envelope.bounce_decision &&
      envelope.bounce_decision.decision === "bounce");
  }

  function fieldRows(envelope) {
    var doc = envelope && envelope.document;
    if (!doc || !doc.fields) { return []; }
    return Object.keys(doc.fields).sort().map(function (name) {
      return { name: name, field: doc.fields[name] };
    });
  }

  function isFlagged(field) {
    if (!field) { return false; }
    if (field.validator_passed === false) { return true; }
    return !!(field.flags && field.flags.length);
  }

  function flaggedRows(envelope) {
    return fieldRows(envelope).filter(function (row) { return isFlagged(row.field); });
  }

  /* The published envelope carries no OCR lines: app/contracts/envelope.py puts
   * line_count on PageResult and nothing else, and verbatim OCR text reaches a
   * consumer through complaint_summary.spans. Read the spans, and also accept a
   * deployment that chose to attach pages[].lines. */
  function ocrLines(envelope) {
    var out = [];
    var pages = (envelope && envelope.pages) || [];
    pages.forEach(function (page) {
      var lines = page.lines || page.ocr_lines || [];
      lines.forEach(function (line) {
        out.push({
          page_no: page.page_no,
          line_index: line.line_index,
          text: line.text,
          bbox: line.bbox,
          origin: "page.lines"
        });
      });
    });
    var summary = envelope && envelope.document && envelope.document.complaint_summary;
    if (summary && summary.spans) {
      summary.spans.forEach(function (span) {
        var evidence = span.evidence || {};
        out.push({
          page_no: evidence.page_no,
          line_index: evidence.span ? evidence.span.line_index : null,
          text: span.text,
          bbox: evidence.bbox,
          origin: "complaint_summary.spans"
        });
      });
    }
    return out;
  }

  function evidenceBoxes(envelope, pageNo) {
    var boxes = [];
    fieldRows(envelope).forEach(function (row) {
      var evidence = row.field && row.field.evidence;
      if (!evidence || !evidence.bbox) { return; }
      if (pageNo && evidence.page_no !== pageNo) { return; }
      boxes.push({
        bbox: evidence.bbox,
        label: row.name,
        flagged: isFlagged(row.field),
        page_no: evidence.page_no
      });
    });
    return boxes;
  }

  /* ------------------------------------------------------------ geometry */

  function bboxBounds(bbox) {
    var xs = [bbox[0], bbox[2], bbox[4], bbox[6]];
    var ys = [bbox[1], bbox[3], bbox[5], bbox[7]];
    var x = Math.min.apply(null, xs);
    var y = Math.min.apply(null, ys);
    return { x: x, y: y, w: Math.max.apply(null, xs) - x, h: Math.max.apply(null, ys) - y };
  }

  var STROKE_PLAIN = "rgba(23, 69, 124, 0.95)";
  var STROKE_FLAGGED = "rgba(210, 135, 43, 0.98)";
  var STROKE_ACTIVE = "rgba(181, 53, 42, 1)";
  var LABEL_INK = "rgba(255, 255, 255, 0.98)";

  function strokeFor(box, activeLabel) {
    if (activeLabel && box.label === activeLabel) { return STROKE_ACTIVE; }
    return box.flagged ? STROKE_FLAGGED : STROKE_PLAIN;
  }

  function drawPolygon(ctx, bbox, scale, colour, label, lineWidth) {
    ctx.beginPath();
    ctx.moveTo(bbox[0] * scale, bbox[1] * scale);
    ctx.lineTo(bbox[2] * scale, bbox[3] * scale);
    ctx.lineTo(bbox[4] * scale, bbox[5] * scale);
    ctx.lineTo(bbox[6] * scale, bbox[7] * scale);
    ctx.closePath();
    ctx.lineWidth = lineWidth || 2;
    ctx.strokeStyle = colour;
    ctx.stroke();
    if (label) {
      var bounds = bboxBounds(bbox);
      var x = bounds.x * scale;
      var y = Math.max(11, bounds.y * scale - 3);
      ctx.font = "11px Consolas, monospace";
      var width = ctx.measureText(label).width + 6;
      ctx.fillStyle = colour;
      ctx.fillRect(x, y - 11, width, 13);
      ctx.fillStyle = LABEL_INK;
      ctx.fillText(label, x + 3, y - 1);
    }
  }

  /* Never trust a measured container width below this: a panel that has not
   * been laid out yet (or is in a hidden pane) reports a few pixels, and a
   * 32-pixel-wide evidence crop is not evidence. */
  var MIN_CANVAS_WIDTH = 320;

  function availableWidth(canvas, override, fallback) {
    var measured = canvas.parentNode ? canvas.parentNode.clientWidth : 0;
    return Math.max(MIN_CANVAS_WIDTH, override || measured || fallback);
  }

  /* Draw the whole page with its evidence boxes. Returns the scale used, so a
   * caller can map a click back into page coordinates. */
  function drawPage(canvas, image, boxes, activeLabel, maxWidth) {
    var limit = availableWidth(canvas, maxWidth, 900);
    var scale = Math.min(1, limit / image.naturalWidth);
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    (boxes || []).forEach(function (box) {
      if (!box.bbox || box.bbox.length !== 8) { return; }
      drawPolygon(ctx, box.bbox, scale, strokeFor(box, activeLabel), box.label, 2);
    });
    return scale;
  }

  /* Draw a padded crop around one bbox - the "field beside its evidence" view. */
  function drawCrop(canvas, image, bbox, options) {
    var opts = options || {};
    var pad = opts.pad === undefined ? 28 : opts.pad;
    var bounds = bboxBounds(bbox);
    var sx = Math.max(0, bounds.x - pad);
    var sy = Math.max(0, bounds.y - pad);
    var sw = Math.min(image.naturalWidth - sx, bounds.w + pad * 2);
    var sh = Math.min(image.naturalHeight - sy, bounds.h + pad * 2);
    if (sw <= 0 || sh <= 0) { return 0; }
    var limit = availableWidth(canvas, opts.maxWidth, 640);
    var scale = Math.min(3, limit / sw);
    canvas.width = Math.max(1, Math.round(sw * scale));
    canvas.height = Math.max(1, Math.round(sh * scale));
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo((bbox[0] - sx) * scale, (bbox[1] - sy) * scale);
    ctx.lineTo((bbox[2] - sx) * scale, (bbox[3] - sy) * scale);
    ctx.lineTo((bbox[4] - sx) * scale, (bbox[5] - sy) * scale);
    ctx.lineTo((bbox[6] - sx) * scale, (bbox[7] - sy) * scale);
    ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = STROKE_ACTIVE;
    ctx.stroke();
    return scale;
  }

  /* --------------------------------------------------------------- image */

  function isPdf(file) {
    return !!file && (file.type === "application/pdf" ||
      /\.pdf$/i.test(file.name || ""));
  }

  function imageFromDataUrl(dataUrl) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () { resolve(image); };
      image.onerror = function () { reject(new Error("the browser could not decode that image")); };
      image.src = dataUrl;
    });
  }

  /* Read the uploaded file in the browser. The service never sends the page
   * image back, so this is the only way the overlay can exist without a second
   * round trip - and it keeps the image on the reviewer's machine. */
  function loadImageFile(file) {
    if (!file) { return Promise.reject(new Error("no file chosen")); }
    if (isPdf(file)) {
      return Promise.reject(new Error(
        "PDF preview is not available: rasterising a PDF needs a library, and " +
        "this page ships no third-party code (the deployment is air-gapped). " +
        "Upload a page image to see evidence boxes, or read the coordinates in " +
        "the fields table."));
    }
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onerror = function () { reject(new Error("could not read that file")); };
      reader.onload = function () {
        imageFromDataUrl(reader.result).then(function (image) {
          resolve({ image: image, dataUrl: reader.result });
        }, reject);
      };
      reader.readAsDataURL(file);
    });
  }

  /* --------------------------------------------------------------- stash */

  /* Hand-off between the two pages.
   *
   * sessionStorage is the primary carrier: it dies with the tab, and the
   * payload is a complaint - name, address, account number. localStorage is
   * written as well, because the review page opens in a *new tab* and a new
   * tab does not always inherit the opener's sessionStorage (notably over
   * file://, where the origin is opaque). The localStorage copy is a one-shot:
   * loadStash() deletes it the moment it reads it, so complainant data does
   * not sit on the reviewer's disk after the hand-over.
   *
   * Failure here is never fatal - the review page falls back to asking for the
   * envelope and the image directly. */
  function writeStash(record) {
    var wrote = false;
    var serialised = JSON.stringify(record);
    try {
      global.sessionStorage.setItem(STASH_KEY, serialised);
      wrote = true;
    } catch (err) { /* quota or a disabled store; try the other one */ }
    try {
      global.localStorage.setItem(STASH_KEY, serialised);
      wrote = true;
    } catch (err) { /* same */ }
    return wrote;
  }

  function saveStash(payload) {
    var record = {
      version: STASH_VERSION,
      saved_at: new Date().toISOString(),
      api_base: payload.apiBase || null,
      envelope: payload.envelope || null,
      image_data_url: payload.imageDataUrl || null,
      file_name: payload.fileName || null
    };
    if (writeStash(record)) {
      return { ok: true, withImage: !!record.image_data_url };
    }
    /* Almost always the image: a page scan as a data URL is megabytes and the
     * quota is a few. Hand over the envelope alone rather than nothing. */
    record.image_data_url = null;
    if (writeStash(record)) {
      return {
        ok: true,
        withImage: false,
        note: "the page image was too large to hand over; choose it again on the review page"
      };
    }
    return { ok: false, withImage: false, note: "browser storage is unavailable" };
  }

  function readStore(store) {
    try {
      var raw = store.getItem(STASH_KEY);
      if (!raw) { return null; }
      var record = JSON.parse(raw);
      return record && record.version === STASH_VERSION ? record : null;
    } catch (err) {
      return null;
    }
  }

  function loadStash() {
    var record = null;
    try { record = readStore(global.sessionStorage); } catch (err) { record = null; }
    if (record) { return record; }
    try {
      record = readStore(global.localStorage);
      if (record) {
        /* One-shot: consume it, and keep a tab-local copy so reloading this
         * page still works. */
        global.localStorage.removeItem(STASH_KEY);
        try {
          global.sessionStorage.setItem(STASH_KEY, JSON.stringify(record));
        } catch (err) { /* the tab-local copy is a convenience, not a need */ }
      }
    } catch (err) {
      record = null;
    }
    return record;
  }

  function clearStash() {
    try { global.sessionStorage.removeItem(STASH_KEY); } catch (err) { /* nothing to clear */ }
    try { global.localStorage.removeItem(STASH_KEY); } catch (err) { /* nothing to clear */ }
  }

  /* ------------------------------------------------------------ keyboard */

  function isTypingTarget(node) {
    if (!node) { return false; }
    var tag = node.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || node.isContentEditable;
  }

  /* Keyboard-first (PRD P1-1): every review action has a key, and the keys are
   * ignored while the reviewer is typing a correction, except Escape. */
  function bindKeys(map) {
    document.addEventListener("keydown", function (event) {
      if (event.ctrlKey || event.metaKey || event.altKey) { return; }
      var key = event.key;
      if (isTypingTarget(event.target) && key !== "Escape") { return; }
      var handler = map[key] || map[String(key).toLowerCase()];
      if (!handler) { return; }
      event.preventDefault();
      handler(event);
    });
  }

  /* ------------------------------------------------------------ download */

  function downloadJSON(filename, payload) {
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var anchor = el("a", { attrs: { download: filename } });
    anchor.href = url;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  /* -------------------------------------------------------------- render */

  function renderBanner(node, envelope) {
    clear(node);
    if (!envelope) { return; }
    var decision = envelope.bounce_decision || {};
    var bounced = decision.decision === "bounce";
    var banner = el("div", { className: "banner " + (bounced ? "bounce" : (envelope.escalation_flag ? "warn" : "ok")) });
    if (bounced) {
      banner.appendChild(el("strong", { text: "Document bounced — nothing was extracted" }));
      banner.appendChild(el("div", { className: "code", text: "reason: " + orDash(decision.reason) }));
      if (decision.detail) {
        banner.appendChild(el("div", { className: "note", text: "detail (internal, not sent to the complainant): " + decision.detail }));
      }
      banner.appendChild(el("div", {
        className: "note",
        text: "app/replies renders the bilingual auto-reply for this reason code."
      }));
    } else {
      banner.appendChild(el("strong", {
        text: envelope.escalation_flag
          ? "Accepted — needs a human: a page escalated or a field is flagged"
          : "Accepted"
      }));
      banner.appendChild(el("div", {
        className: "note",
        text: "result_version " + orDash(envelope.result_version) +
          (envelope.is_final ? " (final)" : " (not final — escalated pages may still land)")
      }));
    }
    node.appendChild(banner);
  }

  function renderOverview(node, envelope) {
    clear(node);
    var processing = envelope.processing || {};
    var doc = envelope.document || {};
    node.appendChild(kvList([
      ["document_id", envelope.document_id],
      ["job_id", orDash(envelope.job_id)],
      ["schema_version", envelope.schema_version],
      ["result_version", envelope.result_version],
      ["is_final", String(!!envelope.is_final)],
      ["provider_profile", envelope.provider_profile],
      ["created_at", envelope.created_at],
      ["document_class", orDash(doc.document_class)],
      ["class confidence", doc.document_class_confidence === undefined ? "—" : pct(doc.document_class_confidence)],
      ["pages", (envelope.pages || []).length],
      ["fields", fieldRows(envelope).length],
      ["flagged fields", flaggedRows(envelope).length],
      ["duration_ms", orDash(processing.duration_ms)],
      ["pages escalated", orDash(processing.pages_escalated)],
      ["tokens used", orDash(processing.tokens_used)]
    ]));
    if (doc.supporting_doc_tags && doc.supporting_doc_tags.length) {
      node.appendChild(el("p", { className: "note", text: "supporting_doc_tags (categorisation stub — NOT production):" }));
      node.appendChild(chipList(doc.supporting_doc_tags));
    }
    if (envelope.warnings && envelope.warnings.length) {
      node.appendChild(el("p", { className: "note", text: "warnings:" }));
      node.appendChild(chipList(envelope.warnings, "bad"));
    }
  }

  function meterClass(score) {
    if (score === null || score === undefined) { return "meter"; }
    if (score < 0.35) { return "meter low"; }
    if (score < 0.6) { return "meter mid"; }
    return "meter";
  }

  function renderPages(node, envelope) {
    clear(node);
    var pages = envelope.pages || [];
    if (!pages.length) {
      node.appendChild(el("p", { className: "note", text: "no page records" }));
      return;
    }
    pages.forEach(function (page) {
      var card = el("div", { className: "page-card" });
      card.appendChild(el("h3", { text: "Page " + page.page_no + " — " + orDash(page.quality_verdict) }));
      var meter = el("div", { className: meterClass(page.quality_score) });
      var bar = el("span");
      bar.style.width = Math.round(Math.max(0, Math.min(1, page.quality_score || 0)) * 100) + "%";
      meter.appendChild(bar);
      card.appendChild(meter);
      card.appendChild(kvList([
        ["quality_score", page.quality_score === undefined ? "—" : pct(page.quality_score)],
        ["ocr_tier", orDash(page.ocr_tier)],
        ["line_count", orDash(page.line_count)],
        ["primary_script", orDash(page.primary_script)],
        ["primary_language", orDash(page.primary_language)],
        ["is_code_mixed", String(!!page.is_code_mixed)],
        ["escalated", String(!!page.escalated)],
        ["shadow_sampled", String(!!page.shadow_sampled)]
      ]));
      if (page.quality_flags && page.quality_flags.length) {
        card.appendChild(chipList(page.quality_flags, "flagged"));
      }
      if (page.repairs_applied && page.repairs_applied.length) {
        card.appendChild(chipList(page.repairs_applied));
      }
      node.appendChild(card);
    });
  }

  var FIELD_COLUMNS = [
    "field", "value", "value_en", "confidence", "source",
    "validator_passed", "flags", "evidence"
  ];

  function renderFields(node, envelope, onShow, selectedRows) {
    clear(node);
    var rows = selectedRows || fieldRows(envelope);
    if (!rows.length) {
      node.appendChild(el("p", { className: "note", text: isBounced(envelope) ? "Document bounced; no fields were extracted." : "No values found in this group." }));
      return;
    }
    var table = el("table");
    var head = el("tr");
    FIELD_COLUMNS.forEach(function (name) { head.appendChild(el("th", { text: name })); });
    table.appendChild(el("thead", { children: [head] }));
    var body = el("tbody");
    rows.forEach(function (row) {
      var field = row.field;
      var tr = el("tr", { className: isFlagged(field) ? "flagged" : "" });
      tr.appendChild(el("td", { text: row.name }));
      tr.appendChild(el("td", { text: orDash(field.value) }));
      tr.appendChild(el("td", { text: orDash(field.value_en) }));
      tr.appendChild(el("td", { className: "mono", text: pct(field.confidence) }));
      tr.appendChild(el("td", { className: "mono", text: orDash(field.source) }));
      tr.appendChild(el("td", { className: "mono", text: tri(field.validator_passed) }));
      var flagCell = el("td");
      flagCell.appendChild(chipList(field.flags || [], "flagged"));
      tr.appendChild(flagCell);
      var evidence = field.evidence || {};
      var evidenceCell = el("td", { className: "mono" });
      evidenceCell.appendChild(el("div", {
        text: "p" + orDash(evidence.page_no) + " " + bboxText(evidence.bbox)
      }));
      if (onShow && evidence.bbox) {
        var button = el("button", { className: "secondary", text: "show" });
        button.type = "button";
        button.addEventListener("click", function () { onShow(row); });
        evidenceCell.appendChild(button);
      }
      tr.appendChild(evidenceCell);
      body.appendChild(tr);
    });
    table.appendChild(body);
    node.appendChild(el("div", { className: "table-scroll", children: [table] }));
  }

  function renderLines(node, envelope) {
    clear(node);
    var lines = ocrLines(envelope);
    if (!lines.length) {
      node.appendChild(el("p", {
        className: "note",
        text: "No OCR text in this envelope. The published contract carries " +
          "line_count per page and verbatim text only through " +
          "complaint_summary.spans; this document has neither."
      }));
      return;
    }
    var summary = envelope.document && envelope.document.complaint_summary;
    if (summary && summary.text) {
      node.appendChild(el("p", { className: "note", text: "complaint_summary (verbatim concatenation of the selected spans):" }));
      node.appendChild(el("pre", { className: "raw", text: summary.text }));
    }
    var list = el("ol", { className: "lines" });
    lines.forEach(function (line) {
      var item = el("li");
      item.appendChild(el("span", { text: orDash(line.text) }));
      item.appendChild(el("span", {
        className: "loc",
        text: "  [p" + orDash(line.page_no) + " line " + orDash(line.line_index) + " · " + line.origin + "]"
      }));
      list.appendChild(item);
    });
    node.appendChild(list);
  }

  global.RUI = {
    DEFAULT_API_BASE: DEFAULT_API_BASE,
    STASH_KEY: STASH_KEY,
    FIELD_COLUMNS: FIELD_COLUMNS,
    q: q,
    qa: qa,
    el: el,
    clear: clear,
    chip: chip,
    chipList: chipList,
    kvList: kvList,
    setStatus: setStatus,
    pct: pct,
    tri: tri,
    orDash: orDash,
    bboxText: bboxText,
    defaultApiBase: defaultApiBase,
    normaliseBase: normaliseBase,
    apiUrl: apiUrl,
    describeError: describeError,
    postDocument: postDocument,
    fetchJob: fetchJob,
    fetchHealth: fetchHealth,
    unwrapEnvelope: unwrapEnvelope,
    isBounced: isBounced,
    fieldRows: fieldRows,
    flaggedRows: flaggedRows,
    isFlagged: isFlagged,
    ocrLines: ocrLines,
    evidenceBoxes: evidenceBoxes,
    bboxBounds: bboxBounds,
    drawPage: drawPage,
    drawCrop: drawCrop,
    isPdf: isPdf,
    loadImageFile: loadImageFile,
    imageFromDataUrl: imageFromDataUrl,
    saveStash: saveStash,
    loadStash: loadStash,
    clearStash: clearStash,
    bindKeys: bindKeys,
    downloadJSON: downloadJSON,
    renderBanner: renderBanner,
    renderOverview: renderOverview,
    renderPages: renderPages,
    renderFields: renderFields,
    renderLines: renderLines
  };
})(window);
