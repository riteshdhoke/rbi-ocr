# 11 — Edit the upload page and shared JavaScript

## A. `app/review_ui/static/review_ui.js`

Replace the entire `postDocument` function with:

```javascript
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
```

In the existing `renderFields` function, make these three edits only:

1. Change the signature to:

```javascript
  function renderFields(node, envelope, onShow, selectedRows) {
```

2. Change `var rows = fieldRows(envelope);` to:

```javascript
    var rows = selectedRows || fieldRows(envelope);
```

3. In the empty-row message, use:

```javascript
text: isBounced(envelope) ? "Document bounced; no fields were extracted." : "No values found in this group."
```

Leave its table and provenance rendering unchanged. Existing review-page callers
still work without `selectedRows`.

## B. `app/review_ui/static/test_page.html`: new input

Find `<p class="status" id="doc-status"></p>`. Replace that one line with:

```html
    <label class="field">Fields to extract (comma separated)
      <input type="text" id="requested-fields" maxlength="4096"
        placeholder="name of complainant, mobile, account number" autocomplete="off">
    </label>
    <p class="hint">Optional. Requested values appear under Extracted fields; additional discoveries appear under Other fields.</p>
    <p class="status" id="doc-status"></p>
```

## C. Same HTML file: output groups

Find `<div id="fields"></div>`. Replace that one line with:

```html
    <p id="missing-fields" class="note"></p>
    <div id="fields"></div>
    <h3>Other fields</h3>
    <div id="other-fields"></div>
```

## D. Same HTML file: render function

Inside its script, replace the entire `renderAll` function with:

```javascript
  function renderAll() {
    var envelope = state.envelope;
    RUI.renderBanner(RUI.q("#banner"), envelope);
    if (!envelope) { return; }
    RUI.renderOverview(RUI.q("#overview"), envelope);
    RUI.renderPages(RUI.q("#pages"), envelope);
    function showField(row) {
      state.activeField = row.originalName || row.name;
      var evidence = row.field.evidence || {};
      if (evidence.page_no) { state.pageNo = evidence.page_no; syncPageSelect(); }
      drawImage();
      canvas.scrollIntoView({ block: "center" });
    }
    var doc = envelope.document || {};
    var allFields = doc.fields || {};
    var requested = envelope.requested_fields || [];
    var selected = [];
    requested.forEach(function (name) {
      ((doc.extracted_fields || {})[name] || []).forEach(function (field, index) {
        var original = Object.keys(allFields).filter(function (key) {
          var candidate = allFields[key];
          return candidate.value === field.value && JSON.stringify(candidate.evidence) === JSON.stringify(field.evidence);
        })[0];
        selected.push({ name: name + (index ? "_" + (index + 1) : ""), originalName: original, field: field });
      });
    });
    RUI.q("#missing-fields").textContent = requested.length
      ? ((doc.missing_fields || []).length ? "Not found: " + doc.missing_fields.join(", ") : "All requested fields found.")
      : "No field list supplied. All discoveries appear under Other fields.";
    if (envelope.bounce_decision && envelope.bounce_decision.decision === "bounce") {
      RUI.q("#missing-fields").textContent = "Document bounced; no values were extracted.";
    }
    RUI.renderFields(RUI.q("#fields"), envelope, showField, selected);
    var other = doc.other_fields || allFields;
    var otherRows = Object.keys(other).sort().map(function (name) { return { name: name, field: other[name] }; });
    RUI.renderFields(RUI.q("#other-fields"), envelope, showField, otherRows);
    RUI.renderLines(RUI.q("#lines"), envelope);
    RUI.q("#raw").textContent = JSON.stringify(state.raw, null, 2);
    buildPageSelect();
    drawImage();
  }
```

## E. Same HTML file: loaded result state

Replace the entire `setEnvelope` function with:

```javascript
  function setEnvelope(payload, note) {
    var envelope = RUI.unwrapEnvelope(payload);
    state.raw = payload;
    if (!envelope) {
      state.envelope = null;
      RUI.setStatus(docStatus,
        "The response did not contain a result envelope (looked for schema_version + bounce_decision). " +
        "See the raw response below.", "error");
      RUI.q("#raw").textContent = JSON.stringify(payload, null, 2);
      return;
    }
    state.envelope = envelope;
    RUI.q("#requested-fields").value = (envelope.requested_fields || []).join(", ");
    state.activeField = null;
    state.pageNo = null;
    RUI.setStatus(docStatus, note || "Envelope loaded.", "");
    renderAll();
  }
```

## F. Same HTML file: upload listener

Replace the complete upload-button listener, stopping before the job-button
listener, with:

```javascript
  RUI.q("#upload-btn").addEventListener("click", function () {
    var file = RUI.q("#file-input").files[0];
    if (!file) {
      RUI.setStatus(docStatus, "Choose a file first.", "error");
      return;
    }
    RUI.setStatus(docStatus, "uploading " + file.name + "…", "busy");
    var synthetic = RUI.q("#synthetic-input").checked;
    RUI.postDocument(apiBase.value, file, fieldName.value, synthetic, RUI.q("#requested-fields").value).then(function (payload) {
      setEnvelope(payload, "Processed " + file.name + ".");
    }, function (error) {
      RUI.setStatus(docStatus, RUI.describeError(error, apiBase.value), "error");
    });
  });
```

## G. `app/review_ui/__init__.py`: cross-platform filename check

Inside `static_path`, replace only the first `if` condition with:

```python
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
```

Keep the existing `raise ValueError(...)` body. This also rejects Windows-style
separators when the application is verified on a POSIX machine.

Final UI check happens after restarting the API in step 12. Hard-refresh the
browser so it loads the updated shared JavaScript; do not open the HTML as file://.
