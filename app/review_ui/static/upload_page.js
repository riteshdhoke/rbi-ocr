/* Upload form and result display. */
(function () {
  "use strict";

  if (!window.RUI || typeof window.RUI.fetchFieldGroups !== "function") {
    document.querySelector("#doc-status").textContent =
      "The upload scripts could not load. Reload the page before extracting.";
    return;
  }

  var state = {
    envelope: null,
    raw: null,
    image: null,
    dataUrl: null,
    fileName: null,
    activeField: null,
    pageNo: null
  };

  var apiBase = RUI.q("#api-base");
  var groupOptions = RUI.q("#group-options");
  var groupStatus = RUI.q("#group-status");
  var overrideModel = RUI.q("#override-model");
  var groupRequest = 0;
  var groupBase;
  var uploadButton = document.querySelector("#upload-btn");
  var uploading = false;
  var loadingGroups = false;
  var connStatus = RUI.q("#conn-status");
  var docStatus = RUI.q("#doc-status");
  var imageStatus = RUI.q("#image-status");
  var canvas = RUI.q("#page-canvas");
  var pageSelect = RUI.q("#page-select");
  var showBoxes = RUI.q("#show-boxes");

  apiBase.value = RUI.defaultApiBase();

  /* ------------------------------------------------------------- render */

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
      ? ((doc.missing_fields || []).length ? "Not extracted reliably: " + doc.missing_fields.join(", ") : "All requested fields found.")
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

  function buildPageSelect() {
    var pages = (state.envelope && state.envelope.pages) || [];
    RUI.clear(pageSelect);
    pages.forEach(function (page) {
      var option = RUI.el("option", { text: "Page " + page.page_no });
      option.value = String(page.page_no);
      pageSelect.appendChild(option);
    });
    if (!state.pageNo && pages.length) { state.pageNo = pages[0].page_no; }
    syncPageSelect();
  }

  function syncPageSelect() {
    if (state.pageNo) { pageSelect.value = String(state.pageNo); }
  }

  function drawImage() {
    if (!state.image) {
      RUI.setStatus(imageStatus, state.envelope
        ? "No page image in this browser. Choose the image file above to draw evidence boxes."
        : "", "busy");
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    var boxes = showBoxes.checked
      ? RUI.evidenceBoxes(state.envelope, state.pageNo)
      : [];
    RUI.drawPage(canvas, state.image, boxes, state.activeField);
    RUI.setStatus(imageStatus,
      state.image.naturalWidth + " × " + state.image.naturalHeight + " px" +
      (boxes.length ? " · " + boxes.length + " evidence box(es) on page " + state.pageNo : " · no evidence boxes on this page"),
      "");
  }

  function renderCompact(payload) {
    var overview = RUI.q("#compact-overview");
    var fields = RUI.q("#compact-fields");
    RUI.clear(overview);
    RUI.clear(fields);
    overview.appendChild(RUI.el("p", { text: "Job ID: " + payload.job_id }));
    overview.appendChild(RUI.el("p", { text: "Pages processed: " +
      (payload.pages_processed.join(", ") || "None") }));
    var names = Object.keys(payload.extracted_data);
    if (!names.length) {
      fields.appendChild(RUI.el("p", { text: "No fields extracted." }));
      return;
    }
    var table = RUI.el("table");
    var header = table.createTHead().insertRow();
    ["Field", "Document label", "Value", "Confidence", "Source page", "Validation"].forEach(function (title) {
      var heading = RUI.el("th", { text: title });
      heading.scope = "col";
      header.appendChild(heading);
    });
    var body = table.createTBody();
    names.forEach(function (name) {
      var field = payload.extracted_data[name];
      var validation = field.validation_status || {};
      var status = validation.passed === true ? "Passed" :
        validation.passed === false ? "Failed" : "No deterministic check";
      var detail = [status, validation.method].concat(validation.flags || []).filter(Boolean);
      var row = body.insertRow();
      [name, field.label, field.value, RUI.pct(field.confidence),
        field.source_page, detail.join(" · ")].forEach(function (value) {
        row.insertCell().textContent = RUI.orDash(value);
      });
    });
    fields.appendChild(table);
  }

  function setResult(payload, note) {
    var envelope = RUI.unwrapEnvelope(payload);
    var compact = payload && typeof payload.job_id === "string" &&
      Array.isArray(payload.pages_processed) && payload.extracted_data &&
      typeof payload.extracted_data === "object" && !Array.isArray(payload.extracted_data);
    state.raw = payload;
    state.envelope = envelope;
    state.activeField = null;
    state.pageNo = null;
    RUI.q("#raw").textContent = JSON.stringify(payload, null, 2);
    RUI.q("#legacy-result").hidden = !envelope;
    RUI.q("#compact-result").hidden = !compact;
    if (compact) {
      RUI.q("#job-id").value = payload.job_id;
      renderCompact(payload);
    } else if (envelope) {
      renderAll();
    } else {
      RUI.setStatus(docStatus, "The response did not contain an extraction result or envelope.", "error");
      return;
    }
    RUI.setStatus(docStatus, note || "Result loaded.", "");
  }

  function syncGroupModels() {
    RUI.qa("select[data-group-model]", groupOptions).forEach(function (select) {
      var checkbox = select.closest(".row").querySelector("input");
      select.disabled = !checkbox.checked || !!overrideModel.value;
    });
  }

  function loadGroups() {
    var base = RUI.normaliseBase(apiBase.value);
    if (base === groupBase) { return; }
    groupBase = base;
    var request = ++groupRequest;
    var previous = new Map(RUI.qa("input", groupOptions).map(function (input) {
      return [input.value, {
        checked: input.checked,
        model: input.closest(".row").querySelector("select").value
      }];
    }));
    loadingGroups = true;
    uploadButton.disabled = true;
    RUI.setStatus(groupStatus, "Loading field groups…", "busy");
    RUI.fetchFieldGroups(base).then(function (groups) {
      if (request !== groupRequest) { return; }
      RUI.clear(groupOptions);
      Object.keys(groups).sort().forEach(function (name) {
        var row = RUI.el("div", { className: "row group-row" });
        var label = RUI.el("label");
        var checkbox = RUI.el("input");
        checkbox.type = "checkbox";
        checkbox.value = name;
        checkbox.addEventListener("change", syncGroupModels);
        label.appendChild(checkbox);
        label.appendChild(RUI.el("span", { text: " " + name }));
        row.appendChild(label);
        var modelLabel = RUI.el("label", { className: "field", text: "Model for " + name });
        var select = RUI.el("select");
        select.dataset.groupModel = name;
        [["", "Inherit"], ["cpu", "CPU"], ["gpu", "GPU"]].forEach(function (entry) {
          var option = RUI.el("option", { text: entry[1] });
          option.value = entry[0];
          select.appendChild(option);
        });
        var selection = previous.get(name);
        if (selection) {
          checkbox.checked = selection.checked;
          select.value = selection.model;
        }
        modelLabel.appendChild(select);
        row.appendChild(modelLabel);
        row.appendChild(RUI.el("span", { className: "hint", text: groups[name].join(", ") }));
        groupOptions.appendChild(row);
      });
      syncGroupModels();
      RUI.setStatus(groupStatus, Object.keys(groups).length ? "" : "No predefined groups available.", "");
    }, function () {
      if (request !== groupRequest) { return; }
      groupBase = undefined;
      RUI.clear(groupOptions);
      RUI.setStatus(groupStatus, "Could not load field groups. Check the API base or enter individual fields.", "error");
    }).finally(function () {
      if (request === groupRequest) {
        loadingGroups = false;
        uploadButton.disabled = uploading;
      }
    });
  }

  function useImageFile(file) {
    if (!file) { return; }
    state.fileName = file.name;
    RUI.loadImageFile(file).then(function (loaded) {
      state.image = loaded.image;
      state.dataUrl = loaded.dataUrl;
      drawImage();
    }, function (error) {
      state.image = null;
      state.dataUrl = null;
      RUI.setStatus(imageStatus, error.message, "error");
    });
  }

  /* -------------------------------------------------------------- wiring */

  RUI.q("#health-btn").addEventListener("click", function () {
    RUI.setStatus(connStatus, "checking…", "busy");
    RUI.fetchHealth(apiBase.value).then(function (payload) {
      RUI.setStatus(connStatus, "healthy: " + JSON.stringify(payload), "");
    }, function (error) {
      RUI.setStatus(connStatus, RUI.describeError(error, apiBase.value), "error");
    });
  });

  apiBase.addEventListener("change", loadGroups);
  overrideModel.addEventListener("change", syncGroupModels);

  RUI.q("#upload-form").addEventListener("submit", function (event) {
    event.preventDefault();
    if (uploadButton.disabled) { return; }
    var file = RUI.q("#file-input").files[0];
    var realm = RUI.q("#realm").value.trim();
    if (!file || !realm) {
      RUI.setStatus(docStatus, "Choose a file and enter a realm.", "error");
      return;
    }
    var config = {
      groups: RUI.qa("input:checked", groupOptions).map(function (checkbox) {
        var model = checkbox.closest(".row").querySelector("select").value;
        return model ? { name: checkbox.value, model: model } : { name: checkbox.value };
      }),
      fields_to_extract: RUI.q("#requested-fields").value.split(",")
        .map(function (field) { return field.trim(); }).filter(Boolean),
      other_fields: RUI.q("#discover-other-fields").checked,
      reactions: RUI.q("#webhook-reaction").checked ? ["webhook"] : []
    };
    if (config.groups.length > 32) {
      RUI.setStatus(docStatus, "Select at most 32 field groups.", "error");
      return;
    }
    if (config.fields_to_extract.length > 128) {
      RUI.setStatus(docStatus, "Enter at most 128 individual fields.", "error");
      return;
    }
    if (overrideModel.value) { config.override_model = overrideModel.value; }
    var size = RUI.q("#file-size");
    if (size.value) {
      config.file_size = size.valueAsNumber;
      if (!Number.isSafeInteger(config.file_size) || config.file_size < 1) {
        RUI.setStatus(docStatus, "Enter a positive whole number of bytes.", "error");
        return;
      }
      if (file.size > config.file_size) {
        RUI.setStatus(docStatus, "The selected file exceeds the requested size limit.", "error");
        return;
      }
    }
    uploading = true;
    uploadButton.disabled = true;
    RUI.setStatus(docStatus, "Uploading " + file.name + "…", "busy");
    RUI.postDocument(apiBase.value, file, RUI.q("#synthetic-input").checked, realm, config).then(function (payload) {
      setResult(payload, "Processed " + file.name + ".");
    }, function (error) {
      RUI.setStatus(docStatus, RUI.describeError(error, apiBase.value), "error");
    }).finally(function () {
      uploading = false;
      uploadButton.disabled = loadingGroups;
    });
  });

  RUI.q("#job-btn").addEventListener("click", function () {
    var jobId = RUI.q("#job-id").value.trim();
    if (!jobId) {
      RUI.setStatus(docStatus, "Enter a job id first.", "error");
      return;
    }
    RUI.setStatus(docStatus, "fetching job " + jobId + "…", "busy");
    RUI.fetchJob(apiBase.value, jobId).then(function (payload) {
      setResult(payload, "Loaded job " + jobId + ".");
    }, function (error) {
      RUI.setStatus(docStatus, RUI.describeError(error, apiBase.value), "error");
    });
  });

  RUI.q("#json-input").addEventListener("change", function (event) {
    var file = event.target.files[0];
    if (!file) { return; }
    file.text().then(function (text) {
      try {
        setResult(JSON.parse(text), "Loaded " + file.name + " from disk.");
      } catch (error) {
        RUI.setStatus(docStatus, "That file is not valid JSON: " + error.message, "error");
      }
    });
  });

  RUI.q("#paste-btn").addEventListener("click", function () {
    var text = RUI.q("#paste-json").value.trim();
    if (!text) {
      RUI.setStatus(docStatus, "Paste a result first.", "error");
      return;
    }
    try {
      setResult(JSON.parse(text), "Rendered pasted JSON.");
    } catch (error) {
      RUI.setStatus(docStatus, "That is not valid JSON: " + error.message, "error");
    }
  });

  RUI.q("#image-input").addEventListener("change", function (event) {
    useImageFile(event.target.files[0]);
  });

  RUI.q("#file-input").addEventListener("change", function (event) {
    useImageFile(event.target.files[0]);
  });

  showBoxes.addEventListener("change", drawImage);
  pageSelect.addEventListener("change", function () {
    state.pageNo = Number(pageSelect.value) || null;
    drawImage();
  });
  RUI.q("#clear-active").addEventListener("click", function () {
    state.activeField = null;
    drawImage();
  });

  RUI.q("#review-btn").addEventListener("click", function () {
    if (!state.envelope) {
      RUI.setStatus(docStatus, "Load an envelope first.", "error");
      return;
    }
    var saved = RUI.saveStash({
      envelope: state.envelope,
      imageDataUrl: state.dataUrl,
      fileName: state.fileName,
      apiBase: RUI.normaliseBase(apiBase.value)
    });
    if (!saved.ok) {
      RUI.setStatus(docStatus,
        "Could not hand the envelope to the review page (" + saved.note +
        "). Open review.html and paste the JSON there.", "error");
      return;
    }
    if (saved.note) { RUI.setStatus(docStatus, saved.note, "busy"); }
    window.open("review.html", "_blank");
  });

  RUI.setStatus(docStatus, "", "");
  loadGroups();

  RUI.bindKeys({
    "u": function () { RUI.q("#file-input").click(); },
    "r": function () { RUI.q("#review-btn").click(); },
    "h": function () { RUI.q("#health-btn").click(); }
  });
})();
