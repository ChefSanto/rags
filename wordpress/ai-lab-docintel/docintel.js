(function ($) {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  var docName = "document", docText = "";

  function status(msg) { $("#ailab-status").text(msg); }

  async function extractPdf(file) {
    status("extracting " + file.name + "…");
    var pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
    var parts = [];
    for (var p = 1; p <= pdf.numPages; p++) {
      var page = await pdf.getPage(p);
      var tc = await page.getTextContent();
      parts.push(tc.items.map(function (i) { return i.str; }).join(" "));
      status("extracting " + file.name + "… page " + p + "/" + pdf.numPages);
    }
    return parts.join("\n");
  }

  function renderResults(fields) {
    var html = "<h2>Extracted fields — review and correct</h2><table class='form-table'>";
    AILAB.fields.forEach(function (f) {
      var v = fields[f.name];
      if (v === null || v === undefined) v = "";
      if (typeof v === "object") v = JSON.stringify(v);
      html += "<tr><th><label>" + esc(f.name) + "</label></th><td>" +
              "<input type='text' class='regular-text ailab-field' data-name='" + esc(f.name) +
              "' value='" + esc(String(v)) + "'></td></tr>";
    });
    html += "</table><p><button class='button button-primary' id='ailab-approve'>Approve &amp; save</button> " +
            "<button class='button' id='ailab-draft'>Save as draft</button></p>";
    $("#ailab-results").html(html);
    $("#ailab-approve").click(function () { save(true); });
    $("#ailab-draft").click(function () { save(false); });
  }

  function collect() {
    var out = {};
    $(".ailab-field").each(function () { out[$(this).data("name")] = $(this).val(); });
    return out;
  }

  function save(approve) {
    status("saving…");
    $.post(AILAB.ajax, {
      action: "ailab_save",
      _ajax_nonce: AILAB.nonce,
      doc_name: docName,
      fields: JSON.stringify(window.__ailabRaw || {}),
      approved: approve ? JSON.stringify(collect()) : ""
    }, function () {
      status(approve ? "approved and saved" : "draft saved");
      location.reload();
    });
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  $("#ailab-drop").click(function () { $("#ailab-file").click(); });
  $("#ailab-drop").on("dragover", function (e) { e.preventDefault(); $(this).css("border-color", "#2271b1"); });
  $("#ailab-drop").on("dragleave drop", function (e) { e.preventDefault(); $(this).css("border-color", "#c3c4c7"); });
  $("#ailab-drop").on("drop", function (e) {
    var f = e.originalEvent.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  $("#ailab-file").on("change", function () { if (this.files[0]) handleFile(this.files[0]); });

  async function handleFile(file) {
    try {
      docName = file.name.replace(/\.pdf$/i, "");
      docText = await extractPdf(file);
      status("text ready (" + docText.length + " chars) — click Extract fields");
    } catch (e) {
      status("failed: " + e.message + " (text-based PDFs only for now)");
    }
  }

  $("#ailab-extract-btn").click(function () {
    var text = docText || $("#ailab-paste").val();
    if (!text.trim()) { status("upload a PDF or paste text first"); return; }
    status("extracting fields…");
    $.post(AILAB.ajax, {
      action: "ailab_extract",
      _ajax_nonce: AILAB.nonce,
      text: text
    }, function (resp) {
      if (resp.error) { status("error: " + resp.error); return; }
      window.__ailabRaw = resp.fields;
      renderResults(resp.fields);
      status("done in " + (resp.elapsed_ms || "?") + " ms — review, correct, approve");
    }).fail(function (xhr) {
      status("error: " + (xhr.responseJSON && xhr.responseJSON.error ? xhr.responseJSON.error : xhr.status));
    });
  });
})(jQuery);
