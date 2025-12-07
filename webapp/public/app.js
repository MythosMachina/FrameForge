document.addEventListener("DOMContentLoaded", () => {
  wireNavigation();
  wireUploadForm();
  wireAutocharForm();
  loadData();
  populateAutocharSelect();
  setInterval(() => {
    refreshQueue();
    refreshHistory();
  }, 5000);
  // Default to dashboard if no hash
  const initial = location.hash.replace("#", "") || "queue";
  setActiveSection(initial);
  if (initial === "autochar") refreshAutochar();
});

async function loadData() {
  await Promise.all([refreshQueue(), refreshHistory()]);
}

function wireNavigation() {
  const navLinks = document.querySelectorAll(".nav a, .cta a");
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        e.preventDefault();
        setActiveSection(href.slice(1));
        history.pushState({}, "", href);
      }
    });
  });
  window.addEventListener("popstate", () => {
    const hash = location.hash.replace("#", "") || "queue";
    setActiveSection(hash);
  });
}

function setActiveSection(id) {
  const sections = document.querySelectorAll("section[data-section]");
  sections.forEach((section) => {
    section.classList.toggle("hidden", section.getAttribute("data-section") !== id);
  });
  const navLinks = document.querySelectorAll(".nav a");
  navLinks.forEach((link) => {
    const href = link.getAttribute("href");
    link.classList.toggle("active", href === `#${id}`);
  });
  if (id === "autochar") {
    refreshAutochar();
  }
}

function formatStepLabel(step) {
  if (!step) return "";
  const label = String(step).toLowerCase();
  const map = {
    queued: "queued",
    unpacked: "unpacked",
    workflow: "workflow",
    rename: "rename",
    cap: "cap/facecap",
    archive: "archive mp4",
    move_capped: "move capped",
    merge_inputs: "merge inputs",
    select: "select",
    cropflip: "crop/flip",
    move_final: "final move",
    autotag: "autotag",
    autotag_watch: "autotag watch",
    train_plan: "train plan",
    train_stage: "train staging",
    train_watch: "train watch",
    train_run: "train run",
    packaging: "packaging",
    cleanup: "cleanup",
    done: "done",
    failed: "failed",
  };
  return map[label] || label;
}

function wireUploadForm() {
  const form = document.querySelector("form.upload-form");
  const fileInput = form?.querySelector('input[type="file"][name="zip"]');
  const fileNameLabel = document.getElementById("selected-file");
  const autocharPresetsSelect = document.getElementById("autochar-presets");
  const uploadMsg = document.getElementById("upload-msg");

  const showFileName = (text, cls = "") => {
    if (!fileNameLabel) return;
    fileNameLabel.textContent = text;
    fileNameLabel.className = `file-name ${cls}`.trim();
  };

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files?.length) {
        const names = Array.from(fileInput.files).map((f) => f.name);
        const label = names.length === 1 ? names[0] : `${names.length} files: ${names.join(", ")}`;
        showFileName(`Selected: ${label}`);
      } else {
        showFileName("");
      }
    });
  }

  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const selectedPresets = autocharPresetsSelect
      ? Array.from(autocharPresetsSelect.selectedOptions).map((o) => o.value)
      : [];
    if (selectedPresets.length) {
      selectedPresets.forEach((p) => fd.append("autocharPresets", p));
    }
    const msg = (text, cls = "") => {
      if (!uploadMsg) return;
      uploadMsg.textContent = text;
      uploadMsg.className = `status-msg ${cls}`;
    };
    const selectedFiles = fileInput?.files ? Array.from(fileInput.files) : [];
    if (!selectedFiles.length) {
      msg("ZIP file is required", "error");
      return;
    }
    msg("Uploading...");
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "upload failed");
      }
      const data = await res.json();
      const runs = data.runs || (data.run ? [data.run] : []);
      const runIds = runs.map((r) => r.runId).join(", ");
      msg(`Queued Run(s): ${runIds}`, "ok");
      const names = selectedFiles.map((f) => f.name);
      const label = names.length === 1 ? names[0] : `${names.length} files: ${names.join(", ")}`;
      showFileName(`Uploaded: ${label}`, "ok");
      setActiveSection("queue");
      await refreshQueue();
    } catch (err) {
      msg(err.message || "upload failed", "error");
      const names = selectedFiles.map((f) => f.name);
      if (names.length) {
        const label = names.length === 1 ? names[0] : `${names.length} files: ${names.join(", ")}`;
        showFileName(`Selected: ${label}`, "error");
      }
    }
  });
}

async function refreshQueue() {
  const container = document.getElementById("queue-list");
  if (!container) return;
  const oldHtml = container.innerHTML;
  try {
    const res = await fetch("/api/queue");
    const data = await res.json();
    const list = data.queue || [];
    const existing = new Map();
    container.querySelectorAll(".card[data-run-id]").forEach((el) => {
      existing.set(el.getAttribute("data-run-id"), el);
    });

    const frag = document.createDocumentFragment();
    if (!list.length) {
      const div = document.createElement("div");
      div.className = "empty";
      div.innerHTML = `<p>No active runs. Submit a dataset to start the pipeline.</p>`;
      frag.appendChild(div);
    } else {
      list.forEach((run) => {
        const key = String(run.runId);
        const existingEl = existing.get(key);
        const card = renderRunCard(run);
        card.setAttribute("data-run-id", key);
        if (!existingEl || existingEl.dataset.stepHint !== card.dataset.stepHint || existingEl.dataset.status !== card.dataset.status) {
          card.classList.add("pulse");
        }
        card.dataset.stepHint = card.dataset.stepHint || "";
        card.dataset.status = (run.status || "").toLowerCase();
        frag.appendChild(card);
        existing.delete(key);
      });
    }
    container.replaceChildren(frag);
  } catch (err) {
    if (container.innerHTML !== oldHtml) {
      container.innerHTML = `<div class="empty"><p>Failed to load queue.</p></div>`;
    }
  }
}

async function refreshHistory() {
  const container = document.getElementById("history-list");
  if (!container) return;
  const existing = new Map();
  container.querySelectorAll(".card[data-run-id]").forEach((el) => {
    existing.set(el.getAttribute("data-run-id"), el);
  });
  const frag = document.createDocumentFragment();
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const list = data.history || [];
    if (!list.length) {
      const div = document.createElement("div");
      div.className = "card empty";
      div.innerHTML = `<p>No finished datasets yet. Completed runs will appear here with download links.</p>`;
      frag.appendChild(div);
    } else {
      list.forEach((run) => {
        const key = String(run.runId);
        const card = renderHistoryCard(run);
        card.setAttribute("data-run-id", key);
        if (!existing.has(key)) {
          card.classList.add("pulse");
        }
        frag.appendChild(card);
      });
    }
    container.replaceChildren(frag);
  } catch (err) {
    container.innerHTML = `<div class="card empty"><p>Failed to load history.</p></div>`;
  }
}

let autocharReqId = 0;
async function refreshAutochar() {
  const container = document.getElementById("autochar-list");
  if (!container) return;
  const reqId = ++autocharReqId;
  try {
    const res = await fetch("/api/autochar");
    const data = await res.json();
    if (reqId !== autocharReqId) return; // stale response
    container.innerHTML = "";
    const seen = new Set();
    const list = (data.presets || []).filter((p) => {
      if (seen.has(p.name)) return false;
      seen.add(p.name);
      return true;
    });
    if (!list.length) {
      container.innerHTML = `<div class="card empty"><p>No presets yet.</p></div>`;
      return;
    }
    list.forEach((p) => container.appendChild(renderAutocharCard(p)));
  } catch (err) {
    if (reqId !== autocharReqId) return;
    container.innerHTML = `<div class="card empty"><p>Failed to load presets.</p></div>`;
  }
}

async function populateAutocharSelect() {
  const select = document.getElementById("autochar-presets");
  if (!select) return;
  try {
    const res = await fetch("/api/autochar");
    const data = await res.json();
    const seen = new Set();
    const list = (data.presets || []).filter((p) => {
      if (seen.has(p.name)) return false;
      seen.add(p.name);
      return true;
    });
    const currentType = document.querySelector('input[name="type"]')?.value || "human";
    select.innerHTML = "";
    list.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      if (p.name === currentType) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (err) {
    // ignore
  }
}

async function deleteRun(id) {
  if (!id) return;
  const proceed = confirm("Run wirklich löschen? (Daten + Eintrag werden entfernt)");
  if (!proceed) return;
  try {
    const res = await fetch(`/api/run/${id}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "delete failed");
    }
    await refreshHistory();
  } catch (err) {
    alert(err.message || "delete failed");
  }
}

function renderRunCard(run) {
  const card = document.createElement("article");
  card.className = "card";
  const steps = ["rename", "cap/facecap", "select", "crop/flip", "autotag", run.flags?.train ? "train" : "finalize"];
  const stageOrder = [
    "queued",
    "unpacked",
    "rename",
    "cap",
    "archive",
    "move_capped",
    "merge_inputs",
    "select",
    "cropflip",
    "move_final",
    "autotag",
    "autotag_watch",
    "train_plan",
    "train_stage",
    "train_watch",
    "train_run",
    "workflow",
    "packaging",
    "cleanup",
    "done",
    "failed",
  ];
  const statusLower = (run.status || "").toLowerCase();
  const lastStep = (run.lastStep || "").toLowerCase();
  const stageIndex = stageOrder.indexOf(lastStep);
  let statusIndex = 0;
  if (statusLower === "done") {
    statusIndex = steps.length - 1;
  } else if (statusLower === "failed") {
    statusIndex = Math.max(0, stageIndex === -1 ? steps.length - 1 : Math.min(steps.length - 1, stageIndex));
  } else if (statusLower === "running") {
    statusIndex = stageIndex === -1 ? 1 : Math.min(steps.length - 1, stageIndex + 1);
  } else {
    statusIndex = 0;
  }
  const stepsHtml = steps
    .map((s, idx) => {
      let cls = "idle";
      if (idx < statusIndex) cls = "done";
      else if (idx === statusIndex) cls = statusLower === "failed" ? "failed" : "active";
      return `<div class="step ${cls}">${s}</div>`;
    })
    .join("");
  const pillClass =
    statusLower === "failed"
      ? "pill-failed"
      : statusLower === "done"
      ? "pill-done"
      : statusLower === "queued"
      ? "pill-queued"
      : "pill-running";
  const pillLabel = statusLower === "failed" ? "failed" : statusLower === "done" ? "done" : statusLower === "queued" ? "queued" : "running";
  const stepHint = lastStep ? ` • ${formatStepLabel(lastStep)}` : "";
  card.dataset.stepHint = stepHint;
  card.dataset.status = statusLower;
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Run ${run.runId}</div>
        <div class="title">${run.name}</div>
      </div>
      <span class="pill ${pillClass}">${pillLabel}${stepHint}</span>
    </div>
    <div class="steps">${stepsHtml}</div>
    <div class="meta">
      <span>type: ${run.type}</span>
      <span>gpu: ${run.flags?.gpu ? "on" : "off"}</span>
      ${run.flags?.autotag ? "<span>autotag</span>" : ""}
      ${run.flags?.autochar ? "<span>autochar</span>" : ""}
      ${run.flags?.facecap ? "<span>facecap</span>" : ""}
      ${run.flags?.train ? "<span>train</span>" : ""}
    </div>
  `;
  return card;
}

function renderHistoryCard(run) {
  const card = document.createElement("article");
  card.className = "card";
  const buttons = [];
  if (run.datasetDownload) buttons.push(`<button class="btn secondary" onclick="window.location='${run.datasetDownload}'">Download dataset</button>`);
  if (run.loraDownload) buttons.push(`<button class="btn secondary" onclick="window.location='${run.loraDownload}'">Download LoRA</button>`);
  buttons.push(`<button class="btn danger" onclick="deleteRun(${run.id})">Delete</button>`);
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Finished</div>
        <div class="title">${run.name}</div>
      </div>
      <span class="pill pill-done">Ready</span>
    </div>
    <div class="meta">
      ${run.flags?.autotag ? "<span>autotag</span>" : ""}
      ${run.flags?.autochar ? "<span>autochar</span>" : ""}
      ${run.flags?.facecap ? "<span>facecap</span>" : ""}
      ${run.flags?.train ? "<span>train</span>" : ""}
    </div>
    <div class="actions">
      ${buttons.join("")}
    </div>
  `;
  return card;
}

// expose deleteRun for inline handlers
window.deleteRun = deleteRun;

function renderAutocharCard(p) {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Preset</div>
        <div class="title">${p.name}</div>
      </div>
      <span class="pill pill-queued">${p.blockPatterns?.length || 0} block • ${p.allowPatterns?.length || 0} allow</span>
    </div>
    <p class="meta">${p.description || ""}</p>
    <div class="actions">
      <button class="btn secondary" data-action="edit" data-id="${p.id}">Edit</button>
      <button class="btn danger" data-action="delete" data-id="${p.id}">Delete</button>
    </div>
  `;
  const btns = card.querySelectorAll("button[data-action]");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      if (action === "edit") {
        fillAutocharForm(p);
      } else if (action === "delete") {
        deleteAutochar(p.id);
      }
    });
  });
  return card;
}

function fillAutocharForm(p) {
  const form = document.getElementById("autochar-form");
  const uploadSelect = document.getElementById("autochar-presets");
  if (!form) return;
  form.querySelector('input[name="id"]').value = p.id || "";
  form.querySelector('input[name="name"]').value = p.name || "";
  form.querySelector('input[name="description"]').value = p.description || "";
  form.querySelector('textarea[name="blockPatterns"]').value = (p.blockPatterns || []).join("\n");
  if (uploadSelect) {
    Array.from(uploadSelect.options).forEach((opt) => {
      opt.selected = opt.value === p.name;
    });
  }
}

async function deleteAutochar(id) {
  if (!id) return;
  const ok = confirm("Preset wirklich löschen?");
  if (!ok) return;
  try {
    const res = await fetch(`/api/autochar/${id}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "delete failed");
    }
    await refreshAutochar();
  } catch (err) {
    alert(err.message || "delete failed");
  }
}

function wireAutocharForm() {
  const form = document.getElementById("autochar-form");
  const resetBtn = document.getElementById("autochar-reset");
  const msgEl = document.getElementById("autochar-msg");
  const uploadSelect = document.getElementById("autochar-presets");
  if (!form) return;
  const setMsg = (text, cls = "") => {
    if (!msgEl) return;
    msgEl.textContent = text;
    msgEl.className = `status-msg ${cls}`;
  };
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const id = fd.get("id");
    const payload = {
      name: fd.get("name"),
      description: fd.get("description"),
      blockPatterns: fd.get("blockPatterns"),
      allowPatterns: "",
    };
    setMsg("Saving...", "");
    try {
      const res = await fetch(id ? `/api/autochar/${id}` : "/api/autochar", {
        method: id ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "save failed");
      }
      setMsg("Saved", "ok");
      form.reset();
      await refreshAutochar();
      if (uploadSelect) {
        // refresh upload select options
        populateAutocharSelect();
      }
    } catch (err) {
      setMsg(err.message || "save failed", "error");
    }
  });
  resetBtn?.addEventListener("click", () => {
    form.reset();
    form.querySelector('input[name="id"]').value = "";
    setMsg("", "");
  });
}
