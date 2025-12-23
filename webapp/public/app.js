document.addEventListener("DOMContentLoaded", () => {
  wireNavigation();
  wireUploadForm();
  wireAutocharForm();
  wireSettings();
  wireManualEditor();
  refreshSystemStatus();
  loadData();
  populateAutocharSelect();
  loadTrainProfiles();
  setInterval(() => {
    refreshQueue();
    refreshHistory();
    refreshSystemStatus();
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
        const target = href.slice(1);
        setActiveSection(target);
        history.pushState({}, "", href);
        if (target === "docs") setDocPage("overview");
      }
    });
  });
  window.addEventListener("popstate", () => {
    const hash = location.hash.replace("#", "") || "queue";
    setActiveSection(hash);
    if (hash === "docs") setDocPage("overview");
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
  if (id === "docs") {
    setDocPage("overview");
  }
}

async function refreshSystemStatus() {
  try {
    const res = await fetch("/api/system/status");
    const data = await res.json();
    const services = data.services || [];
    const pill = document.getElementById("system-status-pill");
    if (pill) {
      const states = services.map((s) => s.state);
      const allOk = states.length > 0 && states.every((s) => s === "OK");
      const allFail = states.length > 0 && states.every((s) => s === "Fail");
      let overall = "OK";
      let label = "All systems up";
      if (allFail) {
        overall = "Fail";
        label = "Full outage";
      } else if (allOk) {
        overall = "OK";
        label = "All systems up";
      } else if (states.some((s) => s === "Fail")) {
        overall = "Degraded";
        label = "Partial outage";
      } else {
        overall = "Degraded";
        label = "Active / waiting";
      }
      const dot = pill.querySelector(".dot");
      const labelNode = pill.querySelector(".orch-label");
      const dotClass =
        overall === "OK"
          ? "dot-green"
          : overall === "Fail"
          ? "dot-red"
          : "dot-yellow";
      if (dot) dot.className = "dot " + dotClass;
      if (labelNode) labelNode.textContent = label;
    }
    const container = document.getElementById("system-status-list");
    if (container) {
      container.innerHTML = services
        .map(
          (s) => `
          <div class="status-card">
            <div class="status-chip chip-${s.state.toLowerCase()}">
              <span class="dot ${s.state === "OK" ? "dot-green" : s.state === "Busy" ? "dot-yellow" : s.state === "Waiting" ? "dot-blue" : "dot-red"}"></span>
              <span>${s.state}</span>
            </div>
            <h4>${s.name}</h4>
            <p class="muted small">${s.message || ""}</p>
          </div>
        `
        )
        .join("");
    }
  } catch (err) {
    console.error("Failed to load system status", err);
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
    train_progress: "train progress",
    train_plan: "train plan",
    train_stage: "train staging",
    train_watch: "train watch",
    train_run: "train run",
    images_only: "images only",
    manual_pause: "manual tagging",
    manual_edit: "manual edit",
    manual_done: "manual done",
    ready_to_train: "ready to train",
    packaging: "packaging",
    cleanup: "cleanup",
    done: "done",
    failed: "failed",
  };
  return map[label] || label;
}

function formatTrainLabel(progress) {
  if (!progress) return "";
  const parts = [];
  if (progress.epoch !== null || progress.epochTotal !== null) {
    const raw = progress.epoch;
    const doneEpoch = raw === null ? null : Math.max(0, raw - 1);
    const cur = doneEpoch === null ? "?" : doneEpoch;
    const tot = progress.epochTotal === null ? "" : `/${progress.epochTotal}`;
    parts.push(`epoch ${cur}${tot}`);
  }
  if (progress.step !== null && progress.stepTotal !== null) {
    parts.push(`step ${progress.step}/${progress.stepTotal}`);
  }
  return parts.join(" • ");
}

function wireUploadForm() {
  const form = document.querySelector("form.upload-form");
  const fileInput = form?.querySelector('input[type="file"][name="zip"]');
  const fileNameLabel = document.getElementById("selected-file");
  const autocharPresetsSelect = document.getElementById("autochar-presets");
  const trainProfileSelect = document.getElementById("train-profile");
  const manualToggle = form?.querySelector('input[name="manualTagging"]');
  const autotagToggle = form?.querySelector('input[name="autotag"]');
  const autocharToggle = form?.querySelector('input[name="autochar"]');
  const tagverifyToggle = form?.querySelector('input[name="tagverify"]');
  const uploadMsg = document.getElementById("upload-msg");
  const stagedList = document.getElementById("staged-list");
  const launchButton = form?.querySelector('button[type="submit"]');

  const stagedUploads = [];
  let stagedLoaded = false;
  if (form) {
    form.setAttribute("novalidate", "true");
  }

  const syncManualToggle = () => {
    if (!manualToggle) return;
    const manualOn = manualToggle.checked;
    if (autotagToggle) {
      autotagToggle.disabled = manualOn;
      if (manualOn) autotagToggle.checked = true;
    }
    if (autocharToggle) autocharToggle.disabled = manualOn;
    if (tagverifyToggle) tagverifyToggle.disabled = manualOn;
    if (manualOn) {
      if (autocharToggle) autocharToggle.checked = false;
      if (tagverifyToggle) tagverifyToggle.checked = false;
    }
  };
  if (manualToggle) {
    manualToggle.addEventListener("change", syncManualToggle);
    syncManualToggle();
  }

  const setUploadMsg = (text, cls = "") => {
    if (!uploadMsg) return;
    uploadMsg.textContent = text;
    uploadMsg.className = `status-msg ${cls}`;
  };

  const renderStaged = () => {
    if (!stagedList) return;
    stagedList.innerHTML = "";
    if (!stagedUploads.length) {
      const div = document.createElement("div");
      div.className = "hint";
      div.textContent = "Drop ZIPs to start staging. Staged items will appear here.";
      stagedList.appendChild(div);
    } else {
      stagedUploads.forEach((u) => {
        const row = document.createElement("div");
        row.className = "staged-item";
        const left = document.createElement("div");
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = u.name;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${formatBytes(u.size)} • ${u.expiresAt ? `expires ${timeUntil(u.expiresAt)}` : "pending"}`;
        left.appendChild(name);
        left.appendChild(meta);

        const right = document.createElement("div");
        right.className = "actions";
        const status = document.createElement("span");
        status.className = `status ${u.status}`;
        status.textContent = u.status === "ready" ? "ready" : u.status;
        right.appendChild(status);
        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.textContent = "Remove";
        removeBtn.addEventListener("click", () => removeStaged(u));
        right.appendChild(removeBtn);

        row.appendChild(left);
        row.appendChild(right);
        stagedList.appendChild(row);
      });
    }
    updateLaunchState();
  };

  const updateLaunchState = () => {
    if (!launchButton) return;
    const hasReady = stagedUploads.some((u) => u.status === "ready");
    launchButton.disabled = !hasReady;
  };

  const addStagedPlaceholder = (file) => {
    const tempId = `temp-${Date.now()}-${Math.random()}`;
    stagedUploads.push({ id: tempId, name: file.name, size: file.size, status: "uploading" });
    renderStaged();
    return tempId;
  };

  const loadStagedFromServer = async () => {
    try {
      const res = await fetch("/api/upload/staged");
      const data = await res.json();
      const uploads = data.uploads || [];
      stagedUploads.splice(0, stagedUploads.length);
      uploads.forEach((u) => {
        stagedUploads.push({
          id: u.uploadId,
          name: u.originalName,
          size: u.size || 0,
          expiresAt: u.expiresAt,
          status: "ready",
        });
      });
      stagedLoaded = true;
      renderStaged();
    } catch (_err) {
      // ignore
    }
  };

  // kick off an initial staged-load when wiring the form
  loadStagedFromServer();

  const removeStaged = async (u) => {
    const idx = stagedUploads.findIndex((x) => x.id === u.id);
    if (idx === -1) return;
    stagedUploads.splice(idx, 1);
    renderStaged();
    if (u.status === "ready" && !u.id.startsWith("temp")) {
      try {
        await fetch(`/api/upload/stage/${u.id}`, { method: "DELETE" });
      } catch (_) {
        /* ignore */
      }
    }
  };

  const stageFile = async (file) => {
    const tempId = addStagedPlaceholder(file);
    const fd = new FormData();
    fd.append("zip", file);
    try {
      const res = await fetch("/api/upload/stage", { method: "POST", body: fd });
      if (!res.ok) throw new Error("stage failed");
      const data = await res.json();
      const upload = data.upload || {};
      const idx = stagedUploads.findIndex((x) => x.id === tempId);
      if (idx !== -1) {
        stagedUploads[idx] = {
          id: upload.id,
          name: upload.name || file.name,
          size: upload.size || file.size,
          expiresAt: upload.expiresAt,
          status: "ready",
        };
      }
      renderStaged();
    } catch (err) {
      const idx = stagedUploads.findIndex((x) => x.id === tempId);
      if (idx !== -1) stagedUploads[idx].status = "error";
      renderStaged();
      setUploadMsg(err.message || "staging failed", "error");
    }
  };

  const showFileName = (text, cls = "") => {
    if (!fileNameLabel) return;
    fileNameLabel.textContent = text;
    fileNameLabel.className = `file-name ${cls}`.trim();
  };

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files?.length) {
        const files = Array.from(fileInput.files);
        const names = files.map((f) => f.name);
        const label = names.length === 1 ? names[0] : `${names.length} files`;
        showFileName(`Staging: ${label}`);
        files.forEach((f) => stageFile(f));
      } else {
        showFileName("");
      }
      fileInput.value = "";
    });
  }

  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const ready = stagedUploads.filter((u) => u.status === "ready");
    if (!ready.length) {
      setUploadMsg("Stage at least one ZIP before launching", "error");
      return;
    }
    const selectedPresets = autocharPresetsSelect
      ? Array.from(autocharPresetsSelect.selectedOptions).map((o) => o.value)
      : [];
    const body = {
      uploads: ready.map((u) => u.id),
      autotag: form.querySelector('input[name="autotag"]')?.checked ?? true,
      autochar: form.querySelector('input[name="autochar"]')?.checked ?? true,
      tagverify: form.querySelector('input[name="tagverify"]')?.checked ?? false,
      facecap: form.querySelector('input[name="facecap"]')?.checked ?? false,
      imagesOnly: form.querySelector('input[name="imagesOnly"]')?.checked ?? false,
      train: form.querySelector('input[name="train"]')?.checked ?? false,
      gpu: form.querySelector('input[name="gpu"]')?.checked ?? true,
      manualTagging: form.querySelector('input[name="manualTagging"]')?.checked ?? false,
      trainProfile: trainProfileSelect?.value || "",
      note: form.querySelector('input[name="note"]')?.value ?? "",
      autocharPresets: selectedPresets,
    };
    setUploadMsg("Queuing staged uploads...");
    try {
      const res = await fetch("/api/upload/commit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "upload failed");
      }
      const data = await res.json();
      const runs = data.runs || (data.run ? [data.run] : []);
      const runIds = runs.map((r) => r.runId).join(", ");
      setUploadMsg(`Queued Run(s): ${runIds}`, "ok");
      showFileName(`Queued: ${runIds}`, "ok");
      stagedUploads.splice(0, stagedUploads.length);
      renderStaged();
      setActiveSection("queue");
      await refreshQueue();
    } catch (err) {
      setUploadMsg(err.message || "upload failed", "error");
      showFileName("Launch failed", "error");
    }
  });
}

async function loadTrainProfiles() {
  const select = document.getElementById("train-profile");
  if (!select) return;
  select.innerHTML = "";
  try {
    const res = await fetch("/api/train-profiles");
    if (!res.ok) throw new Error("failed to load train profiles");
    const data = await res.json();
    const profiles = data.profiles || [];
    if (!profiles.length) {
      select.innerHTML = '<option value="">(no profiles)</option>';
      return;
    }
    const defaultProfile = profiles.find((p) => p.isDefault) || profiles[0];
    profiles.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label ? `${p.name} — ${p.label}` : p.name;
      if (defaultProfile && p.name === defaultProfile.name) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (_err) {
    select.innerHTML = '<option value="">(failed to load)</option>';
  }
}

function wireSettings() {
  const saveBtn = document.getElementById("settings-save");
  const pruneBtn = document.getElementById("settings-prune");
  const msgEl = document.getElementById("settings-msg");
  const queueMsg = document.getElementById("queue-msg");
  const queueModeEl = document.getElementById("queue-mode");
  const queueStatusText = document.getElementById("queue-status-text");
  const queueActive = document.getElementById("queue-active");
  const queueStart = document.getElementById("queue-start");
  const queuePause = document.getElementById("queue-pause");
  const queueStop = document.getElementById("queue-stop");
  const queueRestart = document.getElementById("queue-restart");
  const queueRefresh = document.getElementById("queue-refresh");
  const tabs = document.querySelectorAll(".settings-tabs .tab");
  const panels = document.querySelectorAll(".settings-tab");
  const taggerList = document.getElementById("tagger-model-list");
  const taggerMsg = document.getElementById("tagger-msg");
  const taggerRepoInput = document.getElementById("tagger-repo-id");
  const taggerDownloadBtn = document.getElementById("tagger-download");
  const taggerDefaultDisplay = document.getElementById("tagger-default");
  const taggerPresetSelect = document.getElementById("tagger-preset");
  const taggerPresetStatus = document.getElementById("tagger-preset-status");
  const ids = [
    "capping_fps",
    "capping_jpeg_quality",
    "selection_target_per_character",
    "selection_face_quota",
    "selection_hamming_threshold",
    "selection_hamming_relaxed",
    "autotag_general_threshold",
    "autotag_character_threshold",
    "autotag_max_tags",
    "output_max_images",
    "hf_token",
    "autotag_model_id",
    // trainer settings
    "trainer_base_model",
    "trainer_vae",
    "trainer_resolution",
    "trainer_batch_size",
    "trainer_grad_accum",
    "trainer_epochs",
    "trainer_max_train_steps",
    "trainer_learning_rate",
    "trainer_te_learning_rate",
    "trainer_lr_scheduler",
    "trainer_lr_warmup_steps",
    "trainer_lora_rank",
    "trainer_lora_alpha",
    "trainer_te_lora_rank",
    "trainer_te_lora_alpha",
    "trainer_clip_skip",
    "trainer_network_dropout",
    "trainer_caption_dropout",
    "trainer_shuffle_caption",
    "trainer_keep_tokens",
    "trainer_min_snr_gamma",
    "trainer_noise_offset",
    "trainer_weight_decay",
    "trainer_sample_prompt_1",
    "trainer_sample_prompt_2",
    "trainer_sample_prompt_3",
    "trainer_bucket_min_reso",
    "trainer_bucket_max_reso",
    "trainer_bucket_step",
    "trainer_optimizer",
    "trainer_use_8bit_adam",
    "trainer_gradient_checkpointing",
    "trainer_dataloader_workers",
    "trainer_use_prodigy",
    "trainer_max_grad_norm",
  ];
  const settingsCache = {};

  const setMsg = (text, cls = "") => {
    if (!msgEl) return;
    msgEl.textContent = text;
    msgEl.className = `status-msg ${cls}`;
  };

  const setQueueMsg = (text, cls = "") => {
    if (!queueMsg) return;
    queueMsg.textContent = text;
    queueMsg.className = `status-msg ${cls}`;
  };

  const renderQueueState = (payload = {}) => {
    const mode = payload.mode || "unknown";
    const orch = payload.orchestrator || {};
    if (queueModeEl) queueModeEl.textContent = mode;
    if (queueStatusText) {
      const parts = [];
      if (orch.state) parts.push(orch.state);
      if (orch.message) parts.push(orch.message);
      queueStatusText.textContent = parts.length ? parts.join(" — ") : "status: unknown";
    }
    if (queueActive) queueActive.textContent = orch.activeRunId || "—";
  };

  const loadQueueState = async () => {
    try {
      const res = await fetch("/api/queue/state");
      const data = await res.json();
      renderQueueState(data);
    } catch (err) {
      setQueueMsg("Failed to load queue state", "error");
    }
  };

  const updateQueueState = async (mode) => {
    setQueueMsg("Updating...");
    try {
      const res = await fetch("/api/queue/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Queue update failed");
      renderQueueState(data);
      setQueueMsg(`Queue set to ${data.mode}`, "ok");
    } catch (err) {
      setQueueMsg(err.message || "Queue update failed", "error");
    }
  };

  const fillSettings = (settings) => {
    Object.assign(settingsCache, settings);
    ids.forEach((key) => {
      const input = document.getElementById(`setting-${key}`);
      if (!input) return;
      const val = settings[key];
      if (val === undefined || val === null) return;
      if (input.type === "checkbox") {
        input.checked = String(val).toLowerCase() === "true" || val === true || val === 1;
      } else {
        input.value = val;
      }
    });
    if (taggerDefaultDisplay && settings.autotag_model_id) {
      taggerDefaultDisplay.textContent = settings.autotag_model_id;
    }
  };

  const loadSettings = async () => {
    try {
      const res = await fetch("/api/settings");
      const data = await res.json();
      fillSettings(data.settings || {});
    } catch (err) {
      setMsg("Failed to load settings", "error");
    }
  };

  const collectSettings = () => {
    const payload = {};
    ids.forEach((key) => {
      const input = document.getElementById(`setting-${key}`);
      if (!input) return;
      if (input.type === "checkbox") {
        payload[key] = input.checked;
      } else {
        const val = input.value;
        const num = Number(val);
        payload[key] = Number.isFinite(num) ? num : val;
      }
    });
    return payload;
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) => {
        p.classList.toggle("hidden", p.getAttribute("data-tab-panel") !== target);
      });
    });
  });

  if (queueStart) queueStart.addEventListener("click", () => updateQueueState("running"));
  if (queuePause) queuePause.addEventListener("click", () => updateQueueState("paused"));
  if (queueStop) queueStop.addEventListener("click", () => updateQueueState("stopped"));
  if (queueRestart) queueRestart.addEventListener("click", () => updateQueueState("restart"));
  if (queueRefresh) queueRefresh.addEventListener("click", () => loadQueueState());

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      setMsg("Saving...");
      try {
        const payload = collectSettings();
        const res = await fetch("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Save failed");
        setMsg("Saved", "ok");
      } catch (err) {
        setMsg(err.message || "Save failed", "error");
      }
    });
  }

  if (pruneBtn) {
    pruneBtn.addEventListener("click", async () => {
      if (!confirm("Prune workdirs? This cannot be undone.")) return;
      setMsg("Pruning...");
      try {
        const res = await fetch("/api/prune", { method: "POST" });
        if (!res.ok) throw new Error("Prune failed");
        setMsg("Prune started", "ok");
      } catch (err) {
        setMsg(err.message || "Prune failed", "error");
      }
    });
  }

  loadSettings();
  loadQueueState();

  const setTaggerMsg = (text, cls = "") => {
    if (!taggerMsg) return;
    taggerMsg.textContent = text;
    taggerMsg.className = `status-msg ${cls}`;
  };

  const renderPresetStatus = (modelMap = new Map()) => {
    if (!taggerPresetStatus || !taggerPresetSelect) return;
    const repoId = taggerPresetSelect.value;
    if (!repoId) {
      taggerPresetStatus.textContent = "";
      return;
    }
    const record = modelMap.get(repoId);
    if (!record) {
      taggerPresetStatus.textContent = "Not installed yet.";
      return;
    }
    const status = (record.status || "unknown").toLowerCase();
    const sizeLabel = record.size ? formatBytes(record.size) : "0 B";
    taggerPresetStatus.textContent = `${status} • ${sizeLabel}`;
  };

  const renderTaggerModels = (models = []) => {
    if (!taggerList) return;
    taggerList.innerHTML = "";
    if (!models.length) {
      const div = document.createElement("div");
      div.className = "empty";
      div.textContent = "No local models.";
      taggerList.appendChild(div);
      return;
    }
    models.forEach((m) => {
      const card = document.createElement("div");
      card.className = "card model";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = m.repoId;
      const meta = document.createElement("div");
      meta.className = "meta";
      const size = document.createElement("span");
      size.textContent = formatBytes(m.size);
      const status = document.createElement("span");
      const statusKey = String(m.status || "unknown").toLowerCase();
      status.className = `pill tagger-status tagger-${statusKey}`;
      status.textContent = statusKey;
      meta.appendChild(size);
      meta.appendChild(status);
      const actions = document.createElement("div");
      actions.className = "actions";
      if (settingsCache.autotag_model_id === m.repoId) {
        const badge = document.createElement("span");
        badge.className = "pill badge";
        badge.textContent = "default";
        actions.appendChild(badge);
      }
      const delBtn = document.createElement("button");
      delBtn.className = "btn ghost";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", async () => {
        if (!confirm("Delete model?")) return;
        setTaggerMsg("Deleting...");
        try {
          const res = await fetch(`/api/tagger-models/${m.id}`, { method: "DELETE" });
          if (!res.ok) throw new Error("Delete failed");
          await loadModels();
          setTaggerMsg("Deleted", "ok");
        } catch (err) {
          setTaggerMsg(err.message || "Delete failed", "error");
        }
      });
      const defaultBtn = document.createElement("button");
      defaultBtn.className = "btn secondary";
      defaultBtn.textContent = "Set as default";
      defaultBtn.addEventListener("click", async () => {
        setTaggerMsg("Updating default...");
        try {
          const res = await fetch("/api/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ autotag_model_id: m.repoId }),
          });
          if (!res.ok) throw new Error("Update failed");
          settingsCache.autotag_model_id = m.repoId;
          if (taggerDefaultDisplay) taggerDefaultDisplay.textContent = m.repoId;
          await loadModels();
          setTaggerMsg("Default updated", "ok");
        } catch (err) {
          setTaggerMsg(err.message || "Update failed", "error");
        }
      });
      const refreshBtn = document.createElement("button");
      refreshBtn.className = "btn ghost";
      refreshBtn.textContent = "Re-download";
      refreshBtn.addEventListener("click", async () => {
        taggerRepoInput.value = m.repoId;
        await doDownload();
      });
      actions.appendChild(refreshBtn);
      actions.appendChild(defaultBtn);
      actions.appendChild(delBtn);
      card.appendChild(name);
      card.appendChild(meta);
      card.appendChild(actions);
      taggerList.appendChild(card);
    });
  };

  const loadModels = async () => {
    try {
      const res = await fetch("/api/tagger-models");
      const data = await res.json();
      const items = data.models || [];
      const map = new Map(items.map((m) => [m.repoId, m]));
      renderTaggerModels(data.models || []);
      renderPresetStatus(map);
    } catch (err) {
      setTaggerMsg("Failed to load models", "error");
    }
  };

  const doDownload = async () => {
    const repoId = taggerRepoInput?.value?.trim();
    if (!repoId) {
      setTaggerMsg("Repo ID required", "error");
      return;
    }
    setTaggerMsg("Downloading...");
    try {
      const res = await fetch("/api/tagger-models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repoId }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Download failed");
      }
      await loadModels();
      setTaggerMsg("Downloaded", "ok");
    } catch (err) {
      setTaggerMsg(err.message || "Download failed", "error");
    }
  };

  if (taggerDownloadBtn) {
    taggerDownloadBtn.addEventListener("click", async () => {
      await doDownload();
    });
  }

  if (taggerPresetSelect && taggerRepoInput) {
    taggerPresetSelect.addEventListener("change", async () => {
      if (taggerPresetSelect.value) {
        taggerRepoInput.value = taggerPresetSelect.value;
      }
      await loadModels();
    });
  }

  loadModels();
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function timeUntil(date) {
  const d = new Date(date);
  const diff = d.getTime() - Date.now();
  if (diff <= 0) return "soon";
  const mins = Math.floor(diff / 60000);
  return mins <= 0 ? "under a minute" : `${mins}m left`;
}

const docCache = new Map();

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let inList = false;
  let inOlist = false;
  let inCode = false;
  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (line.startsWith("```")) {
      if (inCode) {
        out.push("</code></pre>");
        inCode = false;
      } else {
        out.push("<pre><code>");
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      out.push(escapeHtml(rawLine));
      continue;
    }
    if (line.startsWith("# ")) {
      out.push(`<h3>${escapeHtml(line.slice(2))}</h3>`);
      continue;
    }
    if (line.startsWith("## ")) {
      out.push(`<h4>${escapeHtml(line.slice(3))}</h4>`);
      continue;
    }
    if (/^\d+\)\s+/.test(line) || /^\d+\.\s+/.test(line)) {
      if (!inOlist) {
        out.push("<ol>");
        inOlist = true;
      }
      out.push(`<li>${escapeHtml(line.replace(/^\d+[\.\)]\s+/, ""))}</li>`);
      continue;
    }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${escapeHtml(line.slice(2))}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (inOlist) {
      out.push("</ol>");
      inOlist = false;
    }
    if (!line.trim()) {
      out.push("");
      continue;
    }
    const withInline = escapeHtml(line)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a class="link" href="$2" target="_blank" rel="noreferrer">$1</a>');
    out.push(`<p>${withInline}</p>`);
  }
  if (inList) out.push("</ul>");
  if (inOlist) out.push("</ol>");
  if (inCode) out.push("</code></pre>");
  return out.join("\n");
}

async function loadDocPage(id) {
  const container = document.getElementById("docs-content");
  if (!container) return;
  if (docCache.has(id)) {
    container.innerHTML = docCache.get(id);
    return;
  }
  container.innerHTML = `<p class="muted">Loading documentation...</p>`;
  try {
    const res = await fetch(`/insite-docs/${id}.md`);
    const text = await res.text();
    if (!res.ok) throw new Error(text || "Failed to load docs");
    const html = renderMarkdown(text);
    docCache.set(id, html);
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<p class="muted">Docs not available.</p>`;
  }
}

function setDocPage(id) {
  const links = document.querySelectorAll(".docs-nav a[data-doc-target]");
  links.forEach((l) => {
    l.classList.toggle("active", l.getAttribute("data-doc-target") === id);
  });
  loadDocPage(id);
}

document.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof Element)) return;
  const docTarget = target.getAttribute("data-doc-target");
  if (docTarget) {
    e.preventDefault();
    setActiveSection("docs");
    setDocPage(docTarget);
    history.pushState({}, "", "#docs");
  }
});

function formatTimestamp(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

function stepsHtmlFor(run, statusLower, lastStep) {
  const steps = run.flags?.imagesOnly
    ? ["rename", "images", "select", "crop/flip", "autotag", run.flags?.train ? "train" : "finalize"]
    : ["rename", "cap/facecap", "select", "crop/flip", "autotag", run.flags?.train ? "train" : "finalize"];
  const stageOrder = [
    "queued",
    "workflow_start",
    "unpacked",
    "rename",
    "cap",
    "images_only",
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
    "train_progress",
    "train_run",
    "workflow",
    "packaging",
    "cleanup",
    "done",
    "failed",
  ];
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
  return steps
    .map((s, idx) => {
      let cls = "idle";
      if (idx < statusIndex) cls = "done";
      else if (idx === statusIndex) cls = statusLower === "failed" ? "failed" : "active";
      return `<div class="step ${cls}">${s}</div>`;
    })
    .join("");
}

function buildRunInfo(run) {
  const parts = [];
  if (run.flags?.gpu !== undefined) parts.push(`gpu: ${run.flags.gpu ? "on" : "off"}`);
  if (run.flags?.autotag) parts.push("autotag");
  if (run.flags?.autochar) parts.push("autochar");
  if (run.flags?.manualTagging) parts.push("manual");
  if (run.flags?.facecap) parts.push("facecap");
  if (run.flags?.imagesOnly) parts.push("images-only");
  if (run.flags?.train) parts.push("train");
  const profile = run.trainProfile || run.flags?.trainProfile;
  if (profile) parts.push(`profile: ${profile}`);
  return parts.join("\n") || "no extra flags";
}

async function refreshQueue() {
  const queueContainer = document.getElementById("queue-list");
  const activeContainer = document.getElementById("active-list");
  if (!queueContainer || !activeContainer) return;
  const oldQueue = queueContainer.innerHTML;
  const oldActive = activeContainer.innerHTML;
  try {
    const res = await fetch("/api/queue");
    const data = await res.json();
    const list = data.queue || [];
    const queueFrag = document.createDocumentFragment();
    const activeFrag = document.createDocumentFragment();
    const queued = list.filter((run) => ["queued"].includes(String(run.status || "").toLowerCase()));
    const active = list.filter((run) => !["queued"].includes(String(run.status || "").toLowerCase()));

    if (!queued.length) {
      const div = document.createElement("div");
      div.className = "empty";
      div.innerHTML = `<p>No queued runs. Submit a dataset to start the pipeline.</p>`;
      queueFrag.appendChild(div);
    } else {
      queued.forEach((run) => {
        const card = renderQueueCard(run);
        card.setAttribute("data-run-id", String(run.runId));
        queueFrag.appendChild(card);
      });
    }

    if (!active.length) {
      const div = document.createElement("div");
      div.className = "empty";
      div.innerHTML = `<p>No active runs. Queue a dataset to begin.</p>`;
      activeFrag.appendChild(div);
    } else {
      active.forEach((run) => {
        const card = renderActiveCard(run);
        card.setAttribute("data-run-id", String(run.runId));
        activeFrag.appendChild(card);
      });
    }
    queueContainer.replaceChildren(queueFrag);
    activeContainer.replaceChildren(activeFrag);
  } catch (err) {
    if (queueContainer.innerHTML !== oldQueue) {
      queueContainer.innerHTML = `<div class="empty"><p>Failed to load queue.</p></div>`;
    }
    if (activeContainer.innerHTML !== oldActive) {
      activeContainer.innerHTML = `<div class="empty"><p>Failed to load active runs.</p></div>`;
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
    select.innerHTML = "";
    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.textContent = "(no preset)";
    noneOpt.selected = true;
    select.appendChild(noneOpt);
    list.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
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

async function stopRun(id) {
  if (!id) return;
  const proceed = confirm("Stop this run? This will kill the job and remove its data.");
  if (!proceed) return;
  try {
    const res = await fetch(`/api/run/${id}/stop`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "stop failed");
    }
    await Promise.all([refreshQueue(), refreshHistory()]);
  } catch (err) {
    alert(err.message || "stop failed");
  }
}

function renderRunCard(run) {
  return renderActiveCard(run);
}

function renderActiveCard(run) {
  const card = document.createElement("article");
  card.className = "card";
  const rawStatus = (run.status || "").toLowerCase();
  const isFailed = rawStatus.startsWith("failed");
  const statusLower = isFailed ? "failed" : rawStatus;
  const rawLastStep = (run.lastStep || "").toLowerCase();
  const lastStep = rawLastStep.startsWith("train_progress") ? "train_progress" : rawLastStep;
  const stepsHtml = stepsHtmlFor(run, statusLower, lastStep);
  const pillClass =
    statusLower === "failed"
      ? "pill-failed"
      : statusLower === "done"
      ? "pill-done"
      : statusLower === "queued"
      ? "pill-queued"
      : "pill-running";
  const pillLabel =
    statusLower === "failed"
      ? "failed"
      : statusLower === "done"
      ? "done"
      : statusLower === "queued"
      ? "queued"
      : "running";
  const progress = run.trainProgress;
  const progressLabel = formatTrainLabel(progress);
  const pct = progress && progress.step !== null && progress.stepTotal ? Math.min(100, Math.max(0, (progress.step / progress.stepTotal) * 100)) : null;
  const stepHint =
    lastStep && lastStep !== "train_progress"
      ? ` • ${formatStepLabel(lastStep)}`
      : "";
  card.dataset.stepHint = stepHint;
  card.dataset.status = statusLower;
  const actions = [];
  if (run.flags?.train && run.id) {
    actions.push(`<button class="btn secondary" onclick='openSamples(${run.id}, ${JSON.stringify(run.runName || run.name || "")})'>Samples</button>`);
  }
  if (statusLower === "manual_tagging") {
    actions.push(`<button class="btn secondary" onclick='openManualEditor(${run.id}, ${JSON.stringify(run.runName || run.name || "")})'>Open editor</button>`);
  }
  if (statusLower === "running" || statusLower === "queued") {
    actions.push(`<button class="btn danger" onclick="stopRun(${run.id})">Stop job</button>`);
  }
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
      <span>created: ${formatTimestamp(run.createdAt)}</span>
      <span>started: ${formatTimestamp(run.startedAt)}</span>
      <span class="info" data-tip="${buildRunInfo(run)}">info</span>
    </div>
    ${
      progress
        ? `<div class="train-progress">
            ${progressLabel ? `<div class="label">${progressLabel}</div>` : ""}
            ${
              pct !== null
                ? `<div class="meter"><span style="width:${pct}%" aria-valuenow="${pct.toFixed(1)}"></span></div>`
                : ""
            }
          </div>`
        : ""
    }
    ${actions.length ? `<div class="actions">${actions.join("")}</div>` : ""}
  `;
  return card;
}

function renderHistoryCard(run) {
  const card = document.createElement("article");
  card.className = "card";
  const statusLower = (run.status || "").toLowerCase();
  const pillClass =
    statusLower === "failed"
      ? "pill-failed"
      : statusLower === "running"
      ? "pill-running"
      : statusLower === "queued"
      ? "pill-queued"
      : "pill-done";
  const pillLabel =
    statusLower === "failed"
      ? "failed"
      : statusLower === "running"
      ? "running"
      : statusLower === "queued"
      ? "queued"
      : "ready";
  const buttons = [];
  buttons.push(`<button class="btn secondary" onclick='openResults(${run.id}, ${JSON.stringify(run.runName || run.name || "")})'>Results</button>`);
  buttons.push(`<button class="btn danger" onclick="deleteRun(${run.id})">Delete</button>`);
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Finished</div>
        <div class="title">${run.name}</div>
      </div>
      <span class="pill ${pillClass}">${pillLabel}</span>
    </div>
    <div class="meta">
      <span>created: ${formatTimestamp(run.createdAt)}</span>
      <span>finished: ${formatTimestamp(run.finishedAt)}</span>
      <span class="info" data-tip="${buildRunInfo(run)}">info</span>
      ${run.flags?.train && run.trainEpochs !== undefined ? `<span>epochs: ${run.trainEpochs || 0}</span>` : ""}
    </div>
    <div class="actions">
      ${buttons.join("")}
    </div>
  `;
  return card;
}

function renderQueueCard(run) {
  const card = document.createElement("article");
  card.className = "card";
  const rawStatus = (run.status || "").toLowerCase();
  const isFailed = rawStatus.startsWith("failed");
  const statusLower = isFailed ? "failed" : rawStatus;
  const pillClass =
    statusLower === "failed"
      ? "pill-failed"
      : statusLower === "done"
      ? "pill-done"
      : statusLower === "queued"
      ? "pill-queued"
      : "pill-running";
  const pillLabel = statusLower === "failed" ? "failed" : statusLower === "done" ? "done" : statusLower === "queued" ? "queued" : "waiting";
  const actions = [];
  actions.push(`<button class="btn danger" onclick="stopRun(${run.id})">Stop job</button>`);
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Run ${run.runId}</div>
        <div class="title">${run.name}</div>
      </div>
      <span class="pill ${pillClass}">${pillLabel}</span>
    </div>
    <div class="meta">
      <span>created: ${formatTimestamp(run.createdAt)}</span>
      <span class="info" data-tip="${buildRunInfo(run)}">info</span>
    </div>
    <div class="actions">${actions.join("")}</div>
  `;
  return card;
}

// expose deleteRun for inline handlers
window.deleteRun = deleteRun;
window.stopRun = stopRun;
window.openSamples = openSamples;
window.openResults = openResults;

function openLightbox(src, label = "") {
  const overlay = document.createElement("div");
  overlay.className = "lightbox-backdrop";
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = `
    <button class="btn ghost lightbox-close" type="button">Close</button>
    <img src="${src}" alt="${label}">
    ${label ? `<div class="lightbox-caption">${label}</div>` : ""}
  `;
  overlay.appendChild(box);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });
  const closeBtn = box.querySelector(".lightbox-close");
  closeBtn?.addEventListener("click", () => overlay.remove());
  document.body.appendChild(overlay);
}

async function openSamples(id, runName = "") {
  try {
    const res = await fetch(`/api/run/${id}/samples`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "failed to load samples");
    }
    const data = await res.json();
    const samples = data.samples || [];
    if (!samples.length) {
      alert("No samples yet for this run.");
      return;
    }
    const overlay = document.createElement("div");
    overlay.className = "modal-backdrop";
    const modal = document.createElement("div");
    modal.className = "modal";
    const head = document.createElement("div");
    head.className = "modal-head";
    head.innerHTML = `
      <div>
        <div class="label">Samples</div>
        <div class="title">${runName || "Run " + id}</div>
      </div>
      <button class="btn ghost" type="button">Close</button>
    `;
    const closeBtn = head.querySelector("button");
    closeBtn?.addEventListener("click", () => overlay.remove());
    const grid = document.createElement("div");
    grid.className = "sample-grid";
    samples.forEach((s) => {
      const fig = document.createElement("figure");
      fig.innerHTML = `
        <img src="${s.url}" alt="${s.name}">
        <figcaption>${s.label || s.name}</figcaption>
      `;
      const img = fig.querySelector("img");
      img?.addEventListener("click", () => openLightbox(s.url, s.label || s.name));
      grid.appendChild(fig);
    });
    modal.appendChild(head);
    modal.appendChild(grid);
    overlay.appendChild(modal);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    document.body.appendChild(overlay);
  } catch (err) {
    alert(err.message || "Failed to load samples");
  }
}

async function openResults(id, runName = "") {
  try {
    const res = await fetch(`/api/run/${id}/results`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "failed to load results");
    }
    const data = await res.json();
    const overlay = document.createElement("div");
    overlay.className = "modal-backdrop";
    const modal = document.createElement("div");
    modal.className = "modal";
    const head = document.createElement("div");
    head.className = "modal-head";
    head.innerHTML = `
      <div>
        <div class="label">Results</div>
        <div class="title">${runName || "Run " + id}</div>
      </div>
      <button class="btn ghost" type="button">Close</button>
    `;
    const closeBtn = head.querySelector("button");
    closeBtn?.addEventListener("click", () => overlay.remove());
    modal.appendChild(head);

    const sections = document.createElement("div");
    sections.className = "results-sections";

    const datasetSection = document.createElement("section");
    datasetSection.className = "results-section";
    datasetSection.innerHTML = `
      <h3>Dataset</h3>
      <div class="results-actions">
        ${data.datasetZip ? `<button class="btn secondary" onclick="window.location='${data.datasetZip}'">Download dataset zip</button>` : `<span class="muted">No dataset zip available.</span>`}
      </div>
    `;
    sections.appendChild(datasetSection);

    const loraSection = document.createElement("section");
    loraSection.className = "results-section";
    const loraButtons = [];
    if (data.loraZip) {
      loraButtons.push(`<button class="btn secondary" onclick="window.location='${data.loraZip}'">Download all LoRAs (zip)</button>`);
    }
    const loraItems = (data.loraFiles || [])
      .map((f) => {
        const label = f.label || (f.epoch === null ? "Final" : `Epoch ${f.epoch}`);
        return `<button class="btn ghost" onclick="window.location='${f.url}'">${label}</button>`;
      })
      .join("");
    loraSection.innerHTML = `
      <h3>LoRA checkpoints</h3>
      <div class="results-actions">${loraButtons.join("") || `<span class="muted">No LoRA zip available.</span>`}</div>
      ${loraItems ? `<div class="result-list">${loraItems}</div>` : `<p class="muted">No checkpoints found.</p>`}
    `;
    sections.appendChild(loraSection);

    const sampleSection = document.createElement("section");
    sampleSection.className = "results-section";
    const sampleZip = data.samplesZip
      ? `<button class="btn secondary" onclick="window.location='${data.samplesZip}'">Download all samples (zip)</button>`
      : `<span class="muted">No sample zip available.</span>`;
    const sampleEpochs = (data.sampleEpochs || [])
      .map((group) => {
        const epochLabel = group.label || (group.epoch === null ? "Other" : `Epoch ${group.epoch}`);
        const zipBtn = group.zip ? `<button class="btn ghost" onclick="window.location='${group.zip}'">Download epoch zip</button>` : `<span class="muted">No zip</span>`;
        const images = (group.images || [])
          .map((img) => `<figure><img src="${img.url}" alt="${img.name}"><figcaption>${img.label || img.name}</figcaption></figure>`)
          .join("");
        return `
          <div class="result-epoch">
            <div class="result-epoch-head">
              <h4>${epochLabel}</h4>
              ${zipBtn}
            </div>
            ${images ? `<div class="sample-grid">${images}</div>` : `<p class="muted">No samples.</p>`}
          </div>
        `;
      })
      .join("");
    sampleSection.innerHTML = `
      <h3>Samples</h3>
      <div class="results-actions">${sampleZip}</div>
      ${sampleEpochs || `<p class="muted">No samples available.</p>`}
    `;
    sections.appendChild(sampleSection);

    modal.appendChild(sections);
    overlay.appendChild(modal);
    overlay.querySelectorAll(".sample-grid img").forEach((img) => {
      img.addEventListener("click", () => {
        const caption = img.closest("figure")?.querySelector("figcaption")?.textContent || "";
        openLightbox(img.getAttribute("src") || "", caption);
      });
    });
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    document.body.appendChild(overlay);
  } catch (err) {
    alert(err.message || "Failed to load results");
  }
}

function wireManualEditor() {
  const searchInput = document.getElementById("manual-search");
  const faceOnly = document.getElementById("manual-face-only");
  const bulkAdd = document.getElementById("manual-bulk-add");
  const bulkRemove = document.getElementById("manual-bulk-remove");
  const applyAdd = document.getElementById("manual-apply-add");
  const applyRemove = document.getElementById("manual-apply-remove");
  const tagList = document.getElementById("manual-tags-list");
  const tagRemoveBtn = document.getElementById("manual-tags-remove");
  const tagRefreshBtn = document.getElementById("manual-tags-refresh");
  const tagFilterInput = document.getElementById("manual-tags-filter");
  const saveBtn = document.getElementById("manual-save");
  const resumeBtn = document.getElementById("manual-resume");
  const backBtn = document.getElementById("manual-back");
  const loadMoreBtn = document.getElementById("manual-load-more");

  const state = {
    runId: null,
    runName: "",
    images: [],
    filtered: [],
    page: 1,
    pageSize: 48,
    selected: new Set(),
    dirty: new Map(),
    tags: [],
    selectedTags: new Set(),
  };

  const msgEl = document.getElementById("manual-msg");
  const grid = document.getElementById("manual-grid");

  const setMsg = (text, cls = "") => {
    if (!msgEl) return;
    msgEl.textContent = text;
    msgEl.className = `status-msg ${cls}`;
  };

  const parseTags = (text) => {
    return String(text || "")
      .toLowerCase()
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean)
      .map((t) => t.replace(/\s+/g, "_"));
  };

  const render = () => {
    if (!grid) return;
    grid.innerHTML = "";
    const start = 0;
    const end = state.page * state.pageSize;
    const slice = state.filtered.slice(start, end);
    slice.forEach((img) => {
      const card = document.createElement("div");
      card.className = "manual-card";
      const checked = state.selected.has(img.path) ? "checked" : "";
      card.innerHTML = `
        <div class="manual-card-head">
          <label class="inline">
            <input type="checkbox" data-path="${img.path}" ${checked}>
            <span>Select</span>
          </label>
          ${img.isFace ? `<span class="pill pill-queued">face</span>` : ""}
        </div>
        <img src="${img.url}" alt="${img.name}">
        <div class="manual-meta">${img.name}</div>
        <textarea data-path="${img.path}" rows="3" placeholder="comma separated tags">${img.caption || ""}</textarea>
      `;
      const checkbox = card.querySelector('input[type="checkbox"]');
      const textarea = card.querySelector("textarea");
      checkbox?.addEventListener("change", () => {
        if (checkbox.checked) state.selected.add(img.path);
        else state.selected.delete(img.path);
      });
      textarea?.addEventListener("input", () => {
        const val = textarea.value;
        img.caption = val;
        state.dirty.set(img.path, val);
      });
      grid.appendChild(card);
    });
    if (loadMoreBtn) {
      loadMoreBtn.disabled = state.filtered.length <= state.page * state.pageSize;
    }
  };

  const renderTags = () => {
    if (!tagList) return;
    tagList.innerHTML = "";
    const needle = String(tagFilterInput?.value || "").trim().toLowerCase();
    const visible = needle
      ? state.tags.filter((item) => item.tag.toLowerCase().includes(needle))
      : state.tags;
    if (!visible.length) {
      tagList.innerHTML = `<p class="muted">No tags found.</p>`;
      return;
    }
    visible.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tag-chip ${state.selectedTags.has(item.tag) ? "active" : ""}`;
      btn.textContent = `${item.tag} · ${item.count}`;
      btn.addEventListener("click", () => {
        if (state.selectedTags.has(item.tag)) {
          state.selectedTags.delete(item.tag);
        } else {
          state.selectedTags.add(item.tag);
        }
        renderTags();
      });
      tagList.appendChild(btn);
    });
  };

  const loadTags = async () => {
    if (!state.runId) return;
    try {
      const res = await fetch(`/api/run/${state.runId}/manual/tags`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Failed to load tags");
      state.tags = data.tags || [];
      state.selectedTags.clear();
      renderTags();
    } catch (err) {
      setMsg(err.message || "Failed to load tags", "error");
    }
  };

  const applyFilters = () => {
    const query = String(searchInput?.value || "").toLowerCase();
    const face = faceOnly?.checked;
    state.filtered = state.images.filter((img) => {
      if (face && !img.isFace) return false;
      if (!query) return true;
      const hay = `${img.name} ${img.caption || ""}`.toLowerCase();
      return hay.includes(query);
    });
    state.page = 1;
    render();
  };

  const applyBulk = (mode) => {
    const input = mode === "add" ? bulkAdd : bulkRemove;
    const tags = parseTags(input?.value || "");
    if (!tags.length) return;
    const targets = state.selected.size
      ? state.images.filter((img) => state.selected.has(img.path))
      : state.filtered;
    for (const img of targets) {
      const current = parseTags(img.caption);
      let next = current;
      if (mode === "add") {
        next = Array.from(new Set([...current, ...tags]));
      } else {
        const remove = new Set(tags);
        next = current.filter((t) => !remove.has(t));
      }
      const joined = next.join(", ");
      img.caption = joined;
      state.dirty.set(img.path, joined);
    }
    applyFilters();
  };

  let removeTimer = null;
  const scheduleRemoveApply = () => {
    if (!bulkRemove) return;
    if (removeTimer) window.clearTimeout(removeTimer);
    removeTimer = window.setTimeout(() => {
      applyBulk("remove");
    }, 400);
  };

  const saveChanges = async () => {
    if (!state.runId) return;
    const updates = Array.from(state.dirty.entries()).map(([path, caption]) => ({ path, caption }));
    if (!updates.length) {
      setMsg("No changes to save.", "ok");
      return;
    }
    setMsg("Saving...");
    try {
      const res = await fetch(`/api/run/${state.runId}/manual/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Save failed");
      setMsg("Saved", "ok");
      state.dirty.clear();
      await loadTags();
    } catch (err) {
      setMsg(err.message || "Save failed", "error");
    }
  };

  const removeSelectedTags = async () => {
    if (!state.runId) return;
    const tags = Array.from(state.selectedTags);
    if (!tags.length) {
      setMsg("Select tags to remove.", "error");
      return;
    }
    setMsg("Removing tags...");
    try {
      const res = await fetch(`/api/run/${state.runId}/manual/tags/remove`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tags }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Remove failed");
      const removeSet = new Set(tags);
      state.images.forEach((img) => {
        const current = parseTags(img.caption);
        const next = current.filter((t) => !removeSet.has(t));
        img.caption = next.join(", ");
      });
      state.dirty.clear();
      applyFilters();
      await loadTags();
      setMsg(`Removed from ${data.updated || 0} images`, "ok");
    } catch (err) {
      setMsg(err.message || "Remove failed", "error");
    }
  };

  const resumePipeline = async () => {
    if (!state.runId) return;
    setMsg("Validating...");
    try {
      const res = await fetch(`/api/run/${state.runId}/manual/commit`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const missing = data.missing?.length ? ` Missing: ${data.missing.length}` : "";
        throw new Error((data.error || "Commit failed") + missing);
      }
      setMsg("Pipeline resumed", "ok");
      setActiveSection("queue");
      refreshQueue();
    } catch (err) {
      setMsg(err.message || "Commit failed", "error");
    }
  };

  if (searchInput) searchInput.addEventListener("input", applyFilters);
  if (faceOnly) faceOnly.addEventListener("change", applyFilters);
  if (applyAdd) applyAdd.addEventListener("click", () => applyBulk("add"));
  if (applyRemove) applyRemove.addEventListener("click", () => applyBulk("remove"));
  if (bulkRemove) bulkRemove.addEventListener("input", scheduleRemoveApply);
  if (tagRemoveBtn) tagRemoveBtn.addEventListener("click", removeSelectedTags);
  if (tagRefreshBtn) tagRefreshBtn.addEventListener("click", loadTags);
  if (tagFilterInput) tagFilterInput.addEventListener("input", renderTags);
  if (saveBtn) saveBtn.addEventListener("click", saveChanges);
  if (resumeBtn) resumeBtn.addEventListener("click", resumePipeline);
  if (backBtn) backBtn.addEventListener("click", () => setActiveSection("queue"));
  if (loadMoreBtn) loadMoreBtn.addEventListener("click", () => {
    state.page += 1;
    render();
  });

  window.openManualEditor = async (id, runName = "") => {
    state.runId = id;
    state.runName = runName;
    state.images = [];
    state.filtered = [];
    state.selected.clear();
    state.dirty.clear();
    setActiveSection("manual");
    setMsg("Loading...");
    try {
      await fetch(`/api/run/${id}/manual/start`, { method: "POST" }).catch(() => null);
      const res = await fetch(`/api/run/${id}/manual/dataset`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Failed to load dataset");
      state.images = data.images || [];
      applyFilters();
      await loadTags();
      setMsg(`Loaded ${state.images.length} images`, "ok");
    } catch (err) {
      setMsg(err.message || "Failed to load dataset", "error");
    }
  };
}

async function openManualEditor(id, runName = "") {
  if (window.openManualEditor) {
    window.openManualEditor(id, runName);
  }
}

function renderAutocharCard(p) {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="label">Preset</div>
        <div class="title">${p.name}</div>
      </div>
      <span class="pill pill-queued">${p.blockPatterns?.length || 0} block</span>
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
