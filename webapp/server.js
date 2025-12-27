import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { execFile, spawn, spawnSync } from 'child_process';
import unzipper from 'unzipper';
import archiver from 'archiver';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Ensure DB URL is present even when .env isn't loaded by the process manager
const DEFAULT_DB_URL = `file:${path.join(__dirname, '..', '_system', 'db', 'db.sqlite')}`;
if (!process.env.DATABASE_URL) {
  process.env.DATABASE_URL = DEFAULT_DB_URL;
}
const DB_BROKER_URL = process.env.DB_BROKER_URL || '';
if (!DB_BROKER_URL) {
  console.error('[init] DB_BROKER_URL is required for webapp DB access.');
  process.exit(1);
}

const app = express();
const PORT = process.env.PORT || 3005;
const HOST = process.env.HOST || '0.0.0.0';
const BUNDLE_ROOT = path.join(__dirname, '..');
const SYSTEM_ROOT = path.join(BUNDLE_ROOT, '_system');
const STORAGE_ROOT = path.join(SYSTEM_ROOT, 'webapp', 'storage');
const UPLOAD_DIR = path.join(STORAGE_ROOT, 'uploads');
const STAGING_DIR = path.join(STORAGE_ROOT, 'staging');
const OUTPUT_DIR = path.join(BUNDLE_ROOT, 'ARCHIVE', 'zips');
const TAGGER_MODELS_DIR = path.join(SYSTEM_ROOT, 'webapp', 'tagger_models');
const LOG_DIR = path.join(__dirname, 'logs');
const ERRORLOG_RETENTION_DAYS = 30;
const ERRORLOG_PRUNE_INTERVAL_MS = 24 * 60 * 60 * 1000;
const WORKER_STALE_SECS = 30;
const STEP_FRESH_SECS = 300;
const SERVICE_PREFIX = path.basename(BUNDLE_ROOT).toLowerCase().includes('frameforge-dev')
  ? 'frameforge-dev'
  : 'frameforge';
const SYSTEMD_SERVICES = {
  broker: `${SERVICE_PREFIX}-db-broker.service`,
  webapp: `${SERVICE_PREFIX}-webapp.service`,
  initiator: `${SERVICE_PREFIX}-initiator.service`,
  orchestrator: `${SERVICE_PREFIX}-orchestrator.service`,
  finisher: `${SERVICE_PREFIX}-finisher.service`,
};
const DOCS_DIR = path.join(BUNDLE_ROOT, 'docs');
const INSITE_DOCS_DIR = path.join(BUNDLE_ROOT, 'insite-docs');
const INPUT_ROOT = path.join(BUNDLE_ROOT, 'INBOX');
const FINAL_OUTPUT = path.join(BUNDLE_ROOT, 'OUTPUTS', 'datasets');
const FINAL_LORA = path.join(BUNDLE_ROOT, 'OUTPUTS', 'loras');
const TRAIN_OUTPUT = path.join(SYSTEM_ROOT, 'trainer', 'output');
const WORKFLOW_WORK = path.join(SYSTEM_ROOT, 'workflow', 'work');
const UPLOADS_DIR = UPLOAD_DIR;
const STAGE_TTL_MS = 5 * 60 * 1000;
const DEFAULT_SETTINGS = {
  capping_fps: 8,
  capping_jpeg_quality: 2,
  selection_target_per_character: 40,
  selection_face_quota: 10,
  selection_hamming_threshold: 6,
  selection_hamming_relaxed: 4,
  output_max_images: 600,
  autotag_general_threshold: 0.55,
  autotag_character_threshold: 0.4,
  autotag_max_tags: 30,
  autotag_model_id: "SmilingWolf/wd-eva02-large-tagger-v3",
  hf_token: "",
  // Trainer defaults (SDXL/Pony)
  trainer_base_model: "darkstorm2150/pony-diffusion-xl-base-1.0",
  trainer_vae: "madebyollin/sdxl-vae-fp16-fix",
  trainer_resolution: 1024,
  trainer_batch_size: 1,
  trainer_grad_accum: 4,
  trainer_epochs: 10,
  trainer_max_train_steps: 6000,
  trainer_learning_rate: 0.0001,
  trainer_te_learning_rate: 0.00005,
  trainer_lr_scheduler: "cosine",
  trainer_lr_warmup_steps: 180,
  trainer_lora_rank: 32,
  trainer_lora_alpha: 32,
  trainer_te_lora_rank: 16,
  trainer_te_lora_alpha: 16,
  trainer_clip_skip: 2,
  trainer_network_dropout: 0.0,
  trainer_caption_dropout: 0.0,
  trainer_shuffle_caption: true,
  trainer_keep_tokens: 1,
  trainer_min_snr_gamma: 5.0,
  trainer_noise_offset: 0.0,
  trainer_weight_decay: 0.01,
  trainer_sample_prompt_1: "front view",
  trainer_sample_prompt_2: "face close up",
  trainer_sample_prompt_3: "sitting, smile",
  trainer_bucket_min_reso: 768,
  trainer_bucket_max_reso: 1024,
  trainer_bucket_step: 64,
  trainer_optimizer: "adamw",
  trainer_use_8bit_adam: true,
  trainer_gradient_checkpointing: true,
  trainer_dataloader_workers: 1,
  trainer_use_prodigy: false,
  trainer_max_grad_norm: 0,
  queue_mode: "running",
  notifications_enabled: false,
  notify_channel_email: false,
  notify_channel_slack: false,
  notify_channel_discord: false,
  notify_channel_webhook: false,
  notify_job_finish: false,
  notify_job_failed: false,
  notify_queue_finish: false,
  smtp_host: "",
  smtp_port: 587,
  smtp_user: "",
  smtp_pass: "",
  smtp_tls: true,
  smtp_ssl: false,
  smtp_from: "",
  smtp_to: "",
  instance_label: "FrameForge Dev",
  instance_url: "",
  slack_webhook_url: "",
  discord_webhook_url: "",
  webhook_url: "",
  webhook_secret: "",
};

function readLogTail(filePath, maxLines = 160) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split(/\r?\n/);
    return lines.slice(-maxLines).join('\n');
  } catch (err) {
    return '';
  }
}

function extractTrainerLogErrors(content) {
  const lines = content.split(/\r?\n/);
  const filtered = [];
  let inTraceback = false;
  for (const line of lines) {
    const trimmed = line.trim();
    const isErrorLine = /ERROR|Exception|FileNotFoundError|CalledProcessError/.test(line);
    const isTraceback = trimmed.startsWith('Traceback (most recent call last):');
    const isProgress = trimmed.startsWith('steps:');
    if (isProgress) {
      continue;
    }
    if (isProgress && inTraceback) {
      inTraceback = false;
    }
    if (isTraceback) {
      inTraceback = true;
      filtered.push(line);
      continue;
    }
    if (inTraceback) {
      filtered.push(line);
      continue;
    }
    if (isErrorLine) {
      filtered.push(line);
    }
  }
  const filteredText = filtered.join('\n').trim();
  let summary = '';
  const missingMatch = content.match(/FileNotFoundError: \[Errno \d+\] No such file or directory: '([^']+)'/);
  if (missingMatch) {
    summary = `Dataset file missing: ${missingMatch[1]}. Training stopped.`;
  } else if (content.includes('Error loading file')) {
    summary = 'Training failed while loading dataset files.';
  } else if (content.includes('CalledProcessError')) {
    summary = 'Training process exited with a non-zero status.';
  }
  return { summary, filteredTail: filteredText };
}

function extractTrainerLogPath(errorLog) {
  const combined = `${errorLog?.errorMessage || ''}\n${errorLog?.errorDetail || ''}\n${errorLog?.logTail || ''}`;
  const match = combined.match(/(\/[^\s]+_system\/trainer\/logs\/[^\s]+\.log)/);
  if (!match) return null;
  const fullPath = path.resolve(match[1]);
  const allowedRoot = path.resolve(SYSTEM_ROOT, 'trainer', 'logs');
  if (!fullPath.startsWith(allowedRoot)) return null;
  if (!fs.existsSync(fullPath)) return null;
  return fullPath;
}

function asBool(val, defaultVal = false) {
  if (val === undefined || val === null) return defaultVal;
  if (typeof val === 'boolean') return val;
  const normalized = String(val).toLowerCase();
  return ['true', '1', 'yes', 'on'].includes(normalized);
}

function normalizeTrainProfileName(val) {
  return String(val || '').toLowerCase().trim();
}

function normalizeSystemdState(raw) {
  const s = String(raw || '').trim().toLowerCase();
  if (s === 'active' || s === 'activating' || s === 'reloading') return 'active';
  if (s === 'failed') return 'failed';
  if (s === 'inactive' || s === 'deactivating') return 'inactive';
  return 'unknown';
}

async function getSystemdState(serviceName) {
  return new Promise((resolve) => {
    execFile('systemctl', ['is-active', serviceName], (err, stdout) => {
      if (err) {
        resolve(normalizeSystemdState(stdout));
        return;
      }
      resolve(normalizeSystemdState(stdout));
    });
  });
}

async function getBrokerHealth(timeoutMs = 1500) {
  if (!DB_BROKER_URL) {
    return { ok: false, message: 'DB_BROKER_URL not configured' };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${DB_BROKER_URL}/health`, { signal: controller.signal });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data?.ok) return { ok: true, message: 'healthy' };
    return { ok: false, message: data?.error || `health ${res.status}` };
  } catch (err) {
    const msg = err?.name === 'AbortError' ? 'health timeout' : err?.message || 'health failed';
    return { ok: false, message: msg };
  } finally {
    clearTimeout(timer);
  }
}

async function hasRecentTrainProgress(thresholdSeconds = STEP_FRESH_SECS) {
  try {
    const runs = await dbQuery(
      "SELECT runId, lastStep FROM Run WHERE status='running' ORDER BY startedAt DESC",
      []
    );
    if (!runs || !runs.length) return false;
    const runIds = runs.map((r) => r.runId);
    const progressMap = await readTrainProgress(runIds);
    const now = Date.now();
    for (const run of runs) {
      const progress = progressMap[run.runId];
      if (progress?.updatedAt) {
        const ageSec = (now - Date.parse(progress.updatedAt)) / 1000;
        if (Number.isFinite(ageSec) && ageSec <= thresholdSeconds) return true;
      }
      const lastStep = String(run.lastStep || "");
      if (lastStep.toLowerCase().startsWith("train_progress")) return true;
    }
    return false;
  } catch {
    return false;
  }
}

function normalizeTrainProfileSettings(input) {
  if (!input) return {};
  if (typeof input === 'string') {
    try {
      const parsed = JSON.parse(input);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (err) {
      throw new Error('invalid settings JSON');
    }
  }
  if (typeof input === 'object') return input;
  return {};
}

async function getTrainProfiles() {
  return dbQuery("SELECT id, name, label, settings, isDefault FROM TrainProfile ORDER BY name ASC", []);
}

async function getDefaultTrainProfileName() {
  const rows = await dbQuery(
    "SELECT name FROM TrainProfile WHERE isDefault=1 ORDER BY id ASC LIMIT 1",
    []
  );
  if (rows?.[0]?.name) return rows[0].name;
  const fallback = await dbQuery("SELECT name FROM TrainProfile ORDER BY id ASC LIMIT 1", []);
  return fallback?.[0]?.name || null;
}

async function resolveTrainProfileName(inputName) {
  const name = normalizeTrainProfileName(inputName);
  if (name) {
    const rows = await dbQuery("SELECT name FROM TrainProfile WHERE name=?", [name]);
    if (rows?.[0]?.name) return rows[0].name;
  }
  return getDefaultTrainProfileName();
}

function detectStep(line = '') {
  const l = line.toLowerCase();
  if (l.includes('step 1: quickrename')) return 'rename';
  if (l.includes('step 2: capping')) return 'cap';
  if (l.includes('images-only input')) return 'images_only';
  if (l.includes('step 2.5: archive')) return 'archive';
  if (l.includes('step 2.5: move capped frames')) return 'move_capped';
  if (l.includes('step 2.5: move remaining source')) return 'merge_inputs';
  if (l.includes('step 3: select caps')) return 'select';
  if (l.includes('step 4: crop and flip')) return 'cropflip';
  if (l.includes('step 5: move finished set')) return 'move_final';
  if (l.includes('step 6: autotag')) return 'autotag';
  if (l.includes('step 6.5: start watcher')) return 'autotag_watch';
  if (l.includes('step 7a: analyze')) return 'train_plan';
  if (l.includes('step 7b: prepare training outputs')) return 'train_stage';
  if (l.includes('step 7c: start watcher')) return 'train_watch';
  if (l.includes('step 7d: training runs')) return 'train_run';
  if (l.includes('workflow_start')) return 'workflow_start';
  return null;
}

function parseTrainProgress(lastStep = '') {
  const text = String(lastStep || '');
  if (!text.toLowerCase().startsWith('train_progress')) return null;
  const stepMatch = text.match(/(?:step\s+)?(\d+)\s*\/\s*(\d+)/i);
  const epochMatch = text.match(/epoch\s+(\d+|\?)\s*(?:\/\s*(\d+))?/i);
  const step = stepMatch ? Number(stepMatch[1]) : null;
  const stepTotal = stepMatch ? Number(stepMatch[2]) : null;
  const epochRaw = epochMatch ? epochMatch[1] : null;
  const epoch = epochRaw && epochRaw !== '?' ? Number(epochRaw) : null;
  const epochTotal = epochMatch && epochMatch[2] ? Number(epochMatch[2]) : null;
  const parts = [];
  if (epoch !== null || epochTotal !== null) {
    const cur = epoch === null ? '?' : epoch;
    const tot = epochTotal === null ? '' : `/${epochTotal}`;
    parts.push(`epoch ${cur}${tot}`);
  }
  if (step !== null && stepTotal !== null) {
    parts.push(`step ${step}/${stepTotal}`);
  }
  return {
    step,
    stepTotal,
    epoch,
    epochTotal,
    label: parts.join(' • '),
  };
}

function ensureBroker() {
  if (!DB_BROKER_URL) {
    throw new Error('DB_BROKER_URL is not configured');
  }
}

async function brokerPost(pathSuffix, payload) {
  ensureBroker();
  const res = await fetch(`${DB_BROKER_URL}${pathSuffix}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || !data?.ok) {
    const msg = data?.error || `broker error ${res.status}`;
    throw new Error(msg);
  }
  return data.data;
}

async function brokerExec(op, args = {}) {
  return brokerPost('/db/exec', { op, args });
}

async function dbQuery(sql, params = []) {
  return brokerPost('/db/query', { op: 'sql_query', args: { sql, params } });
}

async function dbExec(sql, params = []) {
  return brokerPost('/db/exec', { op: 'sql_exec', args: { sql, params } });
}

async function dbCountRunsByStatus(status) {
  const rows = await dbQuery("SELECT COUNT(*) AS count FROM Run WHERE status=?", [status]);
  return Number(rows?.[0]?.count || 0);
}

async function dbFindRunsByStatuses(statuses, orderBySql, limit = null) {
  const placeholders = statuses.map(() => "?").join(",");
  const base = `SELECT * FROM Run WHERE status IN (${placeholders}) ORDER BY ${orderBySql}`;
  const sql = limit ? `${base} LIMIT ?` : base;
  const params = limit ? [...statuses, limit] : statuses;
  return dbQuery(sql, params);
}

async function dbFindRunById(id) {
  const rows = await dbQuery("SELECT * FROM Run WHERE id=?", [id]);
  return rows?.[0] || null;
}

async function dbCreateRun(data) {
  const fields = [
    "runId",
    "name",
    "runName",
    "type",
    "flags",
    "note",
    "status",
    "uploadPath",
    "trainProfile",
  ];
  const values = fields.map((k) => data[k] ?? null);
  await dbExec(
    `INSERT INTO Run (${fields.join(", ")}) VALUES (${fields.map(() => "?").join(", ")})`,
    values
  );
  if (data.status === "queued") {
    await brokerExec("queue_enqueue", { run_id: data.runId });
  }
  const rows = await dbQuery("SELECT * FROM Run WHERE runId=?", [data.runId]);
  return rows?.[0] || null;
}

async function dbUpdateRunFields(id, fields) {
  const keys = Object.keys(fields);
  if (keys.length === 0) return;
  const setSql = keys.map((k) => `${k}=?`).join(", ");
  const params = keys.map((k) => fields[k]);
  params.push(id);
  await dbExec(`UPDATE Run SET ${setSql} WHERE id=?`, params);
}

async function dbFindQueuedRuns() {
  return dbQuery(
    `
    SELECT r.*, q.position AS queuePos
    FROM Run r
    JOIN QueueItem q ON q.runId = r.runId
    WHERE r.status='queued'
    ORDER BY q.position ASC, r.createdAt ASC
    `,
    []
  );
}

async function dbQueueReorder(runId, position) {
  return brokerExec("queue_reorder", { run_id: runId, position });
}

const QUEUE_MODES = ["running", "paused", "stopped"];

function normalizeQueueMode(val, fallback = "running") {
  const v = String(val || "").toLowerCase();
  return QUEUE_MODES.includes(v) ? v : fallback;
}

async function readTrainProgress(runIds = []) {
  if (!Array.isArray(runIds) || runIds.length === 0) return {};
  const placeholders = runIds.map(() => "?").join(",");
  try {
    const rows = await dbQuery(
      `SELECT runId, epoch, epochTotal, step, stepTotal, updatedAt FROM TrainProgress WHERE runId IN (${placeholders})`,
      runIds
    );
    const map = {};
    for (const r of rows || []) {
      map[r.runId] = {
        epoch: r.epoch === null ? null : Number(r.epoch),
        epochTotal: r.epochTotal === null ? null : Number(r.epochTotal),
        step: r.step === null ? null : Number(r.step),
        stepTotal: r.stepTotal === null ? null : Number(r.stepTotal),
        updatedAt: r.updatedAt,
      };
    }
    return map;
  } catch (_err) {
    return {};
  }
}

async function readOrchestratorStatus() {
  try {
    const rows = await dbQuery(
      "SELECT role, pid, state, runId, message, heartbeat FROM WorkerStatus",
      []
    );
    const roles = {};
    for (const row of rows || []) {
      roles[row.role] = row;
    }
    return { roles };
  } catch {
    return null;
  }
}

function deriveOrchestratorState(roles) {
  const states = Object.values(roles || {}).map((r) => String(r?.state || "").toLowerCase());
  if (states.includes("error")) return "error";
  if (states.includes("busy")) return "busy";
  if (states.includes("paused")) return "paused";
  if (states.includes("stopped")) return "stopped";
  if (states.length === 0) return "unknown";
  return "ready";
}

function summarizeOrchestrator(roles) {
  const roleList = Object.values(roles || {});
  const active = roleList.find((r) => r?.runId)?.runId || null;
  const message = roleList.find((r) => r?.message)?.message || "";
  return {
    state: deriveOrchestratorState(roles),
    activeRunId: active,
    message,
    roles,
  };
}

async function getQueueModeSetting() {
  const settings = await getSettingsMap();
  return normalizeQueueMode(settings.queue_mode, "running");
}

async function setQueueMode(mode) {
  const val = normalizeQueueMode(mode, null);
  if (!val) throw new Error("invalid queue mode");
  await dbExec(
    "INSERT INTO Setting (key, value, updatedAt) VALUES (?, ?, CURRENT_TIMESTAMP) " +
      "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updatedAt=CURRENT_TIMESTAMP;",
    ["queue_mode", val]
  );
  return val;
}

function parsePatternText(text = '') {
  const lines = text.split(/\r?\n/);
  const out = [];
  for (const line of lines) {
    const raw = line.trim();
    if (!raw || raw.startsWith('#') || raw.startsWith('//')) continue;
    for (const chunk of raw.split(',')) {
      const item = chunk.trim();
      if (item) out.push(item);
    }
  }
  return out;
}

async function removePath(target) {
  if (!target) return;
  try {
    await fs.promises.rm(target, { recursive: true, force: true });
  } catch (err) {
    console.error('[cleanup] failed to remove', target, err.message);
  }
}

ensureDirs([STORAGE_ROOT, UPLOAD_DIR, STAGING_DIR, OUTPUT_DIR, TAGGER_MODELS_DIR, INPUT_ROOT, LOG_DIR]);
const upload = multer({ dest: UPLOAD_DIR });
const stageUpload = multer({ dest: STAGING_DIR });
const logStream = fs.createWriteStream(path.join(LOG_DIR, 'webapp.log'), { flags: 'a' });

async function ensureTrainProgressTable() {
  try {
    await dbExec(`
      CREATE TABLE IF NOT EXISTS TrainProgress (
        runId TEXT PRIMARY KEY,
        epoch INTEGER,
        epochTotal INTEGER,
        step INTEGER,
        stepTotal INTEGER,
        raw TEXT,
        updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `, []);
  } catch (err) {
    console.error('[db:init] failed to ensure TrainProgress', err);
  }
}
ensureTrainProgressTable();

async function ensureQueueItems() {
  try {
    await brokerExec("queue_backfill", {});
  } catch (err) {
    console.error('[db:init] failed to backfill queue items', err);
  }
}
ensureQueueItems();

async function ensureErrorLogTable() {
  try {
    await dbExec(`
      CREATE TABLE IF NOT EXISTS ErrorLog (
        id INTEGER PRIMARY KEY,
        runId TEXT,
        component TEXT,
        stage TEXT,
        step TEXT,
        errorType TEXT,
        errorCode TEXT,
        errorMessage TEXT,
        errorDetail TEXT,
        logPath TEXT,
        logTail TEXT,
        logMissing BOOLEAN DEFAULT 0,
        createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `, []);
  } catch (err) {
    console.error('[db:init] failed to ensure ErrorLog', err);
  }
}

async function pruneErrorLog() {
  try {
    await dbExec(
      "DELETE FROM ErrorLog WHERE createdAt < datetime('now', ?);",
      [`-${ERRORLOG_RETENTION_DAYS} days`]
    );
  } catch (err) {
    console.error('[db:prune] failed to prune ErrorLog', err);
  }
}

ensureErrorLogTable();
pruneErrorLog();
setInterval(pruneErrorLog, ERRORLOG_PRUNE_INTERVAL_MS);

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use('/docs', express.static(DOCS_DIR));
app.use('/insite-docs', express.static(INSITE_DOCS_DIR));
app.use('/trainer-output', express.static(TRAIN_OUTPUT));
app.use('/final-lora', express.static(FINAL_LORA));


app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.get('/api/orchestrator/status', (_req, res) => {
  readOrchestratorStatus()
    .then((status) => {
      const roles = status?.roles || {};
      res.json({ status: summarizeOrchestrator(roles) });
    })
    .catch(() => res.json({ status: { state: 'unknown', roles: {} } }));
});

app.get('/api/train-profiles', async (_req, res) => {
  try {
    const profiles = await getTrainProfiles();
    const mapped = profiles.map((p) => ({
      ...p,
      settings: JSON.parse(p.settings || '{}'),
    }));
    res.json({ profiles: mapped });
  } catch (err) {
    console.error('[train-profiles:list] failed', err);
    res.status(500).json({ error: 'failed to load train profiles' });
  }
});

app.post('/api/train-profiles', async (req, res) => {
  try {
    const { name, label = '', settings = {}, isDefault = false } = req.body || {};
    if (!name || !String(name).trim()) return res.status(400).json({ error: 'name is required' });
    const normalized = String(name).trim().toLowerCase();
    const settingsObj = normalizeTrainProfileSettings(settings);
    if (isDefault) {
      await dbExec("UPDATE TrainProfile SET isDefault=0", []);
    }
    await dbExec(
      "INSERT INTO TrainProfile (name, label, settings, isDefault, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
      [normalized, String(label || ''), JSON.stringify(settingsObj), isDefault ? 1 : 0]
    );
    const rows = await dbQuery("SELECT id, name, label, settings, isDefault FROM TrainProfile WHERE name=?", [normalized]);
    const created = rows?.[0] || null;
    res.json({ ok: true, profile: created ? { ...created, settings: settingsObj } : null });
  } catch (err) {
    console.error('[train-profiles:create] failed', err);
    res.status(500).json({ error: err.message || 'failed to create train profile' });
  }
});

app.put('/api/train-profiles/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const { name, label = '', settings = {}, isDefault = false } = req.body || {};
    if (!name || !String(name).trim()) return res.status(400).json({ error: 'name is required' });
    const normalized = String(name).trim().toLowerCase();
    const settingsObj = normalizeTrainProfileSettings(settings);
    if (isDefault) {
      await dbExec("UPDATE TrainProfile SET isDefault=0", []);
    }
    await dbExec(
      "UPDATE TrainProfile SET name=?, label=?, settings=?, isDefault=?, updatedAt=CURRENT_TIMESTAMP WHERE id=?",
      [normalized, String(label || ''), JSON.stringify(settingsObj), isDefault ? 1 : 0, id]
    );
    const rows = await dbQuery("SELECT id, name, label, settings, isDefault FROM TrainProfile WHERE id=?", [id]);
    const updated = rows?.[0] || null;
    res.json({ ok: true, profile: updated ? { ...updated, settings: settingsObj } : null });
  } catch (err) {
    console.error('[train-profiles:update] failed', err);
    res.status(500).json({ error: err.message || 'failed to update train profile' });
  }
});

app.delete('/api/train-profiles/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    await dbExec("DELETE FROM TrainProfile WHERE id=?", [id]);
    res.json({ ok: true });
  } catch (err) {
    console.error('[train-profiles:delete] failed', err);
    res.status(500).json({ error: 'failed to delete train profile' });
  }
});

function isWorkerStale(role) {
  if (!role || role.heartbeat === undefined || role.heartbeat === null) return true;
  const age = Math.floor(Date.now() / 1000) - Number(role.heartbeat);
  return Number.isFinite(age) ? age > WORKER_STALE_SECS : true;
}

function mapServiceState(role, hasQueue) {
  if (!role || isWorkerStale(role)) return "Fail";
  const s = String(role.state || "").toLowerCase();
  if (s === "error" || s === "stopped" || s === "paused") return "Fail";
  if (s === "busy") return "Busy";
  if (hasQueue) return "Waiting";
  if (s === "ready" || s === "idle" || s === "blocked" || !s) return "OK";
  return "OK";
}

function roleStatusMessage(role) {
  if (!role) return "offline";
  if (isWorkerStale(role)) return "stale heartbeat";
  return role.message || "";
}

app.get('/api/system/status', async (_req, res) => {
  try {
    const brokerHealth = await getBrokerHealth();
    const [webappState, initiatorState, orchestratorState, finisherState] = await Promise.all([
      getSystemdState(SYSTEMD_SERVICES.webapp),
      getSystemdState(SYSTEMD_SERVICES.initiator),
      getSystemdState(SYSTEMD_SERVICES.orchestrator),
      getSystemdState(SYSTEMD_SERVICES.finisher),
    ]);

    if (!brokerHealth.ok) {
      const services = [
        {
          name: "WebApp",
          state: webappState === 'active' ? "Fail" : "Fail",
          message: webappState === 'active' ? "broker offline" : `service ${webappState}`,
        },
        {
          name: "DB Broker",
          state: "Fail",
          message: brokerHealth.message,
        },
        {
          name: "Initiator",
          state: "Fail",
          message: "broker offline",
        },
        {
          name: "Orchestrator",
          state: "Fail",
          message: "broker offline",
        },
        {
          name: "Finisher",
          state: "Fail",
          message: "broker offline",
        },
      ];
      res.json({ services, queueMode: "unknown" });
      return;
    }

    const mode = await getQueueModeSetting();
    const queued = await dbCountRunsByStatus('queued');
    const queuedInitiated = await dbCountRunsByStatus('queued_initiated');
    const readyFinish = await dbCountRunsByStatus('ready_for_finish');
    const running = await dbCountRunsByStatus('running');
    const readyToTrain = await dbCountRunsByStatus('ready_to_train');
    const manualTagging = await dbCountRunsByStatus('manual_tagging');
    const orchestratorWork = queuedInitiated > 0 || readyToTrain > 0 || running > 0 || manualTagging > 0;
    const initiatorWork = queued > 0;
    const finisherWork = readyFinish > 0;
    const stepsFresh = await hasRecentTrainProgress();

    const resolveState = (systemdState, hasWork, hasSteps) => {
      if (systemdState !== 'active') return { state: 'Fail', message: `service ${systemdState}` };
      if (hasSteps) return { state: 'Busy', message: 'progress active' };
      if (hasWork) return { state: 'Waiting', message: 'no recent progress' };
      return { state: 'OK', message: 'idle' };
    };

    const initiator = resolveState(initiatorState, initiatorWork, false);
    const orchestrator = resolveState(orchestratorState, orchestratorWork, stepsFresh);
    const finisher = resolveState(finisherState, finisherWork, false);
    const services = [
      {
        name: "WebApp",
        state: webappState === 'active' ? "OK" : "Fail",
        message: webappState === 'active' ? (mode === "paused" ? "Queue paused" : "running") : `service ${webappState}`,
      },
      { name: "DB Broker", state: "OK", message: "healthy" },
      {
        name: "Initiator",
        state: initiator.state,
        message: initiator.message,
      },
      {
        name: "Orchestrator",
        state: orchestrator.state,
        message: orchestrator.message,
      },
      {
        name: "Finisher",
        state: finisher.state,
        message: finisher.message,
      },
    ];
    res.json({ services, queueMode: mode });
  } catch (err) {
    console.error('[system:status] failed', err);
    res.status(500).json({ error: 'failed to load system status' });
  }
});

app.get('/api/queue', async (_req, res) => {
  try {
    const queuedItems = await dbFindQueuedRuns();
    const activeItems = await dbFindRunsByStatuses(
      ['queued_initiated', 'running', 'ready_for_finish', 'manual_tagging', 'ready_to_train'],
      "createdAt ASC, id ASC"
    );
    const items = [...queuedItems, ...activeItems];
    const runIds = items.map((r) => r.runId);
    const progressMap = await readTrainProgress(runIds);
    const queue = items.map((r) => {
      const run = withParsedFlags(r);
      const progress = progressMap[run.runId] || parseTrainProgress(run.lastStep);
      if (progress) run.trainProgress = progress;
      if (run.flags?.train) {
        run.trainEpochs = getTrainEpochs(run.runName);
      }
      return run;
    });
    res.json({ queue });
  } catch (err) {
    console.error('[queue] failed', err);
    res.status(500).json({ error: 'failed to load queue' });
  }
});

app.post('/api/queue/reorder', async (req, res) => {
  try {
    const runId = String(req.body?.runId || "").trim();
    const position = Number(req.body?.position);
    if (!runId || !Number.isFinite(position)) {
      return res.status(400).json({ error: 'runId and position required' });
    }
    const rows = await dbQuery("SELECT status FROM Run WHERE runId=?", [runId]);
    const status = rows?.[0]?.status || "";
    if (String(status).toLowerCase() !== "queued") {
      return res.status(400).json({ error: 'run is not queued' });
    }
    await dbQueueReorder(runId, Math.trunc(position));
    res.json({ ok: true });
  } catch (err) {
    console.error('[queue:reorder] failed', err);
    res.status(500).json({ error: 'failed to reorder queue' });
  }
});

app.get('/api/history', async (_req, res) => {
  try {
    const items = await dbFindRunsByStatuses(
      ['done', 'failed', 'failed_initiator', 'failed_worker', 'failed_finish'],
      "finishedAt DESC",
      50
    );
    const runIds = items.map((r) => r.runId);
    const progressMap = await readTrainProgress(runIds);
    const history = items.map((r) => {
      const run = withParsedFlags(r);
      const progress = progressMap[run.runId] || parseTrainProgress(run.lastStep);
      if (progress) run.trainProgress = progress;
      if (run.flags?.train) {
        run.trainEpochs = getTrainEpochs(run.runName);
      }
      return run;
    });
    res.json({ history });
  } catch (err) {
    console.error('[history] failed', err);
    res.status(500).json({ error: 'failed to load history' });
  }
});

app.get('/api/run/:id/progress', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const rows = await dbQuery(
      "SELECT runId, epoch, epochTotal, step, stepTotal, updatedAt, raw FROM TrainProgress WHERE runId=?",
      [run.runId]
    );
    const row = rows && rows[0];
    const progress = row
      ? {
          epoch: row.epoch === null ? null : Number(row.epoch),
          epochTotal: row.epochTotal === null ? null : Number(row.epochTotal),
          step: row.step === null ? null : Number(row.step),
          stepTotal: row.stepTotal === null ? null : Number(row.stepTotal),
          updatedAt: row.updatedAt,
          raw: row.raw,
        }
      : parseTrainProgress(run.lastStep);
    res.json({ progress });
  } catch (err) {
    console.error('[run:progress] failed', err);
    res.status(500).json({ error: 'failed to load progress' });
  }
});

app.get('/api/error-log/:runId', async (req, res) => {
  try {
    const runId = String(req.params.runId || "").trim();
    if (!runId) return res.status(400).json({ error: 'invalid runId' });
    const rows = await dbQuery(
      "SELECT * FROM ErrorLog WHERE runId=? ORDER BY createdAt DESC LIMIT 1",
      [runId]
    );
    const row = rows?.[0] || null;
    if (!row) return res.json({ errorLog: null });
    const trainerLogPath = extractTrainerLogPath(row);
    const trainerLogTail = trainerLogPath ? readLogTail(trainerLogPath, 400) : '';
    const trainerLogParsed = trainerLogTail ? extractTrainerLogErrors(trainerLogTail) : { summary: '', filteredTail: '' };
    res.json({
      errorLog: row,
      trainerLog: trainerLogPath
        ? {
            path: trainerLogPath,
            summary: trainerLogParsed.summary,
            filteredTail: trainerLogParsed.filteredTail,
          }
        : null,
    });
  } catch (err) {
    console.error('[error-log] failed', err);
    res.status(500).json({ error: 'failed to load error log' });
  }
});

app.get('/api/autochar', async (_req, res) => {
  try {
    const items = await dbQuery("SELECT * FROM AutoCharPreset ORDER BY name ASC", []);
    const mapped = items.map((p) => ({
      ...p,
      blockPatterns: JSON.parse(p.blockPatterns || '[]'),
      allowPatterns: JSON.parse(p.allowPatterns || '[]'),
    }));
    res.json({ presets: mapped });
  } catch (err) {
    console.error('[autochar:list] failed', err);
    res.status(500).json({ error: 'failed to load presets' });
  }
});

app.post('/api/autochar', async (req, res) => {
  try {
    const { name, description = '', blockPatterns = '', allowPatterns = '' } = req.body;
    if (!name || !String(name).trim()) return res.status(400).json({ error: 'name is required' });
    const blockArr = Array.isArray(blockPatterns) ? blockPatterns : parsePatternText(blockPatterns);
    const allowArr = []; // allow patterns removed
    await dbExec(
      "INSERT INTO AutoCharPreset (name, description, blockPatterns, allowPatterns, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
      [
        String(name).trim(),
        String(description || ''),
        JSON.stringify(blockArr),
        JSON.stringify(allowArr),
      ]
    );
    const createdRows = await dbQuery("SELECT * FROM AutoCharPreset WHERE name=?", [String(name).trim()]);
    const created = createdRows?.[0] || null;
    res.json({ ok: true, preset: { ...created, blockPatterns: blockArr, allowPatterns: allowArr } });
  } catch (err) {
    console.error('[autochar:create] failed', err);
    res.status(500).json({ error: 'failed to create preset' });
  }
});

app.put('/api/autochar/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const { name, description = '', blockPatterns = '', allowPatterns = '' } = req.body;
    if (!name || !String(name).trim()) return res.status(400).json({ error: 'name is required' });
    const blockArr = Array.isArray(blockPatterns) ? blockPatterns : parsePatternText(blockPatterns);
    const allowArr = []; // allow patterns removed
    await dbExec(
      "UPDATE AutoCharPreset SET name=?, description=?, blockPatterns=?, allowPatterns=?, updatedAt=CURRENT_TIMESTAMP WHERE id=?",
      [
        String(name).trim(),
        String(description || ''),
        JSON.stringify(blockArr),
        JSON.stringify(allowArr),
        id,
      ]
    );
    const rows = await dbQuery("SELECT * FROM AutoCharPreset WHERE id=?", [id]);
    const updated = rows?.[0] || null;
    res.json({ ok: true, preset: { ...updated, blockPatterns: blockArr, allowPatterns: allowArr } });
  } catch (err) {
    console.error('[autochar:update] failed', err);
    res.status(500).json({ error: 'failed to update preset' });
  }
});

app.delete('/api/autochar/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    await dbExec("DELETE FROM AutoCharPreset WHERE id=?", [id]);
    res.json({ ok: true });
  } catch (err) {
    console.error('[autochar:delete] failed', err);
    res.status(500).json({ error: 'failed to delete preset' });
  }
});

app.delete('/api/run/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });

    await removeRunArtifacts(run);
    await dbExec("DELETE FROM Run WHERE id=?", [id]);
    res.json({ ok: true });
  } catch (err) {
    console.error('[delete run] failed', err);
    res.status(500).json({ error: 'delete failed' });
  }
});

app.post('/api/run/:id/stop', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    // Kill workflow processes with RUN_ID env
    try {
      spawnSync('pkill', ['-f', `RUN_ID=${run.runId}`]);
    } catch (_err) {
      // ignore
    }
    // Kill kohya processes for this run (match config file name)
    try {
      const jobToml = `${run.runName}.toml`;
      spawnSync('pkill', ['-f', `sdxl_train_network.py --config_file .*${jobToml}`], { shell: true });
    } catch (_err) {
      // ignore
    }
    // Clean dirs and trainer outputs
    await cleanupWorkingDirs(run);
    await removeRunArtifacts(run);
    await brokerExec("mark_run_status", {
      run_id_db: run.id,
      status: "failed",
      last_step: "failed",
      error: "stopped by user",
      finished: true,
    });
    res.json({ ok: true });
  } catch (err) {
    console.error('[run:stop] failed', err);
    res.status(500).json({ error: 'failed to stop run' });
  }
});

app.get('/api/run/:id/samples', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    if (run.runName.includes('..') || run.runName.includes('/') || run.runName.includes('\\')) {
      return res.status(400).json({ error: 'invalid run name' });
    }
    const samples = collectRunSamples(run.runName);
    res.json({ samples });
  } catch (err) {
    console.error('[run:samples] failed', err);
    res.status(500).json({ error: 'failed to load samples' });
  }
});

app.get('/api/run/:id/results', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    if (run.runName.includes('..') || run.runName.includes('/') || run.runName.includes('\\')) {
      return res.status(400).json({ error: 'invalid run name' });
    }
    const datasetZipName = `${run.name}.zip`;
    const loraZipName = `${run.name}_lora.zip`;
    const datasetZipPath = path.join(OUTPUT_DIR, datasetZipName);
    const loraZipPath = path.join(OUTPUT_DIR, loraZipName);
    const datasetZip = fs.existsSync(datasetZipPath) ? `/api/download/${encodeURIComponent(datasetZipName)}` : run.datasetDownload || null;
    const loraZip = fs.existsSync(loraZipPath) ? `/api/download/${encodeURIComponent(loraZipName)}` : run.loraDownload || null;

    const loraFiles = listLoraFiles(run.runName);
    const sampleGroups = collectRunSampleGroups(run.runName);
    const zipInfo = listSampleZips(run.name);
    const sampleEpochs = sampleGroups.map((group) => {
      const epoch = group.epoch;
      const zip = epoch !== null ? (zipInfo.epochZips[epoch] || null) : null;
      return {
        epoch,
        label: epoch === null ? "Other" : `Epoch ${epoch}`,
        zip,
        images: group.images,
      };
    });

    res.json({
      datasetZip,
      loraZip,
      samplesZip: zipInfo.allZip,
      loraFiles,
      sampleEpochs,
    });
  } catch (err) {
    console.error('[run:results] failed', err);
    res.status(500).json({ error: 'failed to load results' });
  }
});

app.get('/manual-file/:runName/*', (req, res) => {
  const runName = req.params.runName;
  const rel = req.params[0] || '';
  if (!runName || rel.includes('..')) return res.status(400).json({ error: 'invalid path' });
  const root = resolveManualDatasetRoot(runName);
  if (!root) return res.status(404).json({ error: 'not found' });
  const filePath = path.join(root, rel);
  if (!filePath.startsWith(root)) return res.status(400).json({ error: 'invalid path' });
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'missing file' });
  res.sendFile(filePath);
});

app.post('/api/run/:id/manual/start', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    await dbUpdateRunFields(run.id, { status: 'manual_tagging', lastStep: 'manual_pause' });
    await dbExec(
      "UPDATE RunPlan SET status='done', updatedAt=CURRENT_TIMESTAMP WHERE runId=? AND step='manual_pause'",
      [run.runId]
    );
    res.json({ ok: true });
  } catch (err) {
    console.error('[manual:start] failed', err);
    res.status(500).json({ error: 'failed to start manual tagging' });
  }
});

app.get('/api/run/:id/manual/dataset', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    const images = listManualImages(root, run.runName);
    await dbExec(
      "UPDATE RunPlan SET status='done', updatedAt=CURRENT_TIMESTAMP WHERE runId=? AND step='manual_edit'",
      [run.runId]
    );
    res.json({ images, root: run.runName });
  } catch (err) {
    console.error('[manual:dataset] failed', err);
    res.status(500).json({ error: 'failed to load manual dataset' });
  }
});

app.post('/api/run/:id/manual/update', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    const updates = req.body?.updates;
    if (!Array.isArray(updates) || updates.length === 0) {
      return res.status(400).json({ error: 'no updates provided' });
    }
    for (const upd of updates) {
      const rel = String(upd.path || '');
      if (!rel || rel.includes('..')) continue;
      const full = path.join(root, rel);
      if (!full.startsWith(root)) continue;
      const caption = normalizeTags(upd.caption || '');
      const txtPath = full.replace(/\.(jpg|jpeg|png|webp|bmp)$/i, '.txt');
      fs.writeFileSync(txtPath, caption, 'utf-8');
    }
    res.json({ ok: true });
  } catch (err) {
    console.error('[manual:update] failed', err);
    res.status(500).json({ error: 'failed to update captions' });
  }
});

app.get('/api/run/:id/manual/tags', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    const images = listManualImages(root, run.runName);
    const counts = {};
    for (const img of images) {
      const tags = splitNormalizedTags(img.caption);
      for (const tag of tags) {
        counts[tag] = (counts[tag] || 0) + 1;
      }
    }
    const tags = Object.entries(counts)
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return a.tag.localeCompare(b.tag);
      });
    res.json({ tags });
  } catch (err) {
    console.error('[manual:tags] failed', err);
    res.status(500).json({ error: 'failed to load tags' });
  }
});

app.post('/api/run/:id/manual/tags/remove', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    const tags = normalizeTagList(req.body?.tags || []);
    if (!tags.length) return res.status(400).json({ error: 'no tags provided' });
    ensureManualBackup(root);
    const images = listManualImages(root, run.runName);
    const tagSet = new Set(tags);
    let updated = 0;
    for (const img of images) {
      const current = splitNormalizedTags(img.caption);
      const next = current.filter((t) => !tagSet.has(t));
      const nextText = next.join(', ');
      if (nextText !== normalizeTags(img.caption)) {
        const full = path.join(root, img.path);
        const txtPath = full.replace(/\.(jpg|jpeg|png|webp|bmp)$/i, '.txt');
        fs.writeFileSync(txtPath, nextText, 'utf-8');
        updated += 1;
      }
    }
    res.json({ ok: true, updated });
  } catch (err) {
    console.error('[manual:tags:remove] failed', err);
    res.status(500).json({ error: 'failed to remove tags' });
  }
});

app.post('/api/run/:id/manual/commit', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const run = await dbFindRunById(id);
    if (!run) return res.status(404).json({ error: 'not found' });
    const root = resolveManualDatasetRoot(run.runName);
    if (!root) return res.status(404).json({ error: 'manual dataset missing' });
    const images = listManualImages(root, run.runName);
    const missing = images.filter((img) => !normalizeTags(img.caption)).map((img) => img.path);
    if (missing.length) {
      return res.status(400).json({ error: 'missing captions', missing });
    }
    ensureManualBackup(root);
    for (const img of images) {
      const normalized = normalizeTags(img.caption);
      const full = path.join(root, img.path);
      const txtPath = full.replace(/\.(jpg|jpeg|png|webp|bmp)$/i, '.txt');
      fs.writeFileSync(txtPath, normalized, 'utf-8');
    }
    const flags = safeParseFlags(run.flags);
    flags.manualTagging = true;
    flags.manualResume = true;
    await dbUpdateRunFields(run.id, {
      flags: JSON.stringify(flags),
      status: 'ready_to_train',
      lastStep: 'manual_done',
    });
    await dbExec(
      "UPDATE RunPlan SET status='done', updatedAt=CURRENT_TIMESTAMP WHERE runId=? AND step='manual_done'",
      [run.runId]
    );
    res.json({ ok: true });
  } catch (err) {
    console.error('[manual:commit] failed', err);
    res.status(500).json({ error: 'failed to commit manual tagging' });
  }
});

app.get('/api/settings', async (_req, res) => {
  try {
    const settings = await getSettingsMap();
    res.json({ settings });
  } catch (err) {
    console.error('[settings:list] failed', err);
    res.status(500).json({ error: 'failed to load settings' });
  }
});

app.put('/api/settings', async (req, res) => {
  try {
    const payload = req.body || {};
    const allowedKeys = Object.keys(DEFAULT_SETTINGS);
    const updates = {};
    for (const key of allowedKeys) {
      if (payload[key] === undefined || payload[key] === null) continue;
      const val = payload[key];
      if (key === "queue_mode") {
        const norm = normalizeQueueMode(val, null);
        if (!norm) continue;
        updates[key] = norm;
        continue;
      }
      if (typeof val !== 'number' && typeof val !== 'string' && typeof val !== 'boolean') continue;
      updates[key] = val;
    }
    const entries = Object.entries(updates);
    for (const [key, value] of entries) {
      await dbExec(
        "INSERT INTO Setting (key, value, updatedAt) VALUES (?, ?, CURRENT_TIMESTAMP) " +
          "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updatedAt=CURRENT_TIMESTAMP;",
        [key, String(value)]
      );
    }
    const settings = await getSettingsMap();
    res.json({ ok: true, settings });
  } catch (err) {
    console.error('[settings:update] failed', err);
    res.status(500).json({ error: 'failed to save settings' });
  }
});

app.get('/api/queue/state', async (_req, res) => {
  try {
    const mode = await getQueueModeSetting();
    const orch = await readOrchestratorStatus();
    res.json({ mode, orchestrator: summarizeOrchestrator(orch?.roles || {}) });
  } catch (err) {
    console.error('[queue:state] failed', err);
    res.status(500).json({ error: 'failed to read queue state' });
  }
});

app.post('/api/queue/state', async (req, res) => {
  try {
    const { mode, action } = req.body || {};
    let next = mode || action;
    if (String(next).toLowerCase() === 'restart') {
      next = 'running';
    }
    const saved = await setQueueMode(next);
    const orch = await readOrchestratorStatus();
    res.json({ ok: true, mode: saved, orchestrator: summarizeOrchestrator(orch?.roles || {}) });
  } catch (err) {
    console.error('[queue:update] failed', err);
    res.status(400).json({ error: err.message || 'failed to update queue state' });
  }
});

app.post('/api/upload', upload.array('zip', 20), async (req, res) => {
  try {
    const { autotag = 'true', autochar = 'true', facecap = 'false', train = 'false', gpu = 'true', imagesOnly = 'false', tagverify = 'false', manualTagging = 'false', note = '', trainProfile: trainProfileRaw = '' } = req.body;
    const files = req.files || [];
    if (!files.length) {
      return res.status(400).json({ error: 'ZIP file is required' });
    }
    const manualFlag = asBool(manualTagging, false);
    const autocharPresetsRaw = req.body.autocharPresets;
    let autocharPresets = [];
    if (Array.isArray(autocharPresetsRaw)) {
      autocharPresets = autocharPresetsRaw.filter(Boolean).map(String);
    } else if (typeof autocharPresetsRaw === 'string' && autocharPresetsRaw.trim()) {
      autocharPresets = autocharPresetsRaw.split(',').map((s) => s.trim()).filter(Boolean);
    }
    const type = 'general';
    if (!autocharPresets.length && asBool(autochar, true)) {
      autocharPresets = ['general'];
    }
    const trainProfile = await resolveTrainProfileName(trainProfileRaw);
    const autocharPreset = asBool(autochar, true) ? autocharPresets.join(',') : null;
    const flags = {
      autotag: manualFlag ? true : asBool(autotag, true),
      autochar: manualFlag ? false : asBool(autochar, true),
      facecap: asBool(facecap, false),
      train: asBool(train, false),
      gpu: asBool(gpu, true),
      imagesOnly: asBool(imagesOnly, false),
      tagverify: manualFlag ? false : asBool(tagverify, false),
      manualTagging: manualFlag,
      manualResume: false,
      autocharPreset,
      autocharPresets,
      trainProfile,
    };
    const createdRuns = [];
    for (const file of files) {
      const cleanName = sanitizeName(file.originalname.replace(/\.zip$/i, '') || 'dataset');
      const runId = generateRunId();
      const runName = `${runId}_${cleanName}`;
      const destPath = path.join(UPLOAD_DIR, `${runName}.zip`);
      fs.renameSync(file.path, destPath);
      const run = await dbCreateRun({
        runId,
        name: cleanName,
        runName,
        type,
        flags: JSON.stringify(flags),
        note,
        status: 'queued',
        uploadPath: destPath,
        trainProfile,
      });
      createdRuns.push(run);
      log(`[queue] enqueued ${run.runId} ${run.name}`);
    }
    res.json({ ok: true, runs: createdRuns.map(withParsedFlags) });
    log('[queue] in-app queue runner disabled');
  } catch (err) {
    console.error(err);
    log(`[error] upload failed ${err.message}`);
    res.status(500).json({ error: 'upload failed' });
  }
});

app.get('/api/tagger-models', async (_req, res) => {
  try {
    await syncTaggerModels();
    const models = await dbQuery("SELECT * FROM TaggerModel ORDER BY updatedAt DESC", []);
    res.json({ models });
  } catch (err) {
    console.error('[tagger:list] failed', err);
    res.status(500).json({ error: 'failed to load models' });
  }
});

app.post('/api/tagger-models/download', async (req, res) => {
  try {
    const { repoId } = req.body || {};
    if (!repoId || !String(repoId).trim()) return res.status(400).json({ error: 'repoId required' });
    const modelId = String(repoId).trim();
    const safeName = sanitizeName(modelId.replace(/[\\/]/g, '-'));
    const dest = path.join(TAGGER_MODELS_DIR, safeName);
    await dbExec(
      "INSERT INTO TaggerModel (repoId, name, folder, size, status, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) " +
        "ON CONFLICT(repoId) DO UPDATE SET name=excluded.name, folder=excluded.folder, size=excluded.size, status=excluded.status, updatedAt=CURRENT_TIMESTAMP;",
      [modelId, safeName, dest, 0, 'installing']
    );
    await fs.promises.rm(dest, { recursive: true, force: true });
    await fs.promises.mkdir(dest, { recursive: true });
    const settings = await getSettingsMap();
    const token = settings.hf_token ? String(settings.hf_token) : undefined;
    const files = ['selected_tags.csv', 'config.json', 'model.safetensors'];
    let totalSize = 0;
    for (const file of files) {
      const downloaded = await downloadFromHub(modelId, file, dest, token);
      totalSize += downloaded;
    }
    await dbExec(
      "INSERT INTO TaggerModel (repoId, name, folder, size, status, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) " +
        "ON CONFLICT(repoId) DO UPDATE SET name=excluded.name, folder=excluded.folder, size=excluded.size, status=excluded.status, updatedAt=CURRENT_TIMESTAMP;",
      [modelId, safeName, dest, totalSize, 'ready']
    );
    const recordRows = await dbQuery("SELECT * FROM TaggerModel WHERE repoId=?", [modelId]);
    const record = recordRows?.[0] || null;
    res.json({ ok: true, model: record });
  } catch (err) {
    console.error('[tagger:download] failed', err);
    try {
      const modelId = String(req.body?.repoId || '').trim();
      if (modelId) {
        await dbExec(
          "UPDATE TaggerModel SET status=?, size=?, updatedAt=CURRENT_TIMESTAMP WHERE repoId=?",
          ['missing', 0, modelId]
        );
      }
    } catch (_err) {
      // ignore
    }
    res.status(500).json({ error: err.message || 'download failed' });
  }
});

app.delete('/api/tagger-models/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'invalid id' });
    const rows = await dbQuery("SELECT * FROM TaggerModel WHERE id=?", [id]);
    const model = rows?.[0] || null;
    if (!model) return res.status(404).json({ error: 'not found' });
    await removePath(model.folder);
    await dbExec("DELETE FROM TaggerModel WHERE id=?", [id]);
    res.json({ ok: true });
  } catch (err) {
    console.error('[tagger:delete] failed', err);
    res.status(500).json({ error: 'delete failed' });
  }
});

app.post('/api/upload/stage', stageUpload.single('zip'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({ error: 'ZIP file is required' });
    const uploadId = generateRunId();
    const storedName = `${uploadId}.zip`;
    const destPath = path.join(STAGING_DIR, storedName);
    await fs.promises.rename(file.path, destPath);
    const expiresAt = new Date(Date.now() + STAGE_TTL_MS);
    await dbExec(
      "INSERT INTO StagedUpload (uploadId, originalName, storedName, size, expiresAt, createdAt) " +
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
      [uploadId, file.originalname || 'dataset.zip', storedName, file.size || 0, expiresAt.toISOString()]
    );
    res.json({ ok: true, upload: { id: uploadId, name: file.originalname || 'dataset.zip', size: file.size || 0, expiresAt } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'stage failed' });
  }
});

app.delete('/api/upload/stage/:id', async (req, res) => {
  try {
    const uploadId = req.params.id;
    if (!uploadId) return res.status(400).json({ error: 'invalid id' });
    const rows = await dbQuery("SELECT * FROM StagedUpload WHERE uploadId=?", [uploadId]);
    const record = rows?.[0] || null;
    if (!record) return res.status(404).json({ error: 'not found' });
    const filePath = path.join(STAGING_DIR, record.storedName);
    await removePath(filePath);
    await dbExec("DELETE FROM StagedUpload WHERE uploadId=?", [uploadId]);
    res.json({ ok: true });
  } catch (err) {
    console.error('[stage delete] failed', err);
    res.status(500).json({ error: 'delete failed' });
  }
});

app.get('/api/upload/staged', async (_req, res) => {
  try {
    const items = await dbQuery("SELECT * FROM StagedUpload ORDER BY createdAt ASC", []);
    res.json({ uploads: items });
  } catch (err) {
    console.error('[stage list] failed', err);
    res.status(500).json({ error: 'failed to list staged uploads' });
  }
});

app.post('/api/upload/commit', async (req, res) => {
  try {
    const { uploads = [], autotag = true, autochar = true, facecap = false, train = false, gpu = true, imagesOnly = false, tagverify = false, manualTagging = false, note = '', autocharPresets = [], trainProfile: trainProfileRaw = '' } = req.body || {};
    if (!Array.isArray(uploads) || !uploads.length) return res.status(400).json({ error: 'no uploads to commit' });
    const placeholders = uploads.map(() => "?").join(",");
    const rows = await dbQuery(
      `SELECT * FROM StagedUpload WHERE uploadId IN (${placeholders})`,
      uploads
    );
    if (!rows.length) return res.status(400).json({ error: 'no staged uploads found' });
    const now = Date.now();
    const expired = rows.find((r) => new Date(r.expiresAt).getTime() < now);
    if (expired) return res.status(400).json({ error: `staged upload expired: ${expired.uploadId}` });

    const manualFlag = !!manualTagging;
    const flagsAutocharPresets = Array.isArray(autocharPresets) ? autocharPresets.filter(Boolean).map(String) : [];
    const type = flagsAutocharPresets[0] || 'general';
    const trainProfile = await resolveTrainProfileName(trainProfileRaw);
    const flags = {
      autotag: manualFlag ? true : !!autotag,
      autochar: manualFlag ? false : !!autochar,
      facecap: !!facecap,
      train: !!train,
      gpu: !!gpu,
      imagesOnly: !!imagesOnly,
      tagverify: manualFlag ? false : !!tagverify,
      manualTagging: manualFlag,
      manualResume: false,
      autocharPreset: !!autochar && flagsAutocharPresets.length ? flagsAutocharPresets.join(',') : null,
      autocharPresets: flagsAutocharPresets,
      trainProfile,
    };

    const createdRuns = [];
    for (const row of rows) {
      const cleanName = sanitizeName(row.originalName.replace(/\.zip$/i, '') || 'dataset');
      const runId = generateRunId();
      const runName = `${runId}_${cleanName}`;
      const srcPath = path.join(STAGING_DIR, row.storedName);
      const destPath = path.join(UPLOAD_DIR, `${runName}.zip`);
      await fs.promises.rename(srcPath, destPath);
      const run = await dbCreateRun({
        runId,
        name: cleanName,
        runName,
        type,
        flags: JSON.stringify(flags),
        note: String(note || ''),
        status: 'queued',
        uploadPath: destPath,
        trainProfile,
      });
      createdRuns.push(run);
      await dbExec("DELETE FROM StagedUpload WHERE uploadId=?", [row.uploadId]);
      log(`[queue] enqueued ${run.runId} ${run.name} (staged)`);
    }
    res.json({ ok: true, runs: createdRuns.map(withParsedFlags) });
    log('[queue] in-app queue runner disabled (staged)');
  } catch (err) {
    console.error('[commit] failed', err);
    res.status(500).json({ error: 'commit failed' });
  }
});

app.post('/api/prune', async (_req, res) => {
  try {
    const targets = [
      path.join(BUNDLE_ROOT, 'INBOX'),
      path.join(SYSTEM_ROOT, 'workflow', 'capped'),
      path.join(SYSTEM_ROOT, 'workflow', 'work'),
      path.join(SYSTEM_ROOT, 'workflow', 'raw'),
      path.join(SYSTEM_ROOT, 'workflow', 'ready'),
      path.join(BUNDLE_ROOT, 'ARCHIVE', 'mp4'),
      path.join(BUNDLE_ROOT, 'OUTPUTS', 'datasets'),
      path.join(BUNDLE_ROOT, 'OUTPUTS', 'loras'),
      path.join(SYSTEM_ROOT, 'trainer', 'output'),
      path.join(SYSTEM_ROOT, 'trainer', 'logs'),
      path.join(SYSTEM_ROOT, 'trainer', 'dataset'),
      path.join(BUNDLE_ROOT, 'trainer', 'jobs'),
      path.join(BUNDLE_ROOT, 'ARCHIVE', 'zips'),
    ];
    for (const t of targets) {
      await fs.promises.rm(t, { recursive: true, force: true });
      await fs.promises.mkdir(t, { recursive: true });
    }
    await dbExec("DELETE FROM RunPlan", []);
    await dbExec("DELETE FROM TrainProgress", []);
    await dbExec("DELETE FROM WorkerStatus", []);
    await dbExec("DELETE FROM Run", []);
    await dbExec("DELETE FROM StagedUpload", []);
    res.json({ ok: true });
  } catch (err) {
    console.error('[prune] failed', err);
    res.status(500).json({ error: 'prune failed' });
  }
});
app.get('/api/download/:file', (req, res) => {
  const fileName = req.params.file;
  const filePath = path.join(OUTPUT_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'not found' });
  }
  res.download(filePath);
});

app.listen(PORT, HOST, async () => {
  const safeInit = async (label, fn) => {
    try {
      await fn();
    } catch (err) {
      console.error(`[init] ${label} failed`, err);
    }
  };
  await safeInit('reset-stale-runs', resetStaleRuns);
  await safeInit('seed-settings', seedSettings);
  await safeInit('seed-train-profiles', seedTrainProfiles);
  await safeInit('refresh-model-sizes', refreshModelSizes);
  log(`[init] FrameForge web app listening on http://${HOST}:${PORT}`);
  log("[init] In-app queue runner disabled");
  setInterval(cleanStagedUploads, 60000);
});

async function handleRun(run, flags) {
  const destInput = path.join(INPUT_ROOT, run.runName);
  cleanDir(destInput);
  ensureDirs([path.dirname(destInput)]);
  await extractZip(run.uploadPath, destInput);
  await dbUpdateRunFields(run.id, { lastStep: 'unpacked' });
  await dbUpdateRunFields(run.id, { lastStep: 'workflow' });
  await runWorkflow(run, flags);
  await dbUpdateRunFields(run.id, { lastStep: 'packaging' });
  await packageOutputs(run, flags);
  await dbUpdateRunFields(run.id, { lastStep: 'cleanup' });
  await cleanupWorkingDirs(run);
}

async function extractZip(zipPath, destDir) {
  await fs.promises.mkdir(destDir, { recursive: true });
  return new Promise((resolve, reject) => {
    fs.createReadStream(zipPath)
      .pipe(unzipper.Extract({ path: destDir }))
      .on('close', resolve)
      .on('error', reject);
  });
}

async function runWorkflow(run, flags) {
  const args = ['workflow.py'];
  if (flags.autotag) args.push('--autotag');
  if (flags.autochar) args.push('--autochar');
  if (flags.tagverify) args.push('--tagverify');
  if (flags.facecap) args.push('--facecap');
  if (flags.gpu) args.push('--gpu');
  if (flags.train) args.push('--train');
  if (flags.imagesOnly) args.push('--images-only');

  let lastReportedStep = null;
  const markStep = async (step) => {
    if (!step || step === lastReportedStep) return;
    lastReportedStep = step;
    try {
      await dbUpdateRunFields(run.id, { lastStep: step });
    } catch (err) {
      console.error('[step] failed to update lastStep', err.message);
    }
  };

  const parseSteps = (chunk) => {
    const text = chunk.toString();
    text.split(/\r?\n/).forEach((line) => {
      const step = detectStep(line);
      if (step) markStep(step);
    });
  };

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', args, {
      cwd: BUNDLE_ROOT,
      env: {
        ...process.env,
        RUN_ID: run.runId,
        RUN_DB: path.join(STORAGE_ROOT, 'db.sqlite'),
        AUTOCHAR_PRESET: flags.autocharPreset || '',
        DB_BROKER_URL: process.env.DB_BROKER_URL || '',
      },
    });
    proc.stdout.on('data', (d) => { process.stdout.write(`[run ${run.runId}] ${d}`); logStream.write(`[run ${run.runId}] ${d}`); parseSteps(d); });
    proc.stderr.on('data', (d) => { process.stderr.write(`[run ${run.runId}][err] ${d}`); logStream.write(`[run ${run.runId}][err] ${d}`); parseSteps(d); });
    proc.on('close', (code) => {
      if (code === 0) return resolve();
      reject(new Error(`workflow exited with code ${code}`));
    });
    proc.on('error', (err) => reject(err));
  });
}

async function packageOutputs(run, flags) {
  const flagsObj = flags || {};
  const dsDir = path.join(FINAL_OUTPUT, run.runName);
  if (fs.existsSync(dsDir)) {
    const zipPath = path.join(OUTPUT_DIR, `${run.name}.zip`);
    await zipFolder(dsDir, zipPath);
    await dbUpdateRunFields(run.id, { datasetDownload: `/api/download/${path.basename(zipPath)}` });
  }
  const loraDir = path.join(FINAL_LORA, run.runName);
  if (flagsObj.train && fs.existsSync(loraDir)) {
    const zipPath = path.join(OUTPUT_DIR, `${run.name}_lora.zip`);
    await zipFolder(loraDir, zipPath);
    await dbUpdateRunFields(run.id, { loraDownload: `/api/download/${path.basename(zipPath)}` });
  }
}

function zipFolder(srcDir, destZip) {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(destZip);
    const archive = archiver('zip', { zlib: { level: 9 } });
    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
    archive.pipe(output);
    archive.directory(srcDir, false);
    archive.finalize();
  });
}

async function cleanupWorkingDirs(run) {
  const targets = [
    path.join(INPUT_ROOT, run.runName),
    path.join(SYSTEM_ROOT, 'workflow', 'capped', run.runName),
    path.join(SYSTEM_ROOT, 'workflow', 'raw'),
    path.join(SYSTEM_ROOT, 'workflow', 'work', run.runName),
    path.join(SYSTEM_ROOT, 'workflow', 'ready', run.runName),
  ];
  targets.forEach((t) => cleanDir(t));
}

async function removeRunArtifacts(run) {
  const safeName = run.runName;
  const trigger = safeName.includes('_') ? safeName.split('_').slice(1).join('_') : safeName;
  const targets = [
    path.join(INPUT_ROOT, safeName),
    path.join(SYSTEM_ROOT, 'workflow', 'capped', safeName),
    path.join(SYSTEM_ROOT, 'workflow', 'work', safeName),
    path.join(SYSTEM_ROOT, 'workflow', 'raw'),
    path.join(SYSTEM_ROOT, 'workflow', 'ready', safeName),
    path.join(BUNDLE_ROOT, 'OUTPUTS', 'datasets', safeName),
    path.join(BUNDLE_ROOT, 'ARCHIVE', 'mp4'),
    path.join(BUNDLE_ROOT, 'OUTPUTS', 'loras', safeName),
    path.join(SYSTEM_ROOT, 'trainer', 'output', safeName),
    path.join(SYSTEM_ROOT, 'trainer', 'logs', `${safeName}.log`),
    path.join(SYSTEM_ROOT, 'trainer', 'jobs', `${safeName}.toml`),
    path.join(SYSTEM_ROOT, 'trainer', 'jobs', `${safeName}_sample_prompts.txt`),
    path.join(SYSTEM_ROOT, 'trainer', 'dataset', 'images', `1_${trigger}`),
    path.join(UPLOADS_DIR, `${safeName}.zip`),
    path.join(OUTPUT_DIR, `${run.name}.zip`),
    path.join(OUTPUT_DIR, `${run.name}_lora.zip`),
  ];
  for (const t of targets) {
    if (t.includes(path.join('ARCHIVE', 'mp4'))) {
      // remove only files for this runName
      const files = fs.readdirSync(t, { withFileTypes: true }).filter((f) => f.isFile() && f.name.includes(safeName));
      for (const f of files) {
        await removePath(path.join(t, f.name));
      }
      continue;
    }
    await removePath(t);
  }
}

async function resetStaleRuns() {
  // Do not auto-fail running jobs on webapp restart; training/pipeline runs are owned by workers.
  return;
}

async function seedSettings() {
  const existing = await dbQuery("SELECT key FROM Setting", []);
  const map = new Set(existing.map((s) => s.key));
  for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
    if (map.has(key)) continue;
    await dbExec(
      "INSERT INTO Setting (key, value, createdAt, updatedAt) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
      [key, String(value)]
    );
  }
  await syncTaggerModels();
}

async function seedTrainProfiles() {
  const countRows = await dbQuery("SELECT COUNT(*) AS count FROM TrainProfile", []);
  if (Number(countRows?.[0]?.count || 0) > 0) return;
  const profiles = [
    {
      name: "balanced",
      label: "balanced (576p, batch 2x2, rank 24)",
      isDefault: 1,
      settings: {
        trainer_resolution: 576,
        trainer_batch_size: 2,
        trainer_grad_accum: 2,
        trainer_lora_rank: 24,
        trainer_lora_alpha: 24,
        trainer_gradient_checkpointing: true,
        trainer_max_train_steps: 900,
        trainer_dataloader_workers: 4,
        trainer_bucket_min_reso: 64,
        trainer_bucket_max_reso: 640,
        trainer_bucket_step: 64,
      },
    },
    {
      name: "fast",
      label: "fast (512p, batch 2x2, rank 16)",
      isDefault: 0,
      settings: {
        trainer_resolution: 512,
        trainer_batch_size: 2,
        trainer_grad_accum: 2,
        trainer_lora_rank: 16,
        trainer_lora_alpha: 16,
        trainer_gradient_checkpointing: true,
        trainer_max_train_steps: 900,
        trainer_dataloader_workers: 4,
        trainer_bucket_min_reso: 64,
        trainer_bucket_max_reso: 576,
        trainer_bucket_step: 64,
      },
    },
  ];
  for (const p of profiles) {
    await dbExec(
      "INSERT INTO TrainProfile (name, label, settings, isDefault, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
      [p.name, p.label, JSON.stringify(p.settings), p.isDefault ? 1 : 0]
    );
  }
  log("[seed] train profiles inserted");
}

async function seedAutocharPresets() {
  const countRows = await dbQuery("SELECT COUNT(*) AS count FROM AutoCharPreset", []);
  if (Number(countRows?.[0]?.count || 0) > 0) return;
  const presets = [
    { name: "default", description: "base preset", block: [], allow: [] },
    { name: "human", description: "human defaults", block: [], allow: [] },
    { name: "furry", description: "furry defaults", block: [], allow: [] },
    { name: "dragon", description: "dragon defaults", block: [], allow: [] },
    { name: "daemon", description: "daemon defaults", block: [], allow: [] },
  ];
  for (const p of presets) {
    await dbExec(
      "INSERT INTO AutoCharPreset (name, description, blockPatterns, allowPatterns, createdAt, updatedAt) " +
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
      [p.name, p.description, JSON.stringify(p.block), JSON.stringify(p.allow)]
    );
  }
  log("[seed] autochar presets inserted");
}

async function getSettingsMap() {
  const rows = await dbQuery("SELECT key, value FROM Setting", []);
  const map = { ...DEFAULT_SETTINGS };
  for (const row of rows) {
    const raw = row.value;
    const fallback = map[row.key];
    if (raw === "") {
      map[row.key] = raw;
      continue;
    }
    if (typeof fallback === "number") {
      const num = Number(raw);
      map[row.key] = Number.isFinite(num) ? num : fallback;
      continue;
    }
    map[row.key] = raw;
  }
  return map;
}

async function syncTaggerModels() {
  const defaultRepo = DEFAULT_SETTINGS.autotag_model_id;
  const safeName = sanitizeName(defaultRepo.replace(/[\\/]/g, '-'));
  const dest = path.join(TAGGER_MODELS_DIR, safeName);
  let folder = dest;
  let exists = await fs.promises
    .stat(dest)
    .then((s) => s.isDirectory())
    .catch(() => false);
  if (!exists) {
    const cached = await findCachedModel(defaultRepo);
    if (cached) {
      folder = cached;
      exists = true;
    }
  }
  const size = exists ? await dirSize(folder) : 0;
  await dbExec(
    "INSERT INTO TaggerModel (repoId, name, folder, size, status, createdAt, updatedAt) " +
      "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) " +
      "ON CONFLICT(repoId) DO UPDATE SET name=excluded.name, folder=excluded.folder, size=excluded.size, status=excluded.status, updatedAt=CURRENT_TIMESTAMP;",
    [defaultRepo, safeName, folder, size, exists ? 'ready' : 'missing']
  );
}

async function refreshModelSizes() {
  const models = await dbQuery("SELECT * FROM TaggerModel", []);
  for (const m of models) {
    const exists = await fs.promises
      .stat(m.folder)
      .then((s) => s.isDirectory())
      .catch(() => false);
    const size = exists ? await dirSize(m.folder) : 0;
    await dbExec(
      "UPDATE TaggerModel SET size=?, status=?, updatedAt=CURRENT_TIMESTAMP WHERE repoId=?",
      [size, exists ? 'ready' : 'missing', m.repoId]
    );
  }
}

async function cleanStagedUploads() {
  try {
    const now = new Date().toISOString();
    const expired = await dbQuery("SELECT * FROM StagedUpload WHERE expiresAt < ?", [now]);
    for (const row of expired) {
      const fp = path.join(STAGING_DIR, row.storedName);
      await removePath(fp);
      await dbExec("DELETE FROM StagedUpload WHERE uploadId=?", [row.uploadId]);
    }
  } catch (err) {
    console.error('[stage:cleanup] failed', err);
  }
}

async function dirSize(dir) {
  let total = 0;
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      total += await dirSize(full);
    } else if (entry.isFile()) {
      const s = await fs.promises.stat(full);
      total += s.size || 0;
    }
  }
  return total;
}

function ensureDirs(paths) {
  paths.forEach((p) => {
    if (!fs.existsSync(p)) {
      fs.mkdirSync(p, { recursive: true });
    }
  });
}

function cleanDir(target) {
  if (fs.existsSync(target)) {
    fs.rmSync(target, { recursive: true, force: true });
  }
}

function sanitizeName(name) {
  return name.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'dataset';
}

function generateRunId() {
  return String(crypto.randomInt(0, 999999)).padStart(6, '0');
}

function safeParseFlags(str) {
  try {
    return JSON.parse(str || '{}');
  } catch (_err) {
    return {};
  }
}

function withParsedFlags(run) {
  return { ...run, flags: safeParseFlags(run.flags) };
}

function parseSampleLabel(name = "") {
  const m = String(name).match(/_e(\d+)_([0-9]+)/i);
  if (m) {
    const epoch = parseInt(m[1], 10);
    const img = parseInt(m[2], 10);
    if (!Number.isNaN(epoch) && !Number.isNaN(img)) {
      return `Epoch ${epoch}; Image ${img}`;
    }
  }
  return name;
}

function parseSampleEpoch(name = "") {
  const m = String(name).match(/_e(\d{6})_/i);
  if (!m) return null;
  const epoch = parseInt(m[1], 10);
  return Number.isNaN(epoch) ? null : epoch;
}

function parseSampleIndex(name = "") {
  const m = String(name).match(/_e\d{6}_([0-9]+)/i);
  if (!m) return null;
  const idx = parseInt(m[1], 10);
  return Number.isNaN(idx) ? null : idx;
}

function listSampleFiles(dir, urlPrefix) {
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const files = entries.filter((e) => e.isFile() && /\.(png|jpg|jpeg|webp)$/i.test(e.name));
    return files.map((f) => {
      const full = path.join(dir, f.name);
      const stat = fs.statSync(full);
      return {
        name: f.name,
        url: `${urlPrefix}/${encodeURIComponent(f.name)}`,
        mtime: stat.mtimeMs || 0,
        size: stat.size || 0,
        source: dir,
        label: parseSampleLabel(f.name),
      };
    });
  } catch (_err) {
    return [];
  }
}

function collectRunSamples(runName) {
  const safe = String(runName || '');
  const candidates = [
    {
      dir: path.join(TRAIN_OUTPUT, safe, 'samples'),
      urlPrefix: `/trainer-output/${encodeURIComponent(safe)}/samples`,
    },
    {
      dir: path.join(TRAIN_OUTPUT, safe, 'sample'),
      urlPrefix: `/trainer-output/${encodeURIComponent(safe)}/sample`,
    },
    {
      dir: path.join(TRAIN_OUTPUT, safe),
      urlPrefix: `/trainer-output/${encodeURIComponent(safe)}`,
    },
    {
      dir: path.join(FINAL_LORA, safe, 'samples'),
      urlPrefix: `/final-lora/${encodeURIComponent(safe)}/samples`,
    },
    {
      dir: path.join(FINAL_LORA, safe, 'sample'),
      urlPrefix: `/final-lora/${encodeURIComponent(safe)}/sample`,
    },
    {
      dir: path.join(FINAL_LORA, safe),
      urlPrefix: `/final-lora/${encodeURIComponent(safe)}`,
    },
  ];
  const seen = new Set();
  const collected = [];
  for (const cand of candidates) {
    const files = listSampleFiles(cand.dir, cand.urlPrefix);
    for (const f of files) {
      const key = `${f.name}-${f.size}`;
      if (seen.has(key)) continue;
      seen.add(key);
      collected.push(f);
    }
  }
  collected.sort((a, b) => {
    if (a.mtime === b.mtime) return a.name.localeCompare(b.name);
    return b.mtime - a.mtime;
  });
  return collected;
}

function collectRunSampleGroups(runName) {
  const samples = collectRunSamples(runName);
  const grouped = new Map();
  for (const sample of samples) {
    const epoch = parseSampleEpoch(sample.name);
    const key = epoch === null ? "unknown" : String(epoch);
    if (!grouped.has(key)) {
      grouped.set(key, { epoch, images: [] });
    }
    grouped.get(key).images.push(sample);
  }
  const groups = Array.from(grouped.values());
  groups.sort((a, b) => {
    if (a.epoch === null && b.epoch === null) return 0;
    if (a.epoch === null) return 1;
    if (b.epoch === null) return -1;
    return b.epoch - a.epoch;
  });
  for (const group of groups) {
    group.images.sort((a, b) => {
      const ia = parseSampleIndex(a.name);
      const ib = parseSampleIndex(b.name);
      if (ia !== null && ib !== null && ia !== ib) return ia - ib;
      return a.name.localeCompare(b.name);
    });
  }
  return groups;
}

function listSampleZips(runLabel) {
  const files = [];
  try {
    for (const entry of fs.readdirSync(OUTPUT_DIR, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      files.push(entry.name);
    }
  } catch (_err) {
    return { allZip: null, epochZips: {} };
  }
  const prefix = `${runLabel}_samples`;
  const epochZips = {};
  let allZip = null;
  for (const name of files) {
    if (!name.startsWith(prefix) || !name.endsWith(".zip")) continue;
    const epochMatch = name.match(/_samples_e(\d{6})\.zip$/i);
    if (epochMatch) {
      epochZips[parseInt(epochMatch[1], 10)] = `/api/download/${encodeURIComponent(name)}`;
    } else if (name === `${runLabel}_samples.zip`) {
      allZip = `/api/download/${encodeURIComponent(name)}`;
    }
  }
  return { allZip, epochZips };
}

function listLoraFiles(runName) {
  const dir = path.join(FINAL_LORA, runName);
  const results = [];
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".safetensors")) continue;
      const file = entry.name;
      const finalName = `${runName}.safetensors`;
      if (file === finalName) {
        results.push({
          name: file,
          epoch: null,
          label: "Final",
          url: `/final-lora/${encodeURIComponent(runName)}/${encodeURIComponent(file)}`,
        });
        continue;
      }
      const epochMatch = file.match(new RegExp(`^${runName}-([0-9]{6})\\.safetensors$`));
      if (epochMatch) {
        const epoch = parseInt(epochMatch[1], 10);
        results.push({
          name: file,
          epoch,
          label: `Epoch ${epoch}`,
          url: `/final-lora/${encodeURIComponent(runName)}/${encodeURIComponent(file)}`,
        });
      }
    }
  } catch (_err) {
    return [];
  }
  results.sort((a, b) => {
    if (a.epoch === null && b.epoch === null) return 0;
    if (a.epoch === null) return -1;
    if (b.epoch === null) return 1;
    return b.epoch - a.epoch;
  });
  return results;
}

function normalizeTags(text = '') {
  const raw = String(text || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
  if (!raw) return '';
  const parts = raw
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => t.replace(/\s+/g, '_'));
  const seen = new Set();
  const deduped = [];
  for (const tag of parts) {
    if (seen.has(tag)) continue;
    seen.add(tag);
    deduped.push(tag);
  }
  return deduped.join(', ');
}

function splitNormalizedTags(text = '') {
  const normalized = normalizeTags(text);
  if (!normalized) return [];
  return normalized
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
}

function normalizeTagList(tags = []) {
  const cleaned = [];
  for (const tag of tags) {
    const norm = normalizeTags(String(tag || ''));
    if (!norm) continue;
    splitNormalizedTags(norm).forEach((t) => cleaned.push(t));
  }
  return Array.from(new Set(cleaned));
}

function resolveManualDatasetRoot(runName) {
  if (!runName) return null;
  const finalRoot = path.join(FINAL_OUTPUT, runName);
  if (fs.existsSync(finalRoot)) return finalRoot;
  const direct = path.join(WORKFLOW_WORK, runName);
  if (fs.existsSync(direct)) return direct;
  try {
    const entries = fs.readdirSync(WORKFLOW_WORK, { withFileTypes: true });
    const match = entries.find((e) => e.isDirectory() && e.name === runName);
    if (match) return path.join(WORKFLOW_WORK, match.name);
  } catch (_err) {
    return null;
  }
  return null;
}

function listManualImages(rootDir, runName) {
  const images = [];
  const stack = [rootDir];
  const imageExt = /\.(jpg|jpeg|png|webp|bmp)$/i;
  while (stack.length) {
    const dir = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (_err) {
      continue;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        continue;
      }
      if (!entry.isFile() || !imageExt.test(entry.name)) continue;
      const rel = path.relative(rootDir, full);
      const captionPath = full.replace(imageExt, '.txt');
      let caption = '';
      if (fs.existsSync(captionPath)) {
        try {
          caption = fs.readFileSync(captionPath, 'utf-8');
        } catch (_err) {
          caption = '';
        }
      }
      const stat = fs.statSync(full);
      const isFace = rel.toLowerCase().includes('face');
      images.push({
        path: rel,
        name: entry.name,
        caption: caption || '',
        size: stat.size || 0,
        mtime: stat.mtimeMs || 0,
        isFace,
        url: `/manual-file/${encodeURIComponent(runName)}/${encodeURIComponent(rel)}`,
      });
    }
  }
  images.sort((a, b) => a.path.localeCompare(b.path));
  return images;
}

function ensureManualBackup(rootDir) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupDir = path.join(rootDir, `captions_backup_${stamp}`);
  fs.mkdirSync(backupDir, { recursive: true });
  const stack = [rootDir];
  while (stack.length) {
    const dir = stack.pop();
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.startsWith('captions_backup_')) continue;
        stack.push(full);
        continue;
      }
      if (!entry.isFile() || !entry.name.endsWith('.txt')) continue;
      const rel = path.relative(rootDir, full);
      const dest = path.join(backupDir, rel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(full, dest);
    }
  }
  return backupDir;
}

function getTrainEpochs(runName) {
  try {
    const dir = path.join(BUNDLE_ROOT, 'trainer', 'output', runName);
    const files = fs.readdirSync(dir);
    return files.filter((f) => f.endsWith('.safetensors') && f.includes('_epoch')).length;
  } catch (_err) {
    return 0;
  }
}

async function downloadFromHub(repoId, filename, destDir, token) {
  await fs.promises.mkdir(destDir, { recursive: true });
  const py = `
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="${repoId}", filename="${filename}", token=${token ? `"${token}"` : "None"}, local_dir="${destDir}", local_dir_use_symlinks=False)
print(path)
`;
  const env = { ...process.env, HF_HOME: process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface') };
  const proc = spawnSync('python3', ['-c', py], { env });
  if (proc.status !== 0) {
    const stderr = proc.stderr?.toString().trim() || "";
    let msg = "download failed";
    if (stderr) {
      msg = stderr.split(/\r?\n/).slice(-3).join(" ") || msg;
    }
    throw new Error(msg);
  }
  const lines = proc.stdout?.toString().trim().split(/\r?\n/);
  const downloadedPath = lines && lines.length ? lines[lines.length - 1] : path.join(destDir, filename);
  const stat = await fs.promises.stat(downloadedPath);
  return stat.size || 0;
}

async function findCachedModel(repoId) {
  try {
    const base = process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface');
    const repoKey = repoId.replace(/\//g, '--');
    const snapRoot = path.join(base, 'hub', `models--${repoKey}`, 'snapshots');
    const snaps = await fs.promises.readdir(snapRoot, { withFileTypes: true }).catch(() => []);
    const folders = snaps.filter((d) => d.isDirectory()).map((d) => d.name);
    let latest = null;
    let latestTime = 0;
    for (const folder of folders) {
      const fp = path.join(snapRoot, folder);
      const stat = await fs.promises.stat(fp).catch(() => null);
      if (stat && stat.mtimeMs > latestTime) {
        latestTime = stat.mtimeMs;
        latest = fp;
      }
    }
    if (!latest) return null;
    const hasModel = await fs.promises
      .stat(path.join(latest, 'model.safetensors'))
      .then((s) => s.isFile())
      .catch(() => false);
    return hasModel ? latest : null;
  } catch (_err) {
    return null;
  }
}

function log(line) {
  const msg = `${new Date().toISOString()} ${line}\n`;
  process.stdout.write(msg);
  try {
    logStream.write(msg);
  } catch (_err) {
    // ignore
  }
}
