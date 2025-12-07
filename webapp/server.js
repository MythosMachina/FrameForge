import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn } from 'child_process';
import unzipper from 'unzipper';
import archiver from 'archiver';
import { PrismaClient } from '@prisma/client';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3005;
const STORAGE_ROOT = path.join(__dirname, 'storage');
const UPLOAD_DIR = path.join(STORAGE_ROOT, 'uploads');
const OUTPUT_DIR = path.join(STORAGE_ROOT, 'output');
const LOG_DIR = path.join(__dirname, 'logs');
const BUNDLE_ROOT = path.join(__dirname, '..');
const INPUT_ROOT = path.join(BUNDLE_ROOT, '10_input');
const FINAL_OUTPUT = path.join(BUNDLE_ROOT, '60_final_output');
const FINAL_LORA = path.join(BUNDLE_ROOT, '90_final_lora');
const UPLOADS_DIR = UPLOAD_DIR;

function asBool(val, defaultVal = false) {
  if (val === undefined || val === null) return defaultVal;
  if (typeof val === 'boolean') return val;
  const normalized = String(val).toLowerCase();
  return ['true', '1', 'yes', 'on'].includes(normalized);
}

function detectStep(line = '') {
  const l = line.toLowerCase();
  if (l.includes('step 1: quickrename')) return 'rename';
  if (l.includes('step 2: capping')) return 'cap';
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
  return null;
}

function parsePatternText(text = '') {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#') && !l.startsWith('//'));
}

async function removePath(target) {
  if (!target) return;
  try {
    await fs.promises.rm(target, { recursive: true, force: true });
  } catch (err) {
    console.error('[cleanup] failed to remove', target, err.message);
  }
}

ensureDirs([STORAGE_ROOT, UPLOAD_DIR, OUTPUT_DIR, INPUT_ROOT, LOG_DIR]);
const prisma = new PrismaClient();
const upload = multer({ dest: UPLOAD_DIR });
const logStream = fs.createWriteStream(path.join(LOG_DIR, 'webapp.log'), { flags: 'a' });

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let workerActive = false;

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.get('/api/queue', async (_req, res) => {
  const items = await prisma.run.findMany({ where: { status: { in: ['queued', 'running'] } }, orderBy: { createdAt: 'asc' } });
  res.json({ queue: items.map(withParsedFlags) });
});

app.get('/api/history', async (_req, res) => {
  const items = await prisma.run.findMany({ where: { status: { in: ['done', 'failed'] } }, orderBy: { finishedAt: 'desc' }, take: 50 });
  res.json({ history: items.map(withParsedFlags) });
});

app.get('/api/autochar', async (_req, res) => {
  try {
    const items = await prisma.autoCharPreset.findMany({ orderBy: { name: 'asc' } });
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
    const created = await prisma.autoCharPreset.create({
      data: {
        name: String(name).trim(),
        description: String(description || ''),
        blockPatterns: JSON.stringify(blockArr),
        allowPatterns: JSON.stringify(allowArr),
      },
    });
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
    const updated = await prisma.autoCharPreset.update({
      where: { id },
      data: {
        name: String(name).trim(),
        description: String(description || ''),
        blockPatterns: JSON.stringify(blockArr),
        allowPatterns: JSON.stringify(allowArr),
      },
    });
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
    await prisma.autoCharPreset.delete({ where: { id } });
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
    const run = await prisma.run.findUnique({ where: { id } });
    if (!run) return res.status(404).json({ error: 'not found' });

    const uploadZip = path.join(UPLOADS_DIR, `${run.runName}.zip`);
    const datasetDir = path.join(FINAL_OUTPUT, run.runName);
    const loraDir = path.join(FINAL_LORA, run.runName);
    const datasetZip = path.join(OUTPUT_DIR, `${run.name}.zip`);
    const loraZip = path.join(OUTPUT_DIR, `${run.name}_lora.zip`);

    await Promise.all([
      removePath(uploadZip),
      removePath(datasetDir),
      removePath(loraDir),
      removePath(datasetZip),
      removePath(loraZip),
    ]);

    await prisma.run.delete({ where: { id } });
    res.json({ ok: true });
  } catch (err) {
    console.error('[delete run] failed', err);
    res.status(500).json({ error: 'delete failed' });
  }
});

app.post('/api/upload', upload.array('zip', 20), async (req, res) => {
  try {
    const { autotag = 'true', autochar = 'true', facecap = 'false', train = 'false', gpu = 'true', note = '' } = req.body;
    const files = req.files || [];
    if (!files.length) {
      return res.status(400).json({ error: 'ZIP file is required' });
    }
    const autocharPresetsRaw = req.body.autocharPresets;
    let autocharPresets = [];
    if (Array.isArray(autocharPresetsRaw)) {
      autocharPresets = autocharPresetsRaw.filter(Boolean).map(String);
    } else if (typeof autocharPresetsRaw === 'string' && autocharPresetsRaw.trim()) {
      autocharPresets = autocharPresetsRaw.split(',').map((s) => s.trim()).filter(Boolean);
    }
    // Default type/preset if none selected
    const type = autocharPresets[0] || 'general';
    if (!autocharPresets.length && asBool(autochar, true)) {
      autocharPresets = [type];
    }
    const autocharPreset = asBool(autochar, true) ? autocharPresets.join(',') : null;
    const flags = {
      autotag: asBool(autotag, true),
      autochar: asBool(autochar, true),
      facecap: asBool(facecap, false),
      train: asBool(train, false),
      gpu: asBool(gpu, true),
      autocharPreset,
      autocharPresets,
    };
    const createdRuns = [];
    for (const file of files) {
      const cleanName = sanitizeName(file.originalname.replace(/\.zip$/i, '') || 'dataset');
      const runId = generateRunId();
      const runName = `${runId}_${cleanName}`;
      const destPath = path.join(UPLOAD_DIR, `${runName}.zip`);
      fs.renameSync(file.path, destPath);
      const run = await prisma.run.create({
        data: {
          runId,
          name: cleanName,
          runName,
          type,
          flags: JSON.stringify(flags),
          note,
          status: 'queued',
          uploadPath: destPath,
        },
      });
      createdRuns.push(run);
      log(`[queue] enqueued ${run.runId} ${run.name}`);
    }
    res.json({ ok: true, runs: createdRuns.map(withParsedFlags) });
    processQueue();
  } catch (err) {
    console.error(err);
    log(`[error] upload failed ${err.message}`);
    res.status(500).json({ error: 'upload failed' });
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

app.listen(PORT, '0.0.0.0', async () => {
  await resetStaleRuns();
  log(`[init] FrameForge web app listening on http://0.0.0.0:${PORT}`);
  processQueue();
  setInterval(processQueue, 10000);
});

async function processQueue() {
  if (workerActive) return;
  const next = await prisma.run.findFirst({ where: { status: 'queued' }, orderBy: { createdAt: 'asc' } });
  if (!next) return;
  const locked = await prisma.run.updateMany({ where: { id: next.id, status: 'queued' }, data: { status: 'running', startedAt: new Date() } });
  if (!locked.count) return;
  workerActive = true;
  log(`[run ${next.runId}] start ${next.name}`);
  try {
    const flags = safeParseFlags(next.flags);
    await handleRun(next, flags);
    await prisma.run.update({
      where: { id: next.id },
      data: { status: 'done', finishedAt: new Date(), error: null, lastStep: 'done' },
    });
    log(`[run ${next.runId}] done`);
  } catch (err) {
    console.error(`[run ${next.runId}] failed:`, err);
    await prisma.run.update({
      where: { id: next.id },
      data: { status: 'failed', finishedAt: new Date(), error: err.message },
    });
    log(`[run ${next.runId}] failed ${err.message}`);
  } finally {
    workerActive = false;
    setImmediate(processQueue);
  }
}

async function handleRun(run, flags) {
  const destInput = path.join(INPUT_ROOT, run.runName);
  cleanDir(destInput);
  ensureDirs([path.dirname(destInput)]);
  await extractZip(run.uploadPath, destInput);
  await prisma.run.update({ where: { id: run.id }, data: { lastStep: 'unpacked' } });
  await prisma.run.update({ where: { id: run.id }, data: { lastStep: 'workflow' } });
  await runWorkflow(run, flags);
  await prisma.run.update({ where: { id: run.id }, data: { lastStep: 'packaging' } });
  await packageOutputs(run, flags);
  await prisma.run.update({ where: { id: run.id }, data: { lastStep: 'cleanup' } });
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
  if (flags.facecap) args.push('--facecap');
  if (flags.gpu) args.push('--gpu');
  if (flags.train) args.push('--train');

  let lastReportedStep = null;
  const markStep = async (step) => {
    if (!step || step === lastReportedStep) return;
    lastReportedStep = step;
    try {
      await prisma.run.update({ where: { id: run.id }, data: { lastStep: step } });
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
    await prisma.run.update({ where: { id: run.id }, data: { datasetDownload: `/api/download/${path.basename(zipPath)}` } });
  }
  const loraDir = path.join(FINAL_LORA, run.runName);
  if (flagsObj.train && fs.existsSync(loraDir)) {
    const zipPath = path.join(OUTPUT_DIR, `${run.name}_lora.zip`);
    await zipFolder(loraDir, zipPath);
    await prisma.run.update({ where: { id: run.id }, data: { loraDownload: `/api/download/${path.basename(zipPath)}` } });
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
    path.join(INPUT_ROOT, run.type, run.runName),
    path.join(BUNDLE_ROOT, '20_capped_frames', run.type, run.runName),
    path.join(BUNDLE_ROOT, '30_work', 'raw', run.type),
    path.join(BUNDLE_ROOT, '30_work', run.runName),
    path.join(BUNDLE_ROOT, '50_ready_autotag', run.runName),
  ];
  targets.forEach((t) => cleanDir(t));
}

async function resetStaleRuns() {
  await prisma.run.updateMany({ where: { status: 'running' }, data: { status: 'failed', error: 'server restart', finishedAt: new Date() } });
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

function log(line) {
  const msg = `${new Date().toISOString()} ${line}\n`;
  process.stdout.write(msg);
  try {
    logStream.write(msg);
  } catch (_err) {
    // ignore
  }
}
