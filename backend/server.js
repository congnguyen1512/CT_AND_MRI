const http = require('http');
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs/promises');
const fsSync = require('fs');

const app = express();
const BASE_PORT = Number(process.env.PORT) || 3000;
const PORT_TRY_LIMIT = 30;
const BACKEND_ROOT = __dirname;
const PROJECT_ROOT = fsSync.existsSync(path.resolve(BACKEND_ROOT, '..', 'artifacts'))
  ? path.resolve(BACKEND_ROOT, '..')
  : process.cwd();
const FRONTEND_ROOT = path.join(PROJECT_ROOT, 'frontend', 'public');
const INFERENCE_URL = (process.env.INFERENCE_URL || 'http://127.0.0.1:5050').replace(/\/$/, '');

// Enable CORS and basic security headers
app.use((_req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const mime = (file.mimetype || '').toLowerCase();
    const okMime =
      mime === 'image/jpeg' ||
      mime === 'image/png' ||
      mime === 'image/webp' ||
      mime === 'image/jpg';
    const okExt = /\.(jpe?g|png|webp)$/i.test(file.originalname || '');
    cb(null, okMime || okExt);
  },
});
const batchUpload = upload.array('images', 20);

const ARTIFACTS = {
  mri: {
    id: 'mri',
    title: 'MRI - U não',
    labels: 'artifacts/brain_mri_labels.json',
    report: 'artifacts/brain_mri_classification_report.json',
    history: 'artifacts/brain_mri_training_history.json',
  },
  strokeCt: {
    id: 'strokeCt',
    title: 'CT - Đột quỵ não',
    labels: 'artifacts/brain_stroke_ct_labels.json',
    report: 'artifacts/brain_stroke_ct_classification_report.json',
    history: 'artifacts/brain_stroke_ct_training_history.json',
  },
  modality: {
    id: 'modality',
    title: 'Tự động MRI / CT -> chẩn đoán',
    labels: 'model/ct_mri_labels.json',
    history: 'model/ct_mri_training_history.json',
    model: 'model/ct_mri_classifier.keras',
  },
};

const PREDICT_ROUTES = {
  mri: '/predict/mri',
  stroke: '/predict/stroke',
  modality: '/predict/modality',
  auto: '/predict/auto',
};

// Static JSON bundle cache for instantaneous response and 100% reliable Vercel packaging
const STATIC_ARTIFACTS = {
  'artifacts/brain_mri_labels.json': require('../artifacts/brain_mri_labels.json'),
  'artifacts/brain_mri_classification_report.json': require('../artifacts/brain_mri_classification_report.json'),
  'artifacts/brain_mri_training_history.json': require('../artifacts/brain_mri_training_history.json'),
  'artifacts/brain_stroke_ct_labels.json': require('../artifacts/brain_stroke_ct_labels.json'),
  'artifacts/brain_stroke_ct_classification_report.json': require('../artifacts/brain_stroke_ct_classification_report.json'),
  'artifacts/brain_stroke_ct_training_history.json': require('../artifacts/brain_stroke_ct_training_history.json'),
  'model/ct_mri_labels.json': require('../model/ct_mri_labels.json'),
  'model/ct_mri_training_history.json': require('../model/ct_mri_training_history.json'),
};

async function readJson(relPath) {
  if (STATIC_ARTIFACTS[relPath]) {
    return JSON.parse(JSON.stringify(STATIC_ARTIFACTS[relPath]));
  }
  const full = path.join(PROJECT_ROOT, relPath);
  const raw = await fs.readFile(full, 'utf8');
  return JSON.parse(raw);
}

function normalizeModalityLabels(labels) {
  const classToIdx = {};
  Object.entries(labels).forEach(([idx, name]) => {
    classToIdx[name] = Number(idx);
  });
  return {
    class_to_idx: classToIdx,
    idx_to_class: labels,
    image_size: [224, 224],
    model_name: 'mobilenetv2_keras',
  };
}

function normalizeKerasHistory(history) {
  const epochs = Math.max(
    history.accuracy?.length || 0,
    history.loss?.length || 0,
    history.val_accuracy?.length || 0,
    history.val_loss?.length || 0
  );
  return Array.from({ length: epochs }, (_unused, idx) => ({
    epoch: idx + 1,
    train_loss: Number(history.loss?.[idx] || 0),
    train_acc: Number(history.accuracy?.[idx] || 0) * 100,
    val_loss: Number(history.val_loss?.[idx] || 0),
    val_acc: Number(history.val_accuracy?.[idx] || 0) * 100,
  }));
}

app.use(express.static(FRONTEND_ROOT));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

async function requestPrediction(subpath, file) {
  const form = new FormData();
  const blob = new Blob([file.buffer], { type: file.mimetype || 'application/octet-stream' });
  form.append('image', blob, file.originalname || 'upload.jpg');

  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), 120000);
  try {
    const r = await fetch(`${INFERENCE_URL}${subpath}`, {
      method: 'POST',
      body: form,
      signal: ctrl.signal,
    });
    const text = await r.text();
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text || 'Invalid JSON from inference service' };
    }
    if (!r.ok) {
      const detail = body.detail;
      const msg =
        typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
            : body.message || text || r.statusText;
      const error = new Error(msg);
      error.status = r.status;
      throw error;
    }
    return body;
  } finally {
    clearTimeout(tid);
  }
}

async function proxyPredict(subpath, file, res) {
  try {
    const body = await requestPrediction(subpath, file);
    res.json(body);
  } catch (err) {
    const aborted = err.name === 'AbortError';
    res.status(err.status || 503).json({
      error: aborted
        ? 'Inference timeout sau 120s.'
        : err.status
          ? err.message
          : 'Khong ket noi duoc inference API.',
      hint: `Chay rieng: npm run inference (mac dinh ${INFERENCE_URL}). Hoac dat bien INFERENCE_URL.`,
      detail: String(err.message || err),
    });
  }
}

app.get('/api/inference-health', async (_req, res) => {
  try {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 4000);
    const r = await fetch(`${INFERENCE_URL}/health`, { signal: ctrl.signal });
    clearTimeout(tid);
    const j = await r.json();
    res.json({ reachable: true, inferenceUrl: INFERENCE_URL, ...j });
  } catch {
    res.json({
      reachable: false,
      ok: false,
      inferenceUrl: INFERENCE_URL,
      hint: 'Chay terminal: pip install -r requirements.txt && npm run inference',
    });
  }
});

app.post('/api/predict/mri', upload.single('image'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({
      error: 'Thieu file anh. Gui multipart field ten `image` (JPEG/PNG/WebP).',
    });
    return;
  }
  await proxyPredict('/predict/mri', req.file, res);
});

app.post('/api/predict/stroke', upload.single('image'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({
      error: 'Thieu file anh. Gui multipart field ten `image` (JPEG/PNG/WebP).',
    });
    return;
  }
  await proxyPredict('/predict/stroke', req.file, res);
});

app.post('/api/predict/modality', upload.single('image'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({
      error: 'Thieu file anh. Gui multipart field ten `image` (JPEG/PNG/WebP).',
    });
    return;
  }
  await proxyPredict('/predict/modality', req.file, res);
});

app.post('/api/predict/auto', upload.single('image'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({
      error: 'Thieu file anh. Gui multipart field ten `image` (JPEG/PNG/WebP).',
    });
    return;
  }
  await proxyPredict('/predict/auto', req.file, res);
});

app.post('/api/predict/:kind/batch', batchUpload, async (req, res) => {
  const route = PREDICT_ROUTES[req.params.kind];
  if (!route) {
    res.status(404).json({ error: 'Khong co loai du doan', id: req.params.kind });
    return;
  }
  if (!req.files || !req.files.length) {
    res.status(400).json({
      error: 'Thieu file anh. Gui multipart field ten `images` (JPEG/PNG/WebP).',
    });
    return;
  }

  const results = [];
  for (const file of req.files) {
    try {
      const prediction = await requestPrediction(route, file);
      results.push({
        ok: true,
        filename: file.originalname || 'upload.jpg',
        prediction,
      });
    } catch (err) {
      results.push({
        ok: false,
        filename: file.originalname || 'upload.jpg',
        error: err.name === 'AbortError' ? 'Inference timeout sau 120s.' : String(err.message || err),
      });
    }
  }
  res.json({ results });
});

app.get('/api/datasets', async (_req, res) => {
  try {
    const mri = {
      meta: ARTIFACTS.mri,
      labels: await readJson(ARTIFACTS.mri.labels),
      report: await readJson(ARTIFACTS.mri.report),
      history: await readJson(ARTIFACTS.mri.history),
    };
    const strokeCt = {
      meta: ARTIFACTS.strokeCt,
      labels: await readJson(ARTIFACTS.strokeCt.labels),
      report: await readJson(ARTIFACTS.strokeCt.report),
      history: await readJson(ARTIFACTS.strokeCt.history),
    };
    const modality = {
      meta: ARTIFACTS.modality,
      labels: normalizeModalityLabels(await readJson(ARTIFACTS.modality.labels)),
      history: normalizeKerasHistory(await readJson(ARTIFACTS.modality.history)),
    };
    res.json({ mri, strokeCt, modality });
  } catch (err) {
    res.status(500).json({
      error: 'Khong doc duoc file du lieu. Kiem tra thu muc artifacts/ va model/',
      detail: String(err.message),
    });
  }
});

app.get('/api/datasets/:id', async (req, res) => {
  const key = req.params.id === 'stroke-ct' ? 'strokeCt' : req.params.id;
  const spec = ARTIFACTS[key];
  if (!spec) {
    res.status(404).json({ error: 'Khong co dataset', id: req.params.id });
    return;
  }
  try {
    if (key === 'modality') {
      const labels = normalizeModalityLabels(await readJson(spec.labels));
      const history = normalizeKerasHistory(await readJson(spec.history));
      res.json({ meta: spec, labels, history });
      return;
    }
    const labels = await readJson(spec.labels);
    const report = await readJson(spec.report);
    const history = await readJson(spec.history);
    res.json({ meta: spec, labels, report, history });
  } catch (err) {
    res.status(500).json({
      error: 'Khong doc duoc file',
      detail: String(err.message),
    });
  }
});

app.use((err, _req, res, _next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    res.status(400).json({ error: 'File qua lon (toi da 15MB).' });
    return;
  }
  if (err instanceof multer.MulterError) {
    res.status(400).json({ error: err.message || 'Loi upload.' });
    return;
  }
  console.error(err);
  res.status(500).json({ error: 'Loi server.' });
});

function listenWithPortFallback(startPort, attemptsLeft) {
  const server = http.createServer(app);

  server.listen(startPort, () => {
    if (startPort !== BASE_PORT) {
      console.warn(`Cong mac dinh ${BASE_PORT} dang ban, server dung cong ${startPort}.`);
    }
    console.log(`DACN3 dashboard: http://localhost:${startPort}`);
    console.log(`Inference proxy -> ${INFERENCE_URL} (npm run inference)`);
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE' && attemptsLeft > 0) {
      console.warn(`Cong ${startPort} dang duoc dung -> thu cong ${startPort + 1}...`);
      server.close();
      listenWithPortFallback(startPort + 1, attemptsLeft - 1);
      return;
    }
    if (err.code === 'EADDRINUSE') {
      console.error(
        'Het cong thu. Mot tien trinh node khac co the dang chay dashboard.\n' +
          '  - Windows: Task Manager -> ket thuc "Node.js", hoac\n' +
          '  - Chi dinh cong:  $env:PORT=3050; npm start'
      );
      process.exit(1);
      return;
    }
    console.error(err);
    process.exit(1);
  });
}

if (require.main === module) {
  listenWithPortFallback(BASE_PORT, PORT_TRY_LIMIT);
}

module.exports = app;

