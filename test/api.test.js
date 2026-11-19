const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const http = require('http');

const app = require('../backend/server');

test('Artifacts and model JSON files exist and are valid JSON', () => {
  const files = [
    'artifacts/brain_mri_labels.json',
    'artifacts/brain_mri_classification_report.json',
    'artifacts/brain_mri_training_history.json',
    'artifacts/brain_stroke_ct_labels.json',
    'artifacts/brain_stroke_ct_classification_report.json',
    'artifacts/brain_stroke_ct_training_history.json',
    'model/ct_mri_labels.json',
    'model/ct_mri_training_history.json',
  ];

  for (const relPath of files) {
    const fullPath = path.resolve(__dirname, '..', relPath);
    assert.ok(fs.existsSync(fullPath), `File should exist: ${relPath}`);
    const content = fs.readFileSync(fullPath, 'utf8');
    assert.doesNotThrow(() => JSON.parse(content), `File should be valid JSON: ${relPath}`);
  }
});

test('Express API server routes', async (t) => {
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  t.after(() => {
    server.close();
  });

  await t.test('GET /api/health returns ok', async () => {
    const res = await fetch(`${baseUrl}/api/health`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
  });

  await t.test('GET /api/inference-health returns reachable flag', async () => {
    const res = await fetch(`${baseUrl}/api/inference-health`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.ok(typeof body.reachable === 'boolean');
  });

  await t.test('GET /api/datasets returns datasets summary', async () => {
    const res = await fetch(`${baseUrl}/api/datasets`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.ok(body.mri, 'Should contain mri');
    assert.ok(body.strokeCt, 'Should contain strokeCt');
    assert.ok(body.modality, 'Should contain modality');
    assert.ok(Array.isArray(body.mri.history), 'mri history should be array');
    assert.ok(Array.isArray(body.strokeCt.history), 'strokeCt history should be array');
    assert.ok(Array.isArray(body.modality.history), 'modality history should be array');
  });

  await t.test('GET /api/datasets/:id returns individual dataset', async () => {
    const resMri = await fetch(`${baseUrl}/api/datasets/mri`);
    assert.equal(resMri.status, 200);
    const mri = await resMri.json();
    assert.equal(mri.meta.id, 'mri');

    const resCt = await fetch(`${baseUrl}/api/datasets/stroke-ct`);
    assert.equal(resCt.status, 200);
    const ct = await resCt.json();
    assert.equal(ct.meta.id, 'strokeCt');

    const resMod = await fetch(`${baseUrl}/api/datasets/modality`);
    assert.equal(resMod.status, 200);
    const mod = await resMod.json();
    assert.equal(mod.meta.id, 'modality');
  });

  await t.test('Static files are served', async () => {
    const resIndex = await fetch(`${baseUrl}/`);
    assert.equal(resIndex.status, 200);
    const html = await resIndex.text();
    assert.ok(html.includes('NHẬN DIỆN &amp; PHÂN LOẠI ẢNH CT / MRI') || html.includes('CT / MRI'));

    const resCss = await fetch(`${baseUrl}/styles.css`);
    assert.equal(resCss.status, 200);

    const resJs = await fetch(`${baseUrl}/app.js`);
    assert.equal(resJs.status, 200);
  });
});
