(function () {
  const HISTORY_KEY = 'dacn3-prediction-history-v2';
  const THEME_KEY = 'dacn3-theme';
  const UNCLEAR_MARGIN = 0.15;
  const LOW_CONFIDENCE = 0.75;

  const statusEl = document.getElementById('status');
  const themeToggle = document.getElementById('theme-toggle');
  const datasetConfig = {
    mri: {
      title: 'MRI - U não',
      route: 'mri',
      predictHeading: 'Dự đoán ảnh MRI (AlexNet)',
      buttonText: 'Chạy dự đoán MRI',
    },
    strokeCt: {
      title: 'CT - Đột quỵ',
      route: 'stroke',
      predictHeading: 'Dự đoán ảnh CT (ResNet18)',
      buttonText: 'Chạy dự đoán CT',
    },
    modality: {
      title: 'Tự động MRI / CT -> chẩn đoán',
      route: 'auto',
      predictHeading: 'Tự động nhận diện (MobileNetV2) & chẩn đoán',
      buttonText: 'Chạy tự động',
    },
  };
  const panels = {
    mri: document.getElementById('panel-mri'),
    strokeCt: document.getElementById('panel-strokeCt'),
    modality: document.getElementById('panel-modality'),
  };
  const contents = {
    mri: document.getElementById('content-mri'),
    strokeCt: document.getElementById('content-strokeCt'),
    modality: document.getElementById('content-modality'),
  };
  const tabButtons = document.querySelectorAll('.tab');
  const chartRefs = Object.fromEntries(Object.keys(datasetConfig).map((key) => [key, []]));
  const selectedFiles = Object.fromEntries(Object.keys(datasetConfig).map((key) => [key, []]));
  const latestResults = Object.fromEntries(Object.keys(datasetConfig).map((key) => [key, []]));

  let history = loadHistory();
  let inferenceState = null;

  function setStatus(msg, type) {
    statusEl.textContent = msg || '';
    statusEl.className = 'status' + (type ? ` is-${type}` : '');
  }

  function loadHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function saveHistory() {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 100)));
  }

  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY) || 'light';
    document.documentElement.dataset.theme = saved;
    themeToggle.textContent = saved === 'dark' ? '☀️ Giao diện sáng' : '🌙 Giao diện tối';
  }

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    themeToggle.textContent = next === 'dark' ? '☀️ Giao diện sáng' : '🌙 Giao diện tối';
    Object.keys(chartRefs).forEach((key) => {
      chartRefs[key].forEach((chart) => chart.update());
    });
  }

  function destroyCharts(key) {
    chartRefs[key].forEach((c) => c.destroy());
    chartRefs[key] = [];
  }

  function formatPct(x) {
    if (typeof x !== 'number' || Number.isNaN(x)) return '-';
    return (x * 100).toFixed(2) + '%';
  }

  function reportRows(report) {
    const skip = new Set(['accuracy', 'macro avg', 'weighted avg']);
    const summary = report || {};
    const classes = Object.keys(summary).filter((k) => !skip.has(k));
    const rows = classes.map((cls) => ({ cls, ...summary[cls] }));
    return { classes, rows, summary };
  }

  function getTopEntries(probabilities) {
    return Object.entries(probabilities || {})
      .map(([label, probability]) => [label, Number(probability)])
      .sort((a, b) => b[1] - a[1]);
  }

  function isUnclear(top) {
    if (!top.length) return false;
    const first = top[0][1] || 0;
    const second = top[1]?.[1] || 0;
    return first < LOW_CONFIDENCE || first - second < UNCLEAR_MARGIN;
  }

  function datasetTitle(key) {
    return datasetConfig[key]?.title || key;
  }

  function routeFor(key) {
    return datasetConfig[key]?.route || key;
  }

  async function refreshInferenceState() {
    try {
      const res = await fetch('/api/inference-health');
      inferenceState = await res.json();
    } catch {
      inferenceState = { reachable: false, hint: 'Không kiểm tra được Inference API.' };
    }
    return inferenceState;
  }

  function issueForRoute(route) {
    const state = inferenceState;
    if (!state) return '';
    if (!state.reachable) {
      return state.hint || 'Inference API chưa chạy. Hãy mở terminal và chạy: npm run inference';
    }

    const issues = [];
    if ((route === 'auto' || route === 'mri') && !state.mri?.ready) {
      issues.push('Model MRI (.pth) chưa sẵn sàng.');
    }
    if ((route === 'auto' || route === 'stroke') && !state.stroke_ct?.ready) {
      issues.push('Model CT (.pth) chưa sẵn sàng.');
    }
    if ((route === 'auto' || route === 'modality') && !state.modality?.ready) {
      issues.push(state.modality?.error || 'Model nhận diện CT/MRI (.keras) chưa sẵn sàng.');
    }
    return issues.join(' ');
  }

  function renderActionError(message) {
    return `
      <article class="predict-out is-error">
        <h3>Khong the chay du doan</h3>
        <p class="err">${escapeHtml(message)}</p>
      </article>
    `;
  }

  function diagnosisFrom(prediction) {
    return prediction?.diagnosis || prediction || {};
  }

  function modalityFrom(prediction) {
    return prediction?.modality || null;
  }

  function renderPredictZone(key) {
    const config = datasetConfig[key] || datasetConfig.mri;
    const route = routeFor(key);
    return `
      <div class="predict-zone">
        <div class="section-head">
          <div>
            <h2 class="section-title">${escapeHtml(config.predictHeading)}</h2>
            <p class="predict-hint">
              Kéo thả hoặc chọn nhiều ảnh JPEG / PNG / WebP. Tối đa 20 ảnh, mỗi ảnh tối đa 15MB.
            </p>
          </div>
          <div class="report-actions">
            <button type="button" class="btn-secondary" data-action="export-csv">Save CSV</button>
            <button type="button" class="btn-secondary" data-action="export-pdf">Save PDF</button>
          </div>
        </div>
        <form class="predict-form" data-route="${route}">
          <label class="drop-zone" tabindex="0">
            <input type="file" name="images" accept="image/jpeg,image/png,image/webp" multiple required />
            <span class="drop-title">Kéo thả ảnh vào đây</span>
            <span class="drop-sub">hoặc bấm để chọn file</span>
          </label>
          <div class="file-list" data-role="file-list">Chua chon anh.</div>
          <div class="predict-actions">
            <button type="submit" class="btn-predict">${escapeHtml(config.buttonText)}</button>
            <button type="button" class="btn-secondary" data-action="clear-files">Xóa file</button>
          </div>
        </form>
        <div class="predict-result" hidden></div>
      </div>
      <div class="history-zone">
        <div class="section-head">
          <h2 class="section-title">Lịch sử dự đoán</h2>
          <button type="button" class="btn-secondary" data-action="clear-history">Xóa lịch sử</button>
        </div>
        <div class="history-list" data-role="history-list"></div>
      </div>
    `;
  }

  function renderTop2(top) {
    const first = top[0];
    const second = top[1];
    if (!first) return '';
    const unclear = isUnclear(top);
    return `
      <div class="top2 ${unclear ? 'is-unclear' : ''}">
        <div>
          <span class="top-label">Top 1</span>
          <strong>${escapeHtml(first[0])}</strong>
          <span>${formatPct(first[1])}</span>
        </div>
        <div>
          <span class="top-label">Top 2</span>
          <strong>${escapeHtml(second?.[0] || '-')}</strong>
          <span>${formatPct(second?.[1])}</span>
        </div>
        <p>${unclear ? 'Model chưa rõ ràng, nên xem lại top 2.' : 'Độ tin cậy top 1 đang ổn định.'}</p>
      </div>
    `;
  }

  function renderPredictResultItem(item) {
    if (!item.ok) {
      return `
        <article class="predict-out is-error">
          <h3>${escapeHtml(item.filename)}</h3>
          <p class="err">${escapeHtml(item.error || 'Dự đoán thất bại.')}</p>
        </article>
      `;
    }

    const data = item.prediction;
    const diagnosis = diagnosisFrom(data);
    const modality = modalityFrom(data);
    const top = getTopEntries(diagnosis.probabilities);
    const modalityTop = getTopEntries(modality?.probabilities);
    const rows = top
      .map(([name, p]) => `<tr><td>${escapeHtml(name)}</td><td class="num">${formatPct(p)}</td></tr>`)
      .join('');
    const modalityRows = modalityTop
      .map(([name, p]) => `<tr><td>${escapeHtml(name)}</td><td class="num">${formatPct(p)}</td></tr>`)
      .join('');
    const heatmapSrc = diagnosis.heatmap ? `data:image/png;base64,${diagnosis.heatmap}` : null;
    const overlaySrc = diagnosis.overlay ? `data:image/png;base64,${diagnosis.overlay}` : null;
    const heatmapPanel = heatmapSrc
      ? `<div class="image-panel"><strong>Heatmap</strong><img src="${heatmapSrc}" alt="Heatmap" /></div>`
      : '';
    const overlayPanel = overlaySrc
      ? `<div class="image-panel"><strong>Overlay</strong><img src="${overlaySrc}" alt="Overlay" /></div>`
      : '';

    return `
      <article class="predict-out">
        <div class="predict-summary">
          <div>
            <h3>${escapeHtml(item.filename)}</h3>
            ${
              modality
                ? `<p><strong>Loại ảnh:</strong> ${escapeHtml(modality.label)} (${formatPct(Number(modality.confidence))})</p>
                   <p><strong>Model được gọi:</strong> ${escapeHtml(data.selectedTitle || '-')}</p>`
                : ''
            }
            <p><strong>Nhận dự đoán:</strong> ${escapeHtml(diagnosis.label)} (${formatPct(Number(diagnosis.confidence))})</p>
            <p class="sub">Checkpoint: ${escapeHtml(diagnosis.checkpoint || '-')} - ${escapeHtml(diagnosis.device || '')}</p>
          </div>
        </div>
        ${renderTop2(top)}
        <div class="predict-images ${heatmapSrc || overlaySrc ? '' : 'is-single'}">
          <div class="image-panel">
            <strong>Ảnh gốc</strong>
            <img src="${escapeHtml(item.originalSrc || '')}" alt="Ảnh gốc" />
          </div>
          ${heatmapPanel}
          ${overlayPanel}
        </div>
        <table class="report mini">
          <thead><tr><th>Lớp chẩn đoán</th><th class="num">Xác suất</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        ${
          modalityRows
            ? `<table class="report mini modality-table">
                <thead><tr><th>Loại ảnh</th><th class="num">Xác suất</th></tr></thead>
                <tbody>${modalityRows}</tbody>
              </table>`
            : ''
        }
      </article>
    `;
  }

  function addToHistory(key, results) {
    const dataset = datasetTitle(key);
    const entries = results
      .filter((item) => item.ok)
      .map((item) => {
        const diagnosis = diagnosisFrom(item.prediction);
        const modality = modalityFrom(item.prediction);
        const top = getTopEntries(diagnosis.probabilities);
        return {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          createdAt: new Date().toISOString(),
          dataset,
          filename: item.filename,
          label: diagnosis.label,
          confidence: Number(diagnosis.confidence),
          modalityLabel: modality?.label || '',
          modalityConfidence: modality ? Number(modality.confidence) : '',
          selectedTitle: item.prediction.selectedTitle || '',
          top1: top[0]?.[0] || '',
          top1Prob: top[0]?.[1] || 0,
          top2: top[1]?.[0] || '',
          top2Prob: top[1]?.[1] || 0,
          unclear: isUnclear(top),
          checkpoint: diagnosis.checkpoint || '',
          device: diagnosis.device || '',
        };
      });
    history = [...entries, ...history].slice(0, 100);
    saveHistory();
  }

  function renderHistory(key) {
    const root = contents[key];
    const list = root.querySelector('[data-role="history-list"]');
    if (!list) return;
    const dataset = datasetTitle(key);
    const rows = history.filter((item) => item.dataset === dataset).slice(0, 12);
    if (!rows.length) {
      list.innerHTML = '<p class="muted">Chưa có lịch sử cho tab này.</p>';
      return;
    }
    list.innerHTML = rows
      .map(
        (item) => `
        <div class="history-item">
          <div>
            <strong>${escapeHtml(item.filename)}</strong>
            <span>${new Date(item.createdAt).toLocaleString()}</span>
          </div>
          <div>
            <b>${escapeHtml(item.modalityLabel ? `${item.modalityLabel} -> ${item.label}` : item.label)}</b>
            <span>${formatPct(item.confidence)}${item.unclear ? ' - cần xem lại top 2' : ''}</span>
          </div>
        </div>
      `
      )
      .join('');
  }

  function renderSelectedFiles(key) {
    const root = contents[key];
    const list = root.querySelector('[data-role="file-list"]');
    if (!list) return;
    const files = selectedFiles[key];
    if (!files.length) {
      list.textContent = 'Chưa chọn ảnh.';
      return;
    }
    list.innerHTML = files
      .map((file) => `<span>${escapeHtml(file.name)} <small>${Math.round(file.size / 1024)}KB</small></span>`)
      .join('');
  }

  function bindPredictHandlers(key) {
    const root = contents[key];
    const form = root.querySelector('.predict-form');
    const dropZone = root.querySelector('.drop-zone');
    const input = form?.querySelector('input[type="file"]');
    const resultEl = root.querySelector('.predict-result');
    if (!form || !dropZone || !input || !resultEl) return;

    function setFiles(files) {
      selectedFiles[key] = Array.from(files || []).filter((file) => file.type.startsWith('image/'));
      renderSelectedFiles(key);
    }

    input.addEventListener('change', () => setFiles(input.files));

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('is-dragging');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('is-dragging'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('is-dragging');
      setFiles(e.dataTransfer.files);
    });

    root.querySelector('[data-action="clear-files"]').addEventListener('click', () => {
      selectedFiles[key] = [];
      input.value = '';
      renderSelectedFiles(key);
      resultEl.hidden = true;
      resultEl.innerHTML = '';
    });

    root.querySelector('[data-action="export-csv"]').addEventListener('click', () => exportCsv(key));
    root.querySelector('[data-action="export-pdf"]').addEventListener('click', () => exportPdf(key));
    root.querySelector('[data-action="clear-history"]').addEventListener('click', () => {
      const dataset = datasetTitle(key);
      history = history.filter((item) => item.dataset !== dataset);
      saveHistory();
      renderHistory(key);
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const files = selectedFiles[key];
      const route = form.getAttribute('data-route');
      if (!files.length) return;

      resultEl.hidden = false;
      resultEl.innerHTML = '<p class="muted">Đang xử lý dự đoán batch...</p>';
      const btn = form.querySelector('.btn-predict');
      btn.disabled = true;
      try {
        await refreshInferenceState();
        const readinessIssue = issueForRoute(route);
        if (readinessIssue) {
          resultEl.innerHTML = renderActionError(readinessIssue);
          return;
        }

        const fd = new FormData();
        files.forEach((file) => fd.append('images', file));
        const res = await fetch(`/api/predict/${route}/batch`, { method: 'POST', body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          resultEl.innerHTML = `<p class="err">${escapeHtml(data.error || res.statusText)}</p>`;
          return;
        }

        const byName = new Map(files.map((file) => [file.name, URL.createObjectURL(file)]));
        const results = (data.results || []).map((item, idx) => ({
          ...item,
          originalSrc: byName.get(item.filename) || URL.createObjectURL(files[idx]),
        }));
        latestResults[key] = results;
        addToHistory(key, results);
        resultEl.innerHTML = `<div class="batch-grid">${results.map(renderPredictResultItem).join('')}</div>`;
        renderHistory(key);
      } catch (err) {
        resultEl.innerHTML = `<p class="err">${escapeHtml(err.message)}</p>`;
      } finally {
        btn.disabled = false;
      }
    });

    renderSelectedFiles(key);
    renderHistory(key);
  }

  function exportRows(key) {
    const dataset = datasetTitle(key);
    const latest = latestResults[key]
      .filter((item) => item.ok)
      .map((item) => {
        const diagnosis = diagnosisFrom(item.prediction);
        const modality = modalityFrom(item.prediction);
        const top = getTopEntries(diagnosis.probabilities);
        return {
          createdAt: new Date().toISOString(),
          dataset,
          filename: item.filename,
          label: diagnosis.label,
          confidence: Number(diagnosis.confidence),
          modalityLabel: modality?.label || '',
          modalityConfidence: modality ? Number(modality.confidence) : '',
          selectedTitle: item.prediction.selectedTitle || '',
          top1: top[0]?.[0] || '',
          top1Prob: top[0]?.[1] || 0,
          top2: top[1]?.[0] || '',
          top2Prob: top[1]?.[1] || 0,
          unclear: isUnclear(top),
          checkpoint: diagnosis.checkpoint || '',
          device: diagnosis.device || '',
        };
      });
    return latest.length ? latest : history.filter((item) => item.dataset === dataset);
  }

  function exportCsv(key) {
    const rows = exportRows(key);
    if (!rows.length) {
      setStatus('Chưa có kết quả để xuất CSV.', 'warn');
      return;
    }
    const headers = [
      'createdAt',
      'dataset',
      'filename',
      'modalityLabel',
      'modalityConfidence',
      'selectedTitle',
      'label',
      'confidence',
      'top1',
      'top1Prob',
      'top2',
      'top2Prob',
      'unclear',
      'checkpoint',
      'device',
    ];
    const csv = [
      headers.join(','),
      ...rows.map((row) =>
        headers
          .map((h) => `"${String(row[h] ?? '').replace(/"/g, '""')}"`)
          .join(',')
      ),
    ].join('\n');
    downloadText(`dacn3-report-${Date.now()}.csv`, csv, 'text/csv;charset=utf-8');
  }

  function exportPdf(key) {
    const rows = exportRows(key);
    if (!rows.length) {
      setStatus('Chưa có kết quả để xuất PDF.', 'warn');
      return;
    }
    const doc = window.open('', '_blank');
    if (!doc) {
      setStatus('Trình duyệt chặn popup.Hãy cho phép popup để in PDF.', 'warn');
      return;
    }
    doc.document.write(`
      <!doctype html>
      <html>
        <head>
          <title>DACN3 prediction report</title>
          <style>
            body { font-family: Arial, sans-serif; color: #111827; margin: 32px; }
            h1 { font-size: 22px; margin-bottom: 4px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }
            th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; }
            th { background: #f3f4f6; }
          </style>
        </head>
        <body>
          <h1>DACN3 prediction report</h1>
          <p>Generated: ${new Date().toLocaleString()}</p>
          <table>
            <thead><tr><th>Time</th><th>Dataset</th><th>File</th><th>Type</th><th>Model</th><th>Label</th><th>Confidence</th><th>Top 2</th><th>Note</th></tr></thead>
            <tbody>
              ${rows
                .map(
                  (r) => `<tr><td>${escapeHtml(new Date(r.createdAt).toLocaleString())}</td><td>${escapeHtml(r.dataset)}</td><td>${escapeHtml(r.filename)}</td><td>${escapeHtml(r.modalityLabel || '-')} ${r.modalityConfidence !== '' && r.modalityConfidence != null ? `(${formatPct(Number(r.modalityConfidence))})` : ''}</td><td>${escapeHtml(r.selectedTitle || '-')}</td><td>${escapeHtml(r.label)}</td><td>${formatPct(Number(r.confidence))}</td><td>${escapeHtml(r.top2)} (${formatPct(Number(r.top2Prob))})</td><td>${r.unclear ? 'Unclear' : 'OK'}</td></tr>`
                )
                .join('')}
            </tbody>
          </table>
        </body>
      </html>
    `);
    doc.document.close();
    doc.focus();
    setTimeout(() => doc.print(), 250);
  }

  function downloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function renderMetaCard(labels) {
    const names = Object.keys(labels.class_to_idx || {});
    const pills = names.map((n) => `<span class="pill">${escapeHtml(n)}</span>`).join('');
    const size = Array.isArray(labels.image_size) ? `${labels.image_size[0]}x${labels.image_size[1]}` : '-';
    return `
      <div class="cards">
        <div class="card">
          <h3>Kiến Trúc</h3>
          <div class="value">${escapeHtml(labels.model_name || '-')}</div>
          <div class="sub">Mô hình được sử dụng</div>
        </div>
        <div class="card">
          <h3>Kích thước ảnh</h3>
          <div class="value">${escapeHtml(size)}</div>
          <div class="sub">Đầu vào sau tiền xử lý</div>
        </div>
        <div class="card">
          <h3>Số lớp</h3>
          <div class="value">${names.length}</div>
          <div class="sub class-pills">${pills}</div>
        </div>
      </div>
    `;
  }

  function renderReportTable(report) {
    const { rows, summary } = reportRows(report);
    const acc = typeof summary.accuracy === 'number' ? formatPct(summary.accuracy) : '-';
    const body = rows
      .map(
        (r) => `
      <tr>
        <td>${escapeHtml(r.cls)}</td>
        <td class="num">${formatPct(r.precision)}</td>
        <td class="num">${formatPct(r.recall)}</td>
        <td class="num">${formatPct(r['f1-score'])}</td>
        <td class="num">${r.support != null ? Math.round(r.support) : '-'}</td>
      </tr>`
      )
      .join('');
    const extras = ['macro avg', 'weighted avg']
      .filter((k) => summary[k])
      .map((k) => {
        const r = summary[k];
        return `
      <tr class="row-metric">
        <td>${escapeHtml(k)}</td>
        <td class="num">${formatPct(r.precision)}</td>
        <td class="num">${formatPct(r.recall)}</td>
        <td class="num">${formatPct(r['f1-score'])}</td>
        <td class="num">${r.support != null ? Math.round(r.support) : '-'}</td>
      </tr>`;
      })
      .join('');
    return `
      <h2 class="section-title">Báo cáo phân loại</h2>
      <div class="table-wrap">
        <p class="sub report-accuracy">Độ chính xác tổng: <strong>${acc}</strong></p>
        <table class="report">
          <thead>
            <tr>
              <th>Lớp</th>
              <th class="num">Precision</th>
              <th class="num">Recall</th>
              <th class="num">F1</th>
              <th class="num">Support</th>
            </tr>
          </thead>
          <tbody>${body}${extras}</tbody>
        </table>
      </div>
    `;
  }

  function renderCharts(key, hasReport) {
    const chartRootId = `charts-${key}`;
    return `
      <div class="grid-2 chart-grid">
        <div class="chart-box">
          <h3 class="section-title chart-title">Huấn luyện - Loss &amp; Accuracy</h3>
          <canvas id="${chartRootId}-history" height="220"></canvas>
        </div>
        ${hasReport ? `<div class="chart-box">
          <h3 class="section-title chart-title">F1-score theo lớp</h3>
          <canvas id="${chartRootId}-f1" height="220"></canvas>
        </div>` : ''}
      </div>
    `;
  }

  function chartColors() {
    const dark = document.documentElement.dataset.theme === 'dark';
    return {
      tick: dark ? '#aebbd0' : '#52667a',
      grid: dark ? 'rgba(148,163,184,0.18)' : 'rgba(157,174,194,0.45)',
    };
  }

  function attachCharts(key, historyData, report) {
    destroyCharts(key);
    if (typeof Chart === 'undefined') return;
    const colors = chartColors();
    const commonOpts = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: colors.tick } } },
      scales: {
        x: { ticks: { color: colors.tick }, grid: { color: colors.grid } },
        y: { ticks: { color: colors.tick }, grid: { color: colors.grid } },
      },
    };

    const hCanvas = document.getElementById(`charts-${key}-history`);
    if (hCanvas && historyData.length) {
      chartRefs[key].push(
        new Chart(hCanvas.getContext('2d'), {
          type: 'line',
          data: {
            labels: historyData.map((e) => `Epoch ${e.epoch}`),
            datasets: [
              { label: 'Train loss', data: historyData.map((e) => e.train_loss), borderColor: '#5f6fd6', backgroundColor: 'rgba(95,111,214,0.1)', tension: 0.25, yAxisID: 'y' },
              { label: 'Val loss', data: historyData.map((e) => e.val_loss), borderColor: '#d95d66', backgroundColor: 'rgba(217,93,102,0.08)', tension: 0.25, yAxisID: 'y' },
              { label: 'Val acc %', data: historyData.map((e) => e.val_acc), borderColor: '#1f9d8a', tension: 0.25, yAxisID: 'y1' },
            ],
          },
          options: {
            ...commonOpts,
            scales: {
              x: commonOpts.scales.x,
              y: { type: 'linear', position: 'left', title: { display: true, text: 'Loss', color: colors.tick }, ticks: { color: colors.tick }, grid: { color: colors.grid } },
              y1: { type: 'linear', position: 'right', title: { display: true, text: 'Accuracy %', color: colors.tick }, ticks: { color: colors.tick }, grid: { drawOnChartArea: false } },
            },
          },
        })
      );
    }

    const { rows } = reportRows(report);
    const f1Canvas = document.getElementById(`charts-${key}-f1`);
    if (report && f1Canvas && rows.length) {
      chartRefs[key].push(
        new Chart(f1Canvas.getContext('2d'), {
          type: 'bar',
          data: {
            labels: rows.map((r) => r.cls),
            datasets: [{ label: 'F1', data: rows.map((r) => typeof r['f1-score'] === 'number' ? r['f1-score'] : 0), backgroundColor: 'rgba(31, 157, 138, 0.35)', borderColor: '#1f9d8a', borderWidth: 1 }],
          },
          options: {
            ...commonOpts,
            scales: {
              x: commonOpts.scales.x,
              y: { min: 0, max: 1, ticks: { color: colors.tick, callback: (v) => formatPct(Number(v)) }, grid: { color: colors.grid } },
            },
          },
        })
      );
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderPanel(key, bundle) {
    const { meta, labels, report, history: historyData } = bundle;
    const hasReport = Boolean(report);
    contents[key].innerHTML = `
      ${renderMetaCard(labels)}
      ${renderPredictZone(key)}
      ${hasReport ? renderReportTable(report) : ''}
      ${renderCharts(key, hasReport)}
      <p class="dataset-note">Dataset: <strong>${escapeHtml(meta.title)}</strong></p>
    `;
    bindPredictHandlers(key);
    requestAnimationFrame(() => attachCharts(key, Array.isArray(historyData) ? historyData : [], report));
  }

  function switchTab(panelKey) {
    tabButtons.forEach((btn) => {
      const active = btn.dataset.panel === panelKey;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    Object.keys(panels).forEach((k) => {
      const on = k === panelKey;
      panels[k].classList.toggle('is-active', on);
      panels[k].hidden = !on;
    });
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.panel));
  });
  themeToggle.addEventListener('click', toggleTheme);

  async function load() {
    setStatus('Đang tải dữ liệu artifact...');
    try {
      const res = await fetch('/api/datasets');
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || err.error || res.statusText);
      }
      const data = await res.json();
      renderPanel('mri', data.mri);
      renderPanel('strokeCt', data.strokeCt);
      renderPanel('modality', data.modality);
      switchTab('modality');

      let infHint = '';
      let infType = 'ok';
      try {
        const ihj = await refreshInferenceState();
        if (!ihj.reachable) {
          infHint = ' | Inference API chưa chạy (chạy npm run inference để dự đoán ảnh).';
          infType = 'warn';
        } else {
          const parts = [];
          if (!ihj.mri?.ready) parts.push('MRI thiếu file .pth');
          if (!ihj.stroke_ct?.ready) parts.push('CT thiếu file .pth');
          if (!ihj.modality?.ready) {
            parts.push(ihj.modality?.tensorflowReady === false ? 'MRI/CT thiếu TensorFlow' : 'MRI/CT thiếu model .keras');
          }
          if (parts.length) {
            infHint = ' | ' + parts.join('; ') + '.';
            infType = 'warn';
          }
        }
      } catch {
        infHint = ' | Không kiểm tra được inference API.';
        infType = 'warn';
      }
      setStatus('Đã đồng bộ artifact từ máy chủ thành công!' + infHint, infType);
    } catch (e) {
      console.error(e);
      setStatus('Lỗi: ' + e.message, 'error');
      contents.mri.innerHTML =
        '<p>Không tải được API. Kiểm tra đã chạy <code>npm start</code> và có thư mục <code>artifacts/</code>.</p>';
    }
  }

  initTheme();
  switchTab('modality');
  load();
})();
