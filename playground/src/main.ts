/**
 * WarpVector Playground — UI Controller (Enhanced)
 *
 * Features:
 * - Single intent mode (IntentAdapter.tune)
 * - Blend mode with sliders (IntentAdapter.tuneBlended)
 * - Live code snippet display
 * - Batch performance benchmark (IntentAdapter.tuneBatch)
 */
import {
  createDemoState,
  transformWithIntent,
  transformWithBlend,
  runBenchmark,
  cosineSim,
  DIM,
  type DemoState,
  type RankedDoc,
  type BenchmarkResult,
} from './demo-engine.ts';

type Lang = 'en' | 'ja';

const LABELS: Record<Lang, {
  queryLabel: string;
  rankUnit: string;
  runBench: string;
  running: string;
  batchLabel: string;
  individualLabel: string;
  speedupLabel: string;
  opsLabel: string;
}> = {
  en: {
    queryLabel: 'Query: "Apple"',
    rankUnit: 'ranks',
    runBench: '▶ Run Benchmark',
    running: 'Running…',
    batchLabel: 'tuneBatch()',
    individualLabel: 'tune() × N',
    speedupLabel: 'Speedup',
    opsLabel: 'ops/sec',
  },
  ja: {
    queryLabel: 'クエリ: "Apple"',
    rankUnit: '位',
    runBench: '▶ ベンチマーク実行',
    running: '実行中…',
    batchLabel: 'tuneBatch()',
    individualLabel: 'tune() × N',
    speedupLabel: '高速化',
    opsLabel: 'ops/sec',
  },
};

export async function initPlayground(lang: Lang) {
  const canvas = document.getElementById('vectorCanvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const labels = LABELS[lang];

  // Initialize the REAL warpvector engine
  const state = await createDemoState(lang);

  // Show WASM status
  const wasmBadge = document.getElementById('wasmBadge');
  if (wasmBadge) {
    wasmBadge.textContent = state.wasmReady ? 'WASM ✓' : 'JS fallback';
    wasmBadge.style.color = state.wasmReady ? '#10b981' : '#f59e0b';
  }

  // State
  let baseRankings = transformWithIntent(state, null).rankings;
  let currentRankings = baseRankings;
  let currentIntent = 'none';
  let isBlendMode = false;
  let blendWeights: Record<string, number> = { technology: 0, business: 0, medical: 0 };

  // Animation state
  let animFrameId: number | null = null;
  let animStartPositions: { x: number; y: number }[] = [];
  let animTargetPositions: { x: number; y: number }[] = [];
  let animCurrentPositions: { x: number; y: number }[] = [];

  // Normalize coordinates for canvas display
  function normalizePositions(docs: DemoState['docs'], queryPos: { x: number; y: number }) {
    const allX = [...docs.map(d => d.pos.x), queryPos.x];
    const allY = [...docs.map(d => d.pos.y), queryPos.y];
    const minX = Math.min(...allX), maxX = Math.max(...allX);
    const minY = Math.min(...allY), maxY = Math.max(...allY);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = 0.1;
    return {
      normalize: (pos: { x: number; y: number }) => ({
        x: padding + (1 - 2 * padding) * (pos.x - minX) / rangeX,
        y: padding + (1 - 2 * padding) * (pos.y - minY) / rangeY,
      }),
    };
  }

  function initDisplayPositions() {
    const { normalize } = normalizePositions(state.docs, state.query.pos);
    animCurrentPositions = state.docs.map(d => normalize(d.pos));
  }

  initDisplayPositions();
  const { normalize: baseNormalize } = normalizePositions(state.docs, state.query.pos);
  const queryDisplayPos = baseNormalize(state.query.pos);

  // Canvas resize
  function resizeCanvas() {
    const rect = canvas.parentElement!.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }

  // Drawing
  function drawFrame() {
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const p = i / 10;
      ctx.beginPath(); ctx.moveTo(p * w, 0); ctx.lineTo(p * w, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p * h); ctx.lineTo(w, p * h); ctx.stroke();
    }

    // Find top-1
    const top1Id = currentRankings[0]?.id;

    // Connection lines to top-3
    for (let i = 0; i < Math.min(3, currentRankings.length); i++) {
      const doc = currentRankings[i];
      const pos = animCurrentPositions[doc.id];
      const alpha = [0.25, 0.12, 0.06][i];
      ctx.beginPath();
      ctx.moveTo(queryDisplayPos.x * w, queryDisplayPos.y * h);
      ctx.lineTo(pos.x * w, pos.y * h);
      ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Document dots
    state.docs.forEach((doc, i) => {
      const pos = animCurrentPositions[i];
      const x = pos.x * w, y = pos.y * h;
      const isTop1 = doc.id === top1Id;
      const radius = isTop1 ? 7 : 5;

      if (isTop1) {
        const grad = ctx.createRadialGradient(x, y, 0, x, y, 24);
        grad.addColorStop(0, 'rgba(16, 185, 129, 0.3)');
        grad.addColorStop(1, 'rgba(16, 185, 129, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(x, y, 24, 0, Math.PI * 2); ctx.fill();
      }

      ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isTop1 ? '#10b981' : doc.color;
      ctx.fill();
      ctx.strokeStyle = isTop1 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 1.5; ctx.stroke();

      ctx.fillStyle = isTop1 ? '#f1f5f9' : 'rgba(241,245,249,0.5)';
      ctx.font = `${isTop1 ? 500 : 400} 10px Inter, system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText(doc.name, x, y - radius - 6);
    });

    // Query marker
    const qx = queryDisplayPos.x * w, qy = queryDisplayPos.y * h;
    const qGrad = ctx.createRadialGradient(qx, qy, 0, qx, qy, 30);
    qGrad.addColorStop(0, 'rgba(245, 158, 11, 0.25)');
    qGrad.addColorStop(1, 'rgba(245, 158, 11, 0)');
    ctx.fillStyle = qGrad;
    ctx.beginPath(); ctx.arc(qx, qy, 30, 0, Math.PI * 2); ctx.fill();

    ctx.save(); ctx.translate(qx, qy); ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#f59e0b'; ctx.fillRect(-5, -5, 10, 10);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)'; ctx.lineWidth = 1.5;
    ctx.strokeRect(-5, -5, 10, 10); ctx.restore();

    ctx.fillStyle = '#f59e0b'; ctx.font = '600 12px Inter, system-ui';
    ctx.textAlign = 'center'; ctx.fillText(labels.queryLabel, qx, qy - 16);
  }

  // Update rankings UI
  function updateRankingsUI(rankings: RankedDoc[], latencyMs: number) {
    const listEl = document.getElementById('rankingList')!;
    listEl.innerHTML = '';

    document.getElementById('metricLatency')!.textContent = latencyMs.toFixed(2) + 'ms';
    document.getElementById('metricTopSim')!.textContent = rankings[0]?.score.toFixed(3) ?? '—';

    const intentCategory: Record<string, string | null> = {
      none: null, technology: 'tech', business: 'business', medical: 'medical',
    };

    // In blend mode, find the dominant category
    let targetCat: string | null = null;
    if (isBlendMode) {
      let maxW = 0;
      for (const [k, v] of Object.entries(blendWeights)) {
        if (v > maxW) { maxW = v; targetCat = intentCategory[k]; }
      }
      if (maxW < 0.05) targetCat = null;
    } else {
      targetCat = intentCategory[currentIntent];
    }

    let bestImprovement = 0;

    rankings.slice(0, 8).forEach((doc, i) => {
      const vanillaRank = baseRankings.findIndex(d => d.id === doc.id);
      const improvement = vanillaRank - i;
      const isRelevant = targetCat && doc.category === targetCat;
      if (isRelevant && improvement > bestImprovement) bestImprovement = improvement;

      const item = document.createElement('div');
      item.className = 'ranking-item' + (isRelevant ? ' highlight' : '');
      item.innerHTML = `
        <span class="ranking-rank">${i + 1}</span>
        <span class="ranking-name">${doc.name}</span>
        <span class="ranking-score">${doc.score.toFixed(3)}</span>
        <span class="ranking-change ${improvement > 0 ? 'up' : improvement < 0 ? 'down' : 'same'}">
          ${improvement > 0 ? '↑' + improvement : improvement < 0 ? '↓' + Math.abs(improvement) : '—'}
        </span>
      `;
      listEl.appendChild(item);
    });

    const deltaEl = document.getElementById('metricRankDelta')!;
    const isActive = isBlendMode
      ? Object.values(blendWeights).some(v => v > 0.05)
      : currentIntent !== 'none';
    deltaEl.textContent = isActive && bestImprovement > 0
      ? '+' + bestImprovement + ' ' + labels.rankUnit
      : '—';
  }

  // Update code snippet
  function updateCodeSnippet(code: string) {
    const el = document.getElementById('codeSnippet');
    if (el) el.textContent = code;
  }

  // Animate transition
  function animateTransition(latencyMs: number, rankings: RankedDoc[]) {
    const { normalize } = normalizePositions(state.docs, state.query.pos);
    animStartPositions = animCurrentPositions.map(p => ({ ...p }));
    animTargetPositions = state.docs.map(d => normalize(d.pos));

    const duration = 800;
    const startTime = performance.now();

    function step(timestamp: number) {
      const t = Math.min((timestamp - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);

      for (let i = 0; i < state.docs.length; i++) {
        animCurrentPositions[i] = {
          x: animStartPositions[i].x + (animTargetPositions[i].x - animStartPositions[i].x) * ease,
          y: animStartPositions[i].y + (animTargetPositions[i].y - animStartPositions[i].y) * ease,
        };
      }

      drawFrame();
      if (t < 1) animFrameId = requestAnimationFrame(step);
    }

    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(step);

    updateRankingsUI(rankings, latencyMs);
  }

  // Switch single intent
  function switchIntent(intentKey: string) {
    currentIntent = intentKey;
    const { latencyMs, rankings } = transformWithIntent(state, intentKey === 'none' ? null : intentKey);
    currentRankings = rankings;

    // Update intent label
    const btn = document.querySelector(`.intent-btn[data-intent="${intentKey}"]`);
    document.getElementById('intentLabel')!.textContent =
      btn?.querySelector('.intent-btn__name')?.textContent ?? '';

    // Update button states
    document.querySelectorAll('.intent-btn').forEach(b => {
      b.classList.toggle('active', (b as HTMLElement).dataset.intent === intentKey);
    });

    // Code snippet
    const code = intentKey === 'none'
      ? `// No intent applied\nconst result = baseVector;`
      : `const warped = adapter.tune(\n  baseVector,\n  "${intentKey}"\n);`;
    updateCodeSnippet(code);

    animateTransition(latencyMs, rankings);
  }

  // Apply blend from sliders
  function applyBlend() {
    const { latencyMs, rankings, codeSnippet } = transformWithBlend(state, blendWeights);
    currentRankings = rankings;
    updateCodeSnippet(codeSnippet);

    // Update intent label
    const parts: string[] = [];
    for (const [k, v] of Object.entries(blendWeights)) {
      if (v > 0.05) parts.push(`${k} ${Math.round(v * 100)}%`);
    }
    document.getElementById('intentLabel')!.textContent =
      parts.length > 0 ? parts.join(' + ') : (lang === 'ja' ? '通常検索' : 'Vanilla Search');

    animateTransition(latencyMs, rankings);
  }

  // Mode toggle
  const modeToggle = document.getElementById('modeToggle') as HTMLButtonElement | null;
  const singlePanel = document.getElementById('singleIntentPanel');
  const blendPanel = document.getElementById('blendPanel');

  if (modeToggle) {
    modeToggle.addEventListener('click', () => {
      isBlendMode = !isBlendMode;
      modeToggle.textContent = isBlendMode
        ? (lang === 'ja' ? '⇄ シングルモード' : '⇄ Single Mode')
        : (lang === 'ja' ? '⇄ ブレンドモード' : '⇄ Blend Mode');

      if (singlePanel) singlePanel.style.display = isBlendMode ? 'none' : '';
      if (blendPanel) blendPanel.style.display = isBlendMode ? '' : 'none';

      if (isBlendMode) {
        blendWeights = { technology: 0, business: 0, medical: 0 };
        document.querySelectorAll<HTMLInputElement>('.blend-slider').forEach(s => { s.value = '0'; });
        document.querySelectorAll('.blend-value').forEach(v => { v.textContent = '0%'; });
        applyBlend();
      } else {
        switchIntent('none');
      }
    });
  }

  // Blend sliders
  document.querySelectorAll<HTMLInputElement>('.blend-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const intent = slider.dataset.intent!;
      blendWeights[intent] = parseInt(slider.value) / 100;
      const valueEl = slider.parentElement?.querySelector('.blend-value');
      if (valueEl) valueEl.textContent = slider.value + '%';
      applyBlend();
    });
  });

  // Single intent buttons
  document.getElementById('intentGroup')?.addEventListener('click', (e) => {
    if (isBlendMode) return;
    const btn = (e.target as HTMLElement).closest('.intent-btn') as HTMLElement | null;
    if (btn?.dataset.intent) switchIntent(btn.dataset.intent);
  });

  // Benchmark
  const benchBtn = document.getElementById('benchBtn') as HTMLButtonElement | null;
  if (benchBtn) {
    benchBtn.addEventListener('click', () => {
      benchBtn.disabled = true;
      benchBtn.textContent = labels.running;

      // Use requestAnimationFrame to let UI update
      requestAnimationFrame(() => {
        const result = runBenchmark(state, 1000);
        renderBenchmark(result);
        benchBtn.disabled = false;
        benchBtn.textContent = labels.runBench;
      });
    });
  }

  function renderBenchmark(r: BenchmarkResult) {
    const el = document.getElementById('benchResult');
    if (!el) return;
    el.innerHTML = `
      <div class="bench-row">
        <span class="bench-label">${labels.batchLabel}</span>
        <span class="bench-value">${r.batchMs.toFixed(1)}ms</span>
        <span class="bench-ops">${(r.batchOpsPerSec / 1000).toFixed(0)}K ${labels.opsLabel}</span>
      </div>
      <div class="bench-row">
        <span class="bench-label">${labels.individualLabel}</span>
        <span class="bench-value">${r.individualMs.toFixed(1)}ms</span>
        <span class="bench-ops">${(r.individualOpsPerSec / 1000).toFixed(0)}K ${labels.opsLabel}</span>
      </div>
      <div class="bench-speedup">
        <span>${labels.speedupLabel}: </span>
        <span class="bench-speedup-value">${r.speedup.toFixed(1)}×</span>
      </div>
    `;
  }

  // Resize
  window.addEventListener('resize', () => { resizeCanvas(); drawFrame(); });

  // Init
  resizeCanvas();
  drawFrame();
  updateRankingsUI(baseRankings, 0);
  updateCodeSnippet(`// No intent applied\nconst result = baseVector;`);
}
