/**
 * WarpVector Playground — UI Controller
 *
 * Enhanced with Real LLM Embeddings via Transformers.js
 * And Dynamic PCA Camera Projection
 */
import {
  createDemoState,
  transformWithIntent,
  transformWithBlend,
  runBenchmark,
  autoLearnIntents,
  cosineSimilarity,
  projectTo2D,
  updateQuery,
  addCustomIntent,
  DIM,
  CATEGORY_META,
  type DemoState,
  type RankedDoc,
  type BenchmarkResult,
  type AutoLearnResult,
  type EvalMetrics,
  type DataCategory,
} from './demo-engine.ts';

type Lang = 'en' | 'ja';
interface Point2D { x: number; y: number; }

// UI Constants
const RANKING_DISPLAY_COUNT = 8;
const ANIM_DURATION_MS = 900;
const CANVAS_PADDING = 0.15;
const BLEND_ACTIVE_THRESHOLD = 0.05;
const TOP_N_CONNECTIONS = 3;
const HOVER_RADIUS_PX = 10;

const QUANT_SPECS: Record<'none' | 'int8' | 'binary', {
  badge: string; storageGB: number; monthlyCost: number;
}> = {
  none:   { badge: '1536 Bytes/vec', storageGB: 6.0,  monthlyCost: 180 },
  int8:   { badge: '384 Bytes/vec',  storageGB: 1.5,  monthlyCost: 45 },
  binary: { badge: '48 Bytes/vec',   storageGB: 0.18, monthlyCost: 5 },
};

const LABELS: Record<Lang, {
  queryLabel: string;
  rankUnit: string;
  runBench: string;
  running: string;
  batchLabel: string;
  individualLabel: string;
  speedupLabel: string;
  opsLabel: string;
  loadingModel: string;
  embeddingTexts: string;
  customIntentName: string;
  customIntentDesc: string;
  addingIntent: string;
  autoLearnBtn: string;
  autoLearning: string;
  autoLearnDone: string;
  autoLearnCategories: string;
  autoLearnTime: string;
}> = {
  en: {
    queryLabel: 'Query',
    rankUnit: 'ranks',
    runBench: '▶ Run Benchmark',
    running: 'Running…',
    batchLabel: 'tuneBatch()',
    individualLabel: 'tune() × N',
    speedupLabel: 'Speedup',
    opsLabel: 'ops/sec',
    loadingModel: 'Loading LLM Model...',
    embeddingTexts: 'Computing Embeddings...',
    customIntentName: 'Custom Intent',
    customIntentDesc: 'adapter.tune(vec, "custom")',
    addingIntent: 'Adding...',
    autoLearnBtn: '🧠 Auto-learn from Documents',
    autoLearning: '🧠 Learning...',
    autoLearnDone: 'Learned',
    autoLearnCategories: 'Categories',
    autoLearnTime: 'Training Time',
  },
  ja: {
    queryLabel: 'クエリ',
    rankUnit: '位',
    runBench: '▶ ベンチマーク実行',
    running: '実行中…',
    batchLabel: 'tuneBatch()',
    individualLabel: 'tune() × N',
    speedupLabel: '高速化',
    opsLabel: 'ops/sec',
    loadingModel: 'LLMモデル読込中...',
    embeddingTexts: 'ベクトル計算中...',
    customIntentName: 'カスタムインテント',
    customIntentDesc: 'adapter.tune(vec, "custom")',
    addingIntent: '追加中...',
    autoLearnBtn: '🧠 ドキュメントから自動学習',
    autoLearning: '🧠 学習中...',
    autoLearnDone: '学習完了',
    autoLearnCategories: 'カテゴリ',
    autoLearnTime: '学習時間',
  },
};

function getElement(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Element #${id} not found`);
  return el;
}

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (el instanceof HTMLCanvasElement) return el;
  throw new Error(`Canvas #${id} not found`);
}

function isValidQuantMode(v: string): v is 'none' | 'int8' | 'binary' {
  return v === 'none' || v === 'int8' || v === 'binary';
}

/** Wrap an async action with button loading state management */
async function withButtonLoading(
  buttonId: string,
  loadingText: string,
  action: () => Promise<void>,
): Promise<void> {
  const btn = document.getElementById(buttonId);
  let origText = '';
  if (btn instanceof HTMLButtonElement) {
    origText = btn.textContent || '';
    btn.textContent = loadingText;
    btn.disabled = true;
  }
  try {
    await action();
  } finally {
    if (btn instanceof HTMLButtonElement) {
      btn.textContent = origText;
      btn.disabled = false;
    }
  }
}

export async function initPlayground(lang: Lang) {
  const canvas = getCanvas('vectorCanvas');
  const ctx = canvas.getContext('2d')!;
  const labels = LABELS[lang];

  // Loading UI handling
  const loadingOverlay = getElement('loadingOverlay');
  const loadingText = getElement('loadingText');
  const loadingSubtext = getElement('loadingSubtext');

  const progressCallback = (status: string, data?: any) => {
    if (status === 'init_model') {
      loadingText.textContent = labels.loadingModel;
    } else if (status === 'embedding') {
      loadingText.textContent = labels.embeddingTexts;
      loadingSubtext.textContent = 'Generating 384-dimensional vectors...';
    } else if (status === 'progress' && data) {
      if (data.status === 'downloading' || data.status === 'progress') {
        loadingSubtext.textContent = `Downloading: ${data.file} - ${Math.round(data.progress || 0)}%`;
      }
    }
  };

  // Initialize the REAL warpvector engine WITH REAL EMBEDDINGS
  let state: DemoState;
  try {
    state = await createDemoState(lang, progressCallback);
    if (loadingOverlay) loadingOverlay.style.display = 'none'; // Hide overlay
  } catch (err) {
    if (loadingText) loadingText.textContent = 'Error loading model';
    if (loadingSubtext) loadingSubtext.textContent = (err instanceof Error ? err.message : String(err));
    console.error(err);
    return;
  }

  // Show WASM status
  const wasmBadge = document.getElementById('wasmBadge');
  if (wasmBadge) {
    wasmBadge.textContent = state.wasmReady ? 'WASM ✓' : 'JS fallback';
    wasmBadge.style.color = state.wasmReady ? '#10b981' : '#f59e0b';
  }

  // Set initial query input text
  const queryInput = document.getElementById('queryInput');
  if (!(queryInput instanceof HTMLInputElement)) throw new Error('queryInput not found');
  queryInput.value = state.query.text;

  // Build intent buttons dynamically
  function renderIntentButtons() {
    const singlePanel = document.getElementById('intentGroup');
    const blendGroup = document.getElementById('blendGroup');
    if (!singlePanel || !blendGroup) return;
    
    singlePanel.innerHTML = `
      <button class="intent-btn ${currentIntent === 'none' ? 'active' : ''}" data-intent="none">
        <div class="intent-btn__icon" style="background:rgba(100,116,139,0.15);">🔍</div>
        <div class="intent-btn__content">
          <div class="intent-btn__name">${lang === 'ja' ? '通常検索' : 'Vanilla Search'}</div>
          <div class="intent-btn__desc">${lang === 'ja' ? '変換なし（ベースライン）' : 'No transformation applied'}</div>
        </div>
      </button>
    `;
    
    blendGroup.innerHTML = '';

    for (const intent of state.intentsList) {
      // Single button
      const btn = document.createElement('button');
      btn.className = `intent-btn ${currentIntent === intent.key ? 'active' : ''}`;
      btn.dataset.intent = intent.key;
      btn.innerHTML = `
        <div class="intent-btn__icon" style="background:${intent.color};">${intent.icon}</div>
        <div class="intent-btn__content">
          <div class="intent-btn__name">${intent.name}</div>
          <div class="intent-btn__desc">${intent.desc}</div>
        </div>
      `;
      singlePanel.appendChild(btn);

      // Blend slider
      const bItem = document.createElement('div');
      bItem.className = 'blend-item';
      bItem.innerHTML = `
        <div class="blend-item__header">
          <span><span class="blend-icon">${intent.icon}</span> ${intent.name}</span>
          <span class="blend-value">${Math.round((blendWeights[intent.key] || 0) * 100)}%</span>
        </div>
        <input type="range" min="0" max="100" value="${Math.round((blendWeights[intent.key] || 0) * 100)}" class="blend-slider" data-intent="${intent.key}">
      `;
      blendGroup.appendChild(bItem);
    }
    
    // Re-bind slider events
    document.querySelectorAll<HTMLInputElement>('.blend-slider').forEach(slider => {
      slider.addEventListener('input', () => {
        const intentKey = slider.dataset.intent;
        if (!intentKey) return;
        blendWeights[intentKey] = parseInt(slider.value) / 100;
        const valueEl = slider.parentElement?.querySelector('.blend-value');
        if (valueEl) valueEl.textContent = slider.value + '%';
        applyBlend();
      });
    });
  }

  // State
  let baseRankings = transformWithIntent(state, null).rankings;
  let currentRankings = baseRankings;
  let currentIntent = 'none';
  let isBlendMode = false;
  let isCompareMode = false;
  let blendWeights: Record<string, number> = {};
  let lastLatencyMs = 0;

  // Animation state (Dynamic 384D interpolation)
  let animFrameId: number | null = null;
  let animStartVectors: Float32Array[] = [];
  let animStartQueryVector: Float32Array = new Float32Array(DIM);
  let animTargetVectors: Float32Array[] = [];
  let animTargetQueryVector: Float32Array = new Float32Array(DIM);
  
  let animStartBasis1: Float32Array = new Float32Array(DIM);
  let animStartBasis2: Float32Array = new Float32Array(DIM);
  let animTargetBasis1: Float32Array = new Float32Array(DIM);
  let animTargetBasis2: Float32Array = new Float32Array(DIM);

  // Computed 2D positions for the current frame
  let animCurrentPositions: { x: number; y: number }[] = [];
  let animCurrentQueryPos: { x: number; y: number } = { x: 0, y: 0 };

  // Normalize coordinates for canvas display
  function normalizePositions(positions: {x:number, y:number}[], queryPos: {x:number, y:number}) {
    const allX = [...positions.map(p => p.x), queryPos.x];
    const allY = [...positions.map(p => p.y), queryPos.y];
    const minX = Math.min(...allX), maxX = Math.max(...allX);
    const minY = Math.min(...allY), maxY = Math.max(...allY);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    return {
      normalize: (pos: Point2D) => ({
        x: CANVAS_PADDING + (1 - 2 * CANVAS_PADDING) * (pos.x - minX) / rangeX,
        y: CANVAS_PADDING + (1 - 2 * CANVAS_PADDING) * (pos.y - minY) / rangeY,
      }),
    };
  }

  /** Snapshot current state vectors into animation targets */
  function captureAnimTargets() {
    animTargetVectors = state.docs.map(d => new Float32Array(d.currentVector));
    animTargetQueryVector = new Float32Array(state.query.currentVector);
    animTargetBasis1 = new Float32Array(state.basis1);
    animTargetBasis2 = new Float32Array(state.basis2);
  }

  /** Copy current animation targets to starts (for interpolation) */
  function copyTargetsToStarts() {
    animStartVectors = animTargetVectors.map(v => new Float32Array(v));
    animStartQueryVector = new Float32Array(animTargetQueryVector);
    animStartBasis1 = new Float32Array(animTargetBasis1);
    animStartBasis2 = new Float32Array(animTargetBasis2);
  }

  function initDisplayState() {
    captureAnimTargets();
    copyTargetsToStarts();
    updateCurrentFramePositions(1);
  }

  function updateCurrentFramePositions(t: number) {
    const interp = (start: Float32Array, target: Float32Array, out: Float32Array) => {
      for (let i = 0; i < DIM; i++) {
        out[i] = start[i] + (target[i] - start[i]) * t;
      }
    };

    const b1 = new Float32Array(DIM);
    const b2 = new Float32Array(DIM);
    interp(animStartBasis1, animTargetBasis1, b1);
    interp(animStartBasis2, animTargetBasis2, b2);

    // Normalize interpolated basis (optional but good for consistency)
    let n1 = 0, n2 = 0;
    for(let i=0; i<DIM; i++) { n1 += b1[i]*b1[i]; n2 += b2[i]*b2[i]; }
    n1 = Math.sqrt(n1); n2 = Math.sqrt(n2);
    if (n1 > 0) for(let i=0; i<DIM; i++) b1[i] /= n1;
    if (n2 > 0) for(let i=0; i<DIM; i++) b2[i] /= n2;

    const tmpVec = new Float32Array(DIM);
    animCurrentPositions = [];
    for (let i = 0; i < state.docs.length; i++) {
      interp(animStartVectors[i], animTargetVectors[i], tmpVec);
      animCurrentPositions.push(projectTo2D(tmpVec, b1, b2));
    }
    interp(animStartQueryVector, animTargetQueryVector, tmpVec);
    animCurrentQueryPos = projectTo2D(tmpVec, b1, b2);
  }

  // Canvas resize
  function resizeCanvas() {
    const parent = canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }

  // Tooltip
  let hoverDocId: number | null = null;
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const { w, h } = getCanvasSize();

    const { normalize } = normalizePositions(animCurrentPositions, animCurrentQueryPos);

    let found = null;
    for (let i = 0; i < state.docs.length; i++) {
      const pos = normalize(animCurrentPositions[i]);
      const dx = pos.x * w - x;
      const dy = pos.y * h - y;
      if (dx * dx + dy * dy < HOVER_RADIUS_PX * HOVER_RADIUS_PX) {
        found = i;
        break;
      }
    }
    
    if (found !== hoverDocId) {
      hoverDocId = found;
      drawFrame();
    }
  });

  /** Canvas logical size (accounting for device pixel ratio) */
  function getCanvasSize(): { w: number; h: number } {
    const dpr = window.devicePixelRatio || 1;
    return { w: canvas.width / dpr, h: canvas.height / dpr };
  }

  // Drawing sub-functions
  function drawGrid(w: number, h: number) {
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const p = i / 10;
      ctx.beginPath(); ctx.moveTo(p * w, 0); ctx.lineTo(p * w, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p * h); ctx.lineTo(w, p * h); ctx.stroke();
    }
  }

  function drawConnectionLines(w: number, h: number, queryPos: Point2D, normalize: (pos: Point2D) => Point2D) {
    for (let i = 0; i < Math.min(TOP_N_CONNECTIONS, currentRankings.length); i++) {
      const doc = currentRankings[i];
      const pos = normalize(animCurrentPositions[doc.id]);
      const alpha = [0.25, 0.12, 0.06][i];
      ctx.beginPath();
      ctx.moveTo(queryPos.x * w, queryPos.y * h);
      ctx.lineTo(pos.x * w, pos.y * h);
      ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawDocDots(w: number, h: number, normalize: (pos: Point2D) => Point2D, top1Id: number | undefined) {
    state.docs.forEach((doc, i) => {
      const pos = normalize(animCurrentPositions[i]);
      const x = pos.x * w, y = pos.y * h;
      const isTop1 = doc.id === top1Id;
      const isHover = doc.id === hoverDocId;
      const radius = isTop1 ? 7 : (isHover ? 6 : 4);

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

      if (isTop1 || isHover) {
        ctx.fillStyle = isTop1 ? '#f1f5f9' : 'rgba(241,245,249,0.9)';
        ctx.font = `${isTop1 ? 500 : 400} 11px Inter, system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText(doc.name, x, y - radius - 6);
      }

      if (isHover) {
        ctx.fillStyle = '#f1f5f9';
        ctx.font = '11px Inter, system-ui';
        const words = doc.text.split(' ');
        let line = '';
        let yy = y + radius + 14;
        for (let j = 0; j < words.length; j++) {
          const testLine = line + words[j] + ' ';
          if (ctx.measureText(testLine).width > 200 && j > 0) {
            ctx.fillText(line, x, yy);
            line = words[j] + ' ';
            yy += 14;
          } else {
            line = testLine;
          }
        }
        ctx.fillText(line, x, yy);
      }
    });
  }

  function drawQueryMarker(w: number, h: number, queryPos: Point2D) {
    const qx = queryPos.x * w, qy = queryPos.y * h;
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
    ctx.textAlign = 'center'; ctx.fillText(`"${state.query.text}"`, qx, qy - 16);
  }

  // Main draw orchestrator
  function drawFrame() {
    const { w, h } = getCanvasSize();
    ctx.clearRect(0, 0, w, h);

    const { normalize } = normalizePositions(animCurrentPositions, animCurrentQueryPos);
    const queryDisplayPos = normalize(animCurrentQueryPos);
    const top1Id = currentRankings[0]?.id;

    drawGrid(w, h);
    drawConnectionLines(w, h, queryDisplayPos, normalize);
    drawDocDots(w, h, normalize, top1Id);
    drawQueryMarker(w, h, queryDisplayPos);
  }

  // Update rankings UI
  function updateRankingsUI(rankings: RankedDoc[], latencyMs: number) {
    const listEl = getElement('rankingList');
    listEl.innerHTML = '';

    getElement('metricLatency').textContent = latencyMs.toFixed(2) + 'ms';
    getElement('metricTopSim').textContent = rankings[0]?.score.toFixed(3) ?? '—';

    let bestImprovement = 0;

    const renderList = (docs: RankedDoc[], isBaseline: boolean) => {
      let html = '';
      docs.slice(0, RANKING_DISPLAY_COUNT).forEach((doc, i) => {
        const vanillaRank = baseRankings.findIndex(d => d.id === doc.id);
        const improvement = vanillaRank - i;
        const isRelevant = !isBaseline && improvement > 0;
        if (!isBaseline && improvement > bestImprovement) bestImprovement = improvement;

        const highlightClass = isRelevant ? ' highlight' : '';
        const changeHtml = isBaseline 
          ? `<span class="ranking-change same">—</span>`
          : `<span class="ranking-change ${improvement > 0 ? 'up' : improvement < 0 ? 'down' : 'same'}">
              ${improvement > 0 ? '↑' + improvement : improvement < 0 ? '↓' + Math.abs(improvement) : '—'}
             </span>`;

        html += `
          <div class="ranking-item${highlightClass}">
            <span class="ranking-rank">${i + 1}</span>
            <span class="ranking-name" title="${doc.name}">${doc.name}</span>
            <span class="ranking-score">${doc.score.toFixed(3)}</span>
            ${changeHtml}
          </div>
        `;
      });
      return html;
    };

    if (isCompareMode) {
      listEl.className = 'compare-grid';
      listEl.innerHTML = `
        <div class="compare-col">
          <div class="compare-header">${lang === 'ja' ? 'Vanilla (変換前)' : 'Vanilla (Baseline)'}</div>
          <div class="ranking-list">${renderList(baseRankings, true)}</div>
        </div>
        <div class="compare-col">
          <div class="compare-header">${lang === 'ja' ? 'Warped (変換後)' : 'Warped (Current)'}</div>
          <div class="ranking-list">${renderList(rankings, false)}</div>
        </div>
      `;
    } else {
      listEl.className = 'ranking-list';
      listEl.innerHTML = renderList(rankings, false);
    }

    const deltaEl = getElement('metricRankDelta');
    const isActive = isBlendMode
      ? Object.values(blendWeights).some(v => v > BLEND_ACTIVE_THRESHOLD)
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

  // Animate transition (Dynamic 384D Interpolation)
  function animateTransition(latencyMs: number, rankings: RankedDoc[]) {
    copyTargetsToStarts();
    captureAnimTargets();

    const duration = ANIM_DURATION_MS;
    const startTime = performance.now();

    function step(timestamp: number) {
      const t = Math.min((timestamp - startTime) / duration, 1);
      // Smooth easing (cubic out)
      const ease = 1 - Math.pow(1 - t, 3);

      updateCurrentFramePositions(ease);
      drawFrame();

      if (t < 1) {
        animFrameId = requestAnimationFrame(step);
      }
    }

    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(step);

    lastLatencyMs = latencyMs;
    updateRankingsUI(rankings, latencyMs);
  }

  // Recompute base rankings (useful after query change)
  function recomputeBaseRankings() {
    baseRankings = transformWithIntent(state, null).rankings;
  }

  // Apply current state
  function applyCurrentState() {
    if (isBlendMode) {
      applyBlend();
    } else {
      switchIntent(currentIntent);
    }
  }

  // Switch single intent
  function switchIntent(intentKey: string) {
    currentIntent = intentKey;
    const { latencyMs, rankings, metrics } = transformWithIntent(state, intentKey === 'none' ? null : intentKey);
    currentRankings = rankings;

    // Update intent label
    const btn = document.querySelector(`.intent-btn[data-intent="${intentKey}"]`);
    getElement('intentLabel').textContent =
      btn?.querySelector('.intent-btn__name')?.textContent ?? '';

    // Update button states
    document.querySelectorAll('.intent-btn').forEach(b => {
      if (b instanceof HTMLElement) {
        b.classList.toggle('active', b.dataset.intent === intentKey);
      }
    });

    // Code snippet
    const code = intentKey === 'none'
      ? `// No intent applied\nconst result = baseVector;`
      : `const warped = adapter.tune(\n  baseVector,\n  "${intentKey}"\n);`;
    updateCodeSnippet(code);

    updateEvalUI(metrics);

    animateTransition(latencyMs, rankings);
  }

  // Apply blend from sliders
  function applyBlend() {
    const { latencyMs, rankings, codeSnippet, metrics } = transformWithBlend(state, blendWeights);
    currentRankings = rankings;
    updateCodeSnippet(codeSnippet);

    // Update intent label
    const parts: string[] = [];
    for (const [k, v] of Object.entries(blendWeights)) {
      if (v > BLEND_ACTIVE_THRESHOLD) parts.push(`${k} ${Math.round(v * 100)}%`);
    }
    getElement('intentLabel').textContent =
      parts.length > 0 ? parts.join(' + ') : (lang === 'ja' ? '通常検索' : 'Vanilla Search');

    updateEvalUI(metrics);

    animateTransition(latencyMs, rankings);
  }

  // Mode toggle
  const modeToggle = document.getElementById('modeToggle');
  const singlePanel = document.getElementById('singleIntentPanel');
  const blendPanel = document.getElementById('blendPanel');

  if (modeToggle instanceof HTMLButtonElement) {
    modeToggle.addEventListener('click', () => {
      isBlendMode = !isBlendMode;
      modeToggle.textContent = isBlendMode
        ? (lang === 'ja' ? '⇄ シングルモード' : '⇄ Single Mode')
        : (lang === 'ja' ? '⇄ ブレンドモード' : '⇄ Blend Mode');

      if (singlePanel) singlePanel.style.display = isBlendMode ? 'none' : '';
      if (blendPanel) blendPanel.style.display = isBlendMode ? '' : 'none';

      if (isBlendMode) {
        // clear weights
        for (const k of Object.keys(blendWeights)) blendWeights[k] = 0;
        document.querySelectorAll<HTMLInputElement>('.blend-slider').forEach(s => { s.value = '0'; });
        document.querySelectorAll('.blend-value').forEach(v => { v.textContent = '0%'; });
        applyBlend();
      } else {
        switchIntent('none');
      }
    });
  }

  // Whitening Toggle
  const whiteningToggle = document.getElementById('whiteningToggle');
  if (whiteningToggle instanceof HTMLInputElement) {
    whiteningToggle.addEventListener('change', () => {
      state.useWhitening = whiteningToggle.checked;
      recomputeBaseRankings();
      applyCurrentState();
    });
  }

  // Compare Toggle
  const compareToggle = document.getElementById('compareToggle');
  if (compareToggle instanceof HTMLInputElement) {
    compareToggle.addEventListener('change', () => {
      isCompareMode = compareToggle.checked;
      updateRankingsUI(currentRankings, lastLatencyMs);
    });
  }

  // Quantization Radio
  const quantGroup = document.getElementById('quantizationGroup');
  const memoryBadge = document.getElementById('memoryBadge');
  if (quantGroup && memoryBadge) {
    quantGroup.addEventListener('change', (e) => {
      if (e.target instanceof HTMLInputElement) {
        const target = e.target;
        if (target.name !== 'quantMode') return;
        const value = target.value;
        if (isValidQuantMode(value)) {
          state.quantMode = value;
        
          // Update styling
          quantGroup.querySelectorAll('.radio-btn').forEach(l => l.classList.remove('active'));
          target.closest('.radio-btn')?.classList.add('active');

          // Update badge and cost simulator
          const spec = QUANT_SPECS[state.quantMode];
          const baseCost = QUANT_SPECS.none.monthlyCost;
          const savedCost = baseCost - spec.monthlyCost;
          memoryBadge.textContent = spec.badge;

          const costStorage = document.getElementById('costStorage');
          const costMonthly = document.getElementById('costMonthly');
          const costSavingBadge = document.getElementById('costSavingBadge');

          if (costStorage && costMonthly && costSavingBadge) {
            costStorage.textContent = spec.storageGB.toFixed(2) + ' GB';
            costMonthly.textContent = lang === 'ja' ? `約 $${spec.monthlyCost} / 月` : `~$${spec.monthlyCost} / mo`;
            costSavingBadge.textContent = lang === 'ja' ? `$${savedCost} 削減` : `$${savedCost} saved`;
            costSavingBadge.style.color = savedCost > 0 ? '#10b981' : 'inherit';
            costSavingBadge.style.background = savedCost > 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.1)';
          }

          recomputeBaseRankings();
          applyCurrentState();
        }
      }
    });
  }

  // Single intent buttons
  document.getElementById('intentGroup')?.addEventListener('click', (e) => {
    if (isBlendMode) return;
    if (!(e.target instanceof HTMLElement)) return;
    const btn = e.target.closest('.intent-btn');
    if (!(btn instanceof HTMLElement)) return;
    if (btn.dataset.intent) switchIntent(btn.dataset.intent);
  });

  // Query Submit
  document.getElementById('queryForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!queryInput) return;
    const text = queryInput.value.trim();
    if (!text) return;

    await withButtonLoading('querySubmit', '...', async () => {
      await updateQuery(state, text);
    });

    recomputeBaseRankings();
    applyCurrentState();
  });

  // Add Custom Intent
  document.getElementById('customIntentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('customIntentInput');
    if (!(input instanceof HTMLInputElement)) return;
    const text = input.value.trim();
    if (!text) return;

    const hue = Math.floor(Math.random() * 360);
    const color = `hsla(${hue}, 80%, 60%, 0.15)`;

    let key = '';
    await withButtonLoading('customIntentSubmit', labels.addingIntent, async () => {
      key = await addCustomIntent(state, labels.customIntentName, text, '✨', color);
    });
    input.value = '';

    renderIntentButtons();
    if (!isBlendMode) {
      switchIntent(key);
    }
  });

  // Benchmark
  const benchBtn = document.getElementById('benchBtn');
  if (benchBtn instanceof HTMLButtonElement) {
    benchBtn.addEventListener('click', () => {
      benchBtn.disabled = true;
      benchBtn.textContent = labels.running;

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

  // Auto-learn Intents
  document.getElementById('autoLearnBtn')?.addEventListener('click', async () => {
    await withButtonLoading('autoLearnBtn', labels.autoLearning, async () => {
      try {
        const result = await autoLearnIntents(state);
        const resultEl = document.getElementById('autoLearnResult');
        if (resultEl) {
          resultEl.innerHTML = `
            <div class="bench-row" style="border-left: 3px solid rgba(139,92,246,0.6);padding-left:10px;">
              <span class="bench-label">✅ ${labels.autoLearnDone}</span>
            </div>
            <div class="bench-row">
              <span class="bench-label">${labels.autoLearnCategories}</span>
              <span class="bench-value">${result.categories.join(', ')}</span>
            </div>
            <div class="bench-row">
              <span class="bench-label">${labels.autoLearnTime}</span>
              <span class="bench-value">${result.trainingTimeMs.toFixed(0)}ms</span>
            </div>
          `;
        }
        renderIntentButtons();
      } catch (err) {
        console.error('Auto-learn failed:', err);
      }
    });
  });

  // 精度評価UIの更新
  function updateEvalUI(metrics: EvalMetrics) {
    const ndcgEl = document.getElementById('metricNdcg');
    const recallEl = document.getElementById('metricRecall');
    const catEl = document.getElementById('metricExpectedCat');
    const panelEl = document.getElementById('evalPanel');
    if (!ndcgEl || !recallEl || !catEl) return;

    if (!metrics.expectedCategory) {
      if (panelEl) panelEl.style.display = 'none';
      ndcgEl.textContent = recallEl.textContent = catEl.textContent = '—';
      return;
    }

    if (panelEl) panelEl.style.display = 'block';
    ndcgEl.textContent = metrics.ndcg3.toFixed(4);
    recallEl.textContent = (metrics.recall3 * 100).toFixed(1) + '%';

    const meta = CATEGORY_META[metrics.expectedCategory as DataCategory];
    catEl.textContent = meta
      ? `${meta.label[lang]} (${meta.icon})`
      : metrics.expectedCategory;
  }

  // マージされた重みのエクスポート
  const exportBtn = document.getElementById('exportBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!state.lastMergedWeights) {
        alert(lang === 'ja' 
          ? 'エクスポートするマージ済みの重みがありません。インテントを選択するか、ブレンドしてください。' 
          : 'No merged weights to export. Please select or blend intents.');
        return;
      }

      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.lastMergedWeights, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `warpvector_merged_intent_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  // Resize
  window.addEventListener('resize', () => { resizeCanvas(); drawFrame(); });

  // Init
  renderIntentButtons();
  resizeCanvas();
  initDisplayState();
  drawFrame();
  updateRankingsUI(baseRankings, 0);
  updateCodeSnippet(`// No intent applied\nconst result = baseVector;`);
}
