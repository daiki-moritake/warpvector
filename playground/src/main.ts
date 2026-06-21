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
  cosineSim,
  projectTo2D,
  updateQuery,
  addCustomIntent,
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
  loadingModel: string;
  embeddingTexts: string;
  customIntentName: string;
  customIntentDesc: string;
  addingIntent: string;
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
  },
};

export async function initPlayground(lang: Lang) {
  const canvas = document.getElementById('vectorCanvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const labels = LABELS[lang];

  // Loading UI handling
  const loadingOverlay = document.getElementById('loadingOverlay')!;
  const loadingText = document.getElementById('loadingText')!;
  const loadingSubtext = document.getElementById('loadingSubtext')!;

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
    if (loadingSubtext) loadingSubtext.textContent = (err as Error).message;
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
  const queryInput = document.getElementById('queryInput') as HTMLInputElement;
  if (queryInput) queryInput.value = state.query.text;

  // Build intent buttons dynamically
  function renderIntentButtons() {
    const singlePanel = document.getElementById('intentGroup')!;
    const blendGroup = document.getElementById('blendGroup')!;
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
        const intentKey = slider.dataset.intent!;
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
  let blendWeights: Record<string, number> = {};

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
    // Add small margin so points don't clip
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = 0.15;
    return {
      normalize: (pos: { x: number; y: number }) => ({
        x: padding + (1 - 2 * padding) * (pos.x - minX) / rangeX,
        y: padding + (1 - 2 * padding) * (pos.y - minY) / rangeY,
      }),
    };
  }

  function initDisplayState() {
    animTargetVectors = state.docs.map(d => new Float32Array(d.currentVector));
    animTargetQueryVector = new Float32Array(state.query.currentVector);
    animTargetBasis1 = new Float32Array(state.basis1);
    animTargetBasis2 = new Float32Array(state.basis2);
    
    // Copy target to start
    animStartVectors = animTargetVectors.map(v => new Float32Array(v));
    animStartQueryVector = new Float32Array(animTargetQueryVector);
    animStartBasis1 = new Float32Array(animTargetBasis1);
    animStartBasis2 = new Float32Array(animTargetBasis2);
    
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
    const rect = canvas.parentElement!.getBoundingClientRect();
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
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);

    const { normalize } = normalizePositions(animCurrentPositions, animCurrentQueryPos);

    let found = null;
    for (let i = 0; i < state.docs.length; i++) {
      const pos = normalize(animCurrentPositions[i]);
      const dx = pos.x * w - x;
      const dy = pos.y * h - y;
      if (dx * dx + dy * dy < 100) { // 10px radius
        found = i;
        break;
      }
    }
    
    if (found !== hoverDocId) {
      hoverDocId = found;
      drawFrame();
    }
  });

  // Drawing
  function drawFrame() {
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    const { normalize } = normalizePositions(animCurrentPositions, animCurrentQueryPos);
    const queryDisplayPos = normalize(animCurrentQueryPos);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const p = i / 10;
      ctx.beginPath(); ctx.moveTo(p * w, 0); ctx.lineTo(p * w, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p * h); ctx.lineTo(w, p * h); ctx.stroke();
    }

    const top1Id = currentRankings[0]?.id;

    // Connection lines to top-3
    for (let i = 0; i < Math.min(3, currentRankings.length); i++) {
      const doc = currentRankings[i];
      const pos = normalize(animCurrentPositions[doc.id]);
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
      const pos = normalize(animCurrentPositions[i]);
      const x = pos.x * w, y = pos.y * h;
      const isTop1 = doc.id === top1Id;
      const isHover = doc.id === hoverDocId;
      const radius = isTop1 ? 7 : (isHover ? 6 : 4); // Slightly smaller normal dots for more documents

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
      
      // Only draw text if top1 or hovered, to avoid clutter with 30-40 items
      if (isTop1 || isHover) {
        ctx.fillStyle = isTop1 ? '#f1f5f9' : 'rgba(241,245,249,0.9)';
        ctx.font = `${isTop1 ? 500 : 400} 11px Inter, system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText(doc.name, x, y - radius - 6);
      }
      
      // Hover tooltip text
      if (isHover) {
        ctx.fillStyle = '#f1f5f9';
        ctx.font = '11px Inter, system-ui';
        
        // Wrap text
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
    ctx.textAlign = 'center'; ctx.fillText(`"${state.query.text}"`, qx, qy - 16);
  }

  // Update rankings UI
  function updateRankingsUI(rankings: RankedDoc[], latencyMs: number) {
    const listEl = document.getElementById('rankingList')!;
    listEl.innerHTML = '';

    document.getElementById('metricLatency')!.textContent = latencyMs.toFixed(2) + 'ms';
    document.getElementById('metricTopSim')!.textContent = rankings[0]?.score.toFixed(3) ?? '—';

    let bestImprovement = 0;

    rankings.slice(0, 8).forEach((doc, i) => {
      const vanillaRank = baseRankings.findIndex(d => d.id === doc.id);
      const improvement = vanillaRank - i;
      const isRelevant = improvement > 0; // Highlight anything that improved
      if (improvement > bestImprovement) bestImprovement = improvement;

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

  // Animate transition (Dynamic 384D Interpolation)
  function animateTransition(latencyMs: number, rankings: RankedDoc[]) {
    // 1. Copy current TARGETS to STARTS
    animStartVectors = animTargetVectors.map(v => new Float32Array(v));
    animStartQueryVector = new Float32Array(animTargetQueryVector);
    animStartBasis1 = new Float32Array(animTargetBasis1);
    animStartBasis2 = new Float32Array(animTargetBasis2);

    // 2. Set new TARGETS from DemoState
    animTargetVectors = state.docs.map(d => new Float32Array(d.currentVector));
    animTargetQueryVector = new Float32Array(state.query.currentVector);
    animTargetBasis1 = new Float32Array(state.basis1);
    animTargetBasis2 = new Float32Array(state.basis2);

    const duration = 900; // Slightly longer for a majestic camera rotation feel
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
  const whiteningToggle = document.getElementById('whiteningToggle') as HTMLInputElement | null;
  if (whiteningToggle) {
    whiteningToggle.addEventListener('change', () => {
      state.useWhitening = whiteningToggle.checked;
      recomputeBaseRankings();
      applyCurrentState();
    });
  }

  // Quantization Radio
  const quantGroup = document.getElementById('quantizationGroup');
  const memoryBadge = document.getElementById('memoryBadge');
  if (quantGroup && memoryBadge) {
    quantGroup.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      if (target.name === 'quantMode') {
        state.quantMode = target.value as 'none' | 'int8' | 'binary';
        
        // Update styling
        quantGroup.querySelectorAll('.radio-btn').forEach(l => l.classList.remove('active'));
        target.closest('.radio-btn')?.classList.add('active');

        // Update badge
        if (state.quantMode === 'none') {
          memoryBadge.textContent = '1536 Bytes/vec';
        } else if (state.quantMode === 'int8') {
          memoryBadge.textContent = '384 Bytes/vec';
        } else if (state.quantMode === 'binary') {
          memoryBadge.textContent = '48 Bytes/vec';
        }

        recomputeBaseRankings();
        applyCurrentState();
      }
    });
  }

  // Single intent buttons
  document.getElementById('intentGroup')?.addEventListener('click', (e) => {
    if (isBlendMode) return;
    const btn = (e.target as HTMLElement).closest('.intent-btn') as HTMLElement | null;
    if (btn?.dataset.intent) switchIntent(btn.dataset.intent);
  });

  // Query Submit
  document.getElementById('queryForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!queryInput) return;
    const text = queryInput.value.trim();
    if (!text) return;
    
    // Show spinner in input
    const btn = document.getElementById('querySubmit') as HTMLButtonElement;
    const origText = btn.textContent;
    btn.textContent = '...';
    btn.disabled = true;

    await updateQuery(state, text);
    
    btn.textContent = origText;
    btn.disabled = false;
    
    recomputeBaseRankings();
    applyCurrentState();
  });

  // Add Custom Intent
  document.getElementById('customIntentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('customIntentInput') as HTMLInputElement;
    const text = input.value.trim();
    if (!text) return;

    const btn = document.getElementById('customIntentSubmit') as HTMLButtonElement;
    const origText = btn.textContent;
    btn.textContent = labels.addingIntent;
    btn.disabled = true;

    // Generate random hue for icon
    const hue = Math.floor(Math.random() * 360);
    const color = `hsla(${hue}, 80%, 60%, 0.15)`;
    
    const key = await addCustomIntent(state, labels.customIntentName, text, '✨', color);
    
    btn.textContent = origText;
    btn.disabled = false;
    input.value = '';

    // Re-render UI
    renderIntentButtons();
    
    if (!isBlendMode) {
      switchIntent(key);
    }
  });

  // Benchmark
  const benchBtn = document.getElementById('benchBtn') as HTMLButtonElement | null;
  if (benchBtn) {
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
