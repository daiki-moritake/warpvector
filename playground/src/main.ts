/**
 * WarpVector Playground — UI Controller
 *
 * Renders the Canvas visualization and handles user interaction.
 * All vector transformations use the REAL warpvector library.
 */
import { createDemoState, transformWithIntent, cosineSim, type DemoState, type RankedDoc } from './demo-engine.ts';

type Lang = 'en' | 'ja';

const LABELS: Record<Lang, {
  queryLabel: string;
  rankUnit: string;
}> = {
  en: { queryLabel: 'Query: "Apple"', rankUnit: 'ranks' },
  ja: { queryLabel: 'クエリ: "Apple"', rankUnit: '位' },
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

  // Compute initial rankings (no intent)
  let baseRankings = transformWithIntent(state, null).rankings;
  let currentRankings = baseRankings;
  let currentIntent = 'none';

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

  // Initialize display positions
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
    const targetCat = intentCategory[currentIntent];
    let bestImprovement = 0;

    rankings.slice(0, 8).forEach((doc, i) => {
      const vanillaRank = baseRankings.findIndex(d => d.id === doc.id);
      const improvement = vanillaRank - i;
      const isRelevant = targetCat && doc.category === targetCat;
      if (isRelevant && improvement > bestImprovement) bestImprovement = improvement;

      const item = document.createElement('div');
      item.className = 'ranking-item' + (isRelevant && currentIntent !== 'none' ? ' highlight' : '');
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
    deltaEl.textContent = currentIntent === 'none'
      ? '—'
      : bestImprovement > 0 ? '+' + bestImprovement + ' ' + labels.rankUnit : '—';
  }

  // Switch intent with animation
  function switchIntent(intentKey: string) {
    currentIntent = intentKey;

    // Run REAL transformation
    const { latencyMs, rankings } = transformWithIntent(state, intentKey === 'none' ? null : intentKey);
    currentRankings = rankings;

    // Compute new display positions
    const { normalize } = normalizePositions(state.docs, state.query.pos);
    animStartPositions = animCurrentPositions.map(p => ({ ...p }));
    animTargetPositions = state.docs.map(d => normalize(d.pos));

    // Update label
    document.getElementById('intentLabel')!.textContent =
      document.querySelector(`.intent-btn[data-intent="${intentKey}"]`)?.querySelector('.intent-btn__name')?.textContent ?? '';

    // Update button states
    document.querySelectorAll('.intent-btn').forEach(btn => {
      btn.classList.toggle('active', (btn as HTMLElement).dataset.intent === intentKey);
    });

    // Animate
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

  // Events
  document.getElementById('intentGroup')!.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('.intent-btn') as HTMLElement | null;
    if (btn?.dataset.intent) switchIntent(btn.dataset.intent);
  });

  window.addEventListener('resize', () => { resizeCanvas(); drawFrame(); });

  // Initial render
  resizeCanvas();
  drawFrame();
  updateRankingsUI(baseRankings, 0);
}
