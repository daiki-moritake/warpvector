// src/usecases.ts
const isJa = document.documentElement.lang === 'ja';

function renderResult(title: string, meta: string) {
  return `
    <div class="result-item">
      <div class="result-title">${title}</div>
      <div class="result-meta">
        <span>${meta}</span>
      </div>
    </div>
  `;
}

// ---- Use Case 1: Role-Based B2B Search ----
const uc1Results = document.getElementById('uc1-results');
if (uc1Results) {
  // Default state
  uc1Results.innerHTML = isJa ? 
    renderResult('デプロイメントのベストプラクティス', 'スコア: 0.88 • 一般向け') +
    renderResult('Kubernetes デプロイガイド', 'スコア: 0.85 • エンジニア向け') :
    renderResult('Deployment Best Practices', 'Score: 0.88 • General') +
    renderResult('Kubernetes Deployment Guide', 'Score: 0.85 • Engineering');

  document.getElementById('uc1-dev')?.addEventListener('click', () => {
    uc1Results.innerHTML = isJa ?
      renderResult('Kubernetes デプロイガイド', 'スコア: 0.98 • エンジニア向け (Intent: DevOps)') +
      renderResult('CI/CD パイプライン設定手順', 'スコア: 0.95 • エンジニア向け') :
      renderResult('Kubernetes Deployment Guide', 'Score: 0.98 • Eng (Intent: DevOps)') +
      renderResult('CI/CD Pipeline Setup', 'Score: 0.95 • Eng');
  });

  document.getElementById('uc1-sales')?.addEventListener('click', () => {
    uc1Results.innerHTML = isJa ?
      renderResult('A社様 導入事例 (デプロイ期間短縮)', 'スコア: 0.96 • 営業向け (Intent: Sales)') +
      renderResult('エンタープライズ向け製品カタログ', 'スコア: 0.92 • 営業向け') :
      renderResult('Case Study: Company A Deployment', 'Score: 0.96 • Sales (Intent: Sales)') +
      renderResult('Enterprise Product Catalog', 'Score: 0.92 • Sales');
  });
}

// ---- Use Case 2: Offline Edge RAG ----
const uc2Ram = document.getElementById('uc2-ram');
const uc2Latency = document.getElementById('uc2-latency');
const uc2Privacy = document.getElementById('uc2-privacy');
const uc2StatusIcon = document.getElementById('uc2-status-icon');
const uc2StatusText = document.getElementById('uc2-status-text');

document.getElementById('uc2-cloud')?.addEventListener('click', () => {
  if (!uc2Ram || !uc2Latency || !uc2Privacy || !uc2StatusIcon || !uc2StatusText) return;
  uc2Ram.textContent = '614 MB';
  uc2Latency.textContent = isJa ? '120 ms (通信あり)' : '120 ms (Network)';
  uc2Latency.style.color = '#f87171';
  uc2Privacy.textContent = isJa ? '高 (データ送信)' : 'High';
  uc2Privacy.style.color = '#f87171';
  uc2StatusIcon.textContent = '☁️';
  uc2StatusText.textContent = isJa ? 'クラウド接続中' : 'Cloud API Connected';
});

document.getElementById('uc2-local')?.addEventListener('click', () => {
  if (!uc2Ram || !uc2Latency || !uc2Privacy || !uc2StatusIcon || !uc2StatusText) return;
  uc2Ram.textContent = '19.2 MB (96.9% 削減!)';
  uc2Ram.style.color = '#34d399';
  uc2Latency.textContent = isJa ? '0.8 ms (ローカル推論)' : '0.8 ms (Local Inference)';
  uc2Latency.style.color = '#34d399';
  uc2Privacy.textContent = isJa ? 'ゼロ (オフライン)' : 'Zero (Offline)';
  uc2Privacy.style.color = '#34d399';
  uc2StatusIcon.textContent = '📱';
  uc2StatusText.textContent = isJa ? 'オンデバイス (WASM)' : 'On-Device (WASM)';
});

// ---- Use Case 3: Legal Anisotropy ----
const uc3Toggle = document.getElementById('uc3-toggle') as HTMLInputElement;
const uc3Results = document.getElementById('uc3-results');

function updateUc3() {
  if (!uc3Results) return;
  if (uc3Toggle.checked) {
    uc3Results.innerHTML = isJa ?
      renderResult('第7条: 損害賠償の制限', '類似度: 0.820 (正解)') +
      renderResult('第8条: 秘密保持義務', '類似度: 0.415') +
      renderResult('第9条: 契約の解除', '類似度: 0.380') :
      renderResult('Article 7: Limitation of Liability', 'Sim: 0.820 (Correct Match)') +
      renderResult('Article 8: Confidentiality', 'Sim: 0.415') +
      renderResult('Article 9: Termination', 'Sim: 0.380');
  } else {
    uc3Results.innerHTML = isJa ?
      renderResult('第8条: 秘密保持義務', '類似度: 0.992') +
      renderResult('第9条: 契約の解除', '類似度: 0.991') +
      renderResult('第7条: 損害賠償の制限', '類似度: 0.989 (埋もれている)') :
      renderResult('Article 8: Confidentiality', 'Sim: 0.992') +
      renderResult('Article 9: Termination', 'Sim: 0.991') +
      renderResult('Article 7: Limitation of Liability', 'Sim: 0.989 (Buried)');
  }
}
if (uc3Toggle) {
  uc3Toggle.addEventListener('change', updateUc3);
  updateUc3();
}

// ---- Use Case 4: Local Learning ----
let likeCount = 0;
const uc4Status = document.getElementById('uc4-status');
const uc4Next = document.getElementById('uc4-next');

document.getElementById('uc4-like')?.addEventListener('click', () => {
  likeCount++;
  if (!uc4Status || !uc4Next) return;
  
  if (isJa) {
    uc4Status.textContent = `ローカルモデル: ${likeCount}回更新済`;
    uc4Status.className = 'badge badge-green';
    
    if (likeCount === 1) {
      uc4Next.textContent = '最新のAIトレンド (テクノロジー) <- 好みを反映！';
      uc4Next.style.color = '#34d399';
    } else if (likeCount === 2) {
      uc4Next.textContent = '量子暗号の仕組み (テクノロジー) <- さらに特化！';
    } else {
      uc4Next.textContent = 'WarpVectorによるエッジAI (テクノロジー) <- 完璧に適応！';
    }
  } else {
    uc4Status.textContent = `Local Model: Updated ${likeCount}x`;
    uc4Status.className = 'badge badge-green';
    
    if (likeCount === 1) {
      uc4Next.textContent = 'Latest AI Trends (Tech) <- Preferences Applied!';
      uc4Next.style.color = '#34d399';
    } else if (likeCount === 2) {
      uc4Next.textContent = 'How Quantum Crypto Works (Tech) <- Highly Specialized!';
    } else {
      uc4Next.textContent = 'Edge AI with WarpVector (Tech) <- Perfectly Adapted!';
    }
  }
});

// ---- Use Case 5: Semantic Algebra ----
const uc5Results = document.getElementById('uc5-results');
let vConcepts: string[] = [];

function updateUc5() {
  if (!uc5Results) return;
  
  if (vConcepts.length === 0) {
    uc5Results.innerHTML = isJa ?
      renderResult('標準的な赤いナイロンジャケット', 'スコア: 0.91') +
      renderResult('赤いスポーツ用ウィンドブレーカー', 'スコア: 0.88') :
      renderResult('Standard Red Nylon Jacket', 'Score: 0.91') +
      renderResult('Red Sports Windbreaker', 'Score: 0.88');
    return;
  }
  
  const hasVintage = vConcepts.includes('vintage');
  const hasModern = vConcepts.includes('-modern');
  
  if (hasVintage && hasModern) {
    uc5Results.innerHTML = isJa ?
      renderResult('80年代風 赤いコーデュロイジャケット', 'スコア: 0.94 (完璧にマッチ)') +
      renderResult('レトロな赤いウールコート', 'スコア: 0.89') :
      renderResult('80s Style Red Corduroy Jacket', 'Score: 0.94 (Perfect Match)') +
      renderResult('Retro Red Wool Coat', 'Score: 0.89');
  } else if (hasVintage) {
    uc5Results.innerHTML = isJa ?
      renderResult('赤いレザージャケット (ヴィンテージ加工)', 'スコア: 0.92') :
      renderResult('Red Leather Jacket (Distressed Vintage)', 'Score: 0.92');
  } else if (hasModern) {
    uc5Results.innerHTML = isJa ?
      renderResult('クラシックな赤いトレンチコート', 'スコア: 0.90') :
      renderResult('Classic Red Trench Coat', 'Score: 0.90');
  }
}

document.getElementById('uc5-add')?.addEventListener('click', (e) => {
  if (!vConcepts.includes('vintage')) {
    vConcepts.push('vintage');
    (e.target as HTMLElement).style.background = '#ec4899';
    (e.target as HTMLElement).style.color = 'white';
    updateUc5();
  }
});
document.getElementById('uc5-sub')?.addEventListener('click', (e) => {
  if (!vConcepts.includes('-modern')) {
    vConcepts.push('-modern');
    (e.target as HTMLElement).style.background = '#ec4899';
    (e.target as HTMLElement).style.color = 'white';
    updateUc5();
  }
});
document.getElementById('uc5-clear')?.addEventListener('click', () => {
  vConcepts = [];
  const btn1 = document.getElementById('uc5-add');
  const btn2 = document.getElementById('uc5-sub');
  if(btn1) { btn1.style.background = 'transparent'; btn1.style.color = 'inherit'; }
  if(btn2) { btn2.style.background = 'transparent'; btn2.style.color = 'inherit'; }
  updateUc5();
});

// Initialize on load
updateUc5();

// ---- Use Case 6: Model Migration / Alignment ----
const uc6Migrate = document.getElementById('uc6-migrate');
const uc6Status = document.getElementById('uc6-status');
const uc6NewModel = document.getElementById('uc6-new-model');

uc6Migrate?.addEventListener('click', () => {
  if (!uc6Status || !uc6NewModel || !uc6Migrate) return;
  
  const originalText = uc6Migrate.textContent;
  uc6Migrate.textContent = isJa ? '学習中...' : 'Training...';
  uc6Migrate.style.opacity = '0.7';
  
  setTimeout(() => {
    uc6Migrate.textContent = isJa ? '学習完了 (100ペア)' : 'Trained (100 pairs)';
    uc6Migrate.style.background = 'rgba(14,165,233,0.1)';
    uc6Migrate.style.opacity = '1';
    
    uc6NewModel.style.borderColor = '#0ea5e9';
    uc6NewModel.style.background = 'rgba(14,165,233,0.1)';
    
    uc6Status.style.background = 'rgba(16,185,129,0.1)';
    uc6Status.style.borderColor = '#10b981';
    uc6Status.style.color = '#34d399';
    uc6Status.textContent = isJa ? 
      'ステータス: 接続完了 - AlignmentAdapter が 512次元を 1536次元空間へ翻訳' : 
      'Status: Connected - AlignmentAdapter translates 512d to 1536d space';
  }, 800);
});
