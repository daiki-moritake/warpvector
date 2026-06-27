// src/usecases.ts
const isJa = document.documentElement.lang === 'ja';

// ---- Use Case 1: E-commerce Intent Search ----
const uc1Result = document.getElementById('uc1-result')!;
document.getElementById('uc1-none')?.addEventListener('click', () => {
  uc1Result.textContent = isJa ? 
`1. Apple iPhone 15 Pro (類似度: 0.82)
2. 青森県産ふじりんご 1kg (類似度: 0.81)
3. Apple MacBook Air M3 (類似度: 0.79)` :
`1. Apple iPhone 15 Pro (Sim: 0.82)
2. Fresh Fuji Apples 1kg (Sim: 0.81)
3. Apple MacBook Air M3 (Sim: 0.79)`;
});
document.getElementById('uc1-tech')?.addEventListener('click', () => {
  uc1Result.textContent = isJa ?
`[ガジェット Intent適用済 (遅延: 0.8ms)]
1. Apple iPhone 15 Pro (類似度: 0.94)
2. Apple MacBook Air M3 (類似度: 0.91)
3. Apple AirPods Pro (類似度: 0.88)` :
`[Tech Intent Applied (Latency: 0.8ms)]
1. Apple iPhone 15 Pro (Sim: 0.94)
2. Apple MacBook Air M3 (Sim: 0.91)
3. Apple AirPods Pro (Sim: 0.88)`;
});
document.getElementById('uc1-food')?.addEventListener('click', () => {
  uc1Result.textContent = isJa ?
`[食品 Intent適用済 (遅延: 0.7ms)]
1. 青森県産ふじりんご 1kg (類似度: 0.95)
2. 信州産 蜜入りりんご 5kg (類似度: 0.92)
3. 100% ストレートアップルジュース (類似度: 0.86)` :
`[Food Intent Applied (Latency: 0.7ms)]
1. Fresh Fuji Apples 1kg (Sim: 0.95)
2. Honeycrisp Apples 5kg (Sim: 0.92)
3. 100% Pure Apple Juice (Sim: 0.86)`;
});

// ---- Use Case 2: Quantization Edge RAG ----
const uc2Result = document.getElementById('uc2-result')!;
document.getElementById('uc2-float')?.addEventListener('click', () => {
  uc2Result.textContent = isJa ?
`1536次元ベクトルの保存サイズ: 6,144 Bytes
ベクトルDBの推定コスト: 月額 10,000円
Recall@10 精度: 100% (基準)` :
`Storage per 1536-dim vector: 6,144 Bytes
Estimated DB Cost: $100 / month
Recall@10 Accuracy: 100% (Baseline)`;
});
document.getElementById('uc2-int8')?.addEventListener('click', () => {
  uc2Result.textContent = isJa ?
`[Int8 量子化適用済]
1536次元ベクトルの保存サイズ: 1,536 Bytes (75%削減!)
ベクトルDBの推定コスト: 月額 2,500円
Recall@10 精度: 98.2% (ほぼ劣化なし)` :
`[Int8 Quantization Applied]
Storage per 1536-dim vector: 1,536 Bytes (75% Reduction!)
Estimated DB Cost: $25 / month
Recall@10 Accuracy: 98.2% (Near Lossless)`;
});
document.getElementById('uc2-binary')?.addEventListener('click', () => {
  uc2Result.textContent = isJa ?
`[Binary 量子化適用済]
1536次元ベクトルの保存サイズ: 192 Bytes (96.9%削減!)
ベクトルDBの推定コスト: 月額 300円以下 (エッジのメモリに乗るサイズ)
Recall@10 精度: 86.5% (高速な一次検索に最適)` :
`[Binary Quantization Applied]
Storage per 1536-dim vector: 192 Bytes (96.9% Reduction!)
Estimated DB Cost: < $3 / month (Fits in Edge RAM)
Recall@10 Accuracy: 86.5% (Perfect for 1st-stage retrieval)`;
});

// ---- Use Case 3: Domain Whitening ----
const uc3Result = document.getElementById('uc3-result')!;
document.getElementById('uc3-raw')?.addEventListener('click', () => {
  uc3Result.textContent = isJa ?
`クエリ: "パスワードのリセット方法は？"
1位: "アカウントのログイン手順" (類似度: 0.992)
2位: "パスワード忘れに関するFAQ" (類似度: 0.991) <- スコアが近すぎる！
解像度が低く、正確なドキュメントが上位に来ていません。` :
`Query: "How to reset password?"
Top Match: "Account Login Guide" (Sim: 0.992)
2nd Match: "Forgot Password FAQ" (Sim: 0.991) <- Too close!
Resolution is poor.`;
});
document.getElementById('uc3-white')?.addEventListener('click', () => {
  uc3Result.textContent = isJa ?
`[WhiteningAdapter 適用済]
クエリ: "パスワードのリセット方法は？"
1位: "パスワード忘れに関するFAQ" (類似度: 0.850) <- 正解が1位に浮上！
2位: "アカウントのログイン手順" (類似度: 0.612) <- スコアの差が明確に
解像度が劇的に改善されました。` :
`[WhiteningAdapter Applied]
Query: "How to reset password?"
Top Match: "Forgot Password FAQ" (Sim: 0.850) <- Correct match jumped to 1st!
2nd Match: "Account Login Guide" (Sim: 0.612) <- Clear separation!
Resolution significantly improved.`;
});

// ---- Use Case 4: Local Learning ----
const uc4Result = document.getElementById('uc4-result')!;
let learnCount = 0;
document.getElementById('uc4-click')?.addEventListener('click', () => {
  learnCount++;
  uc4Result.textContent = isJa ?
`[InfoNCETrainer] バックグラウンドでローカル重みを更新中...
ローカルモデルの更新回数: ${learnCount} 回` :
`[InfoNCETrainer] Updating local weights in background...
Weights updated: ${learnCount} times.`;
});
document.getElementById('uc4-search')?.addEventListener('click', () => {
  if (learnCount === 0) {
    uc4Result.textContent = isJa ?
`検索クエリ: "映画"
1. 「ローマの休日」 (ロマンス)
2. 「シンドラーのリスト」 (ドラマ)
3. 「ダイ・ハード」 (アクション)
※ 一般的な検索結果です。` :
`Query: "Movie"
1. "Roman Holiday" (Romance)
2. "Schindler's List" (Drama)
3. "Die Hard" (Action)
* Showing generic results.`;
  } else {
    uc4Result.textContent = isJa ?
`検索クエリ: "映画" [パーソナライズ重み適用]
1. 「ダイ・ハード」 (アクション) <- ${learnCount}回の学習で1位に！
2. 「ターミネーター2」 (アクション)
3. 「マッドマックス」 (アクション)
※ サーバーにデータを送らずに好みを学習しました。` :
`Query: "Movie" [Personalized Weights Applied]
1. "Die Hard" (Action) <- Jumped to 1st after ${learnCount} updates!
2. "Terminator 2" (Action)
3. "Mad Max" (Action)
* Learned your preferences completely offline!`;
  }
});

// ---- Use Case 5: VSA Semantic Curation ----
const uc5Result = document.getElementById('uc5-result')!;
let concepts: string[] = [];

function updateVsa() {
  if (concepts.length === 0) {
    uc5Result.textContent = isJa ? `現在のクエリの概念: [空]` : `Current Query Bundle: [Empty]`;
    return;
  }
  let queryStr = concepts.join(' + ').replace('+ -', '- ');
  uc5Result.textContent = isJa ? 
`現在のクエリの概念: [ ${queryStr} ]

[VSA ベクトル合成中 (遅延: 1.2ms)...]
検索結果:
1. 「ブレードランナー 2049」 (完璧にマッチ)
2. 「GHOST IN THE SHELL / 攻殻機動隊」` :
`Current Query Bundle: [ ${queryStr} ]

[VSA Vector Bundling (Latency: 1.2ms)...]
Results:
1. "Blade Runner 2049" (Perfect match)
2. "Ghost in the Shell"`;
}

document.getElementById('uc5-sf')?.addEventListener('click', () => {
  if(!concepts.includes('SF')) concepts.push('SF');
  updateVsa();
});
document.getElementById('uc5-cyber')?.addEventListener('click', () => {
  if(!concepts.includes('Cyberpunk')) concepts.push('Cyberpunk');
  updateVsa();
});
document.getElementById('uc5-dark')?.addEventListener('click', () => {
  if(!concepts.includes('-Dark')) concepts.push('-Dark');
  updateVsa();
});
document.getElementById('uc5-clear')?.addEventListener('click', () => {
  concepts = [];
  updateVsa();
});
