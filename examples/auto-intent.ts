/**
 * IntentMatrixFactory を使った自動Intent行列生成の完全な例
 *
 * このスクリプトは、サンプルベクトルからIntent行列を自動生成し、
 * 検索クエリをドメインに応じて最適化する一連の流れを示します。
 *
 * 実行: bun run examples/auto-intent.ts
 */
import {
  IntentAdapter,
  WarpPipeline,
  cosineSimilarity,
  normalize,
} from "@warpvector/core";
import { IntentMatrixFactory } from "@warpvector/train";

// ========================================
// 1. 擬似的な Embedding 関数
// ========================================
// 実際のアプリケーションでは OpenAI / Cohere / 自前のモデルを使用します
const dim = 64;

function pseudoEmbed(text: string, seed: number): Float32Array {
  const vec = new Float32Array(dim);
  // テキストのハッシュ値をシードとして使用
  const textSeed = text
    .split("")
    .reduce((acc, c) => acc + c.charCodeAt(0), seed);
  for (let i = 0; i < dim; i++) {
    const hash = Math.sin(textSeed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (hash - Math.floor(hash)) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

// ========================================
// 2. ドキュメントコーパスの定義
// ========================================
const documents = [
  {
    id: 1,
    text: "TypeScript の型システムとジェネリクスの活用法",
    domain: "tech",
  },
  { id: 2, text: "WebAssembly で実現する高速なブラウザ演算", domain: "tech" },
  {
    id: 3,
    text: "エッジコンピューティングのアーキテクチャ設計",
    domain: "tech",
  },
  { id: 4, text: "React Server Components の実践パターン", domain: "tech" },
  { id: 5, text: "Kubernetes オートスケーリング戦略", domain: "tech" },
  { id: 6, text: "2024年Q4の売上予測と成長率分析", domain: "business" },
  { id: 7, text: "SaaS企業の顧客獲得コスト最適化戦略", domain: "business" },
  { id: 8, text: "競合他社のプライシング分析レポート", domain: "business" },
  { id: 9, text: "IPO に向けた財務デューデリジェンス", domain: "business" },
  { id: 10, text: "サブスクリプションモデルの解約率改善", domain: "business" },
];

// ドキュメントベクトルの生成
const docVectors = documents.map((doc) => ({
  ...doc,
  vector: pseudoEmbed(doc.text, doc.id),
}));

// ========================================
// 3. IntentMatrixFactory でIntent行列を自動生成
// ========================================
console.log("🔧 IntentMatrixFactory でIntent行列を自動生成中...\n");

const factory = new IntentMatrixFactory(dim);

// Tech カテゴリのサンプル（ドキュメントから取得）
factory.addCategory(
  "tech",
  docVectors.filter((d) => d.domain === "tech").map((d) => d.vector),
);

// Business カテゴリのサンプル
factory.addCategory(
  "business",
  docVectors.filter((d) => d.domain === "business").map((d) => d.vector),
);

// 自動学習（InfoNCE対照学習で最適なアフィン変換を学習）
const intents = await factory.build({
  training: { epochs: 100, learningRate: 0.01, patience: 10 },
});

console.log("✅ Intent行列の生成完了");
console.log(
  `   tech:     matrix ${(intents.tech.matrix as Float32Array).length} 要素, routingVector あり`,
);
console.log(
  `   business: matrix ${(intents.business.matrix as Float32Array).length} 要素, routingVector あり\n`,
);

// ========================================
// 4. パイプラインを構築
// ========================================
const pipeline = new WarpPipeline(dim).addIntent(intents);

// ========================================
// 5. 検索の実行: Vanilla vs Intent Warping
// ========================================
const query = "最新のクラウドインフラ技術";
const queryVector = pseudoEmbed(query, 9999);

console.log(`📝 検索クエリ: "${query}"\n`);

// --- Vanilla 検索（変換なし） ---
console.log("═══ Vanilla 検索（変換なし） ═══");
const vanillaResults = docVectors
  .map((doc) => ({
    ...doc,
    score: cosineSimilarity(queryVector, doc.vector),
  }))
  .sort((a, b) => b.score - a.score);

vanillaResults.slice(0, 5).forEach((r, i) => {
  const marker = r.domain === "tech" ? "🔧" : "💼";
  console.log(`  ${i + 1}. ${marker} [${r.score.toFixed(4)}] ${r.text}`);
});

// --- Intent Warping: tech 意図 ---
console.log("\n═══ Intent Warping: tech 意図 ═══");
const techQueryWarped = (await pipeline.run(queryVector, {
  intent: "tech",
})) as Float32Array;
const techResults = docVectors
  .map((doc) => ({
    ...doc,
    score: cosineSimilarity(techQueryWarped, doc.vector),
  }))
  .sort((a, b) => b.score - a.score);

techResults.slice(0, 5).forEach((r, i) => {
  const marker = r.domain === "tech" ? "🔧" : "💼";
  console.log(`  ${i + 1}. ${marker} [${r.score.toFixed(4)}] ${r.text}`);
});

// --- Intent Warping: business 意図 ---
console.log("\n═══ Intent Warping: business 意図 ═══");
const bizQueryWarped = (await pipeline.run(queryVector, {
  intent: "business",
})) as Float32Array;
const bizResults = docVectors
  .map((doc) => ({
    ...doc,
    score: cosineSimilarity(bizQueryWarped, doc.vector),
  }))
  .sort((a, b) => b.score - a.score);

bizResults.slice(0, 5).forEach((r, i) => {
  const marker = r.domain === "tech" ? "🔧" : "💼";
  console.log(`  ${i + 1}. ${marker} [${r.score.toFixed(4)}] ${r.text}`);
});

// --- Auto-blending（自動ルーティング） ---
console.log("\n═══ Auto-blending（自動ルーティング） ═══");
const adapter = new IntentAdapter(dim);
adapter.addIntent("tech", intents.tech);
adapter.addIntent("business", intents.business);
const autoWarped = adapter.tuneAutoBlended(queryVector);
const autoResults = docVectors
  .map((doc) => ({
    ...doc,
    score: cosineSimilarity(autoWarped, doc.vector),
  }))
  .sort((a, b) => b.score - a.score);

autoResults.slice(0, 5).forEach((r, i) => {
  const marker = r.domain === "tech" ? "🔧" : "💼";
  console.log(`  ${i + 1}. ${marker} [${r.score.toFixed(4)}] ${r.text}`);
});

// ========================================
// 6. サマリー
// ========================================
const vanillaTechCount = vanillaResults
  .slice(0, 5)
  .filter((r) => r.domain === "tech").length;
const warpedTechCount = techResults
  .slice(0, 5)
  .filter((r) => r.domain === "tech").length;

console.log("\n════════════════════════════════════════");
console.log("📊 サマリー");
console.log(`   Vanilla 検索 Top5 の tech 文書数:    ${vanillaTechCount}/5`);
console.log(`   Intent Warping Top5 の tech 文書数:  ${warpedTechCount}/5`);
console.log("════════════════════════════════════════\n");
