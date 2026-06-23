import pc from 'picocolors';
import { writePackageJson, writeFile, getExecCommand } from '../scaffold';
import type { TemplateDefinition } from './types';

const INTENT_TEMPLATE = `\
/**
 * WarpVector Minimal Intent Search
 *
 * IntentMatrixFactory を使ってサンプルベクトルから
 * 最適なIntent行列を自動生成し、検索を改善するデモです。
 *
 * 実行: npx tsx src/index.ts
 */
import { IntentAdapter, cosineSimilarity, normalize } from "@warpvector/core";
import { IntentMatrixFactory } from "@warpvector/ml";

// --- 1. 疑似 Embedding ---
// 実際には OpenAI / Cohere / HuggingFace の埋め込みモデルを使用します
const dim = 64;

function embed(text: string, seed: number): Float32Array {
  const vec = new Float32Array(dim);
  const s = text.split("").reduce((a, c) => a + c.charCodeAt(0), seed);
  for (let i = 0; i < dim; i++) {
    const h = Math.sin(s * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (h - Math.floor(h)) * 2 - 1;
  }
  const norm = Math.sqrt(vec.reduce((a, v) => a + v * v, 0));
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

// --- 2. ドキュメント定義 ---
const docs = [
  { text: "TypeScript generics and type inference", domain: "tech" },
  { text: "WebAssembly for high-performance browser apps", domain: "tech" },
  { text: "Kubernetes auto-scaling strategies", domain: "tech" },
  { text: "Q4 revenue forecast and growth analysis", domain: "business" },
  { text: "Customer acquisition cost optimization", domain: "business" },
  { text: "Competitive pricing analysis report", domain: "business" },
].map((d, i) => ({ ...d, vec: embed(d.text, i) }));

// --- 3. Intent行列を自動生成 ---
const factory = new IntentMatrixFactory(dim);
factory.addCategory("tech", docs.filter(d => d.domain === "tech").map(d => d.vec));
factory.addCategory("business", docs.filter(d => d.domain === "business").map(d => d.vec));

const intents = await factory.build({
  training: { epochs: 100, learningRate: 0.01, patience: 10 }
});

// --- 4. 検索を実行 ---
const query = "cloud infrastructure technology";
const qVec = embed(query, 9999);

console.log("\\n🔍 Query:", query, "\\n");

// Vanilla検索
console.log("═══ Vanilla Search ═══");
const vanilla = docs.map(d => ({ ...d, score: cosineSimilarity(qVec, d.vec) }))
  .sort((a, b) => b.score - a.score);
vanilla.forEach((r, i) => {
  const icon = r.domain === "tech" ? "🔧" : "💼";
  console.log(\`  \${i + 1}. \${icon} [\${r.score.toFixed(4)}] \${r.text}\`);
});

// Intent Warping (tech)
const adapter = new IntentAdapter(dim);
adapter.addIntent("tech", intents.tech);
adapter.addIntent("business", intents.business);

const warped = adapter.tune(qVec, "tech");
console.log("\\n═══ Intent Warping (tech) ═══");
const results = docs.map(d => ({ ...d, score: cosineSimilarity(warped, d.vec) }))
  .sort((a, b) => b.score - a.score);
results.forEach((r, i) => {
  const icon = r.domain === "tech" ? "🔧" : "💼";
  console.log(\`  \${i + 1}. \${icon} [\${r.score.toFixed(4)}] \${r.text}\`);
});

console.log("\\n✅ Done! Edit src/index.ts to use your own embedding model.\\n");
`;

const README = `\
# \${name} — WarpVector Minimal Intent Search

WarpVector を使ったインテントベース検索の最小構成プロジェクトです。

## Getting Started

\\\`\\\`\\\`bash
npm install
npx tsx src/index.ts
\\\`\\\`\\\`

## カスタマイズ

1. **\\\`embed()\\\` 関数を差し替える**: OpenAI / Cohere / HuggingFace の埋め込みモデルに接続
2. **ドキュメントを追加**: \\\`docs\\\` 配列に実際のデータを追加
3. **カテゴリを拡張**: \\\`factory.addCategory()\\\` で新しいドメインを追加

## ドキュメント

- [Getting Started](https://github.com/daiki-moritake/warpvector/blob/main/docs/getting-started.md)
- [IntentMatrixFactory](https://github.com/daiki-moritake/warpvector/blob/main/docs/17-intent-matrix-factory.md)
- [API Reference](https://github.com/daiki-moritake/warpvector/blob/main/docs/api-reference.md)
`;

export const minimalIntentTemplate: TemplateDefinition = {
  id: 'minimal-intent',

  choice: {
    title: pc.bold('Minimal Intent Search') + pc.dim(' — Auto-learn intent matrices, zero config'),
    value: 'minimal-intent',
    description: 'Best for getting started quickly',
  },

  generate(dir: string, name: string) {
    writePackageJson(dir, {
      name,
      version: '0.1.0',
      private: true,
      scripts: {
        start: 'tsx src/index.ts',
        dev: 'tsx watch src/index.ts',
      },
      dependencies: {
        warpvector: '^0.2.0',
        '@warpvector/core': '^0.2.0',
        '@warpvector/ml': '^0.2.0',
      },
      devDependencies: {
        typescript: '^5.0.0',
        tsx: '^4.0.0',
        '@types/node': '^20',
      },
    });

    writeFile(dir, 'src/index.ts', INTENT_TEMPLATE);
    writeFile(dir, 'README.md', README);
  },

  getNextSteps(pm: string) {
    const execCmd = getExecCommand(pm);
    return [`${execCmd} tsx src/index.ts`];
  },
};
