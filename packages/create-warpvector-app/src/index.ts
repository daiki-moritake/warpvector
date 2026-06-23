import { Command } from 'commander';
import prompts from 'prompts';
import pc from 'picocolors';
import fs from 'fs';
import path from 'path';

const program = new Command();

program
  .name('create-warpvector-app')
  .description('Scaffold a new WarpVector application')
  .version('0.2.0');

program.action(async () => {
  console.log(pc.blue('🌌 Welcome to WarpVector!'));
  console.log('Let\'s create a new vector search application.\n');

  const response = await prompts([
    {
      type: 'text',
      name: 'projectName',
      message: 'What is your project named?',
      initial: 'my-warpvector-app'
    },
    {
      type: 'select',
      name: 'template',
      message: 'Which template would you like to use?',
      choices: [
        {
          title: pc.bold('Minimal Intent Search') + pc.dim(' — Auto-learn intent matrices, zero config'),
          value: 'minimal-intent',
          description: 'Best for getting started quickly'
        },
        {
          title: pc.bold('Next.js + Prisma + pgvector') + pc.dim(' — Full-stack RAG app'),
          value: 'next-prisma-pgvector',
          description: 'Production-ready with PostgreSQL'
        },
        {
          title: pc.bold('Cloudflare Workers (Edge)') + pc.dim(' — Sub-millisecond inference'),
          value: 'cloudflare-worker',
          description: 'Deploy to edge in seconds'
        },
      ]
    },
    {
      type: 'select',
      name: 'packageManager',
      message: 'Which package manager do you use?',
      choices: [
        { title: 'bun', value: 'bun' },
        { title: 'npm', value: 'npm' },
        { title: 'pnpm', value: 'pnpm' },
        { title: 'yarn', value: 'yarn' },
      ]
    }
  ]);

  if (!response.projectName || !response.template) {
    console.log(pc.red('Operation cancelled.'));
    process.exit(1);
  }

  const projectDir = path.resolve(process.cwd(), response.projectName);

  if (fs.existsSync(projectDir)) {
    console.log(pc.red(`\nError: Directory ${response.projectName} already exists.`));
    process.exit(1);
  }

  console.log(pc.cyan(`\nCreating project in ${projectDir}...`));
  fs.mkdirSync(projectDir, { recursive: true });

  switch (response.template) {
    case 'minimal-intent':
      createMinimalTemplate(projectDir, response.projectName);
      break;
    case 'next-prisma-pgvector':
      createNextTemplate(projectDir, response.projectName);
      break;
    case 'cloudflare-worker':
      createWorkerTemplate(projectDir, response.projectName);
      break;
  }

  // Create shared config files
  createTsConfig(projectDir);

  console.log(pc.green('\n✔ Project created successfully!\n'));
  console.log('Next steps:');
  const pm = response.packageManager || 'npm';
  const runCmd = pm === 'npm' ? 'npx' : pm;
  console.log(pc.cyan(`  cd ${response.projectName}`));
  console.log(pc.cyan(`  ${pm} install`));
  if (response.template === 'minimal-intent') {
    console.log(pc.cyan(`  ${runCmd} tsx src/index.ts`));
  } else if (response.template === 'cloudflare-worker') {
    console.log(pc.cyan(`  ${pm === 'npm' ? 'npm run' : pm} dev`));
  } else {
    console.log(pc.cyan(`  ${pm === 'npm' ? 'npm run' : pm} dev`));
  }

  console.log(pc.dim('\n📖 Docs: https://github.com/daiki-moritake/warpvector'));
  console.log(pc.dim('🎮 Playground: https://daiki-moritake.github.io/warpvector/\n'));
});

program.parse();

// ========================================
// Template: Minimal Intent Search
// ========================================
function createMinimalTemplate(dir: string, name: string) {
  const packageJson = {
    name,
    version: "0.1.0",
    private: true,
    scripts: {
      "start": "tsx src/index.ts",
      "dev": "tsx watch src/index.ts"
    },
    dependencies: {
      "warpvector": "^0.2.0",
      "@warpvector/core": "^0.2.0",
      "@warpvector/ml": "^0.2.0"
    },
    devDependencies: {
      "typescript": "^5.0.0",
      "tsx": "^4.0.0",
      "@types/node": "^20"
    }
  };
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(packageJson, null, 2));

  fs.mkdirSync(path.join(dir, 'src'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'src/index.ts'), MINIMAL_INTENT_TEMPLATE.trim());
  fs.writeFileSync(path.join(dir, 'README.md'), MINIMAL_README.trim());
}

const MINIMAL_INTENT_TEMPLATE = `
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

const MINIMAL_README = `
# ${'{name}'} — WarpVector Minimal Intent Search

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

// ========================================
// Template: Next.js + Prisma + pgvector
// ========================================
function createNextTemplate(dir: string, name: string) {
  const packageJson = {
    name,
    version: "0.1.0",
    private: true,
    scripts: {
      "dev": "next dev",
      "build": "next build",
      "start": "next start"
    },
    dependencies: {
      "next": "14.2.0",
      "react": "^18",
      "react-dom": "^18",
      "warpvector": "^0.2.0",
      "@warpvector/core": "^0.2.0",
      "@warpvector/ml": "^0.2.0",
      "@prisma/client": "^5.0.0",
      "sql-template-tag": "^5.2.1"
    },
    devDependencies: {
      "typescript": "^5",
      "@types/node": "^20",
      "@types/react": "^18",
      "@types/react-dom": "^18",
      "prisma": "^5.0.0"
    }
  };
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(packageJson, null, 2));

  // App structure
  fs.mkdirSync(path.join(dir, 'src/app'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'src/app/page.tsx'), `
import { WarpPipeline } from 'warpvector';

export default function Home() {
  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>🌌 WarpVector Next.js Template</h1>
      <p>Edit <code>src/app/page.tsx</code> to get started.</p>
    </main>
  );
}
  `.trim());

  // Prisma schema
  fs.mkdirSync(path.join(dir, 'prisma'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'prisma/schema.prisma'), `
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model Document {
  id        String   @id @default(cuid())
  content   String
  embedding Unsupported("vector(1536)")?
}
  `.trim());

  // .env.example
  fs.writeFileSync(path.join(dir, '.env.example'), `DATABASE_URL="postgresql://user:password@localhost:5432/warpvector?schema=public"\n`);
}

// ========================================
// Template: Cloudflare Workers
// ========================================
function createWorkerTemplate(dir: string, name: string) {
  const packageJson = {
    name,
    version: "0.1.0",
    private: true,
    scripts: {
      "dev": "wrangler dev",
      "deploy": "wrangler deploy"
    },
    dependencies: {
      "warpvector": "^0.2.0",
      "@warpvector/core": "^0.2.0"
    },
    devDependencies: {
      "wrangler": "^3.0.0",
      "typescript": "^5.0.0",
      "@cloudflare/workers-types": "^4.0.0"
    }
  };
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(packageJson, null, 2));

  fs.mkdirSync(path.join(dir, 'src'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'src/index.ts'), `
import { IntentAdapter, cosineSimilarity, normalize } from '@warpvector/core';

export interface Env {
  // KV namespace for persisting adapter state
  WARP_KV?: KVNamespace;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === '/search' && request.method === 'POST') {
      const body = await request.json() as { query: number[]; intent?: string };

      const adapter = new IntentAdapter(384);
      // 💡 インテント重みを KV/D1/環境変数から読み込む場合:
      //    const state = await env.WARP_KV?.get("adapter-state");
      //    if (state) adapter = IntentAdapter.importState(JSON.parse(state));

      const result = adapter.tune(body.query, body.intent || "default");
      return Response.json({ vector: Array.from(result) });
    }

    return new Response('🌌 WarpVector Edge Function Ready! POST /search to begin.');
  },
};
  `.trim());

  fs.writeFileSync(path.join(dir, 'wrangler.toml'), `
name = "${name}"
main = "src/index.ts"
compatibility_date = "2024-01-01"
  `.trim());
}

// ========================================
// Shared: tsconfig.json
// ========================================
function createTsConfig(dir: string) {
  const tsconfig = {
    compilerOptions: {
      target: "ES2022",
      module: "ESNext",
      moduleResolution: "bundler",
      esModuleInterop: true,
      strict: true,
      skipLibCheck: true,
      outDir: "dist",
      declaration: true
    },
    include: ["src"]
  };
  fs.writeFileSync(path.join(dir, 'tsconfig.json'), JSON.stringify(tsconfig, null, 2));
}
