import pc from 'picocolors';
import { writePackageJson, writeFile, getRunCommand } from '../scaffold';
import type { TemplateDefinition } from './types';

const WORKER_SRC = `\
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
};`;

function wranglerToml(name: string): string {
  return `\
name = "${name}"
main = "src/index.ts"
compatibility_date = "2024-01-01"`;
}

export const cloudflareTemplate: TemplateDefinition = {
  id: 'cloudflare-worker',

  choice: {
    title: pc.bold('Cloudflare Workers (Edge)') + pc.dim(' — Sub-millisecond inference'),
    value: 'cloudflare-worker',
    description: 'Deploy to edge in seconds',
  },

  generate(dir: string, name: string) {
    writePackageJson(dir, {
      name,
      version: '0.1.0',
      private: true,
      scripts: {
        dev: 'wrangler dev',
        deploy: 'wrangler deploy',
      },
      dependencies: {
        warpvector: '^0.2.0',
        '@warpvector/core': '^0.2.0',
      },
      devDependencies: {
        wrangler: '^3.0.0',
        typescript: '^5.0.0',
        '@cloudflare/workers-types': '^4.0.0',
      },
    });

    writeFile(dir, 'src/index.ts', WORKER_SRC);
    writeFile(dir, 'wrangler.toml', wranglerToml(name));
  },

  getNextSteps(pm: string) {
    const runCmd = getRunCommand(pm);
    return [`${runCmd} dev`];
  },
};
