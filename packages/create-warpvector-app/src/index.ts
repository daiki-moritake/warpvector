import { Command } from 'commander';
import prompts from 'prompts';
import pc from 'picocolors';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

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
        { title: 'Next.js + Prisma + pgvector', value: 'next-prisma-pgvector' },
        { title: 'Cloudflare Workers (Edge)', value: 'cloudflare-worker' }
      ]
    }
  ]);

  if (!response.projectName) {
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

  // For the purpose of this CLI, we will create the template files dynamically
  // If this was a production CLI, we would copy them from a templates folder or github repo.
  
  if (response.template === 'next-prisma-pgvector') {
    createNextTemplate(projectDir, response.projectName);
  } else {
    createWorkerTemplate(projectDir, response.projectName);
  }

  console.log(pc.green('\n✔ Project created successfully!\n'));
  console.log('Next steps:');
  console.log(pc.cyan(`  cd ${response.projectName}`));
  console.log(pc.cyan('  npm install'));
  console.log(pc.cyan('  npm run dev'));
});

program.parse();

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

  // basic structure
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
}

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
      "warpvector": "^0.2.0"
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
import { IntentAdapter } from 'warpvector';

export interface Env {}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const adapter = new IntentAdapter(384);
    
    return new Response('🌌 WarpVector Edge Function Ready!');
  },
};
  `.trim());
  
  fs.writeFileSync(path.join(dir, 'wrangler.toml'), `
name = "${name}"
main = "src/index.ts"
compatibility_date = "2024-01-01"
  `.trim());
}
