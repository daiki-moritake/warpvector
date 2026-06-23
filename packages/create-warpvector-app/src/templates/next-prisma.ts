import pc from 'picocolors';
import { writePackageJson, writeFile, getRunCommand } from '../scaffold';
import type { TemplateDefinition } from './types';

const PAGE_TSX = `\
import { WarpPipeline } from 'warpvector';

export default function Home() {
  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>🌌 WarpVector Next.js Template</h1>
      <p>Edit <code>src/app/page.tsx</code> to get started.</p>
    </main>
  );
}`;

const PRISMA_SCHEMA = `\
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
}`;

const ENV_EXAMPLE = `DATABASE_URL="postgresql://user:password@localhost:5432/warpvector?schema=public"`;

export const nextPrismaTemplate: TemplateDefinition = {
  id: 'next-prisma-pgvector',

  choice: {
    title: pc.bold('Next.js + Prisma + pgvector') + pc.dim(' — Full-stack RAG app'),
    value: 'next-prisma-pgvector',
    description: 'Production-ready with PostgreSQL',
  },

  generate(dir: string, name: string) {
    writePackageJson(dir, {
      name,
      version: '0.1.0',
      private: true,
      scripts: {
        dev: 'next dev',
        build: 'next build',
        start: 'next start',
      },
      dependencies: {
        next: '14.2.0',
        react: '^18',
        'react-dom': '^18',
        warpvector: '^0.2.0',
        '@warpvector/core': '^0.2.0',
        '@warpvector/ml': '^0.2.0',
        '@prisma/client': '^5.0.0',
        'sql-template-tag': '^5.2.1',
      },
      devDependencies: {
        typescript: '^5',
        '@types/node': '^20',
        '@types/react': '^18',
        '@types/react-dom': '^18',
        prisma: '^5.0.0',
      },
    });

    writeFile(dir, 'src/app/page.tsx', PAGE_TSX);
    writeFile(dir, 'prisma/schema.prisma', PRISMA_SCHEMA);
    writeFile(dir, '.env.example', ENV_EXAMPLE);
  },

  getNextSteps(pm: string) {
    const runCmd = getRunCommand(pm);
    return [`${runCmd} dev`];
  },
};
