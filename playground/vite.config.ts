import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  base: './',
  resolve: {
    alias: {
      '@warpvector/core': resolve(__dirname, '../packages/core/src/index.ts'),
      '@warpvector/ml': resolve(__dirname, '../packages/ml/src/index.ts'),
      '@warpvector/extras': resolve(__dirname, '../packages/extras/src/index.ts'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        ja: resolve(__dirname, 'ja.html'),
      },
    },
  },
});
