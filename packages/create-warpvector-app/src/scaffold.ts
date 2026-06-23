import fs from 'fs';
import path from 'path';

/**
 * 指定ディレクトリに package.json を書き出す
 */
export function writePackageJson(dir: string, config: Record<string, unknown>): void {
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(config, null, 2));
}

/**
 * 親ディレクトリを自動作成してテンプレートファイルを書き出す
 */
export function writeTemplateFile(dir: string, relativePath: string, content: string): void {
  const fullPath = path.join(dir, relativePath);
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  fs.writeFileSync(fullPath, content.trim());
}

/**
 * 共通の tsconfig.json を生成する
 */
export function createTsConfig(dir: string): void {
  const tsconfig = {
    compilerOptions: {
      target: 'ES2022',
      module: 'ESNext',
      moduleResolution: 'bundler',
      esModuleInterop: true,
      strict: true,
      skipLibCheck: true,
      outDir: 'dist',
      declaration: true,
    },
    include: ['src'],
  };
  fs.writeFileSync(path.join(dir, 'tsconfig.json'), JSON.stringify(tsconfig, null, 2));
}

/**
 * パッケージマネージャーに応じた実行コマンドを返す
 */
export function getRunCommand(pm: string): string {
  return pm === 'npm' ? 'npm run' : pm;
}

/**
 * パッケージマネージャーに応じた npx 相当コマンドを返す
 */
export function getExecCommand(pm: string): string {
  switch (pm) {
    case 'pnpm': return 'pnpm exec';
    case 'yarn': return 'yarn';
    case 'bun':  return 'bun';
    default:     return 'npx';
  }
}
