/**
 * sync-versions.ts
 *
 * ルートの package.json の version をすべてのサブパッケージに同期するスクリプト。
 * バージョンの唯一の信頼源 (single source of truth) はルートの package.json です。
 *
 * 使い方: bun run scripts/sync-versions.ts
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from "fs";
import { join } from "path";

const ROOT = join(import.meta.dir, "..");
const rootPkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf-8"));
const version: string = rootPkg.version;

const packagesDir = join(ROOT, "packages");
const subPackages = readdirSync(packagesDir).filter((name) =>
  statSync(join(packagesDir, name)).isDirectory(),
);

let synced = 0;

for (const name of subPackages) {
  const pkgPath = join(packagesDir, name, "package.json");
  try {
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    if (pkg.version !== version) {
      const oldVersion = pkg.version;
      pkg.version = version;
      writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n", "utf-8");
      console.log(`  ✅ ${pkg.name}: ${oldVersion} → ${version}`);
      synced++;
    } else {
      console.log(`  ✓  ${pkg.name}: ${version} (already synced)`);
    }
  } catch {
    // package.json が存在しないディレクトリはスキップ
  }
}

console.log(
  `\nDone. ${synced} package(s) updated to v${version}.`,
);
