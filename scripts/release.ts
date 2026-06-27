import { $, file } from "bun";
import { join } from "path";
import { existsSync } from "fs";

async function run() {
  // 1. package.jsonから現在のバージョンを取得
  const pkgFile = file("package.json");
  const pkg = await pkgFile.json();
  const version = pkg.version;

  console.log(`🚀 Starting release process for v${version}...`);

  // 2. リリースノートの存在チェック
  const notesPath = join("release-notes", `v${version}.md`);
  if (!existsSync(notesPath)) {
    console.error(`\n❌ Error: Release notes for v${version} not found!`);
    console.error(`   Please create \`${notesPath}\` with the changelog before releasing.\n`);
    process.exit(1);
  }

  console.log(`✅ Found release notes for v${version}.`);

  // 3. テストと型チェックの実行（リリースの安全性を担保）
  console.log(`\n🧪 Running typecheck and tests...`);
  await $`bun run typecheck`;
  await $`bun test`;

  // 4. ワークスペース内のバージョン同期
  console.log(`\n📦 Syncing versions across workspaces...`);
  await $`bun run sync:versions`;

  // 5. コミットとタグの作成
  const status = await $`git status --porcelain`.text();
  if (status.trim() !== "") {
    console.log(`\n📝 Committing version updates...`);
    await $`git commit -am "chore(release): バージョンを ${version} に更新"`;
  } else {
    console.log(`\nℹ️ No new changes to commit for this release.`);
  }

  console.log(`\n🏷️  Creating git tag v${version}...`);
  const tagExists = await $`git tag -l v${version}`.text();
  if (tagExists.trim() === `v${version}`) {
    console.log(`ℹ️ Tag v${version} already exists. Skipping tag creation.`);
  } else {
    await $`git tag -a v${version} -m "Release v${version}"`;
    console.log(`✅ Created tag v${version}.`);
  }

  // 6. リモートへのプッシュとGitHubリリースの作成
  console.log(`\n🚀 Pushing to remote repository...`);
  await $`git push origin main`;
  await $`git push origin v${version}`;

  console.log(`\n🚀 Creating GitHub Release...`);
  await $`gh release create v${version} -F ${notesPath} -t "Release v${version}"`;

  console.log(`\n🎉 All done! Release v${version} is fully published to GitHub.`);
}

run().catch((error) => {
  console.error(`\n❌ Release failed:`, error);
  process.exit(1);
});
