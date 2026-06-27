import { $, file } from "bun";
import { join } from "path";
import { existsSync } from "fs";

async function run() {
  // 1. package.jsonから現在のバージョンを取得
  const pkgFile = file("package.json");
  const pkg = await pkgFile.json();
  const version = pkg.version;

  console.log(`🚀 v${version} のリリース処理を開始します...`);

  // 2. リリースノートの存在チェック
  const notesPath = join("release-notes", `v${version}.md`);
  if (!existsSync(notesPath)) {
    console.error(`\n❌ エラー: v${version} のリリースノートが見つかりません！`);
    console.error(`   リリース前に \`${notesPath}\` を作成し、変更内容を記述してください。\n`);
    process.exit(1);
  }

  console.log(`✅ v${version} のリリースノートを確認しました。`);

  // 3. テストと型チェックの実行（リリースの安全性を担保）
  console.log(`\n🧪 型チェックとテストを実行しています...`);
  await $`bun run typecheck`;
  await $`bun test`;

  // 4. ワークスペース内のバージョン同期
  console.log(`\n📦 ワークスペース内のバージョンを同期しています...`);
  await $`bun run sync:versions`;

  // 5. コミットとタグの作成
  const status = await $`git status --porcelain`.text();
  if (status.trim() !== "") {
    console.log(`\n📝 バージョン更新をコミットしています...`);
    await $`git commit -am "chore(release): バージョンを ${version} に更新"`;
  } else {
    console.log(`\nℹ️ コミットする新しい変更はありません。`);
  }

  console.log(`\n🏷️  gitタグ v${version} を作成しています...`);
  const tagExists = await $`git tag -l v${version}`.text();
  if (tagExists.trim() === `v${version}`) {
    console.log(`ℹ️ タグ v${version} は既に存在するため作成をスキップします。`);
  } else {
    await $`git tag -a v${version} -m "Release v${version}"`;
    console.log(`✅ タグ v${version} を作成しました。`);
  }

  // 6. リモートへのプッシュとGitHubリリースの作成
  console.log(`\n🚀 リモートリポジトリへプッシュしています...`);
  await $`git push origin main`;
  await $`git push origin v${version}`;

  console.log(`\n🚀 GitHubリリースを作成しています...`);
  await $`gh release create v${version} -F ${notesPath} -t "Release v${version}"`;

  console.log(`\n🎉 完了しました！ v${version} のリリースがGitHubに公開されました。`);
}

run().catch((error) => {
  console.error(`\n❌ リリース処理が失敗しました:`, error);
  process.exit(1);
});
