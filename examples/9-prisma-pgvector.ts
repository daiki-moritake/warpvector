
import { withWarpVector } from "../src/integrations/prisma";
import { WhiteningAdapter } from "../src/WhiteningAdapter";

console.log("--- Prisma + pgvector 統合デモ ---");
console.log("WarpVectorによるリアルタイムベクトル推論（Whitening補正等）と、");
console.log("Prismaを用いたデータベース検索を自動的に連携させる拡張機能です。\n");

// 1. WarpVector のアダプターを用意（ここでは等方化フィルタを使用）
const adapter = new WhiteningAdapter(3);
// ダミー学習 (本来はデータベースの過去のベクトル群で学習する)
adapter.update([1, 2, 3]);
adapter.update([1, 2, 4]);
adapter.update([2, 2, 3]);

console.log("WarpVector アダプターの準備完了");

// 2. Prisma Client に WarpVector 拡張機能を適用
// 実際のDB接続がなくてもSQLの生成まではモックで確認可能
// （ライブラリのリポジトリ内には Prisma スキーマがないため、モッククライアントを使用します）
const basePrisma = {
  $extends: function(ext: any) {
    const extension = ext({
      $queryRawUnsafe: async (sql: string) => {
        console.log("--- 実際にデータベースに送信されるSQL ---");
        console.log(sql.trim());
        console.log("-----------------------------------------");
        return [{ id: 1, content: "ダミーの検索結果", score: 0.99 }];
      },
      $extends: (e: any) => e
    });
    return {
      document: {
        ...extension.model.$allModels,
        $name: "Document"
      }
    };
  }
} as any;

// 拡張機能を適用
const prisma = basePrisma.$extends(
  withWarpVector({
    adapter: adapter,
    vectorField: "embedding", // pgvector の保存先カラム名
    distanceOperator: "<=>",  // コサイン距離
  })
);

// 3. 生の検索クエリをそのまま渡して検索実行
const rawSearchVector = [1.5, 2.0, 3.5];

console.log("🔍 生のベクトルで検索を実行します:", rawSearchVector);
console.log("  -> 内部で WarpVector.tune() が自動的にかかり、PostgreSQLに投げられます。\n");

// 【重要】開発者は通常の Prisma クエリのようにメソッドを呼ぶだけ！
// document モデルはスキーマが存在しないため、any型でキャストして呼び出します（デモ用）
const result = await (prisma as any).document.searchByVector({
  vector: rawSearchVector,
  topK: 3,
  where: "category = 'science'",
});

console.log("\n📦 検索結果:");
console.log(result);

console.log("\n💡 結論: WarpVector の変換とベクターDB (pgvector) の検索が、一つのメソッドで美しく統合されました！");
