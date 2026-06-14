import { Prisma } from "@prisma/client/extension";
import { VectorDBAdapter } from "../db";
import { WarpAdapter } from "../WarpAdapter";

export interface WarpPrismaConfig {
  /**
   * WarpVectorのアダプター (IntentAdapter, WhiteningAdapter など)
   */
  adapter: WarpAdapter;
  /**
   * pgvector のベクトルデータが保存されているカラム名
   * デフォルト: "embedding"
   */
  vectorField?: string;
  /**
   * 類似度検索に使用する距離関数
   * '<->' : ユークリッド距離 (デフォルト)
   * '<#>' : 内積 (マイナス)
   * '<=>' : コサイン距離
   */
  distanceOperator?: "<->" | "<#>" | "<=>";
}

/**
 * Prisma Client に対して WarpVector の推論と pgvector のベクトル検索を
 * 透過的に実行するメソッド `searchByVector` を追加する拡張機能です。
 */
export const withWarpVector = (config: WarpPrismaConfig) => {
  const vectorField = config.vectorField ?? "embedding";
  const distanceOp = config.distanceOperator ?? "<->";

  return Prisma.defineExtension((client) => {
    return client.$extends({
      name: "warpvector",
      model: {
        $allModels: {
          /**
           * 生のベクトルを受け取り、WarpVectorで変換した後に pgvector 検索を行います。
           *
           * @param args.vector 生のベクトル (変換前)
           * @param args.topK 取得する最大件数 (デフォルト: 10)
           * @param args.where 追加のフィルタリング条件 (例: "category = 'science'")
           */
          async searchByVector<T>(
            this: T,
            args: {
              vector: number[] | Float32Array;
              topK?: number;
              where?: string;
            },
          ) {
            const context = Prisma.getExtensionContext(this);
            // prismaのモデル名 -> テーブル名 (正確なマッピングには @@map も考慮されるべきだが簡易的に $name を使用)
            const tableName = (context as any).$name;

            // 1. WarpVectorによる推論（リアルタイム変換）
            const tunedVector = config.adapter.tune(args.vector);
            const pgVectorStr = VectorDBAdapter.toPgvector(tunedVector);

            const limit = args.topK ?? 10;
            const whereClause = args.where ? `WHERE ${args.where}` : "";

            // 2. Prisma の $queryRawUnsafe を使って SQL を実行
            const sql = `
              SELECT *
              FROM "${tableName}"
              ${whereClause}
              ORDER BY "${vectorField}" ${distanceOp} '${pgVectorStr}'::vector
              LIMIT ${limit};
            `;

            return (client as any).$queryRawUnsafe(sql);
          },
        },
      },
    });
  });
};
