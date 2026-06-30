import { Prisma } from "@prisma/client/extension";
import { VectorDBFormatter } from "@warpvector/core";
import { WarpAdapter } from "@warpvector/core";
import sqlTemplate, { Sql, raw } from "sql-template-tag";

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
           * @param args.where 追加のフィルタリング条件 (安全な sql-template-tag Sql オブジェクト)
           */
          async searchByVector<T>(
            this: T,
            args: {
              vector: number[] | Float32Array;
              topK?: number;
              where?: Sql;
            },
          ) {
            const context = Prisma.getExtensionContext(this);
            // prismaのモデル名 -> テーブル名 (正確なマッピングには @@map も考慮されるべきだが簡易的に $name を使用)
            const tableName = (context as any).$name || "document";

            // 1. テーブル名とカラム名の識別子バリデーション (英数字とアンダースコアのみ許容)
            if (!/^[a-zA-Z0-9_]+$/.test(tableName)) {
              throw new Error(`Invalid table name: ${tableName}`);
            }
            if (!/^[a-zA-Z0-9_]+$/.test(vectorField)) {
              throw new Error(`Invalid vector field name: ${vectorField}`);
            }

            // 2. 距離演算子のバリデーション
            if (!["<->", "<#>", "<=>"].includes(distanceOp)) {
              throw new Error(`Invalid distance operator: ${distanceOp}`);
            }

            // 3. limit (topK) の数値バリデーション
            const limit = args.topK ?? 10;
            if (typeof limit !== "number" || isNaN(limit) || limit < 0) {
              throw new Error("Invalid topK value.");
            }

            // 4. WarpVectorによる推論（リアルタイム変換）
            const tunedVector = config.adapter.tune(args.vector);
            const pgVectorStr = VectorDBFormatter.toPgvector(tunedVector);

            // 5. 安全な SQL プレースホルダーオブジェクトの組み立て
            const rawTable = raw(`"${tableName}"`);
            const rawField = raw(`"${vectorField}"`);
            const rawOp = raw(distanceOp);

            let sqlQuery: Sql;
            if (args.where) {
              sqlQuery = sqlTemplate`
                SELECT *
                FROM ${rawTable}
                WHERE ${args.where}
                ORDER BY ${rawField} ${rawOp} ${pgVectorStr}::vector
                LIMIT ${limit};
              `;
            } else {
              sqlQuery = sqlTemplate`
                SELECT *
                FROM ${rawTable}
                ORDER BY ${rawField} ${rawOp} ${pgVectorStr}::vector
                LIMIT ${limit};
              `;
            }

            return (client as any).$queryRaw(sqlQuery);
          },
        },
      },
    });
  });
};
