import { expect, test, describe } from "bun:test";
import { withWarpVector } from "../src/prisma";
import { WarpAdapter } from "@warpvector/core";
import sql from "sql-template-tag";

// モック用のアダプター (ベクトルの値を2倍にするだけ)
class MockAdapter implements WarpAdapter {
  tune(vector: number[] | Float32Array): Float32Array {
    const arr = Array.from(vector);
    return new Float32Array(arr.map((v) => v * 2));
  }
}

describe("WarpPrismaExtension", () => {
  test("generates correct SQL and transforms vector", async () => {
    const mockAdapter = new MockAdapter();
    const rawVector = [1.0, 2.0, 3.0];

    // PrismaClientのモック
    let capturedSql: any = null;

    // ベースとなるモッククライアント
    const baseClient: any = {
      $queryRaw: async (sql: any) => {
        capturedSql = sql;
        return [{ id: 1, content: "Mock Result" }];
      },
      $extends: (e: any) => e,
    };

    const mockPrismaClient: any = {
      $extends: (ext: any) => {
        // extは(client) => client.$extends(...) という関数
        const extension = ext(baseClient);

        return {
          document: {
            ...extension.model.$allModels,
            $name: "Document",
          },
          ...baseClient,
        };
      },
    };

    const prismaExtension = withWarpVector({
      adapter: mockAdapter,
      vectorField: "my_embedding",
      distanceOperator: "<=>",
    });

    const client = mockPrismaClient.$extends(prismaExtension);

    // テスト対象のメソッドを呼び出す (where 句に sql-template-tag の sql を指定)
    const results = await client.document.searchByVector({
      vector: rawVector,
      topK: 5,
      where: sql`category = ${"science"}`,
    });

    expect(results).toEqual([{ id: 1, content: "Mock Result" }]);

    // 生成されたSQLの検証 (capturedSql は Sql)
    expect(capturedSql).not.toBeNull();
    const text = capturedSql.text;
    const values = capturedSql.values;

    expect(text).toContain("SELECT *");
    expect(text).toContain('FROM "Document"');
    expect(text).toContain("WHERE category = $1");
    // $2::vector ではなく、Prisma拡張の展開順により $2 が pgVectorStr、
    // そして $3 が limit になる。
    // text表現を確認する：
    // SELECT * FROM "Document" WHERE category = $1 ORDER BY "my_embedding" <=> $2::vector LIMIT $3;
    expect(text).toContain('ORDER BY "my_embedding" <=> $2::vector');
    expect(text).toContain("LIMIT $3");

    expect(values[0]).toBe("science");
    expect(values[1]).toBe("[2, 4, 6]");
    expect(values[2]).toBe(5);
  });
});
