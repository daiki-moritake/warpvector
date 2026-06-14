import { expect, test, describe } from "bun:test";
import { withWarpVector, WarpAdapter } from "../src/integrations/prisma";
import { PrismaClient } from "@prisma/client";

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
    let capturedSql = "";

    // ベースとなるモッククライアント
    const baseClient: any = {
      $queryRawUnsafe: async (sql: string) => {
        capturedSql = sql;
        return [{ id: 1, content: "Mock Result" }];
      },
      $extends: (e: any) => e,
    };

    const mockPrismaClient: any = {
      $extends: (ext: any) => {
        // extは(client) => client.$extends(...) という関数
        // この client にベースクライアントの機能を渡す
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

    // withWarpVectorは Prisma.defineExtension を返す
    const prismaExtension = withWarpVector({
      adapter: mockAdapter,
      vectorField: "my_embedding",
      distanceOperator: "<=>",
    });

    const client = mockPrismaClient.$extends(prismaExtension);

    // テスト対象のメソッドを呼び出す
    const results = await client.document.searchByVector({
      vector: rawVector,
      topK: 5,
      where: "category = 'science'",
    });

    expect(results).toEqual([{ id: 1, content: "Mock Result" }]);

    // 生成されたSQLの検証
    // vector [1,2,3] は MockAdapter により [2,4,6] に変換されるはず
    expect(capturedSql).toContain("SELECT *");
    expect(capturedSql).toContain('FROM "Document"');
    expect(capturedSql).toContain("WHERE category = 'science'");
    expect(capturedSql).toContain(
      "ORDER BY \"my_embedding\" <=> '[2, 4, 6]'::vector",
    );
    expect(capturedSql).toContain("LIMIT 5");
  });
});
