import { expect, test, describe } from "bun:test";
import { WarpEmbeddings } from "../src/integrations/langchain";
import { Embeddings } from "@langchain/core/embeddings";
import { IntentAdapter } from "../src/IntentAdapter";

// ダミーのベース Embeddings 実装
class MockBaseEmbeddings extends Embeddings {
  async embedDocuments(documents: string[]): Promise<number[][]> {
    // 常に同じベクトルを返すモック
    return documents.map(() => [1.0, 0.5, 0.0]);
  }
  async embedQuery(document: string): Promise<number[]> {
    return [1.0, 0.5, 0.0];
  }
}

describe("WarpEmbeddings (LangChain Integration)", () => {
  test("embedQuery applies intent transformation", async () => {
    const base = new MockBaseEmbeddings({});

    // x, y を入れ替える行列を定義
    const intentWeights = {
      matrix: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    };

    const adapter = new IntentAdapter({ myIntent: intentWeights });

    const warpEmbeddings = new WarpEmbeddings({
      baseEmbeddings: base,
      adapter: adapter,
      intentName: "myIntent",
    });

    const queryRes = await warpEmbeddings.embedQuery("Hello");
    // [1.0, 0.5, 0.0] が [0.5, 1.0, 0.0] に変換されるはず
    expect(queryRes[0]).toBeCloseTo(0.5);
    expect(queryRes[1]).toBeCloseTo(1.0);
    expect(queryRes[2]).toBeCloseTo(0.0);
  });

  test("embedDocuments passes through without transformation", async () => {
    const base = new MockBaseEmbeddings({});

    const intentWeights = {
      matrix: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    };

    const adapter = new IntentAdapter({ myIntent: intentWeights });

    const warpEmbeddings = new WarpEmbeddings({
      baseEmbeddings: base,
      adapter: adapter,
      intentName: "myIntent",
    });

    const docsRes = await warpEmbeddings.embedDocuments(["doc1", "doc2"]);
    // ドキュメントは変換されずそのまま [1.0, 0.5, 0.0] となる
    expect(docsRes[0][0]).toBe(1.0);
    expect(docsRes[0][1]).toBe(0.5);
    expect(docsRes[0][2]).toBe(0.0);
  });

  test("can dynamically switch intent", async () => {
    const base = new MockBaseEmbeddings({});

    const intent1 = {
      matrix: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    };

    const intent2 = {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [1, 1, 1],
    };

    const adapter = new IntentAdapter({ intent1: intent1, intent2: intent2 });

    const warpEmbeddings = new WarpEmbeddings({
      baseEmbeddings: base,
      adapter: adapter,
      intentName: "intent1",
    });

    warpEmbeddings.setIntent("intent2");
    const queryRes = await warpEmbeddings.embedQuery("Hello");
    // [1.0, 0.5, 0.0] に [1, 1, 1] を足すので [2.0, 1.5, 1.0] になるはず
    expect(queryRes[0]).toBeCloseTo(2.0);
    expect(queryRes[1]).toBeCloseTo(1.5);
    expect(queryRes[2]).toBeCloseTo(1.0);
  });
});
