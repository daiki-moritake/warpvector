import { describe, expect, test } from "bun:test";
import { IntentAdapter } from "@warpvector/core";
import {
  WarpLlamaIndexEmbeddings,
  LlamaIndexBaseEmbedding,
} from "../src/llama-index";

// Mock BaseEmbedding for testing
class MockLlamaEmbeddings implements LlamaIndexBaseEmbedding {
  async getTextEmbedding(text: string): Promise<number[]> {
    return [0.1, 0.2, 0.3];
  }
  async getQueryEmbedding(query: string): Promise<number[]> {
    return [0.1, 0.2, 0.3]; // Same vector for simplicity
  }
}

describe("WarpLlamaIndexEmbeddings (LlamaIndex Integration)", () => {
  const mockBase = new MockLlamaEmbeddings();

  test("getTextEmbedding passes through without transformation", async () => {
    const adapter = new IntentAdapter(3);
    const warpEmbeddings = new WarpLlamaIndexEmbeddings({
      baseEmbeddings: mockBase,
      adapter,
      intentName: "test",
    });

    const result = await warpEmbeddings.getTextEmbedding("hello");
    expect(result).toEqual([0.1, 0.2, 0.3]);
  });

  test("getQueryEmbedding applies intent transformation", async () => {
    const adapter = new IntentAdapter(3);
    adapter.addIntent("test", {
      matrix: [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ],
      bias: [1, 1, 1],
    });

    const warpEmbeddings = new WarpLlamaIndexEmbeddings({
      baseEmbeddings: mockBase,
      adapter,
      intentName: "test",
    });

    // base is [0.1, 0.2, 0.3]
    // tune -> [0.1*2 + 1, 0.2*2 + 1, 0.3*2 + 1] = [1.2, 1.4, 1.6]
    const result = await warpEmbeddings.getQueryEmbedding("query");

    expect(result[0]).toBeCloseTo(1.2);
    expect(result[1]).toBeCloseTo(1.4);
    expect(result[2]).toBeCloseTo(1.6);
  });
});
