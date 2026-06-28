import { expect, test, describe } from "bun:test";
import { MlpAdapter, WhiteningAdapter } from "@warpvector/ml";
import { BaseTrainer, IntentTrainer } from "@warpvector/train";
import {
  WarpPipeline,
  VectorDBAdapter,
  getWasmAllocatorOffset,
  int8DotProduct,
  globalWasmPool,
} from "@warpvector/core";
import { QuantizationAdapter } from "../src/adapters/QuantizationAdapter";
import { withWarpVector } from "@warpvector/prisma";
import sql from "sql-template-tag";

describe("Advanced Safety, Memory allocation, and Quantization Tests", () => {
  test("1536 and 3072 dimension MLP inference without buffer overrun", async () => {
    // 1536次元
    const dim1536 = 1536;
    const w1 = new Float32Array(dim1536 * 64);
    w1.fill(0.01);
    const b1 = new Float32Array(64);
    b1.fill(0.0);

    const mlp1536 = new MlpAdapter([
      { matrix: w1, bias: b1, activation: "relu" },
    ]);
    await mlp1536.init();

    const input1536 = new Float32Array(dim1536);
    input1536.fill(1.0);

    const out1536 = mlp1536.tune(input1536);
    expect(out1536.length).toBe(64);
    // 1.0 * 0.01 = 0.01、それが 1536 次元分足されるので、各要素の和は 15.36 に近いはず
    expect(out1536[0]).toBeCloseTo(15.36, 1);

    // 3072次元
    const dim3072 = 3072;
    const w2 = new Float32Array(dim3072 * 16);
    w2.fill(0.005);
    const b2 = new Float32Array(16);
    b2.fill(0.1);

    const mlp3072 = new MlpAdapter([
      { matrix: w2, bias: b2, activation: "linear" },
    ]);
    await mlp3072.init();

    const input3072 = new Float32Array(dim3072);
    input3072.fill(2.0);

    const out3072 = mlp3072.tune(input3072);
    expect(out3072.length).toBe(16);
    // 2.0 * 0.005 = 0.01、それが 3072 次元分足される = 30.72。それにバイアス 0.1 を加算して 30.82
    expect(out3072[0]).toBeCloseTo(30.82, 1);
  });

  test("Pipeline with multi WASM-based adapters does not collide in memory", async () => {
    const dim = 128;
    const w = new Float32Array(dim * dim);
    w.fill(0.05);
    const b = new Float32Array(dim);
    b.fill(0.0);

    // MlpAdapter と WhiteningAdapter を両方使うパイプライン
    const mlpAdapter = new MlpAdapter([
      { matrix: w, bias: b, activation: "linear" },
    ]);
    const whiteningAdapter = new WhiteningAdapter(dim, {
      learningRate: 0.1,
      numComponents: 2,
    });
    const pipeline = new WarpPipeline(dim)
      .addStep("MlpAdapter", mlpAdapter)
      .addStep("WhiteningAdapter", whiteningAdapter);

    await pipeline.init();

    const input = new Float32Array(dim);
    input.fill(1.0);

    // 実行が正常に終わり、メモリアドレスの衝突でクラッシュしないこと
    const output = await pipeline.run(input);
    expect(output.length).toBe(dim);
  });

  test("QuantizationAdapter dynamic scaling and dot product restoration", () => {
    const dim = 8;
    const adapter = new QuantizationAdapter({
      type: "int8",
      dim,
      dynamic: true,
    });

    // 絶対値の最大値が 0.05 と非常に小さなベクトル
    const rawVector1 = [0.01, -0.05, 0.02, 0.03, -0.01, 0.04, 0.0, 0.01];
    const rawVector2 = [0.02, 0.01, -0.04, 0.01, 0.02, -0.03, 0.01, 0.02];

    const q1 = adapter.encode(new Float32Array(rawVector1)) as Int8Array;
    const q2 = adapter.encode(new Float32Array(rawVector2)) as Int8Array;

    // サイズが dim + 4 (12バイト) になっているはず
    expect(q1.length).toBe(12);

    // 最後の 4 バイトから最大値が正しく読み出せること
    const view1 = new DataView(q1.buffer, q1.byteOffset, q1.byteLength);
    const maxVal1 = view1.getFloat32(dim, true);
    expect(maxVal1).toBeCloseTo(0.05, 5);

    // 動的スケーリングによるドット積の復元
    const expectedDot = rawVector1.reduce(
      (sum, v, i) => sum + v * rawVector2[i],
      0,
    );
    const approxDot = int8DotProduct(q1, q2);

    // 近似された内積が実数空間の内積に近いはず
    expect(approxDot).toBeCloseTo(expectedDot, 4);

    // 絶対値の最大値が 1200.0 と 1000.0 を超える大きなベクトル
    const largeVector1 = [1000.0, -1200.0, 500.0, 300.0, -100.0, 400.0, 0.0, 100.0];
    const largeVector2 = [200.0, 100.0, -400.0, 100.0, 200.0, -300.0, 100.0, 200.0];

    const qLarge1 = adapter.encode(new Float32Array(largeVector1)) as Int8Array;
    const qLarge2 = adapter.encode(new Float32Array(largeVector2)) as Int8Array;

    const expectedLargeDot = largeVector1.reduce(
      (sum, v, i) => sum + v * largeVector2[i],
      0,
    );
    const approxLargeDot = int8DotProduct(qLarge1, qLarge2);

    // 最大値が 1000.0 を超えても、正しく dynamic 量子化と判定され、
    // ドット積がスケールを考慮して近似復元されること (誤差 2% 未満)
    const relativeError = Math.abs(approxLargeDot - expectedLargeDot) / Math.abs(expectedLargeDot);
    expect(relativeError).toBeLessThan(0.02);
  });

  test("Prisma integration blocks potential SQL injection and invalid parameters", async () => {
    // モック用のアダプター
    const adapter: any = { tune: (v: any) => new Float32Array(v) };

    const extension = withWarpVector({
      adapter,
      vectorField: "embedding",
      distanceOperator: "<=>",
    });

    const mockClient: any = {
      $extends: (ext: any) => {
        const extObj = ext({
          $extends: (e: any) => e,
          $queryRaw: async () => [],
        });
        return {
          document: {
            ...extObj.model.$allModels,
            $name: "Document",
          },
        };
      },
    };

    const client = mockClient.$extends(extension);

    // 1. Prisma.Sql 形式での正常動作テスト
    const results = await client.document.searchByVector({
      vector: [0.1, 0.2],
      where: sql`category = ${"science"}`,
      topK: 5,
    });
    expect(results).toEqual([]);

    // 2. 不正な topK パラメータ
    expect(
      client.document.searchByVector({
        vector: [0.1, 0.2],
        topK: -5,
      }),
    ).rejects.toThrow("Invalid topK value.");
  });

  test("VectorDBAdapter.toPgvector converts binary vector to binary bit-string", () => {
    // Binary量子化されたベクトルのモック (16ビット分 = 2バイト)
    const binaryVector = new Uint8Array([240, 15]); // 240 = 11110000, 15 = 00001111
    const pgvectorStr = VectorDBAdapter.toPgvector(binaryVector);

    expect(pgvectorStr).toBe("1111000000001111");
  });

  test("MlpAdapter consecutive execution does not grow WASM memory offset (leak-free)", async () => {
    const dim = 16;
    const w = new Float32Array(dim * dim);
    w.fill(0.01);
    const b = new Float32Array(dim);

    const mlp = new MlpAdapter([{ matrix: w, bias: b, activation: "linear" }]);
    await mlp.init();

    const input = new Float32Array(dim);
    input.fill(1.0);

    const offsetBefore = getWasmAllocatorOffset();

    for (let i = 0; i < 50; i++) {
      mlp.tune(input);
    }

    const offsetAfter = getWasmAllocatorOffset();
    expect(offsetAfter).toBe(offsetBefore);
  });

  test("BaseTrainer concurrent training runs without WASM memory collision", async () => {
    type Example = { source: number[]; target: number[] };
    type Weights = { matrix: Float32Array; bias: Float32Array };
    class DummyTrainer extends BaseTrainer<Example, Weights> {
      get sourceDimension() {
        return 2;
      }
      get targetDimension() {
        return 2;
      }
      calculateLoss() {
        return 0;
      }
      adamStep() {
        // Mock step
      }
      toWeights(matrix: Float32Array, bias: Float32Array): Weights {
        return { matrix, bias };
      }
    }

    const trainer1 = new DummyTrainer();
    trainer1.addExample({ source: [1, 2], target: [2, 4] });
    trainer1.addExample({ source: [3, 6], target: [6, 12] });

    const trainer2 = new DummyTrainer();
    trainer2.addExample({ source: [2, 3], target: [4, 6] });
    trainer2.addExample({ source: [4, 5], target: [8, 10] });

    // IntentTrainer のインスタンスを用意
    const intentTrainer = new IntentTrainer(2);
    const initialWeights = {
      matrix: new Float32Array([1, 0, 0, 1]),
      bias: new Float32Array([0, 0]),
    };

    // train と並行して updateOnline を呼び出し、WASM メモリ衝突でクラッシュしないことを確認
    const [w1, w2, wOnline] = await Promise.all([
      trainer1.train({ epochs: 10 }),
      trainer2.train({ epochs: 10 }),
      intentTrainer.updateOnline(initialWeights, {
        input: [1, 2],
        target: [2, 4],
      }),
    ]);

    expect(w1).toBeDefined();
    expect(w2).toBeDefined();
    expect(wOnline).toBeDefined();
  });

  test("MlpAdapter inference across different WASM contexts works correctly", async () => {
    const dim = 16;
    const w = new Float32Array(dim * dim);
    w.fill(0.01);
    const b = new Float32Array(dim);
    b.fill(0.1);

    const mlp = new MlpAdapter([{ matrix: w, bias: b, activation: "linear" }]);
    await mlp.init();

    const runInContext = (val: number) => {
      const wasmCtx = globalWasmPool.acquire();
      try {
        globalWasmPool.setCurrentSyncContext(wasmCtx);
        const input = new Float32Array(dim);
        input.fill(val);
        const output = mlp.tune(input);
        const expected = val * 0.16 + 0.1;
        expect(output[0]).toBeCloseTo(expected, 4);
      } finally {
        globalWasmPool.clearCurrentSyncContext();
        globalWasmPool.release(wasmCtx);
      }
    };

    // 順次、異なるコンテキストで実行
    runInContext(1.0);
    runInContext(2.0);
    runInContext(3.0);
  });

  test("MlpAdapter handles advanced WASM memory offset in active context safely without crash", async () => {
    const dim = 16;
    const w = new Float32Array(dim * dim);
    w.fill(0.01);
    const b = new Float32Array(dim);
    b.fill(0.1);

    const mlp = new MlpAdapter([{ matrix: w, bias: b, activation: "linear" }]);
    await mlp.init();

    const wasmCtx = globalWasmPool.acquire();
    try {
      globalWasmPool.setCurrentSyncContext(wasmCtx);

      // 他のアダプターの実行などを模倣して、アクティブなコンテキストのメモリを大きく進める (10MB)
      wasmCtx.allocate(10000000); // 10MB 確保

      const input = new Float32Array(dim);
      input.fill(1.0);

      // 修正後は、メモリが適切に拡張されたインスタンス上で実行されるため、
      // クラッシュせずに正常に推論結果（expected = 1.0 * 0.16 + 0.1 = 0.26）が得られる
      const output = mlp.tune(input);
      expect(output[0]).toBeCloseTo(0.26, 4);
    } finally {
      globalWasmPool.clearCurrentSyncContext();
      globalWasmPool.release(wasmCtx);
    }
  });
});
