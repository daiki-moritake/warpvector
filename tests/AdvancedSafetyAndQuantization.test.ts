import { expect, test, describe } from "bun:test";
import { MlpAdapter } from "../src/MlpAdapter";
import { WhiteningAdapter } from "../src/WhiteningAdapter";
import { WarpPipeline } from "../src/WarpPipeline";
import { QuantizationAdapter } from "../src/QuantizationAdapter";
import { withWarpVector } from "../src/integrations/prisma";
import { VectorDBAdapter } from "../src/db";

describe("Advanced Safety, Memory allocation, and Quantization Tests", () => {
  
  test("1536 and 3072 dimension MLP inference without buffer overrun", async () => {
    // 1536次元
    const dim1536 = 1536;
    const w1 = new Float32Array(dim1536 * 64);
    w1.fill(0.01);
    const b1 = new Float32Array(64);
    b1.fill(0.0);

    const mlp1536 = new MlpAdapter([
      { matrix: w1, bias: b1, activation: "relu" }
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
      { matrix: w2, bias: b2, activation: "linear" }
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
    const pipeline = new WarpPipeline(dim)
      .addMlp([{ matrix: w, bias: b, activation: "linear" }])
      .addWhitening({ learningRate: 0.1, numComponents: 2 });

    await pipeline.init();

    const input = new Float32Array(dim);
    input.fill(1.0);

    // 実行が正常に終わり、メモリアドレスの衝突でクラッシュしないこと
    const output = pipeline.run(input);
    expect(output.length).toBe(dim);
  });

  test("QuantizationAdapter dynamic scaling and dot product restoration", () => {
    const dim = 8;
    const adapter = new QuantizationAdapter({ type: "int8", dim, dynamic: true });
    
    // 絶対値の最大値が 0.05 と非常に小さなベクトル
    const rawVector1 = [0.01, -0.05, 0.02, 0.03, -0.01, 0.04, 0.0, 0.01];
    const rawVector2 = [0.02, 0.01, -0.04, 0.01, 0.02, -0.03, 0.01, 0.02];

    const q1 = adapter.tune(rawVector1) as Int8Array;
    const q2 = adapter.tune(rawVector2) as Int8Array;

    // サイズが dim + 4 (12バイト) になっているはず
    expect(q1.length).toBe(12);

    // 最後の 4 バイトから最大値が正しく読み出せること
    const view1 = new DataView(q1.buffer, q1.byteOffset, q1.byteLength);
    const maxVal1 = view1.getFloat32(dim, true);
    expect(maxVal1).toBeCloseTo(0.05, 5);

    // 動的スケーリングによるドット積の復元
    const expectedDot = rawVector1.reduce((sum, v, i) => sum + v * rawVector2[i], 0);
    const approxDot = QuantizationAdapter.int8DotProduct(q1, q2);

    // 近似された内積が実数空間の内積に近いはず
    expect(approxDot).toBeCloseTo(expectedDot, 4);
  });

  test("Prisma integration blocks potential SQL injection and invalid parameters", async () => {
    // モック用のアダプター
    const adapter: any = { tune: (v: any) => new Float32Array(v) };

    const extension = withWarpVector({
      adapter,
      vectorField: "embedding",
      distanceOperator: "<=>"
    });

    const mockClient: any = {
      $extends: (ext: any) => {
        const extObj = ext({
          $extends: (e: any) => e,
          $queryRawUnsafe: async () => []
        });
        return {
          document: {
            ...extObj.model.$allModels,
            $name: "Document"
          }
        };
      }
    };

    const client = mockClient.$extends(extension);

    // 1. 不正な where 句 (セミコロンによるSQLインジェクション) はエラーになること
    expect(
      client.document.searchByVector({
        vector: [0.1, 0.2],
        where: "category = 'science'; DROP TABLE 'Document';"
      })
    ).rejects.toThrow("Potential SQL injection detected in where clause.");

    // 2. コメントアウト記号を用いた SQL インジェクションもブロックされること
    expect(
      client.document.searchByVector({
        vector: [0.1, 0.2],
        where: "category = 'science' --"
      })
    ).rejects.toThrow("Potential SQL injection detected in where clause.");

    // 3. 不正な topK パラメータ
    expect(
      client.document.searchByVector({
        vector: [0.1, 0.2],
        topK: -5
      })
    ).rejects.toThrow("Invalid topK value.");
  });

  test("VectorDBAdapter.toPgvector converts binary vector to binary bit-string", () => {
    // Binary量子化されたベクトルのモック (16ビット分 = 2バイト)
    const binaryVector = new Uint8Array([240, 15]); // 240 = 11110000, 15 = 00001111
    const pgvectorStr = VectorDBAdapter.toPgvector(binaryVector);
    
    expect(pgvectorStr).toBe("1111000000001111");
  });
});
