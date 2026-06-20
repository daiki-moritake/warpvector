import { expect, test, describe } from "bun:test";
import { IntentAdapter } from "../src/adapters/IntentAdapter";

describe("IntentAdapter Binary Serialization", () => {
  test("exports and imports intent perfectly", () => {
    // 次元数3のダミーアダプター
    const adapter1 = new IntentAdapter(3);

    // Flat Float32Array での直接追加テスト
    const flatMatrix = new Float32Array([
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ]);
    const bias = new Float32Array([0.1, 0.2, 0.3]);
    const routing = new Float32Array([1.0, 0.0, 0.0]);

    adapter1.addIntent("myIntent", {
      matrix: flatMatrix,
      bias: bias,
      routingVector: routing,
    });

    // 1. エクスポート
    const binary = adapter1.exportIntentBinary("myIntent");

    // 2. 別の空のアダプターにインポート
    const adapter2 = new IntentAdapter(3);
    adapter2.importIntentBinary("importedIntent", binary);

    // 3. 同じ入力に対して同じ出力になるか確認
    const input = [1.0, 1.0, 1.0];
    const res1 = adapter1.tune(input, "myIntent");
    const res2 = adapter2.tune(input, "importedIntent");

    expect(res1[0]).toBeCloseTo(res2[0]);
    expect(res1[1]).toBeCloseTo(res2[1]);
    expect(res1[2]).toBeCloseTo(res2[2]);

    // 4. routingVector も正しく復元されているか確認（AutoBlend が動くか）
    const autoBlendRes1 = adapter1.tuneAutoBlended(input);
    const autoBlendRes2 = adapter2.tuneAutoBlended(input);

    expect(autoBlendRes1[0]).toBeCloseTo(autoBlendRes2[0]);
  });

  test("throws error on dimension mismatch during import", () => {
    const adapter1 = new IntentAdapter(3);
    adapter1.addIntent("intent1", {
      matrix: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      bias: [0.0, 0.0, 0.0],
    });
    const binary = adapter1.exportIntentBinary("intent1");

    const adapter2 = new IntentAdapter(4); // 異なる次元数
    expect(() => {
      adapter2.importIntentBinary("imported", binary);
    }).toThrow("Dimension mismatch");
  });
});
