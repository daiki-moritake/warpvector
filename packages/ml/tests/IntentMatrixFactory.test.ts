import { describe, it, expect } from "bun:test";
import { IntentMatrixFactory } from "../src/factory/IntentMatrixFactory";
import { IntentAdapter, cosineSimilarity } from "@warpvector/core";

/**
 * テスト用のヘルパー: 指定したシードに基づく疑似ランダムなベクトルを生成する。
 * 各テスト呼び出しで再現可能なベクトルを生成するために使用。
 */
function generateSyntheticVector(
  dim: number,
  seed: number,
  direction: Float32Array | null = null,
  noise: number = 0.3,
): Float32Array {
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    // 簡易的な疑似ランダム生成（再現可能）
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = hash - Math.floor(hash) - 0.5;
  }

  // direction が指定されている場合は、そちらにバイアスをかける
  if (direction) {
    for (let i = 0; i < dim; i++) {
      vec[i] = direction[i] * (1 - noise) + vec[i] * noise;
    }
  }

  // L2 正規化
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dim; i++) vec[i] /= norm;
  }

  return vec;
}

/**
 * カテゴリの「方向」を表す基底ベクトルを生成する。
 * 各カテゴリは異なる方向を持つようにする。
 */
function generateCategoryDirection(dim: number, categoryIndex: number): Float32Array {
  const dir = new Float32Array(dim);
  // カテゴリごとに異なる次元レンジに強い成分を持つベクトルを生成
  const startDim = Math.floor((categoryIndex * dim) / 4);
  const endDim = Math.floor(((categoryIndex + 1) * dim) / 4);
  for (let i = startDim; i < endDim && i < dim; i++) {
    dir[i] = 1.0;
  }
  // 正規化
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += dir[i] * dir[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dim; i++) dir[i] /= norm;
  }
  return dir;
}

describe("IntentMatrixFactory", () => {
  const dim = 32; // テスト用の低次元

  describe("基本API", () => {
    it("カテゴリを追加できる", () => {
      const factory = new IntentMatrixFactory(dim);
      const vec = generateSyntheticVector(dim, 1);
      factory.addCategory("tech", [vec]);
      expect(factory.getCategoryNames()).toEqual(["tech"]);
      expect(factory.getSampleCount("tech")).toBe(1);
    });

    it("同じカテゴリにサンプルを追加するとアキュムレートされる", () => {
      const factory = new IntentMatrixFactory(dim);
      factory.addCategory("tech", [generateSyntheticVector(dim, 1)]);
      factory.addCategory("tech", [generateSyntheticVector(dim, 2)]);
      expect(factory.getSampleCount("tech")).toBe(2);
    });

    it("メソッドチェーンが使える", () => {
      const factory = new IntentMatrixFactory(dim);
      const result = factory
        .addCategory("a", [generateSyntheticVector(dim, 1)])
        .addCategory("b", [generateSyntheticVector(dim, 2)]);
      expect(result).toBe(factory);
    });

    it("getTotalSampleCount で全カテゴリの合計サンプル数を返す", () => {
      const factory = new IntentMatrixFactory(dim);
      factory.addCategory("a", [
        generateSyntheticVector(dim, 1),
        generateSyntheticVector(dim, 2),
      ]);
      factory.addCategory("b", [generateSyntheticVector(dim, 3)]);
      expect(factory.getTotalSampleCount()).toBe(3);
    });

    it("存在しないカテゴリの getSampleCount は 0 を返す", () => {
      const factory = new IntentMatrixFactory(dim);
      expect(factory.getSampleCount("nonexistent")).toBe(0);
    });
  });

  describe("バリデーション", () => {
    it("次元数が0以下の場合はエラー", () => {
      expect(() => new IntentMatrixFactory(0)).toThrow();
      expect(() => new IntentMatrixFactory(-1)).toThrow();
    });

    it("次元数が小数の場合はエラー", () => {
      expect(() => new IntentMatrixFactory(1.5)).toThrow();
    });

    it("空のベクトル配列を渡すとエラー", () => {
      const factory = new IntentMatrixFactory(dim);
      expect(() => factory.addCategory("tech", [])).toThrow();
    });

    it("次元数が一致しないベクトルを渡すとエラー", () => {
      const factory = new IntentMatrixFactory(dim);
      const wrongDimVec = new Float32Array(dim + 1);
      expect(() => factory.addCategory("tech", [wrongDimVec])).toThrow();
    });

    it("カテゴリが1つしかない場合は build() でエラー", async () => {
      const factory = new IntentMatrixFactory(dim);
      factory.addCategory("tech", [generateSyntheticVector(dim, 1)]);
      await expect(factory.build()).rejects.toThrow("At least 2 categories");
    });

    it("カテゴリが0個の場合は build() でエラー", async () => {
      const factory = new IntentMatrixFactory(dim);
      await expect(factory.build()).rejects.toThrow("At least 2 categories");
    });
  });

  describe("build() — Intent行列の自動生成", () => {
    it("2カテゴリから IntentWeights を正常に生成できる", async () => {
      const factory = new IntentMatrixFactory(dim);

      const dir1 = generateCategoryDirection(dim, 0);
      const dir2 = generateCategoryDirection(dim, 1);

      factory.addCategory("tech", [
        generateSyntheticVector(dim, 1, dir1),
        generateSyntheticVector(dim, 2, dir1),
        generateSyntheticVector(dim, 3, dir1),
      ]);
      factory.addCategory("business", [
        generateSyntheticVector(dim, 10, dir2),
        generateSyntheticVector(dim, 11, dir2),
        generateSyntheticVector(dim, 12, dir2),
      ]);

      const intents = await factory.build({
        training: { epochs: 50, learningRate: 0.01 },
      });

      // 両方のカテゴリが存在すること
      expect(intents).toHaveProperty("tech");
      expect(intents).toHaveProperty("business");

      // matrix と bias が正しい形状であること
      const tech = intents["tech"];
      expect(tech.matrix).toBeInstanceOf(Float32Array);
      expect((tech.matrix as Float32Array).length).toBe(dim * dim);
      expect(tech.bias).toBeInstanceOf(Float32Array);
      expect((tech.bias as Float32Array).length).toBe(dim);
    });

    it("routingVector がデフォルトで生成される", async () => {
      const factory = new IntentMatrixFactory(dim);

      const dir1 = generateCategoryDirection(dim, 0);
      const dir2 = generateCategoryDirection(dim, 2);

      factory.addCategory("a", [
        generateSyntheticVector(dim, 1, dir1),
        generateSyntheticVector(dim, 2, dir1),
      ]);
      factory.addCategory("b", [
        generateSyntheticVector(dim, 10, dir2),
        generateSyntheticVector(dim, 11, dir2),
      ]);

      const intents = await factory.build({
        training: { epochs: 30 },
      });

      // routingVector が生成されていること
      expect(intents["a"].routingVector).toBeDefined();
      expect(intents["b"].routingVector).toBeDefined();

      const routingA = intents["a"].routingVector!;
      const routingB = intents["b"].routingVector!;

      // routingVector は正規化されていること（L2ノルムが≒1.0）
      let normA = 0;
      for (let i = 0; i < routingA.length; i++) normA += routingA[i] * routingA[i];
      expect(Math.abs(Math.sqrt(normA) - 1.0)).toBeLessThan(0.01);
    });

    it("generateRoutingVectors: false の場合は routingVector を生成しない", async () => {
      const factory = new IntentMatrixFactory(dim);

      const dir1 = generateCategoryDirection(dim, 0);
      const dir2 = generateCategoryDirection(dim, 2);

      factory.addCategory("a", [
        generateSyntheticVector(dim, 1, dir1),
        generateSyntheticVector(dim, 2, dir1),
      ]);
      factory.addCategory("b", [
        generateSyntheticVector(dim, 10, dir2),
        generateSyntheticVector(dim, 11, dir2),
      ]);

      const intents = await factory.build({
        generateRoutingVectors: false,
        training: { epochs: 30 },
      });

      expect(intents["a"].routingVector).toBeUndefined();
      expect(intents["b"].routingVector).toBeUndefined();
    });

    it("生成された IntentWeights は IntentAdapter に投入できる", async () => {
      const factory = new IntentMatrixFactory(dim);

      const dir1 = generateCategoryDirection(dim, 0);
      const dir2 = generateCategoryDirection(dim, 2);

      factory.addCategory("tech", [
        generateSyntheticVector(dim, 1, dir1),
        generateSyntheticVector(dim, 2, dir1),
        generateSyntheticVector(dim, 3, dir1),
      ]);
      factory.addCategory("biz", [
        generateSyntheticVector(dim, 10, dir2),
        generateSyntheticVector(dim, 11, dir2),
        generateSyntheticVector(dim, 12, dir2),
      ]);

      const intents = await factory.build({
        training: { epochs: 50 },
      });

      // IntentAdapter に投入しても例外が発生しないこと
      const adapter = new IntentAdapter(dim);
      adapter.addIntent("tech", intents["tech"]);
      adapter.addIntent("biz", intents["biz"]);

      // tune が正常に実行できること
      const testVec = generateSyntheticVector(dim, 99);
      const techResult = adapter.tune(testVec, "tech");
      const bizResult = adapter.tune(testVec, "biz");

      // 結果が正しい次元数であること
      expect(techResult.length).toBe(dim);
      expect(bizResult.length).toBe(dim);

      // techResult と bizResult が異なる変換であること（全く同一でないこと）
      let same = true;
      for (let i = 0; i < dim; i++) {
        if (Math.abs(techResult[i] - bizResult[i]) > 1e-6) {
          same = false;
          break;
        }
      }
      expect(same).toBe(false);
    });
  });

  describe("3カテゴリ以上", () => {
    it("3カテゴリから IntentWeights を生成できる", async () => {
      const factory = new IntentMatrixFactory(dim);

      const dir1 = generateCategoryDirection(dim, 0);
      const dir2 = generateCategoryDirection(dim, 1);
      const dir3 = generateCategoryDirection(dim, 2);

      factory.addCategory("tech", [
        generateSyntheticVector(dim, 1, dir1),
        generateSyntheticVector(dim, 2, dir1),
      ]);
      factory.addCategory("business", [
        generateSyntheticVector(dim, 10, dir2),
        generateSyntheticVector(dim, 11, dir2),
      ]);
      factory.addCategory("legal", [
        generateSyntheticVector(dim, 20, dir3),
        generateSyntheticVector(dim, 21, dir3),
      ]);

      const intents = await factory.build({
        training: { epochs: 50 },
      });

      expect(Object.keys(intents).sort()).toEqual([
        "business",
        "legal",
        "tech",
      ]);

      // 全ての Intent が有効な重みを持つこと
      for (const name of ["tech", "business", "legal"]) {
        expect(intents[name].matrix).toBeInstanceOf(Float32Array);
        expect(intents[name].bias).toBeInstanceOf(Float32Array);
        expect(intents[name].routingVector).toBeDefined();
      }
    });
  });
});
