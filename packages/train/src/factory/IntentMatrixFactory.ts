import { IntentWeights } from "@warpvector/core";
import {
  assertDimension,
  normalize,
  cosineSimilarity,
  innerProduct,
} from "@warpvector/core";
import type { InfoNCEExample, TripletExample } from "@warpvector/core";
import { InfoNCETrainer } from "../trainers/InfoNCETrainer";

/**
 * Negative サンプリング戦略。
 * - "uniform": 均等サンプリング（既存の動作）
 * - "hard": anchor に最も近い（最も紛らわしい）negative を優先選択
 * - "semi-hard": positive より遠いが、ランダムよりは近い negative を選択
 */
export type NegativeStrategy = "uniform" | "hard" | "semi-hard";
import type { BaseTrainingOptions } from "../trainers/types";

/**
 * IntentMatrixFactory のビルドオプション。
 */
export interface IntentMatrixFactoryOptions {
  /**
   * 学習時のハイパーパラメータ。
   * `autoTune: true` を設定すると、最適な学習率を自動探索します。
   * @default { epochs: 200, learningRate: 0.01, autoTune: true, patience: 15 }
   */
  training?: BaseTrainingOptions;

  /**
   * InfoNCE の温度パラメータ。
   * 小さいほど正解と不正解の分離が鋭くなりますが、勾配が不安定になる場合があります。
   * @default 0.07
   */
  temperature?: number;

  /**
   * 各カテゴリの平均ベクトルを routingVector として自動設定するかどうか。
   * true にすると、`tuneAutoBlended()` で入力ベクトルに最も近いカテゴリが自動選択されます。
   * @default true
   */
  generateRoutingVectors?: boolean;

  /**
   * 各 anchor に対する negative サンプルの最大数。
   * 値が大きいほど学習は安定しますが、計算コストが増加します。
   * @default 7
   */
  maxNegativesPerAnchor?: number;

  /**
   * Negative サンプリング戦略。
   * - "uniform": 均等サンプリング（デフォルト、既存の動作）
   * - "hard": anchor に最も近い negative を優先。紛らわしい境界ケースに集中して学習
   * - "semi-hard": positive より遠いが近めの negative を選択。学習の安定性と効率のバランス
   * @default "uniform"
   */
  negativeStrategy?: NegativeStrategy;
}

/**
 * カテゴリごとのサンプルベクトルから、最適な IntentWeights（行列Wとバイアスb）を
 * 自動生成するファクトリクラスです。
 *
 * 内部では InfoNCE（対照学習）を用いて、「同じカテゴリのベクトルを近づけ、
 * 異なるカテゴリのベクトルを遠ざける」ようにアフィン変換を最適化します。
 *
 * ユーザーは各カテゴリにサンプルベクトルを 5〜10 個程度追加するだけで、
 * IntentAdapter にそのまま投入できる IntentWeights が得られます。
 *
 * @example
 * ```typescript
 * import { IntentMatrixFactory } from 'warpvector/ml';
 *
 * const factory = new IntentMatrixFactory(1536);
 *
 * // カテゴリごとにサンプルベクトルを追加
 * factory.addCategory("tech", [
 *   await embed("TypeScript runtime performance"),
 *   await embed("WebAssembly SIMD optimization"),
 *   await embed("Edge computing latency"),
 * ]);
 * factory.addCategory("business", [
 *   await embed("Q4 revenue forecast"),
 *   await embed("Market share analysis"),
 *   await embed("Customer acquisition cost"),
 * ]);
 *
 * // 自動で最適な Intent 行列を学習
 * const intents = await factory.build();
 * // → { tech: { matrix, bias, routingVector }, business: { matrix, bias, routingVector } }
 *
 * // そのままパイプラインに投入
 * const pipeline = new WarpPipeline(1536).addIntent(intents);
 * ```
 */
export class IntentMatrixFactory {
  private readonly dimension: number;
  private readonly categories: Map<string, (number[] | Float32Array)[]>;

  /**
   * @param dimension 入力ベクトルの次元数
   */
  constructor(dimension: number) {
    if (dimension <= 0 || !Number.isInteger(dimension)) {
      throw new Error(
        `Dimension must be a positive integer, got ${dimension}.`,
      );
    }
    this.dimension = dimension;
    this.categories = new Map();
  }

  /**
   * カテゴリにサンプルベクトルを追加します。
   * 同じカテゴリに対して複数回呼び出すと、サンプルが追加（アキュムレート）されます。
   *
   * @param categoryName カテゴリ名（= Intent 名として使用されます）
   * @param vectors サンプルベクトルの配列（各ベクトルは dimension 次元）
   * @throws 次元数が一致しないベクトルが含まれている場合にエラーをスローします。
   */
  public addCategory(
    categoryName: string,
    vectors: (number[] | Float32Array)[],
  ): this {
    if (vectors.length === 0) {
      throw new Error(
        `At least one sample vector is required for category '${categoryName}'.`,
      );
    }

    for (let i = 0; i < vectors.length; i++) {
      assertDimension(
        vectors[i],
        this.dimension,
        `Category '${categoryName}' vector[${i}]`,
      );
    }

    const existing = this.categories.get(categoryName) ?? [];
    existing.push(...vectors);
    this.categories.set(categoryName, existing);

    return this;
  }

  /**
   * 登録済みの全カテゴリ名を返します。
   */
  public getCategoryNames(): string[] {
    return Array.from(this.categories.keys());
  }

  /**
   * 指定カテゴリに登録されているサンプル数を返します。
   */
  public getSampleCount(categoryName: string): number {
    return this.categories.get(categoryName)?.length ?? 0;
  }

  /**
   * 全カテゴリの合計サンプル数を返します。
   */
  public getTotalSampleCount(): number {
    let total = 0;
    for (const vectors of this.categories.values()) {
      total += vectors.length;
    }
    return total;
  }

  /**
   * 登録されたカテゴリとサンプルベクトルから、各カテゴリ用の IntentWeights を自動生成します。
   *
   * 学習プロセス:
   * 1. カテゴリ間の対照学習データ（InfoNCE 形式）を自動構築
   * 2. InfoNCETrainer で最適なアフィン変換（W, b）を学習
   * 3. 各カテゴリの平均ベクトルを routingVector として設定（オプション）
   *
   * @param options ビルドオプション
   * @returns 各カテゴリ名をキーとした IntentWeights のマッピング
   * @throws カテゴリが2つ未満の場合にエラーをスローします。
   */
  public async build(
    options: IntentMatrixFactoryOptions = {},
  ): Promise<Record<string, IntentWeights>> {
    const categoryNames = Array.from(this.categories.keys());

    if (categoryNames.length < 2) {
      throw new Error(
        "At least 2 categories are required to build intent matrices. " +
          `Currently ${categoryNames.length} category is registered.`,
      );
    }

    const temperature = options.temperature ?? 0.07;
    const generateRouting = options.generateRoutingVectors ?? true;
    const maxNegsPerAnchor = options.maxNegativesPerAnchor ?? 7;
    const negativeStrategy = options.negativeStrategy ?? "uniform";

    const trainingOptions: BaseTrainingOptions = {
      epochs: 200,
      learningRate: 0.01,
      autoTune: true,
      patience: 15,
      ...options.training,
    };

    // Step 1: カテゴリごとの平均ベクトル（セントロイド）を計算
    const centroids = new Map<string, Float32Array>();
    for (const [name, vectors] of this.categories.entries()) {
      centroids.set(name, this.computeCentroid(vectors));
    }

    // Step 2: 各カテゴリ用の InfoNCE 学習データを構築し、学習
    const result: Record<string, IntentWeights> = {};

    for (const categoryName of categoryNames) {
      const categoryVectors = this.categories.get(categoryName)!;
      const otherCategoryNames = categoryNames.filter(
        (n) => n !== categoryName,
      );

      // このカテゴリ用の InfoNCE サンプルを構築
      const examples = this.buildInfoNCEExamples(
        categoryName,
        categoryVectors,
        otherCategoryNames,
        maxNegsPerAnchor,
        negativeStrategy,
      );

      // InfoNCETrainer でカテゴリ特化のアフィン変換を学習
      const trainer = new InfoNCETrainer(this.dimension);
      for (const example of examples) {
        trainer.addExample(example);
      }

      const weights = await trainer.train({
        ...trainingOptions,
        temperature,
      } as BaseTrainingOptions & { temperature?: number });

      // routingVector を設定
      if (generateRouting) {
        const centroid = centroids.get(categoryName)!;
        weights.routingVector = normalize(centroid);
      }

      result[categoryName] = weights;
    }

    return result;
  }

  /**
   * 特定のカテゴリ用の InfoNCE 学習データを構築します。
   *
   * 「anchor = カテゴリ内サンプル」「positive = 同カテゴリの別サンプル」「negatives = 他カテゴリのサンプル」
   * という対照学習ペアを生成します。
   */
  private buildInfoNCEExamples(
    categoryName: string,
    categoryVectors: (number[] | Float32Array)[],
    otherCategoryNames: string[],
    maxNegatives: number,
    negativeStrategy: NegativeStrategy = "uniform",
  ): InfoNCEExample[] {
    const examples: InfoNCEExample[] = [];

    // 他カテゴリから negative サンプルを集める
    const allNegatives: (number[] | Float32Array)[] = [];
    for (const otherName of otherCategoryNames) {
      const otherVectors = this.categories.get(otherName)!;
      allNegatives.push(...otherVectors);
    }

    for (let i = 0; i < categoryVectors.length; i++) {
      const anchor = categoryVectors[i];

      // positive: 同カテゴリの他のサンプルからランダムに1つ選択
      // サンプルが1つしかない場合は、anchor 自身を微小ノイズ付きで使用
      let positive: number[] | Float32Array;
      if (categoryVectors.length > 1) {
        const posIdx = i === 0 ? 1 : i - 1;
        positive = categoryVectors[posIdx];
      } else {
        // 唯一のサンプル → 自身のコピー（学習信号は弱いが、クラッシュは避ける）
        positive = anchor;
      }

      // negatives: 他カテゴリからサンプリング（戦略に応じて選択方法が変わる）
      const negatives = this.sampleNegatives(
        allNegatives,
        maxNegatives,
        i, // シードとして使用
        negativeStrategy,
        anchor,
        positive,
      );

      if (negatives.length === 0) {
        continue;
      }

      examples.push({ anchor, positive, negatives });
    }

    return examples;
  }

  /**
   * negative サンプルをサンプリングします。
   * 戦略に応じて、均等 / ハード / セミハード の選択方法を使い分けます。
   *
   * @param allNegatives 全ての negative 候補ベクトル
   * @param maxNegatives 選択する最大数
   * @param seed 乱数シード（再現性のため）
   * @param strategy サンプリング戦略
   * @param anchor 現在の anchor ベクトル（hard/semi-hard で使用）
   * @param positive 現在の positive ベクトル（semi-hard で使用）
   */
  private sampleNegatives(
    allNegatives: (number[] | Float32Array)[],
    maxNegatives: number,
    seed: number,
    strategy: NegativeStrategy = "uniform",
    anchor?: number[] | Float32Array,
    positive?: number[] | Float32Array,
  ): (number[] | Float32Array)[] {
    if (allNegatives.length <= maxNegatives) {
      return allNegatives;
    }

    if (strategy === "uniform" || !anchor) {
      // ストライドベースの均等サンプリング（決定的・再現可能）
      const result: (number[] | Float32Array)[] = [];
      const stride = allNegatives.length / maxNegatives;
      for (let i = 0; i < maxNegatives; i++) {
        const idx = Math.floor((i * stride + seed) % allNegatives.length);
        result.push(allNegatives[idx]);
      }
      return result;
    }

    // anchor との類似度を計算してソート
    const scored = allNegatives.map((neg, idx) => ({
      neg,
      idx,
      similarity: cosineSimilarity(anchor, neg),
    }));

    if (strategy === "hard") {
      // Hard Negative Mining: anchor に最も近い（最も紛らわしい）negative を選択
      scored.sort((a, b) => b.similarity - a.similarity);
      return scored.slice(0, maxNegatives).map((s) => s.neg);
    }

    // Semi-Hard Negative Mining:
    // positive との類似度より低いが、なるべく近い negative を選択
    // 条件: sim(anchor, neg) < sim(anchor, pos) かつ sim が高い順
    const posSim = positive
      ? cosineSimilarity(anchor, positive)
      : Infinity;

    // positive より遠い negative を、近い順にソート
    const semiHardCandidates = scored
      .filter((s) => s.similarity < posSim)
      .sort((a, b) => b.similarity - a.similarity);

    if (semiHardCandidates.length >= maxNegatives) {
      return semiHardCandidates.slice(0, maxNegatives).map((s) => s.neg);
    }

    // Semi-hard だけで足りない場合、hard negative で補完
    const result = semiHardCandidates.map((s) => s.neg);
    const hardFallback = scored
      .filter((s) => s.similarity >= posSim)
      .sort((a, b) => b.similarity - a.similarity);
    for (const item of hardFallback) {
      if (result.length >= maxNegatives) break;
      result.push(item.neg);
    }

    return result.slice(0, maxNegatives);
  }

  /**
   * ベクトル群のセントロイド（平均ベクトル）を計算します。
   */
  private computeCentroid(vectors: (number[] | Float32Array)[]): Float32Array {
    const dim = this.dimension;
    const centroid = new Float32Array(dim);

    for (const vec of vectors) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += vec[i];
      }
    }

    const n = vectors.length;
    for (let i = 0; i < dim; i++) {
      centroid[i] /= n;
    }

    return centroid;
  }
}
