import {
  assertDimension,
  normalize,
  slerp,
} from "@warpvector/core";

/**
 * VectorAugmentor の設定オプション。
 */
export interface AugmentOptions {
  /**
   * 使用する拡張戦略の配列。
   * @default ["noise"]
   */
  strategy?: AugmentStrategy[];

  /**
   * 各サンプルから生成する拡張ベクトルの数。
   * @default 3
   */
  multiplier?: number;

  /**
   * ガウシアンノイズの標準偏差。
   * 小さすぎると拡張効果がなく、大きすぎると意味が変質する。
   * @default 0.01
   */
  noiseSigma?: number;

  /**
   * ドロップアウト率（0.0 〜 1.0）。
   * 各次元がゼロ化される確率。
   * @default 0.1
   */
  dropoutRate?: number;

  /**
   * Slerp 補間で使用する t 値の範囲 [min, max]。
   * 各補間で範囲内の値がランダムに選ばれます。
   * @default [0.2, 0.8]
   */
  slerpRange?: [number, number];

  /**
   * 拡張後にL2正規化を適用するかどうか。
   * コサイン類似度ベースの検索では true が推奨されます。
   * @default true
   */
  normalizeOutput?: boolean;
}

export type AugmentStrategy = "noise" | "dropout" | "slerp-interpolation";

/**
 * メルセンヌ・ツイスタ (MT19937) のシード初期化などに使われる定番の乗数。
 * ビットミキシングによる乱数状態の初期化に使用します。
 */
const PRNG_MULTIPLIER = 1812433253;

/** 32ビット符号なし整数の最大値 + 1 (2^32)。一様乱数を [0, 1) の範囲に正規化するために使用します。 */
const UINT32_MAX_PLUS_1 = 4294967296;

/** 決定的な擬似乱数 (Xorshift) のシフトパラメータ */
const XORSHIFT_SHIFT_1 = 23;
const XORSHIFT_SHIFT_2 = 17;
const XORSHIFT_SHIFT_3 = 26;

/** 対数関数のゼロ除算 (-Infinity) を防ぐための微小値 */
const EPSILON = 1e-10;

/** バッチ処理時のシード生成用ハッシュ関数に使用する素数 */
const SEED_HASH_PRIME_1 = 31;
const SEED_HASH_PRIME_2 = 7;
const SEED_HASH_PRIME_3 = 13;

/**
 * 決定的な擬似乱数生成器 (xorshift128+)。
 * 再現可能な拡張結果を保証します。
 */
class SeededRng {
  private s0: number;
  private s1: number;

  constructor(seed: number) {
    this.s0 = seed | 0 || 1;
    this.s1 = (Math.imul(seed, PRNG_MULTIPLIER) + 1) | 0 || 2;
  }

  /** 0.0 〜 1.0 の一様乱数 */
  next(): number {
    let s1 = this.s0;
    const s0 = this.s1;
    this.s0 = s0;
    s1 ^= s1 << XORSHIFT_SHIFT_1;
    s1 ^= s1 >>> XORSHIFT_SHIFT_2;
    s1 ^= s0;
    s1 ^= s0 >>> XORSHIFT_SHIFT_3;
    this.s1 = s1;
    return ((this.s0 + this.s1) >>> 0) / UINT32_MAX_PLUS_1;
  }

  /** Box-Muller 変換による正規分布乱数 */
  nextGaussian(): number {
    const u1 = this.next() || EPSILON; // log(0) 防止
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

/**
 * ベクトル空間におけるデータ拡張（Data Augmentation）を行うユーティリティクラス。
 *
 * 少量のサンプルベクトルから、意味的に近い変異ベクトルを生成することで、
 * 学習データの不足を補い、IntentMatrixFactory や各種 Trainer の学習効率を向上させます。
 *
 * @example
 * ```typescript
 * import { VectorAugmentor } from "warpvector/train";
 *
 * const augmentor = new VectorAugmentor(1536);
 *
 * // 3サンプルから15の拡張ベクトルを生成
 * const augmented = augmentor.augmentBatch(originalVectors, {
 *   strategy: ["noise", "slerp-interpolation"],
 *   multiplier: 5,
 * });
 * ```
 */
export class VectorAugmentor {
  private readonly dimension: number;

  /**
   * @param dimension ベクトルの次元数
   */
  constructor(dimension: number) {
    if (dimension <= 0 || !Number.isInteger(dimension)) {
      throw new Error(
        `Dimension must be a positive integer, got ${dimension}.`,
      );
    }
    this.dimension = dimension;
  }

  /**
   * 単一のベクトルにガウシアンノイズを注入します。
   *
   * @param vector 元のベクトル
   * @param options オプション
   * @param seed 乱数シード（再現性のため）
   * @returns ノイズが追加された新しいベクトル
   */
  public addNoise(
    vector: number[] | Float32Array,
    options: { sigma?: number; normalizeOutput?: boolean } = {},
    seed: number = 42,
  ): Float32Array {
    assertDimension(vector, this.dimension, "VectorAugmentor.addNoise input");
    const sigma = options.sigma ?? 0.01;
    const rng = new SeededRng(seed);
    const result = new Float32Array(this.dimension);

    for (let i = 0; i < this.dimension; i++) {
      result[i] = vector[i] + sigma * rng.nextGaussian();
    }

    if (options.normalizeOutput ?? true) {
      return normalize(result);
    }
    return result;
  }

  /**
   * ベクトルにドロップアウト（ランダム次元のゼロ化 + 再正規化）を適用します。
   *
   * @param vector 元のベクトル
   * @param options オプション
   * @param seed 乱数シード
   * @returns ドロップアウトが適用された新しいベクトル
   */
  public dropout(
    vector: number[] | Float32Array,
    options: { rate?: number; normalizeOutput?: boolean } = {},
    seed: number = 42,
  ): Float32Array {
    assertDimension(vector, this.dimension, "VectorAugmentor.dropout input");
    const rate = options.rate ?? 0.1;
    const rng = new SeededRng(seed);
    const result = new Float32Array(this.dimension);

    // ドロップアウトのスケーリングファクター (inverted dropout)
    const scale = 1.0 / (1.0 - rate);

    for (let i = 0; i < this.dimension; i++) {
      result[i] = rng.next() < rate ? 0 : vector[i] * scale;
    }

    if (options.normalizeOutput ?? true) {
      return normalize(result);
    }
    return result;
  }

  /**
   * 2つのベクトルを球面線形補間 (Slerp) で補間します。
   *
   * @param v1 始点ベクトル
   * @param v2 終点ベクトル
   * @param options オプション
   * @returns 補間されたベクトル
   */
  public interpolate(
    v1: number[] | Float32Array,
    v2: number[] | Float32Array,
    options: { t?: number; normalizeOutput?: boolean } = {},
  ): Float32Array {
    assertDimension(v1, this.dimension, "VectorAugmentor.interpolate v1");
    assertDimension(v2, this.dimension, "VectorAugmentor.interpolate v2");
    const t = options.t ?? 0.5;
    const result = slerp(v1, v2, t);

    if (options.normalizeOutput ?? true) {
      return normalize(result);
    }
    return result;
  }

  /**
   * 複数のサンプルベクトルから、データ拡張された新しいベクトル群を一括生成します。
   * 元のベクトルも結果に含まれます。
   *
   * @param vectors 元のサンプルベクトルの配列
   * @param options 拡張オプション
   * @returns 元のベクトル + 拡張ベクトルの配列
   */
  public augmentBatch(
    vectors: (number[] | Float32Array)[],
    options: AugmentOptions = {},
  ): (number[] | Float32Array)[] {
    if (vectors.length === 0) {
      return [];
    }

    for (let i = 0; i < vectors.length; i++) {
      assertDimension(
        vectors[i],
        this.dimension,
        `VectorAugmentor.augmentBatch vector[${i}]`,
      );
    }

    const strategies = options.strategy ?? ["noise"];
    const multiplier = options.multiplier ?? 3;
    const sigma = options.noiseSigma ?? 0.01;
    const dropoutRate = options.dropoutRate ?? 0.1;
    const slerpRange = options.slerpRange ?? [0.2, 0.8];
    const shouldNormalize = options.normalizeOutput ?? true;

    // 元のベクトルを結果に含める
    const result: (number[] | Float32Array)[] = [...vectors];
    let seedCounter = 0;

    for (let m = 0; m < multiplier; m++) {
      for (let i = 0; i < vectors.length; i++) {
        // 使用する戦略をラウンドロビンで選択
        const strategyIdx = (m + i) % strategies.length;
        const strategy = strategies[strategyIdx];
        const seed = ++seedCounter * SEED_HASH_PRIME_1 + i * SEED_HASH_PRIME_2 + m * SEED_HASH_PRIME_3;

        switch (strategy) {
          case "noise": {
            result.push(
              this.addNoise(
                vectors[i],
                { sigma, normalizeOutput: shouldNormalize },
                seed,
              ),
            );
            break;
          }
          case "dropout": {
            result.push(
              this.dropout(
                vectors[i],
                { rate: dropoutRate, normalizeOutput: shouldNormalize },
                seed,
              ),
            );
            break;
          }
          case "slerp-interpolation": {
            // 他のベクトルとの補間
            const rng = new SeededRng(seed);
            const otherIdx = Math.floor(rng.next() * vectors.length);
            const other =
              otherIdx === i
                ? vectors[(i + 1) % vectors.length]
                : vectors[otherIdx];
            const t =
              slerpRange[0] + rng.next() * (slerpRange[1] - slerpRange[0]);
            result.push(
              this.interpolate(vectors[i], other, {
                t,
                normalizeOutput: shouldNormalize,
              }),
            );
            break;
          }
          default:
            throw new Error(`Unknown augmentation strategy: ${strategy}`);
        }
      }
    }

    return result;
  }
}
