import type { IntentWeights } from "@warpvector/core";

/**
 * クライアントから送信される重み更新情報。
 */
export interface ClientUpdate {
  /** クライアントのローカル学習済み重み */
  weights: IntentWeights;
  /** そのクライアントの学習回数（信頼度として使用） */
  interactionCount: number;
}

/**
 * 複数ユーザー/クライアントの学習済み重みを FedAvg (Federated Averaging) で
 * 集約するユーティリティクラス。
 *
 * 各クライアントはローカルで学習した重みとインタラクション数を送信し、
 * サーバー（またはコーディネーター）がこのクラスで集約して
 * 新しいグローバルベースラインを生成します。
 *
 * 集約アルゴリズム:
 * ```
 * W_new = W_base + Σ (count_i / total_count) * (W_i - W_base)
 * ```
 *
 * @example
 * ```typescript
 * const aggregator = new FederatedAggregator(baseWeights, 1536);
 *
 * // クライアントA: 100回学習済み
 * aggregator.submitUpdate({ weights: clientAWeights, interactionCount: 100 });
 *
 * // クライアントB: 50回学習済み
 * aggregator.submitUpdate({ weights: clientBWeights, interactionCount: 50 });
 *
 * // FedAvg で集約 → 新しいベースライン
 * const newBase = aggregator.aggregate();
 *
 * // 次のラウンドの準備
 * aggregator.reset(newBase);
 * ```
 */
export class FederatedAggregator {
  private baseWeights: IntentWeights;
  private readonly dimension: number;
  private updates: ClientUpdate[] = [];

  /**
   * FederatedAggregator を作成します。
   *
   * @param baseWeights 現在のグローバルベースライン重み
   * @param dimension ベクトルの次元数
   */
  constructor(baseWeights: IntentWeights, dimension: number) {
    this.baseWeights = baseWeights;
    this.dimension = dimension;
  }

  /**
   * クライアントの重み更新を登録します。
   *
   * @param update クライアントの重みとインタラクション数
   * @throws {Error} interactionCount が 0 以下の場合
   */
  public submitUpdate(update: ClientUpdate): void {
    if (update.interactionCount <= 0) {
      throw new Error("interactionCount must be greater than 0.");
    }
    this.updates.push(update);
  }

  /**
   * 登録された全クライアントの重みを FedAvg で集約し、新しいベースラインを返します。
   *
   * 各クライアントの寄与は interactionCount に比例します。
   * より多く学習したクライアントほど、集約結果への影響が大きくなります。
   *
   * @returns 集約された新しい IntentWeights
   * @throws {Error} 登録されたクライアントが0件の場合
   */
  public aggregate(): IntentWeights {
    if (this.updates.length === 0) {
      throw new Error("No client updates submitted. Call submitUpdate first.");
    }

    const dim = this.dimension;
    const matrixSize = dim * dim;

    // ベースの matrix と bias を取得
    const baseMatrix = this.toFloat32(this.baseWeights.matrix, matrixSize);
    const baseBias = this.toFloat32(this.baseWeights.bias, dim);

    // 結果用バッファ（ベースからスタート）
    const resultMatrix = new Float32Array(baseMatrix);
    const resultBias = new Float32Array(baseBias);

    // 総インタラクション数
    const totalCount = this.updates.reduce(
      (sum, u) => sum + u.interactionCount,
      0,
    );

    // 各クライアントの差分を重み付き加算
    for (const update of this.updates) {
      const scale = update.interactionCount / totalCount;
      const clientMatrix = this.toFloat32(update.weights.matrix, matrixSize);
      const clientBias = this.toFloat32(update.weights.bias, dim);

      // resultMatrix += scale * (clientMatrix - baseMatrix)
      for (let i = 0; i < matrixSize; i++) {
        resultMatrix[i] += scale * (clientMatrix[i] - baseMatrix[i]);
      }

      // resultBias += scale * (clientBias - baseBias)
      for (let i = 0; i < dim; i++) {
        resultBias[i] += scale * (clientBias[i] - baseBias[i]);
      }
    }

    return {
      ...this.baseWeights,
      matrix: resultMatrix,
      bias: resultBias,
    };
  }

  /**
   * 登録をクリアし、次の集約ラウンドの準備をします。
   *
   * @param newBaseWeights 新しいベースライン重み（省略時は現在のベースを維持）
   */
  public reset(newBaseWeights?: IntentWeights): void {
    if (newBaseWeights) {
      this.baseWeights = newBaseWeights;
    }
    this.updates = [];
  }

  /**
   * 登録済みクライアント数。
   */
  public get clientCount(): number {
    return this.updates.length;
  }

  /**
   * number[] | number[][] | Float32Array を Float32Array に正規化するヘルパー。
   */
  private toFloat32(
    arr: number[] | number[][] | Float32Array,
    expectedLength: number,
  ): Float32Array {
    if (arr instanceof Float32Array) {
      return arr;
    }
    const result = new Float32Array(expectedLength);
    // 2D array (number[][]) → flatten
    if (Array.isArray(arr) && arr.length > 0 && Array.isArray(arr[0])) {
      let idx = 0;
      for (const row of arr as number[][]) {
        for (const val of row) {
          if (idx < expectedLength) result[idx++] = val;
        }
      }
      return result;
    }
    // 1D array (number[])
    const flat = arr as number[];
    for (let i = 0; i < Math.min(flat.length, expectedLength); i++) {
      result[i] = flat[i];
    }
    return result;
  }
}
