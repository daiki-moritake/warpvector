import {
  type WarpAdapter,
  assertDimension,
  normalize,
  innerProduct,
  addScaledVector,
  getWasmInstance,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  initWasm,
  withWasmMemoryStack,
  safeJsonParse,
  assertPositiveInt,
  assertNonNegativeInt,
  assertObject,
  assertNumberArray,
  assertArray,
} from "@warpvector/core";

export interface SoftWhiteningConfig {
  /**
   * 学習率 (0.0 ~ 1.0)
   * 平均と主成分、固有値を更新する際の指数移動平均の重み。
   * デフォルトは 0.01
   */
  learningRate?: number;

  /**
   * 追跡・フィルタリングするトップ主成分（PC）の数。
   * デフォルトは 5
   */
  numComponents?: number;

  /**
   * 逆熱拡散の巻き戻し時間（シャープネスの強さ）。
   * 0 の場合は元のベクトルのまま。
   * 大きくするほど、拡散した意味（分散の大きい主成分）が強く抑制され、特有の鋭い意味が強調されます。
   * デフォルトは 1.0
   */
  tau?: number;

  /**
   * 推論後のベクトルを L2 正規化するかどうか。
   * 逆拡散によってノルムが変化するため、Cosine Similarity を用いる場合は true が推奨されます。
   * デフォルトは true
   */
  normalizeOutput?: boolean;
}

/**
 * SoftWhiteningAdapter は、ストリーミングデータに対して
 * 固有空間での逆熱方程式（Inverse Heat Equation）に基づくスペクトルフィルタリングを実行し、
 * LLM特有の「意味の拡散（コンテキストの混ざり合い）」を解消して、
 * 真の意図（シャープな特徴）を抽出するアダプターです。
 */
export class SoftWhiteningAdapter implements WarpAdapter {
  public dim: number;
  public mean: Float32Array;
  public components: Float32Array[];
  public eigenvalues: Float32Array;

  private count: number = 0;
  private learningRate: number;
  private numComponents: number;
  public tau: number;
  public normalizeOutput: boolean;

  public async init(): Promise<void> {
    await initWasm();
  }

  /**
   * 新しい SoftWhiteningAdapter を作成します。
   * @param dim ベクトルの次元数
   * @param config 設定オプション
   */
  constructor(dim: number, config: SoftWhiteningConfig = {}) {
    this.dim = dim;
    this.learningRate = config.learningRate ?? 0.01;
    this.numComponents = config.numComponents ?? 5;
    this.tau = config.tau ?? 1.0;
    this.normalizeOutput = config.normalizeOutput ?? true;

    if (this.tau < 0) {
      throw new Error("SoftWhiteningAdapter: tau must be non-negative.");
    }
    if (this.numComponents <= 0) {
      throw new Error("SoftWhiteningAdapter: numComponents must be positive.");
    }

    this.mean = new Float32Array(dim);
    this.eigenvalues = new Float32Array(this.numComponents);
    this.components = [];

    // 主成分ベクトルをランダムに初期化（正規化済み）
    // 固有値は微小な値で初期化
    for (let k = 0; k < this.numComponents; k++) {
      const pc = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        pc[i] = Math.random() * 2 - 1;
      }
      this.components.push(normalize(pc));
      this.eigenvalues[k] = 1e-6; // 小さい初期値
    }
  }

  /**
   * 入力ベクトルを用いて、平均、主成分、固有値（分散）をオンライン更新します。
   *
   * @param vector 学習用の入力ベクトル
   */
  public update(vector: number[] | Float32Array): void {
    assertDimension(vector, this.dim, "SoftWhiteningAdapter.update");

    // 1. オンライン平均の更新 (指数移動平均)
    if (this.count === 0) {
      for (let i = 0; i < this.dim; i++) {
        this.mean[i] = vector[i];
      }
    } else {
      for (let i = 0; i < this.dim; i++) {
        this.mean[i] =
          (1 - this.learningRate) * this.mean[i] +
          this.learningRate * vector[i];
      }
    }
    this.count++;

    // 2. 平均を引いたゼロセンタリングされたベクトル x
    const x = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      x[i] = vector[i] - this.mean[i];
    }

    // --- JS側で固有値を更新 (EMA of projection squared) ---
    // x の現在の各成分への射影の二乗を固有値（分散）として指数移動平均でトラッキング
    for (let k = 0; k < this.numComponents; k++) {
      const projection = innerProduct(this.components[k], x);
      const variance = projection * projection;
      this.eigenvalues[k] =
        (1 - this.learningRate) * this.eigenvalues[k] +
        this.learningRate * variance;
    }

    // 3. Sanger's Rule (Generalized Hebbian Algorithm) による複数の主成分の直交抽出
    const x_residual = new Float32Array(x);

    const instance = getWasmInstance();

    if (instance && instance.exports.sangerUpdateWasm) {
      const memory = instance.exports.memory as WebAssembly.Memory;
      withWasmMemoryStack(() => {
        const componentsSize = this.numComponents * this.dim * 4;
        const xResidualSize = this.dim * 4;
        const componentsPtr = allocateWasmMemory(componentsSize);
        const xResidualPtr = allocateWasmMemory(xResidualSize);

        const f32 = new Float32Array(memory.buffer);
        for (let k = 0; k < this.numComponents; k++) {
          const comp = this.components[k];
          for (let i = 0; i < this.dim; i++) {
            f32[componentsPtr / 4 + k * this.dim + i] = comp[i];
          }
        }
        writeFloat32ArrayToWasm(memory, x_residual, xResidualPtr);

        const sangerUpdateWasm = instance.exports
          .sangerUpdateWasm as CallableFunction;
        sangerUpdateWasm(
          componentsPtr,
          xResidualPtr,
          this.dim,
          this.numComponents,
          this.learningRate,
        );

        for (let k = 0; k < this.numComponents; k++) {
          for (let i = 0; i < this.dim; i++) {
            this.components[k][i] = f32[componentsPtr / 4 + k * this.dim + i];
          }
        }
      });
    } else {
      // WASMが使えない場合のフォールバック
      for (let k = 0; k < this.numComponents; k++) {
        const w = this.components[k];
        let y = innerProduct(w, x_residual);

        addScaledVector(w, x_residual, this.learningRate * y);
        addScaledVector(w, w, -this.learningRate * y * y);

        this.components[k] = normalize(w);
        addScaledVector(x_residual, this.components[k], -y);
      }
    }
  }

  /**
   * 推論 (Inverse Diffusion / Sharpening):
   * 逆熱核 (Inverse Heat Kernel) に基づくスペクトルフィルタリングを用いて、
   * 拡散された（分散の大きい）成分を減衰させ、シャープな特徴を抽出します。
   *
   * @param vector 推論対象のベクトル
   * @returns シャープニングされた新しいベクトル
   */
  public tune(vector: number[] | Float32Array): Float32Array {
    assertDimension(vector, this.dim, "SoftWhiteningAdapter.tune");

    let result = new Float32Array(this.dim);

    // 1. 平均の減算 (Zero-centering)
    for (let i = 0; i < this.dim; i++) {
      result[i] = vector[i] - this.mean[i];
    }

    // 2. 逆拡散フィルタリング
    // x_sharp = x - \sum_{k} (1 - exp(-\tau \lambda_k)) (x \cdot w_k) w_k
    for (let k = 0; k < this.numComponents; k++) {
      const w = this.components[k];
      const lambda = this.eigenvalues[k];
      
      const projection = innerProduct(result, w);
      
      // 減衰係数: 固有値が大きい（より拡散している）成分ほど大きく引く
      const attenuation = 1.0 - Math.exp(-this.tau * lambda);
      
      addScaledVector(result, w, -attenuation * projection);
    }

    if (this.normalizeOutput) {
      // ゼロベクトルによるゼロ除算を防ぐため、安全に正規化
      let sumSq = 0;
      for (let i = 0; i < this.dim; i++) {
        sumSq += result[i] * result[i];
      }
      if (sumSq > 1e-12) {
        const norm = Math.sqrt(sumSq);
        for (let i = 0; i < this.dim; i++) {
          result[i] /= norm;
        }
      }
    }

    return result;
  }

  /**
   * 複数のベクトル（バッチ）に対して一括で推論を行います。
   * ループのオーバーヘッドを削減し、スループットを最大化します。
   *
   * @param vectors 変換対象のベクトル配列
   * @returns シャープニングされた新しいベクトルの配列
   */
  public tuneBatch(vectors: Float32Array[]): Float32Array[] {
    const batchSize = vectors.length;
    const results = new Array<Float32Array>(batchSize);

    // 事前に減衰係数を計算しておく
    const attenuations = new Float32Array(this.numComponents);
    for (let k = 0; k < this.numComponents; k++) {
      attenuations[k] = 1.0 - Math.exp(-this.tau * this.eigenvalues[k]);
    }

    for (let i = 0; i < batchSize; i++) {
      const vec = vectors[i];
      assertDimension(vec, this.dim, "SoftWhiteningAdapter.tuneBatch");

      const result = new Float32Array(this.dim);
      for (let j = 0; j < this.dim; j++) {
        result[j] = vec[j] - this.mean[j];
      }

      for (let k = 0; k < this.numComponents; k++) {
        const w = this.components[k];
        const projection = innerProduct(result, w);
        const attenuation = attenuations[k];
        
        // inline addScaledVector to avoid function call overhead
        const scale = -attenuation * projection;
        for (let j = 0; j < this.dim; j++) {
          result[j] += scale * w[j];
        }
      }

      if (this.normalizeOutput) {
        let sumSq = 0;
        for (let j = 0; j < this.dim; j++) {
          sumSq += result[j] * result[j];
        }
        if (sumSq > 1e-12) {
          const norm = Math.sqrt(sumSq);
          for (let j = 0; j < this.dim; j++) {
            result[j] /= norm;
          }
        }
      }

      results[i] = result;
    }

    return results;
  }

  /**
   * 現在の学習状態をシリアライズして出力します。
   */
  public exportState(): string {
    return JSON.stringify({
      dim: this.dim,
      count: this.count,
      learningRate: this.learningRate,
      numComponents: this.numComponents,
      tau: this.tau,
      normalizeOutput: this.normalizeOutput,
      mean: Array.from(this.mean),
      eigenvalues: Array.from(this.eigenvalues),
      components: this.components.map((c) => Array.from(c)),
    });
  }

  /**
   * シリアライズされた学習状態から SoftWhiteningAdapter を復元します。
   */
  public static importState(stateJson: string): SoftWhiteningAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "SoftWhiteningAdapter"),
      "root",
    );
    const dim = assertPositiveInt(data.dim, "dim");
    const learningRate =
      typeof data.learningRate === "number" ? data.learningRate : 0.01;
    const numComponents = assertPositiveInt(
      data.numComponents,
      "numComponents",
    );
    const tau = typeof data.tau === "number" ? data.tau : 1.0;
    const normalizeOutput = typeof data.normalizeOutput === "boolean" ? data.normalizeOutput : true;
    const count = assertNonNegativeInt(data.count, "count");
    const mean = assertNumberArray(data.mean, "mean");
    const eigenvalues = assertNumberArray(data.eigenvalues, "eigenvalues");
    const components = assertArray(data.components, "components");

    const adapter = new SoftWhiteningAdapter(dim, {
      learningRate,
      numComponents,
      tau,
      normalizeOutput,
    });
    adapter.count = count;
    adapter.mean = new Float32Array(mean);
    adapter.eigenvalues = new Float32Array(eigenvalues);
    adapter.components = components.map(
      (c: unknown, i: number) =>
        new Float32Array(assertNumberArray(c, `components[${i}]`)),
    );
    return adapter;
  }
}
