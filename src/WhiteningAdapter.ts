import { assertDimension, normalize, reject } from "./utils";

export interface WhiteningConfig {
  /**
   * 学習率 (0.0 ~ 1.0)
   * 平均と主成分を更新する際の指数移動平均の重み。
   * デフォルトは 0.01
   */
  learningRate?: number;
  
  /**
   * 除去するトップ主成分（PC）の数。
   * 通常、事前学習モデルのコーン（偏り）問題を取り除くには、
   * 上位 1〜2 個の主成分を除去する（All-but-the-Top）だけで十分な効果があります。
   * デフォルトは 1
   */
  numComponents?: number;
}

/**
 * WhiteningAdapter は、ストリーミングデータに対して
 * Oja's Rule によるオンラインPCAを実行し、
 * ベクトル空間の等方化（Whitening / Anisotropy Reduction）を行うアダプターです。
 */
export class WhiteningAdapter {
  public dim: number;
  public mean: Float32Array;
  public components: Float32Array[];
  
  private count: number = 0;
  private learningRate: number;
  private numComponents: number;

  /**
   * 新しい WhiteningAdapter を作成します。
   * @param dim ベクトルの次元数
   * @param config 設定オプション
   */
  constructor(dim: number, config: WhiteningConfig = {}) {
    this.dim = dim;
    this.learningRate = config.learningRate ?? 0.01;
    this.numComponents = config.numComponents ?? 1;

    this.mean = new Float32Array(dim);
    this.components = [];
    
    // 主成分ベクトルをランダムに初期化（正規化済み）
    for (let k = 0; k < this.numComponents; k++) {
      const pc = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        pc[i] = Math.random() * 2 - 1;
      }
      this.components.push(normalize(pc));
    }
  }

  /**
   * 入力ベクトルを用いて、平均と主成分をオンライン更新します。
   * (Oja's Rule + Generalized Hebbian Algorithm)
   * 
   * @param vector 学習用の入力ベクトル
   */
  public update(vector: number[] | Float32Array): void {
    assertDimension(vector, this.dim, "WhiteningAdapter.update");

    // 1. オンライン平均の更新 (指数移動平均)
    // 最初の入力のときはそのまま平均とする
    if (this.count === 0) {
      for (let i = 0; i < this.dim; i++) {
        this.mean[i] = vector[i];
      }
    } else {
      for (let i = 0; i < this.dim; i++) {
        this.mean[i] = (1 - this.learningRate) * this.mean[i] + this.learningRate * vector[i];
      }
    }
    this.count++;

    // 2. 平均を引いたゼロセンタリングされたベクトル x
    const x = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      x[i] = vector[i] - this.mean[i];
    }

    // 3. Sanger's Rule (Generalized Hebbian Algorithm) による複数の主成分の直交抽出
    // 各成分ごとに、それより上位の成分の寄与を取り除きながら Oja's rule を適用する
    const x_residual = new Float32Array(x); // コピー
    
    for (let k = 0; k < this.numComponents; k++) {
      const w = this.components[k];
      
      // y = w^T * x_residual (射影成分)
      let y = 0;
      for (let i = 0; i < this.dim; i++) {
        y += w[i] * x_residual[i];
      }

      // Oja's Rule: Δw = learningRate * y * (x_residual - y * w)
      for (let i = 0; i < this.dim; i++) {
        w[i] += this.learningRate * y * (x_residual[i] - y * w[i]);
      }

      // 安定性のために明示的に再正規化
      this.components[k] = normalize(w);

      // 下位の主成分計算のために、現在の主成分(w)の寄与分を x_residual から引く (Gram-Schmidt 直交化)
      // x_residual = x_residual - y * w
      for (let i = 0; i < this.dim; i++) {
        x_residual[i] -= y * this.components[k][i];
      }
    }
  }

  /**
   * 推論 (Whitening の適用):
   * 入力ベクトルから平均を引き、偏りの原因である上位主成分を除去します (All-but-the-Top)。
   * 
   * @param vector 推論・補正対象のベクトル
   * @returns 等方化・ゼロセンタリングされた新しいベクトル
   */
  public tune(vector: number[] | Float32Array): Float32Array {
    assertDimension(vector, this.dim, "WhiteningAdapter.tune");

    let result = new Float32Array(this.dim);

    // 1. 平均の減算
    for (let i = 0; i < this.dim; i++) {
      result[i] = vector[i] - this.mean[i];
    }

    // 2. 主成分の除去 (Orthogonal Rejection)
    for (let k = 0; k < this.numComponents; k++) {
      const w = this.components[k];
      // reject(result, w) だが、w は既に正規化済みなので計算を最適化できる
      // u・u = 1 なので、スカラー値は単なる内積 v・u
      let dot = 0;
      for (let i = 0; i < this.dim; i++) {
        dot += result[i] * w[i];
      }
      for (let i = 0; i < this.dim; i++) {
        result[i] -= dot * w[i];
      }
    }

    // 出力ベクトルは最終的に正規化して返すことが多いが、
    // ユースケースによっては正規化前が必要なこともあるため、ここではそのまま返す。
    // 必要なら呼び出し側で `normalize(result)` を行う。
    return result;
  }
}
