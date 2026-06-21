import { innerProduct, normalize } from "@warpvector/core";

export interface MultipathScatteringConfig {
  /**
   * 波の多重散乱の減衰率 (0.0 ~ 1.0)。
   * 値が大きいほど、遠くのドキュメント（多くのホップを経た経路）からの影響を強く受けます。
   * (PageRank や Random Walk with Restart における 1 - テレポート確率 に相当します)
   * デフォルトは 0.85
   */
  alpha?: number;

  /**
   * 候補ドキュメント間の類似度グラフを構築する際の、エッジの足切りしきい値。
   * この値未満の類似度は0とみなされ、グラフが疎（Sparse）になります。
   * デフォルトは 0.0 (すべての正の類似度を考慮)
   */
  threshold?: number;

  /**
   * 定常場を求めるための反復計算（多重経路の追跡）の最大回数。
   * デフォルトは 20
   */
  maxIterations?: number;

  /**
   * 収束判定のための許容誤差。
   * デフォルトは 1e-6
   */
  tolerance?: number;
}

export interface MultipathScatteringResult {
  /** 元の候補配列におけるインデックス */
  originalIndex: number;
  /** 多重散乱場理論（Random Walk with Restart）によって集約された新しいスコア */
  score: number;
  /** 初期のコサイン類似度スコア */
  initialScore: number;
  /** 候補ベクトル */
  vector: Float32Array;
}

/**
 * MultipathScatteringReranker
 * 
 * 物理学における「多重経路散乱場理論」のアイデアにインスパイアされたリランカー。
 * ベクトル空間上の候補ドキュメント群を散乱のネットワーク（グラフ）とみなし、
 * 初期スコア（波源からの入射波）が多重反射を繰り返して定常状態に至る過程を、
 * Random Walk with Restart (PPR) を用いてシミュレーションします。
 * これにより、孤立したノイズドキュメントのスコアは減衰し、
 * 多数の類似ドキュメントから「多重経路で支持されている」真の意図（波源）のスコアが際立ちます。
 */
export class MultipathScatteringReranker {
  public alpha: number;
  public threshold: number;
  public maxIterations: number;
  public tolerance: number;

  constructor(config: MultipathScatteringConfig = {}) {
    this.alpha = config.alpha ?? 0.85;
    this.threshold = config.threshold ?? 0.0;
    this.maxIterations = config.maxIterations ?? 20;
    this.tolerance = config.tolerance ?? 1e-6;

    if (this.alpha < 0 || this.alpha >= 1) {
      throw new Error("MultipathScatteringReranker: alpha must be in [0, 1).");
    }
    if (this.maxIterations < 1) {
      throw new Error("MultipathScatteringReranker: maxIterations must be at least 1.");
    }
  }

  /**
   * 候補ベクトル群に対して多重経路散乱波によるリランキングを行います。
   * 
   * @param query クエリベクトル（initialScoresがある場合はnull可）
   * @param candidates 検索システム等から返された候補ベクトル群
   * @param initialScores （任意）計算済みの初期コサイン類似度スコア
   * @returns スコア降順でソートされたMultipathScatteringResultの配列
   */
  public rerank(
    query: Float32Array | number[] | null,
    candidates: (Float32Array | number[])[],
    initialScores?: number[]
  ): MultipathScatteringResult[] {
    const N = candidates.length;
    if (N === 0) return [];

    if (!query && (!initialScores || initialScores.length !== N)) {
      throw new Error("MultipathScatteringReranker: Must provide either 'query' or a valid 'initialScores' array.");
    }

    // L2正規化
    const CNorm = candidates.map(c => normalize(new Float32Array(c)));

    // 1. 初期の観測場 S0 (Initial Field)
    let S0 = new Float32Array(N);
    if (initialScores && initialScores.length === N) {
      S0.set(initialScores);
    } else if (query) {
      const qNorm = normalize(new Float32Array(query));
      for (let i = 0; i < N; i++) {
        S0[i] = Math.max(0, innerProduct(qNorm, CNorm[i])); // 非負に制限
      }
    }

    if (N === 1) {
      return [{ originalIndex: 0, score: S0[0], initialScore: S0[0], vector: CNorm[0] }];
    }

    // S0 を確率分布として正規化（総和を1にする）
    let sumS0 = 0;
    for (let i = 0; i < N; i++) sumS0 += S0[i];
    if (sumS0 > 0) {
      for (let i = 0; i < N; i++) S0[i] /= sumS0;
    } else {
      // S0が全て0の場合は均等分布
      const uniform = 1.0 / N;
      for (let i = 0; i < N; i++) S0[i] = uniform;
    }

    // 2. 散乱推移マトリクスの構築 P
    // W[i][j] = max(0, cos_sim(c_i, c_j) - threshold)
    const W = new Float32Array(N * N);
    const D = new Float32Array(N);
    const dim = CNorm[0].length;
    const threshold = this.threshold;

    for (let i = 0; i < N; i++) {
      const ci = CNorm[i];
      for (let j = i + 1; j < N; j++) {
        const cj = CNorm[j];
        let sim = 0;
        for (let k = 0; k < dim; k++) {
          sim += ci[k] * cj[k];
        }

        const w = Math.max(0, sim - threshold);
        if (w > 0) {
          W[i * N + j] = w;
          W[j * N + i] = w; // 対称
          D[i] += w;
          D[j] += w;
        }
      }
    }

    // P[i][j] = jからiへの遷移確率 とする。
    // RWRの式は S_{t+1} = alpha * P * S_t + (1 - alpha) * S_0 なので、
    // グラフのランダムウォークとして、P_{ij} = W_{ij} / D_j （列和が1）とする。
    const P = new Float32Array(N * N);
    for (let j = 0; j < N; j++) {
      if (D[j] > 0) {
        for (let i = 0; i < N; i++) {
          P[i * N + j] = W[i * N + j] / D[j];
        }
      } else {
        // 孤立ノードからの遷移は自分自身に戻るとする（吸収状態）
        P[j * N + j] = 1.0;
      }
    }

    // 3. 多重経路散乱場 (Random Walk with Restart) の計算
    // S_{t+1} = alpha * P * S_t + (1 - alpha) * S_0
    let currentS = new Float32Array(S0);
    const nextS = new Float32Array(N);
    const alpha = this.alpha;
    const oneMinusAlpha = 1.0 - alpha;

    for (let iter = 0; iter < this.maxIterations; iter++) {
      let maxDiff = 0;
      
      for (let i = 0; i < N; i++) {
        let pSum = 0;
        const rowOffset = i * N;
        for (let j = 0; j < N; j++) {
          pSum += P[rowOffset + j] * currentS[j];
        }
        nextS[i] = alpha * pSum + oneMinusAlpha * S0[i];
        
        const diff = Math.abs(nextS[i] - currentS[i]);
        if (diff > maxDiff) maxDiff = diff;
      }
      
      currentS.set(nextS);
      if (maxDiff < this.tolerance) {
        break; // 収束
      }
    }

    // 4. 結果のフォーマットとソート
    // 定常確率分布になっているため、スケールを戻すために元のS0の最大値などに合わせることもできるが、
    // 順位（相対スコア）として使うためそのまま出力する。
    const results: MultipathScatteringResult[] = [];
    for (let i = 0; i < N; i++) {
      results.push({
        originalIndex: i,
        score: currentS[i],
        initialScore: initialScores && initialScores.length === N ? initialScores[i] : S0[i] * sumS0, // 元のスケールのスコアがあれば戻す
        vector: CNorm[i]
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }
}
