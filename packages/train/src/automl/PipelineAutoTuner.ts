import { WarpPipeline, InputVector, OutputVector } from "@warpvector/core";
import { SearchExample, getPositiveRank } from "./metrics";

export type MetricType = "MRR" | "Recall@1" | "Recall@5" | "Recall@10";

export interface TuneConfig<TParams extends Record<string, any>> {
  /** 
   * ハイパーパラメータの探索空間（Grid） 
   * 例: { tau: [0.5, 1.0], numComponents: [1, 5] }
   */
  searchSpace: { [K in keyof TParams]: TParams[K][] };
  
  /** 
   * 与えられたパラメータから WarpPipeline を組み立てるファクトリ関数 
   */
  pipelineBuilder: (params: TParams) => WarpPipeline;
  
  /** 
   * 評価指標（デフォルトは MRR） 
   */
  metric?: MetricType;
  
  /**
   * 進捗を報告するコールバック関数
   */
  onProgress?: (current: number, total: number, bestScore: number) => void;
}

export interface TuneResult<TParams> {
  bestParams: TParams;
  bestScore: number;
  bestPipeline: WarpPipeline;
  allResults: { params: TParams; score: number }[];
}

/**
 * 検証用データセットを用いて、最も検索精度（MRR等）が高くなる WarpPipeline の
 * 構成（ハイパーパラメータ）を自動探索（AutoML）するユーティリティ。
 */
export class PipelineAutoTuner {
  private dataset: SearchExample<InputVector>[];

  constructor(dataset: SearchExample<InputVector>[]) {
    this.dataset = dataset;
    if (dataset.length === 0) {
      throw new Error("Dataset for AutoTuner cannot be empty.");
    }
  }

  private async evaluatePipeline(pipeline: WarpPipeline, metric: MetricType): Promise<number> {
    if (this.dataset.length === 0) return 0;

    let scoreSum = 0;

    // データセット全体を一度に変換してメモリ配列に保持するのではなく、
    // 1件ずつ推論してスコアを加算することで O(1) のメモリオーバーヘッドを実現します。
    for (const ex of this.dataset) {
      const transformedQuery = await pipeline.run(ex.query);
      const transformedPositive = await pipeline.run(ex.positive);
      const transformedNegatives = ex.negatives
        ? await Promise.all(ex.negatives.map(neg => pipeline.run(neg)))
        : [];

      const rank = getPositiveRank(transformedQuery, transformedPositive, transformedNegatives);

      switch (metric) {
        case "MRR":
          scoreSum += 1.0 / rank;
          break;
        case "Recall@1":
          if (rank <= 1) scoreSum += 1;
          break;
        case "Recall@5":
          if (rank <= 5) scoreSum += 1;
          break;
        case "Recall@10":
          if (rank <= 10) scoreSum += 1;
          break;
        default:
          throw new Error(`Unknown metric: ${metric}`);
      }
    }

    return scoreSum / this.dataset.length;
  }

  /**
   * 探索空間の全ての組み合わせ（直積）を生成するヘルパーメソッド
   */
  private generateGrid<TParams extends Record<string, any>>(
    space: { [K in keyof TParams]: TParams[K][] }
  ): TParams[] {
    const keys = Object.keys(space) as (keyof TParams)[];
    const combinations: TParams[] = [];

    const helper = (index: number, current: Partial<TParams>) => {
      if (index === keys.length) {
        combinations.push({ ...current } as TParams);
        return;
      }
      const key = keys[index];
      const values = space[key];
      for (const val of values) {
        current[key] = val;
        helper(index + 1, current);
      }
    };

    helper(0, {});
    return combinations;
  }

  /**
   * グリッドサーチを用いて最適なパイプライン構成を探索します。
   */
  public async tuneGrid<TParams extends Record<string, any>>(
    config: TuneConfig<TParams>
  ): Promise<TuneResult<TParams>> {
    const metric = config.metric || "MRR";
    const grid = this.generateGrid(config.searchSpace);
    
    if (grid.length === 0) {
      throw new Error("Search space generated 0 combinations.");
    }

    let bestScore = -Infinity;
    let bestParams: TParams = grid[0];
    let bestPipeline: WarpPipeline | null = null;
    const allResults: { params: TParams; score: number }[] = [];

    for (let i = 0; i < grid.length; i++) {
      const params = grid[i];
      const pipeline = config.pipelineBuilder(params);
      
      // パイプラインの初期化（WASMロードなどが必要な場合）
      await pipeline.init();

      const score = await this.evaluatePipeline(pipeline, metric);
      allResults.push({ params, score });

      if (score > bestScore) {
        bestScore = score;
        bestParams = params;
        bestPipeline = pipeline;
      }

      if (config.onProgress) {
        config.onProgress(i + 1, grid.length, bestScore);
      }
    }

    if (!bestPipeline) {
      throw new Error("Failed to build or evaluate any pipeline.");
    }

    return {
      bestParams,
      bestScore,
      bestPipeline,
      allResults
    };
  }
}
