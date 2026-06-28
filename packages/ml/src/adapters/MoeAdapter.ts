import {
  cosineSimilarity,
  type WarpAdapter,
  type InputVector,
  type TransformOutput,
  type AdapterState,
  AdapterRegistry,
} from "@warpvector/core";

export interface ExpertDefinition {
  /** 一意のエキスパートID */
  id: string;
  /** エキスパートとなるWarpAdapterインスタンス */
  adapter: WarpAdapter;
  /**
   * コサイン類似度ルーターを使用する場合の、このエキスパートのドメインを代表する重心ベクトル。
   */
  centroid?: number[] | Float32Array;
}

export type RoutingStrategy = "cosine" | "custom";

export interface MoeAdapterConfig {
  experts: ExpertDefinition[];
  /** ルーティング戦略。デフォルトは "cosine" */
  routingStrategy?: RoutingStrategy;
  /** カスタムルーター関数。入力ベクトルを受け取り、エキスパートIDを返す */
  customRouter?: (vector: Float32Array) => string;
}

/**
 * Mixture of Experts (MoE) アダプター。
 * 入力ベクトルに応じて、動的に適切なサブアダプタ（エキスパート）にルーティングします。
 * Gating Network として、各エキスパートのセントロイド（重心）とのコサイン類似度を用いたハードルーティングを行います。
 */
export class MoeAdapter implements WarpAdapter {
  private experts: Map<string, WarpAdapter> = new Map();
  private centroids: Map<string, Float32Array> = new Map();
  private config: MoeAdapterConfig;

  constructor(config: MoeAdapterConfig) {
    this.config = config;
    for (const expert of config.experts) {
      this.experts.set(expert.id, expert.adapter);
      if (expert.centroid) {
        this.centroids.set(expert.id, new Float32Array(expert.centroid));
      }
    }

    if (this.experts.size === 0) {
      throw new Error("MoeAdapter requires at least one expert.");
    }
  }

  /**
   * 内包する全エキスパートの初期化を非同期に実行します。
   */
  async init(): Promise<void> {
    for (const expert of this.experts.values()) {
      if (expert.init) {
        await expert.init();
      }
    }
  }

  /**
   * 入力ベクトルを評価し、担当すべきエキスパートIDを返します。
   */
  private route(vectorArray: Float32Array): string {
    if (this.config.routingStrategy === "custom" && this.config.customRouter) {
      return this.config.customRouter(vectorArray);
    }

    // Default to cosine similarity routing (Hard routing)
    let bestExpert = "";
    let bestScore = -Infinity;

    for (const [id, centroid] of this.centroids.entries()) {
      const score = cosineSimilarity(vectorArray, centroid);
      if (score > bestScore) {
        bestScore = score;
        bestExpert = id;
      }
    }

    // セントロイドが定義されていない、またはスコアが計算できない場合は最初のエキスパートへフォールバック
    if (!bestExpert) {
      return this.config.experts[0].id;
    }

    return bestExpert;
  }

  tune(vector: InputVector, context?: string): TransformOutput {
    const vectorArray =
      vector instanceof Float32Array ? vector : new Float32Array(vector);

    // エキスパートの選択 (Gating)
    const expertId = this.route(vectorArray);
    const expert = this.experts.get(expertId);

    if (!expert) {
      throw new Error(`Expert ${expertId} not found`);
    }

    // 選択されたエキスパートでのみ推論を実行 (疎な処理)
    return expert.tune(vectorArray, context);
  }

  tuneBatch(vectors: InputVector[], context?: string): TransformOutput[] {
    return vectors.map((v) => this.tune(v, context));
  }

  exportState(): AdapterState {
    const state: any = {
      type: "MoeAdapter",
      routingStrategy: this.config.routingStrategy,
      experts: [],
    };
    for (const expert of this.config.experts) {
      state.experts.push({
        id: expert.id,
        centroid: expert.centroid ? Array.from(expert.centroid) : undefined,
        adapterType: expert.adapter.constructor.name, // Will be needed for importState
        adapterState: expert.adapter.exportState
          ? expert.adapter.exportState()
          : undefined,
      });
    }
    return JSON.stringify(state);
  }

  /**
   * JSON文字列またはオブジェクトから MoeAdapter を復元します。
   * 内部のエキスパートは AdapterRegistry を通じて動的に復元されます。
   */
  static importState(state: AdapterState): MoeAdapter {
    const parsed = typeof state === "string" ? JSON.parse(state) : state;
    if (parsed.type !== "MoeAdapter") {
      throw new Error(`Invalid state type for MoeAdapter: ${parsed.type}`);
    }


    const experts: ExpertDefinition[] = parsed.experts.map((exp: any) => {
      const importFn = AdapterRegistry.get(exp.adapterType);
      if (!importFn) {
        throw new Error(
          `Cannot import MoeAdapter: unknown adapter type '${exp.adapterType}' for expert '${exp.id}'. Did you register it?`,
        );
      }
      return {
        id: exp.id,
        centroid: exp.centroid,
        adapter: importFn(exp.adapterState),
      };
    });

    return new MoeAdapter({
      experts,
      routingStrategy: parsed.routingStrategy,
    });
  }
}
