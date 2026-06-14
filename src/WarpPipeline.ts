import { WarpAdapter } from "./WarpAdapter";
import { IntentAdapter, IntentWeights } from "./IntentAdapter";
import { LoraIntentAdapter, LoraIntentWeights } from "./LoraIntentAdapter";
import { WhiteningAdapter } from "./WhiteningAdapter";
import { ProjectionAdapter, ProjectionWeights } from "./ProjectionAdapter";
import { MlpAdapter, MlpLayer } from "./MlpAdapter";
import { QuantizationAdapter, QuantizationType } from "./QuantizationAdapter";
import { VectorDBAdapter } from "./db";

export interface PipelineStep {
  type: string;
  adapter: WarpAdapter;
}

export interface PipelineState {
  type: string;
  state: any;
}

export interface RunContext {
  intent?: string;
  version?: string;
}

/**
 * WarpPipeline (統一インターフェース)
 *
 * 複数の WarpAdapter を直感的なビルダーパターンで数珠つなぎ（チェーン）し、
 * データパイプラインとして一括で実行・保存・復元するためのラッパークラスです。
 */
export class WarpPipeline {
  private steps: PipelineStep[] = [];

  constructor(public inputDim: number) {}

  /**
   * IntentAdapter (意図による線形変換) をパイプラインに追加します。
   */
  public addIntent(intents?: Record<string, IntentWeights>): this {
    const adapter = new IntentAdapter(intents || this.inputDim);
    this.steps.push({ type: "IntentAdapter", adapter });
    return this;
  }

  /**
   * LoraIntentAdapter (低ランク適応による線形変換) をパイプラインに追加します。
   */
  public addLoraIntent(
    rank: number,
    intents?: Record<string, LoraIntentWeights>
  ): this {
    const adapter = new LoraIntentAdapter(this.inputDim, rank, intents);
    this.steps.push({ type: "LoraIntentAdapter", adapter });
    return this;
  }

  /**
   * WhiteningAdapter (PCAによる空間的偏りの除去) をパイプラインに追加します。
   */
  public addWhitening(options?: { learningRate?: number; numComponents?: number }): this {
    const adapter = new WhiteningAdapter(this.inputDim, options);
    this.steps.push({ type: "WhiteningAdapter", adapter });
    return this;
  }

  /**
   * ProjectionAdapter (次元圧縮) をパイプラインに追加します。
   */
  public addProjection(outputDim: number, projections?: Record<string, ProjectionWeights>): this {
    const adapter = new ProjectionAdapter(this.inputDim, outputDim, projections);
    this.steps.push({ type: "ProjectionAdapter", adapter });
    // パイプラインの後続の入力次元を更新
    this.inputDim = outputDim;
    return this;
  }

  /**
   * MlpAdapter (多層ニューラルネットワーク / 非線形推論) をパイプラインに追加します。
   */
  public addMlp(layers: MlpLayer[]): this {
    const adapter = new MlpAdapter(layers);
    this.steps.push({ type: "MlpAdapter", adapter });
    
    // パイプラインの後続の入力次元を更新
    const lastLayer = layers[layers.length - 1];
    if (lastLayer.matrix instanceof Float32Array) {
      this.inputDim = lastLayer.bias.length;
    } else {
      this.inputDim = lastLayer.matrix.length;
    }
    
    return this;
  }

  /**
   * QuantizationAdapter (ベクトル量子化 / 圧縮) をパイプラインに追加します。
   * これは通常、パイプラインの最後のステップとして使用されます。
   */
  public quantize(type: QuantizationType): this {
    const adapter = new QuantizationAdapter({ type, dim: this.inputDim });
    this.steps.push({ type: "QuantizationAdapter", adapter });
    return this;
  }

  /**
   * パイプライン内に WASM などの非同期初期化を必要とするアダプタが含まれている場合、
   * それらを一括でセットアップします。
   */
  public async init(): Promise<void> {
    for (const step of this.steps) {
      if (typeof (step.adapter as any).init === "function") {
        await (step.adapter as any).init();
      }
    }
  }

  /**
   * パイプラインを順次実行し、入力ベクトルを最終的な表現に変換します。
   *
   * @param vector 変換元のベースベクトル
   * @param context インテントやバージョンなどのコンテキスト情報
   * @returns パイプラインを通過した最終的なベクトル (Float32Array または Uint8Array/Int8Array)
   */
  public run(vector: number[] | Float32Array, context?: RunContext): any {
    let currentVector: any = vector;
    
    for (const step of this.steps) {
      if (step.adapter instanceof QuantizationAdapter) {
        // 量子化アダプタにはコンテキストは不要
        currentVector = step.adapter.tune(currentVector as any);
      } else {
        // インテントを必要とするアダプタ (Intent, Projection等)
        currentVector = step.adapter.tune(currentVector, context?.intent || "default");
      }
    }

    return currentVector;
  }

  /**
   * 複数のベクトル（バッチ）を一括でパイプラインに通します。
   * 内部の tuneBatch が実装されているアダプタでは WASM/SIMD による高速処理が適用されます。
   *
   * @param vectors 変換元のベースベクトルの配列
   * @param context インテントやバージョンなどのコンテキスト情報
   * @returns 変換されたベクトルの配列
   */
  public runBatch(vectors: (number[] | Float32Array)[], context?: RunContext): any[] {
    let currentVectors: any[] = vectors;

    for (const step of this.steps) {
      if (typeof (step.adapter as any).tuneBatch === "function") {
        // tuneBatch メソッドがある場合は一括処理を委譲
        if (step.adapter instanceof QuantizationAdapter) {
          currentVectors = (step.adapter as any).tuneBatch(currentVectors);
        } else {
          currentVectors = (step.adapter as any).tuneBatch(currentVectors, context?.intent || "default");
        }
      } else {
        // tuneBatch がない場合は通常のループ処理へフォールバック
        currentVectors = currentVectors.map(vec => {
          if (step.adapter instanceof QuantizationAdapter) {
            return step.adapter.tune(vec as any);
          } else {
            return step.adapter.tune(vec, context?.intent || "default");
          }
        });
      }
    }

    return currentVectors;
  }

  /**
   * ベクトル変換から特定データベース向けのフォーマットまでを1回の呼び出しで行います。
   *
   * @param vector 変換元のベースベクトル
   * @param dbOptions フォーマットの指定オプション
   * @param context パイプラインのコンテキスト
   * @returns 指定されたデータベース形式のオブジェクトや文字列
   */
  public runAndFormat(
    vector: number[] | Float32Array,
    dbOptions: { format: "pgvector" | "pinecone" | "redis", topK?: number, filter?: Record<string, any> },
    context?: RunContext
  ): any {
    const tunedVector = this.run(vector, context);

    switch (dbOptions.format) {
      case "pgvector":
        return VectorDBAdapter.toPgvector(tunedVector as number[] | Float32Array);
      case "pinecone":
        return VectorDBAdapter.toPineconeQuery(
          tunedVector as number[] | Float32Array, 
          dbOptions.topK, 
          dbOptions.filter
        );
      case "redis":
        return VectorDBAdapter.toRedis(tunedVector as number[] | Float32Array);
      default:
        throw new Error(`Unknown format: ${dbOptions.format}`);
    }
  }

  /**
   * パイプライン内の全アダプタの状態（学習済みの重みなど）を JSON 化可能な配列として出力します。
   * これにより、DBやRedis等への永続化が容易になります。
   */
  public exportState(): PipelineState[] {
    return this.steps.map(step => {
      // 全てのアダプタは exportState を実装している前提
      const state = (step.adapter as any).exportState ? (step.adapter as any).exportState() : null;
      return {
        type: step.type,
        state
      };
    });
  }

  /**
   * エクスポートされた JSON 状態から、パイプラインを完全に復元（再構築）します。
   * @param states exportState で出力された配列
   * @returns 復元された新しい WarpPipeline インスタンス
   */
  public static importState(states: PipelineState[]): WarpPipeline {
    if (!states || states.length === 0) {
      throw new Error("No states provided to import.");
    }

    // 初期の入力次元は、復元しながら推測する（各アダプタの復元メソッドに依存）
    // とりあえずダミー値 0 で初期化し、必要に応じて設定
    const pipeline = new WarpPipeline(0);

    for (const step of states) {
      let adapter: WarpAdapter;
      switch (step.type) {
        case "IntentAdapter":
          adapter = IntentAdapter.importState(step.state);
          break;
        case "LoraIntentAdapter":
          adapter = LoraIntentAdapter.importState(step.state);
          break;
        case "WhiteningAdapter":
          adapter = WhiteningAdapter.importState(step.state);
          break;
        case "ProjectionAdapter":
          adapter = ProjectionAdapter.importState(step.state);
          break;
        case "MlpAdapter":
          adapter = MlpAdapter.importState(step.state);
          break;
        case "QuantizationAdapter":
          adapter = QuantizationAdapter.importState(step.state);
          break;
        default:
          throw new Error(`Unknown adapter type: ${step.type}`);
      }
      pipeline.steps.push({ type: step.type, adapter });
    }

    // 復元後、最新のアダプタの出力次元を inputDim として設定しておく
    // (次のステップを追加する場合に備えて)
    if (pipeline.steps.length > 0) {
      // 便宜上、復元では正確な inputDim を後から取るのは難しい場合があるため、
      // ひとまず維持する。ただし、基本的には復元後にチェーンを追加することは少ない想定。
    }

    return pipeline;
  }
}
