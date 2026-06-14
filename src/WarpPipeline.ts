import { WarpAdapter, InputVector, OutputVector, AdapterState } from "./WarpAdapter";
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
  state: AdapterState | null;
}

export interface RunContext {
  intent?: string;
  version?: string;
}

export interface FormatOptions {
  format: string;
  topK?: number;
  filter?: Record<string, unknown>;
  [key: string]: unknown;
}

/**
 * WarpPipeline (統一インターフェース)
 *
 * 複数の WarpAdapter を直感的なビルダーパターンで数珠つなぎ（チェーン）し、
 * データパイプラインとして一括で実行・保存・復元するためのラッパークラスです。
 */
export class WarpPipeline {
  private steps: PipelineStep[] = [];
  
  /**
   * アダプタの復元関数を保持するレジストリ。
   * カスタムアダプタをパイプラインで利用・復元可能にするために使用します。
   */
  private static adapterRegistry = new Map<string, (state: AdapterState) => WarpAdapter>();

  /**
   * カスタムアダプタをパイプラインのレジストリに登録します。
   * これにより importState でカスタムアダプタを復元可能になります。
   * 
   * @param type アダプタの識別子 (例: "MyCustomAdapter")
   * @param importFn 状態オブジェクトからアダプタインスタンスを復元する関数
   */
  public static registerAdapter(type: string, importFn: (state: AdapterState) => WarpAdapter): void {
    WarpPipeline.adapterRegistry.set(type, importFn);
  }

  /**
   * フォーマット変換ロジックを保持するレジストリ。
   */
  private static formatRegistry = new Map<string, (vector: InputVector, options: Record<string, unknown>) => unknown>();

  /**
   * カスタムの出力フォーマットを登録します。
   * これにより、ユーザー独自のDB形式（Milvus, Weaviateなど）への変換を動的に追加できます。
   * 
   * @param format フォーマット名 (例: "pgvector")
   * @param formatFn 変換を行うコールバック関数
   */
  public static registerFormat(format: string, formatFn: (vector: InputVector, options: Record<string, unknown>) => unknown): void {
    WarpPipeline.formatRegistry.set(format, formatFn);
  }

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
  public addWhitening(options?: import("./WhiteningAdapter").WhiteningConfig): this {
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
   * カスタムアダプタを直接パイプラインの末尾に追加します。
   * (ビルダーパターンで独自の拡張アダプタを組み込む際に使用します)
   * 
   * @param type アダプタの識別子（レジストリ登録名と一致させることを推奨）
   * @param adapter WarpAdapterを実装したインスタンス
   */
  public addStep(type: string, adapter: WarpAdapter): this {
    this.steps.push({ type, adapter });
    return this;
  }

  /**
   * パイプライン内に WASM などの非同期初期化を必要とするアダプタが含まれている場合、
   * それらを一括でセットアップします。
   */
  public async init(): Promise<void> {
    for (const step of this.steps) {
      if (typeof step.adapter.init === "function") {
        await step.adapter.init();
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
  public run(vector: InputVector, context?: RunContext): OutputVector {
    let currentVector: OutputVector = vector as OutputVector; // Initial input is Vector, but OutputVector handles ArrayTypes
    
    for (const step of this.steps) {
      // 全てのアダプタにコンテキストを渡す（不要なアダプタは内部で無視する）
      currentVector = step.adapter.tune(currentVector as InputVector, context?.intent || "default");
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
  public runBatch(vectors: InputVector[], context?: RunContext): OutputVector[] {
    let currentVectors: OutputVector[] = vectors as OutputVector[];

    for (const step of this.steps) {
      if (typeof step.adapter.tuneBatch === "function") {
        // tuneBatch メソッドがある場合は一括処理を委譲
        currentVectors = step.adapter.tuneBatch(currentVectors as InputVector[], context?.intent || "default");
      } else {
        // tuneBatch がない場合は通常のループ処理へフォールバック
        currentVectors = currentVectors.map(vec => step.adapter.tune(vec as InputVector, context?.intent || "default"));
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
    vector: InputVector,
    dbOptions: FormatOptions,
    context?: RunContext
  ): unknown {
    const tunedVector = this.run(vector, context);

    const formatFn = WarpPipeline.formatRegistry.get(dbOptions.format);
    if (!formatFn) {
      throw new Error(`Unknown format: ${dbOptions.format}. Did you forget to register it?`);
    }

    return formatFn(tunedVector as InputVector, dbOptions);
  }

  /**
   * パイプライン内の全アダプタの状態（学習済みの重みなど）を JSON 化可能な配列として出力します。
   * これにより、DBやRedis等への永続化が容易になります。
   */
  public exportState(): PipelineState[] {
    return this.steps.map(step => {
      const state = typeof step.adapter.exportState === "function" ? step.adapter.exportState() : null;
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
      const importFn = WarpPipeline.adapterRegistry.get(step.type);
      if (!importFn) {
        throw new Error(`Unknown adapter type: ${step.type}. Did you forget to register it via WarpPipeline.registerAdapter?`);
      }
      
      const adapter = importFn(step.state as AdapterState);
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

// 組み込みアダプタを初期登録
WarpPipeline.registerAdapter("IntentAdapter", (state) => IntentAdapter.importState(state as string));
WarpPipeline.registerAdapter("LoraIntentAdapter", (state) => LoraIntentAdapter.importState(state as string));
WarpPipeline.registerAdapter("WhiteningAdapter", (state) => WhiteningAdapter.importState(state as string));
WarpPipeline.registerAdapter("ProjectionAdapter", (state) => ProjectionAdapter.importState(state as string));
WarpPipeline.registerAdapter("MlpAdapter", (state) => MlpAdapter.importState(state as string));
WarpPipeline.registerAdapter("QuantizationAdapter", (state) => QuantizationAdapter.importState(state as string));

// 組み込みフォーマットを初期登録
WarpPipeline.registerFormat("pgvector", (vec, _opts) => VectorDBAdapter.toPgvector(vec));
WarpPipeline.registerFormat("pinecone", (vec, opts) => VectorDBAdapter.toPineconeQuery(vec, opts.topK as number, opts.filter as Record<string, unknown>));
WarpPipeline.registerFormat("redis", (vec, _opts) => VectorDBAdapter.toRedis(vec));
