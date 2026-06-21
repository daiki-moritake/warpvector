import {
  WarpAdapter,
  FinalStageAdapter,
  InputVector,
  OutputVector,
  AdapterState,
} from "../interfaces/WarpAdapter";
import { IntentAdapter, IntentWeights } from "../adapters/IntentAdapter";
import {
  LoraIntentAdapter,
  LoraIntentWeights,
} from "../adapters/LoraIntentAdapter";
import {
  ProjectionAdapter,
  ProjectionWeights,
} from "../adapters/ProjectionAdapter";
import { VectorDBAdapter } from "../adapters/VectorDBAdapter";
import { AdapterRegistry } from "./AdapterRegistry";
import { FormatRegistry } from "./FormatRegistry";

export interface PipelineStep {
  type: string;
  adapter: WarpAdapter;
}

export interface PipelineState {
  type: string;
  state: AdapterState | null;
}

export interface FinalStageState {
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
  private finalStage?: { type: string; adapter: FinalStageAdapter };

  /**
   * カスタムアダプタをパイプラインのレジストリに登録します。
   * これにより importState でカスタムアダプタを復元可能になります。
   *
   * @param type アダプタの識別子 (例: "MyCustomAdapter")
   * @param importFn 状態オブジェクトからアダプタインスタンスを復元する関数
   */
  public static registerAdapter(
    type: string,
    importFn: (state: AdapterState) => WarpAdapter,
  ): void {
    AdapterRegistry.register(type, importFn);
  }

  /**
   * カスタムの出力フォーマットを登録します。
   * これにより、ユーザー独自のDB形式（Milvus, Weaviateなど）への変換を動的に追加できます。
   *
   * @param format フォーマット名 (例: "pgvector")
   * @param formatFn 変換を行うコールバック関数
   */
  public static registerFormat(
    format: string,
    formatFn: (vector: OutputVector, options: FormatOptions) => unknown,
  ): void {
    FormatRegistry.register(format, formatFn);
  }

  constructor(public inputDim: number) {}

  /**
   * 量子化などの最終変換（FinalStageAdapter）をパイプライン末尾に設定します。
   * パイプラインの run() 実行時、全ての WarpAdapter による変換が完了した後に
   * FinalStageAdapter.encode() が呼ばれます。
   *
   * @param type アダプタの識別子 (例: "QuantizationAdapter")
   * @param adapter FinalStageAdapter を実装したインスタンス
   */
  public setFinalStage(type: string, adapter: FinalStageAdapter): this {
    this.finalStage = { type, adapter };
    return this;
  }

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
    intents?: Record<string, LoraIntentWeights>,
  ): this {
    const adapter = new LoraIntentAdapter(this.inputDim, rank, intents);
    this.steps.push({ type: "LoraIntentAdapter", adapter });
    return this;
  }

  /**
   * ProjectionAdapter (次元圧縮) をパイプラインに追加します。
   */
  public addProjection(
    outputDim: number,
    projections?: Record<string, ProjectionWeights>,
  ): this {
    const adapter = new ProjectionAdapter(
      this.inputDim,
      outputDim,
      projections,
    );
    this.steps.push({ type: "ProjectionAdapter", adapter });
    // パイプラインの後続の入力次元を更新
    this.inputDim = outputDim;
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
    // ステップが空でfinalStageもない場合は不要な変換を避ける
    if (this.steps.length === 0 && !this.finalStage) {
      return vector instanceof Float32Array ? vector : new Float32Array(vector);
    }

    let currentVector: Float32Array =
      vector instanceof Float32Array ? vector : new Float32Array(vector);

    for (const step of this.steps) {
      // 全てのアダプタにコンテキストを渡す（不要なアダプタは内部で無視する）
      const result = step.adapter.tune(
        currentVector,
        context?.intent || "default",
      );
      // WarpAdapter の中間段は常に Float32Array を返すことを期待
      currentVector = result as Float32Array;
    }

    // 最終段（量子化等）が設定されている場合、encode() を適用
    if (this.finalStage) {
      return this.finalStage.adapter.encode(currentVector);
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
  public runBatch(
    vectors: InputVector[],
    context?: RunContext,
  ): OutputVector[] {
    const batchSize = vectors.length;
    let currentVectors = new Array<Float32Array>(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const v = vectors[i];
      currentVectors[i] = v instanceof Float32Array ? v : new Float32Array(v);
    }

    for (const step of this.steps) {
      if (typeof step.adapter.tuneBatch === "function") {
        // tuneBatch メソッドがある場合は一括処理を委譲
        currentVectors = step.adapter.tuneBatch(
          currentVectors,
          context?.intent || "default",
        ) as Float32Array[];
      } else {
        // tuneBatch がない場合は通常のループ処理へフォールバック
        for (let i = 0; i < batchSize; i++) {
          currentVectors[i] = step.adapter.tune(
            currentVectors[i],
            context?.intent || "default",
          ) as Float32Array;
        }
      }
    }

    // 最終段（量子化等）が設定されている場合、encode() を適用
    if (this.finalStage) {
      const results = new Array<OutputVector>(batchSize);
      for (let i = 0; i < batchSize; i++) {
        results[i] = this.finalStage!.adapter.encode(currentVectors[i]);
      }
      return results;
    }

    return currentVectors;
  }

  /**
   * ストリームやイテレータから順次ベクトルを受け取り、バッチ処理と組み合わせてパイプライン推論を実行します。
   * メモリに巨大な配列を保持しないため、数百万件のベクトルでも OOM (メモリ枯渇) せずに高速変換可能です。
   *
   * @param vectorStream 変換元のベクトルの非同期イテレータ または イテレータ
   * @param options コンテキストやバッチサイズのオプション
   * @returns 変換されたベクトルの非同期ジェネレータ
   */
  public async *runStream(
    vectorStream: AsyncIterable<InputVector> | Iterable<InputVector>,
    options?: { context?: RunContext; batchSize?: number },
  ): AsyncGenerator<OutputVector, void, unknown> {
    const batchSize = options?.batchSize ?? 128;
    const context = options?.context;
    let buffer: InputVector[] = [];

    for await (const vector of vectorStream) {
      buffer.push(vector);
      if (buffer.length >= batchSize) {
        const results = this.runBatch(buffer, context);
        for (const res of results) {
          yield res;
        }
        buffer = [];
      }
    }

    if (buffer.length > 0) {
      const results = this.runBatch(buffer, context);
      for (const res of results) {
        yield res;
      }
    }
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
    context?: RunContext,
  ): unknown {
    const tunedVector = this.run(vector, context);

    const formatFn = FormatRegistry.get(dbOptions.format);
    if (!formatFn) {
      throw new Error(
        `Unknown format: ${dbOptions.format}. Did you forget to register it?`,
      );
    }

    return formatFn(tunedVector, dbOptions);
  }

  /**
   * パイプライン内の全アダプタの状態（学習済みの重みなど）を JSON 化可能な配列として出力します。
   * これにより、DBやRedis等への永続化が容易になります。
   */
  public exportState(): {
    steps: PipelineState[];
    finalStage?: FinalStageState;
  } {
    const steps = this.steps.map((step) => {
      const state =
        typeof step.adapter.exportState === "function"
          ? step.adapter.exportState()
          : null;
      return {
        type: step.type,
        state,
      };
    });

    let finalStage: FinalStageState | undefined;
    if (this.finalStage) {
      finalStage = {
        type: this.finalStage.type,
        state:
          typeof this.finalStage.adapter.exportState === "function"
            ? this.finalStage.adapter.exportState()
            : null,
      };
    }

    return { steps, finalStage };
  }

  /**
   * エクスポートされた JSON 状態から、パイプラインを完全に復元（再構築）します。
   * @param states exportState で出力された配列
   * @returns 復元された新しい WarpPipeline インスタンス
   */
  public static importState(
    data:
      | PipelineState[]
      | { steps: PipelineState[]; finalStage?: FinalStageState },
  ): WarpPipeline {
    // 後方互換: PipelineState[] (旧形式) と { steps, finalStage } (新形式) の両方を受け付ける
    const states = Array.isArray(data) ? data : data.steps;
    const finalStageState = Array.isArray(data) ? undefined : data.finalStage;

    if (!states || states.length === 0) {
      throw new Error("No states provided to import.");
    }

    // 初期の入力次元は、復元しながら推測する（各アダプタの復元メソッドに依存）
    // とりあえずダミー値 0 で初期化し、必要に応じて設定
    const pipeline = new WarpPipeline(0);

    for (const step of states) {
      const importFn = AdapterRegistry.get(step.type);
      if (!importFn) {
        throw new Error(
          `Unknown adapter type: ${step.type}. Did you forget to register it via WarpPipeline.registerAdapter?`,
        );
      }

      const adapter = importFn(step.state as AdapterState);
      pipeline.steps.push({ type: step.type, adapter });
    }

    // FinalStage の復元
    if (finalStageState) {
      const importFn = AdapterRegistry.getFinalStage(finalStageState.type);
      if (importFn) {
        const adapter = importFn(finalStageState.state as AdapterState);
        pipeline.finalStage = { type: finalStageState.type, adapter };
      }
    }

    return pipeline;
  }

  /**
   * FinalStageAdapter をレジストリに登録します。
   */
  public static registerFinalStage(
    type: string,
    importFn: (state: AdapterState) => FinalStageAdapter,
  ): void {
    AdapterRegistry.registerFinalStage(type, importFn);
  }
}

// core パッケージに含まれるアダプタのみ初期登録
WarpPipeline.registerAdapter("IntentAdapter", (state) =>
  IntentAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("LoraIntentAdapter", (state) =>
  LoraIntentAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("ProjectionAdapter", (state) =>
  ProjectionAdapter.importState(state as string),
);

// 組み込みフォーマットを初期登録
WarpPipeline.registerFormat("pgvector", (vec, _opts) =>
  VectorDBAdapter.toPgvector(vec),
);
WarpPipeline.registerFormat("pinecone", (vec, opts) =>
  VectorDBAdapter.toPineconeQuery(
    vec,
    opts.topK as number,
    opts.filter as Record<string, unknown>,
  ),
);
WarpPipeline.registerFormat("redis", (vec, _opts) =>
  VectorDBAdapter.toRedis(vec),
);
