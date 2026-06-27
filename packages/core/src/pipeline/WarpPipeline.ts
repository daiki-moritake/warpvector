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
import { WarpPipelineError } from "../errors";
import { MetricsCollector, type PipelineMetrics } from "../metrics";
import { wasmMutex, initWasm } from "../wasm/wasm-loader";
import { WarpTracer } from "../telemetry/WarpTracer";
import { globalWasmPool } from "../wasm/WasmPool";

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
 * WarpPipeline の初期化オプション。
 */
export interface PipelineOptions {
  /**
   * true にすると、run() / runBatch() の初回呼び出し時に自動で init() を実行します。
   * デフォルト: true
   *
   * @example
   * ```typescript
   * // init() を呼ばなくても初回 run() で自動初期化される
   * const pipeline = new WarpPipeline(1536, { autoInit: true });
   * const result = await pipeline.run(vector); // 内部で init() が呼ばれる
   * ```
   */
  autoInit?: boolean;
}

/**
 * dryRun() が返す各ステップの中間結果。
 */
export interface DryRunStepResult {
  /** ステップの型名 */
  step: string;
  /** ステップの出力ベクトル */
  output: OutputVector;
  /** ステップの実行時間（ミリ秒） */
  durationMs: number;
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
  private _initialized: boolean = false;
  private _autoInit: boolean;
  private _initPromise: Promise<void> | null = null;
  private _metrics: MetricsCollector = new MetricsCollector();

  /**
   * パイプラインのメトリクス収集コレクターを取得します。
   * デフォルトでは無効です。`pipeline.metrics.enable()` で有効にしてください。
   *
   * @example
   * ```typescript
   * pipeline.metrics.enable();
   * pipeline.run(vector);
   * console.log(pipeline.metrics.getMetrics());
   * ```
   */
  public get metrics(): MetricsCollector {
    return this._metrics;
  }

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

  /**
   * @param inputDim 入力ベクトルの次元数
   * @param options パイプラインの初期化オプション
   */
  private _inputDim: number;

  /**
   * パイプラインの入力次元数を取得します。
   */
  public get inputDim(): number {
    return this._inputDim;
  }

  /**
   * @param inputDim 入力ベクトルの次元数
   * @param options パイプラインの初期化オプション
   */
  constructor(
    inputDim: number,
    options?: PipelineOptions,
  ) {
    this._inputDim = inputDim;
    this._autoInit = options?.autoInit ?? true;
  }

  /**
   * 量子化などの最終変換（FinalStageAdapter）をパイプライン末尾に設定します。
   * パイプラインの run() 実行時、全ての WarpAdapter による変換が完了した後に
   * FinalStageAdapter.encode() が呼ばれます。
   *
   * @param typeOrAdapter アダプタの識別子 (例: "QuantizationAdapter") または FinalStageAdapter インスタンス
   * @param adapter FinalStageAdapter を実装したインスタンス（第一引数が文字列の場合のみ必要）
   */
  public setFinalStage(typeOrAdapter: string | FinalStageAdapter, adapterParam?: FinalStageAdapter): this {
    let type: string;
    let adapter: FinalStageAdapter;

    if (typeof typeOrAdapter === "string") {
      type = typeOrAdapter;
      if (!adapterParam) throw new Error("adapterParam is required when first argument is a string.");
      adapter = adapterParam;
    } else {
      adapter = typeOrAdapter;
      type = adapter.constructor.name;
    }

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
    this._inputDim = outputDim;
    return this;
  }

  /**
   * カスタムアダプタを直接パイプラインの末尾に追加します。
   * (ビルダーパターンで独自の拡張アダプタを組み込む際に使用します)
   *
   * @param typeOrAdapter アダプタの識別子 または WarpAdapter インスタンス
   * @param adapter WarpAdapterを実装したインスタンス（第一引数が文字列の場合のみ必要）
   */
  public addStep(typeOrAdapter: string | WarpAdapter, adapterParam?: WarpAdapter): this {
    let type: string;
    let adapter: WarpAdapter;

    if (typeof typeOrAdapter === "string") {
      type = typeOrAdapter;
      if (!adapterParam) throw new Error("adapterParam is required when first argument is a string.");
      adapter = adapterParam;
    } else {
      adapter = typeOrAdapter;
      type = adapter.constructor.name;
    }

    this.steps.push({ type, adapter });
    return this;
  }

  /**
   * パイプライン内に WASM などの非同期初期化を必要とするアダプタが含まれている場合、
   * それらを一括でセットアップします。
   */
  public async init(): Promise<void> {
    if (this._initialized) return;
    
    // WasmPoolはランタイムで必須になったため初期化する
    await initWasm();

    for (const step of this.steps) {
      if (typeof step.adapter.init === "function") {
        await step.adapter.init();
      }
    }
    this._initialized = true;
  }

  /**
   * autoInit が有効な場合、初回呼び出しで自動的に init() を実行します。
   * すでに初期化済みの場合は何もしません。
   */
  private async ensureInitialized(): Promise<void> {
    if (this._initialized) return;
    if (!this._autoInit) return;
    if (!this._initPromise) {
      this._initPromise = this.init();
    }
    await this._initPromise;
  }

  /**
   * パイプラインを順次実行し、入力ベクトルを最終的な表現に変換します。
   *
   * @param vector 変換元のベースベクトル
   * @param context インテントやバージョンなどのコンテキスト情報
   * @returns パイプラインを通過した最終的なベクトル (Float32Array または Uint8Array/Int8Array)
   */
  public async run(vector: InputVector, context?: RunContext): Promise<OutputVector> {
    await this.ensureInitialized();
    // ステップが空でfinalStageもない場合は不要な変換を避ける
    if (this.steps.length === 0 && !this.finalStage) {
      return vector instanceof Float32Array ? vector : new Float32Array(vector);
    }

    const wasmCtx = globalWasmPool.acquire();
    try {
      globalWasmPool.setCurrentSyncContext(wasmCtx);
      const stopRun = this._metrics.startRun();

    let currentVector: Float32Array =
      vector instanceof Float32Array ? vector : new Float32Array(vector);

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const stopStep = this._metrics.startStep(step.type);
      try {
        // 全てのアダプタにコンテキストを渡す（不要なアダプタは内部で無視する）
        const result = step.adapter.tune(
          currentVector,
          context?.intent || "default",
        );
        // WarpAdapter の中間段は常に Float32Array (TransformOutput) を返すことを期待
        if (!(result instanceof Float32Array)) {
          throw new Error(`Intermediate adapter ${step.type} must return Float32Array.`);
        }
        currentVector = result;
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          i,
          step.type,
          { cause: e },
        );
      }
      stopStep?.();
    }

    // 最終段（量子化等）が設定されている場合、encode() を適用
    if (this.finalStage) {
      const stopFinal = this._metrics.startStep(this.finalStage.type);
      try {
        const result = this.finalStage.adapter.encode(currentVector);
        stopFinal?.();
        stopRun?.();
        return result;
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          this.steps.length,
          this.finalStage.type,
          { cause: e },
        );
      }
    }

      stopRun?.();
      return currentVector;
    } finally {
      globalWasmPool.clearCurrentSyncContext();
      globalWasmPool.release(wasmCtx);
    }
  }

  /**
   * 複数のベクトル（バッチ）を一括でパイプラインに通します。
   * 内部の tuneBatch が実装されているアダプタでは WASM/SIMD による高速処理が適用されます。
   *
   * @param vectors 変換元のベースベクトルの配列
   * @param context インテントやバージョンなどのコンテキスト情報
   * @returns 変換されたベクトルの配列
   */
  public async runBatch(
    vectors: InputVector[],
    context?: RunContext,
  ): Promise<OutputVector[]> {
    await this.ensureInitialized();
    const batchSize = vectors.length;

    const wasmCtx = globalWasmPool.acquire();
    try {
      globalWasmPool.setCurrentSyncContext(wasmCtx);
      const stopBatch = this._metrics.startBatchRun(batchSize);

      let currentVectors = new Array<Float32Array>(batchSize);
      for (let i = 0; i < batchSize; i++) {
        const v = vectors[i];
        currentVectors[i] = v instanceof Float32Array ? v : new Float32Array(v);
      }

      for (let si = 0; si < this.steps.length; si++) {
      const step = this.steps[si];
      const stopStep = this._metrics.startStep(step.type);
      try {
        if (typeof step.adapter.tuneBatchAsync === "function") {
          // tuneBatchAsync メソッドがある場合は非同期一括処理を委譲（WebGPU等）
          const results = await step.adapter.tuneBatchAsync(
            currentVectors,
            context?.intent || "default",
          );
          for (let i = 0; i < batchSize; i++) {
            if (!(results[i] instanceof Float32Array)) {
               throw new Error(`Intermediate adapter ${step.type} must return Float32Array in tuneBatchAsync.`);
            }
          }
          currentVectors = results as Float32Array[];
        } else if (typeof step.adapter.tuneBatch === "function") {
          // tuneBatch メソッドがある場合は同期一括処理を委譲（WASM/SIMD等）
          const results = step.adapter.tuneBatch(
            currentVectors,
            context?.intent || "default",
          );
          for (let i = 0; i < batchSize; i++) {
            if (!(results[i] instanceof Float32Array)) {
               throw new Error(`Intermediate adapter ${step.type} must return Float32Array in tuneBatch.`);
            }
          }
          currentVectors = results as Float32Array[];
        } else {
          // tuneBatch がない場合は通常のループ処理へフォールバック
          for (let i = 0; i < batchSize; i++) {
            const result = step.adapter.tune(
              currentVectors[i],
              context?.intent || "default",
            );
            if (!(result instanceof Float32Array)) {
               throw new Error(`Intermediate adapter ${step.type} must return Float32Array.`);
            }
            currentVectors[i] = result;
          }
        }
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          si,
          step.type,
          { cause: e },
        );
      }
      stopStep?.();
    }

    // 最終段（量子化等）が設定されている場合、encode() を適用
    if (this.finalStage) {
      const stopFinal = this._metrics.startStep(this.finalStage.type);
      try {
        const results = new Array<OutputVector>(batchSize);
        for (let i = 0; i < batchSize; i++) {
          results[i] = this.finalStage!.adapter.encode(currentVectors[i]);
        }
        stopFinal?.();
        stopBatch?.();
        return results;
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          this.steps.length,
          this.finalStage.type,
          { cause: e },
        );
      }
    }

      stopBatch?.();
      return currentVectors;
    } finally {
      globalWasmPool.clearCurrentSyncContext();
      globalWasmPool.release(wasmCtx);
    }
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
    options?: { context?: RunContext; batchSize?: number; maxBufferBatches?: number },
  ): AsyncGenerator<OutputVector, void, unknown> {
    // 自動初期化
    await this.ensureInitialized();

    const batchSize = options?.batchSize ?? 128;
    const context = options?.context;
    let buffer: InputVector[] = [];

    for await (const vector of vectorStream) {
      buffer.push(vector);
      if (buffer.length >= batchSize) {
        // WASM Instance PoolによりrunBatch内部で自動的に排他制御・並行処理される
        const batch = buffer;
        buffer = [];
        const results = await this.runBatch(batch, context);
        for (const res of results) {
          yield res;
        }
      }
    }

    if (buffer.length > 0) {
      const results = await this.runBatch(buffer, context);
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
  public async runAndFormat<T = unknown>(
    vector: InputVector,
    dbOptions: FormatOptions,
    context?: RunContext,
  ): Promise<T> {
    const tunedVector = await this.run(vector, context);

    const formatFn = FormatRegistry.get(dbOptions.format);
    if (!formatFn) {
      throw new Error(
        `Unknown format: ${dbOptions.format}. Did you forget to register it?`,
      );
    }

    return formatFn(tunedVector, dbOptions) as T;
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
   * パイプラインの構成を人間が読める文字列として返します。
   * デバッグ時にパイプラインの構成を確認するのに便利です。
   *
   * @example
   * ```typescript
   * console.log(pipeline.inspect());
   * // Pipeline [1536-dim]
   * //   Step 0: MlpAdapter
   * //   Step 1: IntentAdapter
   * //   Final: QuantizationAdapter
   * ```
   */
  public inspect(): string {
    const lines: string[] = [`Pipeline [${this.inputDim}-dim]`];
    for (let i = 0; i < this.steps.length; i++) {
      lines.push(`  Step ${i}: ${this.steps[i].type}`);
    }
    if (this.finalStage) {
      lines.push(`  Final: ${this.finalStage.type}`);
    }
    if (this.steps.length === 0 && !this.finalStage) {
      lines.push("  (empty pipeline)");
    }
    return lines.join("\n");
  }

  /**
   * パイプラインの各ステップの中間出力と実行時間を返すデバッグ用メソッド。
   * 本番環境ではなく、開発中のデバッグ・動作確認にのみ使用してください。
   *
   * @param vector 変換元のベースベクトル
   * @param context コンテキスト情報
   * @returns 各ステップの出力と実行時間の配列
   *
   * @example
   * ```typescript
   * const results = pipeline.dryRun(testVector, { intent: "tech" });
   * results.forEach(r => {
   *   console.log(`${r.step}: ${r.durationMs.toFixed(2)}ms, dim=${r.output.length}`);
   * });
   * ```
   */
  public async dryRun(
    vector: InputVector,
    context?: RunContext,
  ): Promise<DryRunStepResult[]> {
    await this.ensureInitialized();
    const results: DryRunStepResult[] = [];

    let currentVector: Float32Array =
      vector instanceof Float32Array ? vector : new Float32Array(vector);

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const start = performance.now();
      try {
        const result = step.adapter.tune(
          currentVector,
          context?.intent || "default",
        );
        if (!(result instanceof Float32Array)) {
          throw new Error(`Intermediate adapter ${step.type} must return Float32Array.`);
        }
        currentVector = result;
        results.push({
          step: step.type,
          output: result,
          durationMs: performance.now() - start,
        });
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          i,
          step.type,
          { cause: e },
        );
      }
    }

    if (this.finalStage) {
      const start = performance.now();
      try {
        const result = this.finalStage.adapter.encode(currentVector);
        results.push({
          step: this.finalStage.type,
          output: result,
          durationMs: performance.now() - start,
        });
      } catch (e) {
        throw new WarpPipelineError(
          (e as Error).message,
          this.steps.length,
          this.finalStage.type,
          { cause: e },
        );
      }
    }

    return results;
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
