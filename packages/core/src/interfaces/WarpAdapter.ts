export type InputVector = number[] | Float32Array;
/** 空間変換アダプタの出力型（パイプライン中間段階、常にFloat32Array） */
export type TransformOutput = Float32Array;
/** 最終出力型（量子化後を含む） */
export type OutputVector = Float32Array | Int8Array | Uint8Array;
export type AdapterState = Record<string, unknown>;

/**
 * WarpVector のすべてのベクトル変換アダプターに共通するインターフェース。
 * 外部の統合ライブラリ（Prisma, LangChainなど）は、このインターフェースを通じて
 * さまざまなアダプター（IntentAdapter, MlpAdapter, WhiteningAdapter など）を
 * 透過的に扱うことができます。
 *
 * @example
 * ```typescript
 * // IntentAdapter での使用例
 * const adapter: WarpAdapter = new IntentAdapter({
 *   techFocus: {
 *     matrix: [[1.0, 0.0], [0.0, 2.5]],
 *     bias: [0.0, 0.3],
 *   },
 * });
 * const result = adapter.tune([0.5, 0.3], "techFocus");
 * ```
 *
 * @example
 * ```typescript
 * // カスタムアダプターの実装
 * class MyAdapter implements WarpAdapter {
 *   tune(vector: InputVector, context?: string): OutputVector {
 *     const v = vector instanceof Float32Array ? vector : new Float32Array(vector);
 *     // カスタムの変換ロジック
 *     return v;
 *   }
 * }
 * ```
 */
export interface WarpAdapter {
  /**
   * 与えられたベクトルを変換（ワープ）します。
   *
   * @param vector 変換前のベクトル (number[] または Float32Array)
   * @param context オプションのコンテキスト情報 (意図の名前、バージョンなど)
   * @returns 変換後のベクトル (常に Float32Array)
   *
   * @example
   * ```typescript
   * const adapter = new IntentAdapter({ tech: { matrix: [...], bias: [...] } });
   * const warped = adapter.tune(rawVector, "tech");
   * ```
   */
  tune(vector: InputVector, context?: string): TransformOutput;

  /**
   * 複数のベクトルを一括で変換します（オプション実装）
   * WASMやSIMDを使用した最適化処理を提供します。
   *
   * @param vectors 変換前のベクトルの配列
   * @param context オプションのコンテキスト情報
   *
   * @example
   * ```typescript
   * const vectors = [vector1, vector2, vector3];
   * const results = adapter.tuneBatch?.(vectors, "tech") ?? vectors.map(v => adapter.tune(v, "tech"));
   * ```
   */
  tuneBatch?(vectors: InputVector[], context?: string): TransformOutput[];

  /**
   * 複数のベクトルを一括で非同期に変換します（オプション実装）
   * WebGPU Compute Shader などの非同期ハードウェアアクセラレーションを提供します。
   *
   * @param vectors 変換前のベクトルの配列
   * @param context オプションのコンテキスト情報
   */
  tuneBatchAsync?(
    vectors: InputVector[],
    context?: string,
  ): Promise<TransformOutput[]>;

  /**
   * 非同期での初期化処理（オプション実装）
   * WASMのロードなどが必要なアダプタで実装します。
   *
   * @example
   * ```typescript
   * const mlp = new MlpAdapter(layers);
   * await mlp.init(); // WASM を初期化
   * ```
   */
  init?(): Promise<void>;

  /**
   * アダプタの状態（学習済み重みなど）をエクスポートします（オプション実装）
   * importState に渡して完全に復元可能なJSON形式を返します。
   *
   * @example
   * ```typescript
   * const state = adapter.exportState();
   * // Redis等に保存
   * await redis.set("adapter_state", JSON.stringify(state));
   *
   * // 復元
   * const restored = IntentAdapter.importState(state as string);
   * ```
   */
  exportState?(): AdapterState;
}

/**
 * 量子化など、パイプラインの最終段にのみ配置可能なアダプタのインターフェース。
 * 入力はFloat32Array、出力は任意の型付き配列（Int8Array, Uint8Arrayなど）。
 *
 * WarpPipeline.setFinalStage() で使用します。
 *
 * @example
 * ```typescript
 * const quantizer = new QuantizationAdapter({ type: "int8", dim: 1536 });
 * const pipeline = new WarpPipeline(1536)
 *   .addIntent(intents)
 *   .setFinalStage("QuantizationAdapter", quantizer);
 * ```
 */
export interface FinalStageAdapter {
  /**
   * Float32Array ベクトルを最終的な出力形式にエンコードします。
   *
   * @param vector 変換済みのFloat32Arrayベクトル
   * @returns エンコード後のベクトル (Int8Array, Uint8Array等)
   */
  encode(vector: Float32Array): OutputVector;

  /** アダプタの状態をエクスポートします */
  exportState?(): AdapterState;
}
