import { IntentWeights } from "@warpvector/core";
import {
  IntentMatrixFactory,
  IntentMatrixFactoryOptions,
} from "./IntentMatrixFactory";

/**
 * テキスト埋め込み関数の型。
 * 外部の Embedding API（OpenAI, Cohere 等）を呼び出す関数を渡します。
 */
export type EmbedFn = (text: string) => Promise<number[] | Float32Array>;

/**
 * LlmIntentBootstrapper のオプション。
 */
export interface LlmIntentBootstrapperOptions {
  /**
   * IntentMatrixFactory に渡すビルドオプション。
   */
  factoryOptions?: IntentMatrixFactoryOptions;

  /**
   * API呼び出しの並列度。
   * rate limit を避けるため、デフォルトは3。
   * @default 3
   */
  concurrency?: number;

  /**
   * API呼び出し間のディレイ（ms）。
   * rate limit 対策として使用します。
   * @default 0
   */
  delayMs?: number;
}

/**
 * テキスト記述だけで Intent 行列を Zero-Shot 生成するブートストラッパー。
 *
 * ユーザーはサンプルベクトルを1つも用意する必要がありません。
 * カテゴリごとの自然言語テキスト記述を渡すだけで、
 * 内部で Embedding API を呼び出してベクトル化し、
 * IntentMatrixFactory で最適な変換行列を自動学習します。
 *
 * @example
 * ```typescript
 * import { LlmIntentBootstrapper } from "warpvector/train";
 *
 * const bootstrapper = new LlmIntentBootstrapper(1536, {
 *   embedFn: async (text) => {
 *     const resp = await openai.embeddings.create({
 *       input: text,
 *       model: "text-embedding-3-small",
 *     });
 *     return resp.data[0].embedding;
 *   },
 * });
 *
 * const intents = await bootstrapper.generate({
 *   tech: [
 *     "software engineering and programming",
 *     "cloud infrastructure and DevOps",
 *     "machine learning algorithms",
 *   ],
 *   business: [
 *     "revenue analysis and financial reports",
 *     "market strategy and competitive analysis",
 *     "customer acquisition and retention",
 *   ],
 * });
 *
 * // そのままパイプラインに投入
 * const pipeline = new WarpPipeline(1536).addIntent(intents);
 * ```
 */
export class LlmIntentBootstrapper {
  private readonly dimension: number;
  private readonly embedFn: EmbedFn;
  private readonly options: LlmIntentBootstrapperOptions;

  /**
   * @param dimension ベクトルの次元数（使用する Embedding モデルの出力次元に合わせる）
   * @param config embedFn と追加オプション
   */
  constructor(
    dimension: number,
    config: { embedFn: EmbedFn } & LlmIntentBootstrapperOptions,
  ) {
    if (dimension <= 0 || !Number.isInteger(dimension)) {
      throw new Error(
        `Dimension must be a positive integer, got ${dimension}.`,
      );
    }
    if (typeof config.embedFn !== "function") {
      throw new Error("embedFn must be a function.");
    }

    this.dimension = dimension;
    this.embedFn = config.embedFn;
    this.options = config;
  }

  /**
   * カテゴリごとのテキスト記述から Intent 行列を自動生成します。
   *
   * @param categories カテゴリ名とテキスト記述のマッピング
   * @returns 各カテゴリの IntentWeights
   * @throws カテゴリが2つ未満の場合、または embedFn がエラーを返した場合
   */
  public async generate(
    categories: Record<string, string[]>,
  ): Promise<Record<string, IntentWeights>> {
    const categoryNames = Object.keys(categories);

    if (categoryNames.length < 2) {
      throw new Error(
        "At least 2 categories are required. " +
          `Currently ${categoryNames.length} category is provided.`,
      );
    }

    // 全テキストの収集
    const allTexts: { category: string; text: string }[] = [];
    for (const [category, texts] of Object.entries(categories)) {
      if (texts.length === 0) {
        throw new Error(
          `Category '${category}' must have at least one text description.`,
        );
      }
      for (const text of texts) {
        allTexts.push({ category, text });
      }
    }

    // 埋め込み実行（並列度制限付き）
    const concurrency = this.options.concurrency ?? 3;
    const delayMs = this.options.delayMs ?? 0;
    const embeddings = new Map<string, (number[] | Float32Array)[]>();

    for (const name of categoryNames) {
      embeddings.set(name, []);
    }

    // チャンク単位で並列実行
    for (let i = 0; i < allTexts.length; i += concurrency) {
      const chunk = allTexts.slice(i, i + concurrency);

      const results = await Promise.all(
        chunk.map(async ({ text }) => {
          const vec = await this.embedFn(text);
          return vec;
        }),
      );

      for (let j = 0; j < chunk.length; j++) {
        const vec = results[j];
        if (vec.length !== this.dimension) {
          throw new Error(
            `Embedding dimension mismatch: expected ${this.dimension}, got ${vec.length} ` +
              `for text "${chunk[j].text.substring(0, 50)}..."`,
          );
        }
        embeddings.get(chunk[j].category)!.push(vec);
      }

      // rate limit 対策のディレイ
      if (delayMs > 0 && i + concurrency < allTexts.length) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }

    // IntentMatrixFactory で学習
    const factory = new IntentMatrixFactory(this.dimension);
    for (const [category, vectors] of embeddings.entries()) {
      factory.addCategory(category, vectors);
    }

    return factory.build(this.options.factoryOptions);
  }

  /**
   * 単一カテゴリのテキスト記述をベクトル化のみ行います（学習は行わない）。
   * テスト用やデバッグ用に使用します。
   *
   * @param texts テキスト記述の配列
   * @returns ベクトルの配列
   */
  public async embedTexts(
    texts: string[],
  ): Promise<(number[] | Float32Array)[]> {
    const results: (number[] | Float32Array)[] = [];
    const concurrency = this.options.concurrency ?? 3;
    const delayMs = this.options.delayMs ?? 0;

    for (let i = 0; i < texts.length; i += concurrency) {
      const chunk = texts.slice(i, i + concurrency);
      const embeddings = await Promise.all(
        chunk.map((text) => this.embedFn(text)),
      );
      results.push(...embeddings);

      if (delayMs > 0 && i + concurrency < texts.length) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }

    return results;
  }
}
