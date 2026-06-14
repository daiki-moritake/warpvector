import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings";
import { IntentAdapter } from "@warpvector/core";
import { Activation } from "@warpvector/core";

/**
 * WarpEmbeddings の設定オプション
 */
export interface WarpEmbeddingsOptions extends EmbeddingsParams {
  /**
   * ラップするベースの Embeddings インスタンス（例: OpenAIEmbeddings）
   * 初期の密ベクトル（Dense Vector）の生成に使用されます。
   */
  baseEmbeddings: Embeddings;

  /**
   * 初期化済みの WarpVector IntentAdapter。
   * 意図（インテント）の変換行列とバイアスを保持しています。
   */
  adapter: IntentAdapter;

  /**
   * (オプション) 検索クエリに使用する初期の意図（インテント）名。
   * 指定しない場合、検索を実行する前に `setIntent()` で設定する必要があります。
   */
  intentName?: string;

  /**
   * (オプション) アフィン変換後に適用する非線形活性化関数（'relu', 'sigmoid', 'tanh'など）。
   */
  activation?: Activation;

  /**
   * (オプション) 単一の意図を指定する代わりに、自己アテンションベースの
   * 動的意図合成（Auto-blending）を使用するかどうか。
   * true の場合、クエリ実行時に `intentName` は無視されます。
   */
  autoBlend?: boolean;
}

/**
 * WarpEmbeddings は、LangChain の BaseEmbeddings インスタンスをラップするクラスです。
 * `embedQuery` メソッドをインターセプトし、ベースベクトルを生成した後に、
 * 現在のコンテキスト/意図に基づいて動的なアフィン変換（WarpVector）を適用します。
 *
 * VectorStore へ正確なセマンティクスとして保存（インデックス）できるよう、
 * ドキュメントは変換されず、通常通り埋め込まれます。
 * ユーザーの「検索クエリ」に対してのみワープが適用されます。
 *
 * @example
 * ```typescript
 * import { OpenAIEmbeddings } from "@langchain/openai";
 * import { IntentAdapter } from "warpvector";
 * import { WarpEmbeddings } from "warpvector/integrations/langchain";
 *
 * const baseEmbeddings = new OpenAIEmbeddings();
 * const adapter = new IntentAdapter({ ... });
 *
 * const embeddings = new WarpEmbeddings({
 *   baseEmbeddings,
 *   adapter,
 *   intentName: "riskAnalysis"
 * });
 *
 * // このラップされた embeddings インスタンスを LangChain の VectorStore に直接渡せます！
 * const vectorStore = new MemoryVectorStore(embeddings);
 * ```
 */
export class WarpEmbeddings extends Embeddings {
  private baseEmbeddings: Embeddings;
  private adapter: IntentAdapter;
  private intentName?: string;
  private activation?: Activation;
  private autoBlend: boolean;

  constructor(options: WarpEmbeddingsOptions) {
    super(options);
    this.baseEmbeddings = options.baseEmbeddings;
    this.adapter = options.adapter;
    this.intentName = options.intentName;
    this.activation = options.activation;
    this.autoBlend = options.autoBlend ?? false;
  }

  /**
   * 実行時にアクティブな意図（インテント）を動的に切り替えます。
   */
  public setIntent(intentName: string, activation?: Activation): void {
    this.intentName = intentName;
    this.activation = activation;
    this.autoBlend = false; // 明示的な意図が設定された場合は自動ブレンドを無効化
  }

  /**
   * 自己アテンション型の動的意図合成（Auto-blending）を有効化または無効化します。
   */
  public setAutoBlend(enabled: boolean): void {
    this.autoBlend = enabled;
  }

  /**
   * ベースの embeddings モデルを使用して、ドキュメントを通常通り埋め込みます。
   * VectorDB には客観的な空間データを含める必要があるため、インデックスされるドキュメントはワープ（変換）しません。
   */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    return this.baseEmbeddings.embedDocuments(documents);
  }

  /**
   * ユーザーのクエリを埋め込み、WarpVector によるアフィン変換を適用します。
   */
  async embedQuery(document: string): Promise<number[]> {
    // 1. 基盤となる LLM / モデルからベースの密ベクトルを生成
    const baseVector = await this.baseEmbeddings.embedQuery(document);

    // 2. WarpVector 変換を適用
    let warped: Float32Array;
    if (this.autoBlend) {
      warped = this.adapter.tuneAutoBlended(baseVector, this.activation);
    } else {
      if (!this.intentName) {
        throw new Error(
          "WarpEmbeddings: intentName が設定されておらず、autoBlend も false です。setIntent() を呼び出すか autoBlend を有効にしてください。",
        );
      }
      warped = this.adapter.tune(baseVector, this.intentName, this.activation);
    }

    // LangChain が期待する標準の JavaScript 配列として返す
    return Array.from(warped);
  }
}
