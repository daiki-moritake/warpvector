import { IntentAdapter } from "../IntentAdapter";
import { Activation } from "../utils";

/**
 * LlamaIndex の BaseEmbedding インターフェース互換
 * 依存関係を避けるため、Duck Typing 用のインターフェースを定義しています。
 */
export interface LlamaIndexBaseEmbedding {
  getTextEmbedding(text: string): Promise<number[]>;
  getQueryEmbedding(query: string): Promise<number[]>;
  getTextEmbeddings?(texts: string[]): Promise<number[][]>;
}

export interface WarpLlamaIndexEmbeddingsOptions {
  /**
   * ラップするベースの Embeddings インスタンス（LlamaIndex の BaseEmbedding 継承クラス）
   * 初期の密ベクトル（Dense Vector）の生成に使用されます。
   */
  baseEmbeddings: LlamaIndexBaseEmbedding;

  /**
   * 初期化済みの WarpVector IntentAdapter。
   */
  adapter: IntentAdapter;

  /**
   * (オプション) 検索クエリに使用する初期の意図（インテント）名。
   */
  intentName?: string;

  /**
   * (オプション) アフィン変換後に適用する非線形活性化関数。
   */
  activation?: Activation;

  /**
   * (オプション) 自己アテンションベースの動的意図合成（Auto-blending）を使用するかどうか。
   */
  autoBlend?: boolean;
}

/**
 * WarpLlamaIndexEmbeddings は、LlamaIndex の BaseEmbedding インスタンスをラップします。
 * `getQueryEmbedding` メソッドをインターセプトし、ベースベクトルを生成した後に
 * 現在のコンテキスト/意図に基づいて動的なアフィン変換（WarpVector）を適用します。
 */
export class WarpLlamaIndexEmbeddings implements LlamaIndexBaseEmbedding {
  private baseEmbeddings: LlamaIndexBaseEmbedding;
  private adapter: IntentAdapter;
  private intentName?: string;
  private activation?: Activation;
  private autoBlend: boolean;

  constructor(options: WarpLlamaIndexEmbeddingsOptions) {
    this.baseEmbeddings = options.baseEmbeddings;
    this.adapter = options.adapter;
    this.intentName = options.intentName;
    this.activation = options.activation;
    this.autoBlend = options.autoBlend ?? false;
  }

  public setIntent(intentName: string, activation?: Activation): void {
    this.intentName = intentName;
    this.activation = activation;
    this.autoBlend = false;
  }

  public setAutoBlend(enabled: boolean): void {
    this.autoBlend = enabled;
  }

  /**
   * ドキュメントの埋め込み（インデックス作成用）は変換を行いません。
   */
  async getTextEmbedding(text: string): Promise<number[]> {
    return this.baseEmbeddings.getTextEmbedding(text);
  }

  /**
   * 複数ドキュメントの埋め込み
   */
  async getTextEmbeddings(texts: string[]): Promise<number[][]> {
    if (this.baseEmbeddings.getTextEmbeddings) {
      return this.baseEmbeddings.getTextEmbeddings(texts);
    }
    return Promise.all(texts.map((t) => this.getTextEmbedding(t)));
  }

  /**
   * クエリの埋め込み（検索用）にのみ WarpVector 変換を適用します。
   */
  async getQueryEmbedding(query: string): Promise<number[]> {
    const baseVector = await this.baseEmbeddings.getQueryEmbedding(query);

    let warped: Float32Array;
    if (this.autoBlend) {
      warped = this.adapter.tuneAutoBlended(baseVector, this.activation);
    } else {
      if (!this.intentName) {
        throw new Error(
          "WarpLlamaIndexEmbeddings: intentName が設定されておらず、autoBlend も false です。",
        );
      }
      warped = this.adapter.tune(baseVector, this.intentName, this.activation);
    }

    return Array.from(warped);
  }
}
