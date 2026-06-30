import { TripletExample, InfoNCEExample } from "../interfaces/training";

/**
 * 検索結果のインプレッション（表示）情報。
 * ユーザーに検索結果が表示されたタイミングで記録します。
 */
export interface SearchImpression {
  /** 検索クエリのベクトル */
  queryVector: Float32Array;
  /** 表示された検索結果のベクトル配列 */
  resultVectors: Float32Array[];
  /** 記録時刻 (Date.now()) */
  timestamp: number;
}

/**
 * ユーザーのフィードバックシグナル。
 * クリック、スキップ、滞在時間などの暗黙的フィードバックを表します。
 */
export interface FeedbackSignal {
  /** 対象の impressionId（recordImpression の戻り値） */
  impressionId: string;
  /** 結果配列内のインデックス */
  resultIndex: number;
  /** フィードバックの種類 */
  type: "click" | "skip" | "dwell";
  /** dwell の場合は滞在時間 (ms) */
  value?: number;
}

/**
 * FeedbackCollector のオプション。
 */
export interface FeedbackCollectorOptions {
  /** dwell（滞在時間）を positive とみなす閾値 (ms)。デフォルト: 3000 */
  dwellThresholdMs?: number;
  /** インプレッションの最大保持数。デフォルト: 200 */
  maxImpressions?: number;
}

/**
 * 内部で保持するインプレッション情報（フィードバック付き）。
 */
interface StoredImpression extends SearchImpression {
  id: string;
  feedbacks: FeedbackSignal[];
}

/**
 * ユーザーの暗黙的フィードバック（クリック、スキップ、滞在時間）を自動収集し、
 * TripletTrainer / InfoNCETrainer 用の学習データに変換するクラス。
 *
 * @example
 * ```typescript
 * const collector = new FeedbackCollector();
 *
 * // 検索結果を表示した時に記録
 * const impressionId = collector.recordImpression({
 *   queryVector: queryVec,
 *   resultVectors: [doc1Vec, doc2Vec, doc3Vec],
 *   timestamp: Date.now(),
 * });
 *
 * // ユーザーが doc1 をクリックした時
 * collector.recordFeedback({
 *   impressionId,
 *   resultIndex: 0,
 *   type: 'click',
 * });
 *
 * // 学習データに変換
 * const examples = collector.toTripletExamples();
 * ```
 */
export class FeedbackCollector {
  private impressions = new Map<string, StoredImpression>();
  private impressionOrder: string[] = [];
  private idCounter = 0;
  private readonly dwellThresholdMs: number;
  private readonly maxImpressions: number;

  constructor(options: FeedbackCollectorOptions = {}) {
    this.dwellThresholdMs = options.dwellThresholdMs ?? 3000;
    this.maxImpressions = options.maxImpressions ?? 200;
  }

  /**
   * 検索結果のインプレッション（表示）を記録します。
   *
   * @param impression インプレッション情報
   * @returns 生成された impressionId。以降の recordFeedback で使用します。
   */
  public recordImpression(impression: SearchImpression): string {
    const id = `imp_${++this.idCounter}_${impression.timestamp}`;

    this.impressions.set(id, {
      ...impression,
      id,
      feedbacks: [],
    });
    this.impressionOrder.push(id);

    // 上限を超えたら古いインプレッションを削除
    while (this.impressionOrder.length > this.maxImpressions) {
      const oldId = this.impressionOrder.shift()!;
      this.impressions.delete(oldId);
    }

    return id;
  }

  /**
   * ユーザーのフィードバックを記録します。
   *
   * @param signal フィードバックシグナル
   * @throws {Error} 指定された impressionId が存在しない場合
   * @throws {Error} resultIndex が範囲外の場合
   */
  public recordFeedback(signal: FeedbackSignal): void {
    const impression = this.impressions.get(signal.impressionId);
    if (!impression) {
      throw new Error(
        `Impression not found: ${signal.impressionId}. It may have been evicted due to maxImpressions limit.`,
      );
    }
    if (
      signal.resultIndex < 0 ||
      signal.resultIndex >= impression.resultVectors.length
    ) {
      throw new Error(
        `resultIndex ${signal.resultIndex} out of range [0, ${impression.resultVectors.length - 1}]`,
      );
    }
    impression.feedbacks.push(signal);
  }

  /**
   * 蓄積されたフィードバックを TripletExample の配列に変換します。
   *
   * 変換ルール:
   * - click → positive、同一インプレッション内の未クリック結果 → negative
   * - dwell（閾値以上） → positive、同一インプレッション内の他結果 → negative
   * - skip → 明示的な negative（他に positive がある場合のみ有効）
   *
   * 1つのインプレッションから複数の TripletExample が生成される場合があります。
   *
   * @returns TripletExample の配列
   */
  public toTripletExamples(): TripletExample[] {
    const examples: TripletExample[] = [];

    for (const impression of this.impressions.values()) {
      const { positiveIndices, negativeIndices } =
        this.classifyResults(impression);

      if (positiveIndices.length === 0 || negativeIndices.length === 0) {
        continue;
      }

      // 各 positive × 各 negative のペアを生成
      for (const posIdx of positiveIndices) {
        for (const negIdx of negativeIndices) {
          examples.push({
            anchor: impression.queryVector,
            positive: impression.resultVectors[posIdx],
            negative: impression.resultVectors[negIdx],
          });
        }
      }
    }

    return examples;
  }

  /**
   * 蓄積されたフィードバックを InfoNCEExample の配列に変換します。
   *
   * 1つの positive に対し、同一インプレッション内の全 negative をまとめます。
   *
   * @returns InfoNCEExample の配列
   */
  public toInfoNCEExamples(): InfoNCEExample[] {
    const examples: InfoNCEExample[] = [];

    for (const impression of this.impressions.values()) {
      const { positiveIndices, negativeIndices } =
        this.classifyResults(impression);

      if (positiveIndices.length === 0 || negativeIndices.length === 0) {
        continue;
      }

      const negVectors = negativeIndices.map(
        (i) => impression.resultVectors[i],
      );

      for (const posIdx of positiveIndices) {
        examples.push({
          anchor: impression.queryVector,
          positive: impression.resultVectors[posIdx],
          negatives: negVectors,
        });
      }
    }

    return examples;
  }

  /**
   * インプレッションバッファをクリアします。
   * 学習データへの変換後に呼び出してください。
   */
  public flush(): void {
    this.impressions.clear();
    this.impressionOrder = [];
  }

  /**
   * 現在バッファに蓄積されているインプレッション数。
   */
  public get pendingCount(): number {
    return this.impressions.size;
  }

  /**
   * フィードバック付きインプレッション数
   * （少なくとも1つのフィードバックが記録されたインプレッション）。
   */
  public get actionableCount(): number {
    let count = 0;
    for (const imp of this.impressions.values()) {
      if (imp.feedbacks.length > 0) count++;
    }
    return count;
  }

  /**
   * インプレッション内の結果を positive / negative に分類します。
   */
  private classifyResults(impression: StoredImpression): {
    positiveIndices: number[];
    negativeIndices: number[];
  } {
    const positiveSet = new Set<number>();
    const negativeSet = new Set<number>();

    for (const fb of impression.feedbacks) {
      switch (fb.type) {
        case "click":
          positiveSet.add(fb.resultIndex);
          break;
        case "dwell":
          if (fb.value !== undefined && fb.value >= this.dwellThresholdMs) {
            positiveSet.add(fb.resultIndex);
          } else {
            negativeSet.add(fb.resultIndex);
          }
          break;
        case "skip":
          negativeSet.add(fb.resultIndex);
          break;
      }
    }

    // positive に含まれるものは negative から除外
    for (const posIdx of positiveSet) {
      negativeSet.delete(posIdx);
    }

    // フィードバックがない結果は negative とみなす（クリックされなかった = 無関係）
    if (positiveSet.size > 0) {
      for (let i = 0; i < impression.resultVectors.length; i++) {
        if (!positiveSet.has(i) && !negativeSet.has(i)) {
          negativeSet.add(i);
        }
      }
    }

    return {
      positiveIndices: Array.from(positiveSet),
      negativeIndices: Array.from(negativeSet),
    };
  }
}
