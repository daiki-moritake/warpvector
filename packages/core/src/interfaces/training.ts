/**
 * 学習データのペア（Anchor, Positive, Negative）
 * @interface TripletExample
 */
export interface TripletExample {
  /** 基準となるベクトル（検索クエリなど） */
  anchor: number[] | Float32Array;
  /** Anchorに近づけたい正解ベクトル（クリックされた商品など） */
  positive: number[] | Float32Array;
  /** Anchorから遠ざけたい不正解ベクトル（スルーされた商品など） */
  negative: number[] | Float32Array;
}

/**
 * 学習データのペア（Anchor, Positive, 複数のNegatives）
 * @interface InfoNCEExample
 */
export interface InfoNCEExample {
  /** 基準となるベクトル（検索クエリなど） */
  anchor: number[] | Float32Array;
  /** Anchorに近づけたい正解ベクトル（クリックされた商品など） */
  positive: number[] | Float32Array;
  /** Anchorから遠ざけたい不正解ベクトルの配列（スルーされた商品群など） */
  negatives: (number[] | Float32Array)[];
}
