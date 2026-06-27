---
title: "TypeScriptだけで構築するベクトルのオートチューニング（AutoML）とRAG検索精度評価の裏側"
emoji: "🤖"
type: "tech"
topics: ["rag", "llm", "typescript", "automl", "vectorsearch"]
published: true
---

## はじめに

ベクトル検索や RAG（Retrieval-Augmented Generation）のシステムを本番運用する際、最も頭を悩ませるのが「**どうすれば検索精度を最大化できるか？**」というパラメータ調整です。

- ベクトルの次元数を何次元に圧縮すべきか？
- 異方性補正（Whitening）の学習率や主成分数はいくつが最適か？
- コサイン類似度だけで不十分な場合、クエリの「意図（Intent）」に合わせた空間変形行列をどう設計するか？

これらのハイパーパラメータの調整は、これまで「**勘と度胸**」で行われるか、あるいは **Python のデータ分析エコシステム（Optuna, scikit-learn 等）** に頼らざるを得ませんでした。

しかし、RAG のミドルウェアやアプリケーションサーバーを Node.js や Bun、Cloudflare Workers 等の JavaScript / TypeScript 環境で構築している場合、評価やチューニングのためだけに Python 環境を用意したり、プロセス間通信を挟むのは大きな開発・運用コストになります。

WarpVector の最新リリース（v0.4.0 / v0.5.0）では、 **JS/TS だけで動作する「AutoML（PipelineAutoTuner）」** と、 **検索精度（NDCG@K や MRR）を定量測定する「RAG 評価キット」** を導入しました。本記事では、この仕組みと、実稼働を支える安全な量子化設計（SafeQuantizationAdapter）の裏側を解説します。

---

## 📊 TypeScript で実装する RAG 評価キット（`@warpvector/eval`）

精度の高いチューニングを行うための大前提は、「**現在の検索精度を定量的に測定できること**」です。WarpVector では、情報検索（Information Retrieval）の代表的な指標である **NDCG@K** や **MRR（Mean Reciprocal Rank）** を TypeScript のみで計算できる軽量な評価パッケージを提供しています。

### 代表的な指標の TS 実装

MRR は「最初の関連ドキュメントが何番目にヒットしたか」の逆数を表し、NDCG@K は「上位 K 件のランキングの質の高さ」を `[0, 1]` の範囲で評価します。

以下は、`@warpvector/eval` に組み込まれている評価指標のコード実装のイメージです。

```typescript
/**
 * MRR (Mean Reciprocal Rank) の計算
 * 最初の関連ドキュメントの順位の逆数。
 */
export function reciprocalRank(relevanceScores: number[]): number {
  for (let i = 0; i < relevanceScores.length; i++) {
    if (relevanceScores[i] > 0) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * DCG@K (Discounted Cumulative Gain) の計算
 * 上位K件の関連度スコアを、位置に応じて対数減衰させて加算する。
 */
function dcgAtK(relevanceScores: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, relevanceScores.length); i++) {
    dcg += relevanceScores[i] / Math.log2(i + 2);
  }
  return dcg;
}

/**
 * NDCG@K (Normalized DCG) の計算
 * 実際のDCG@Kを、理想的なランキング（降順ソート）のDCG@Kで割って正規化する。
 */
export function ndcgAtK(relevanceScores: number[], k: number): number {
  const actualDCG = dcgAtK(relevanceScores, k);
  const idealScores = [...relevanceScores].sort((a, b) => b - a);
  const idealDCG = dcgAtK(idealScores, k);

  if (idealDCG === 0) return 0;
  return actualDCG / idealDCG;
}
```

これらの評価処理は、CLI からも直接実行可能です。検索結果と正解データ（Ground Truth）の JSON を用意すれば、以下のコマンドで即座にレポートが出力されます。

```bash
npx warpvector-eval --input results.json --k 10 --format markdown
```

---

## 🤖 グリッドサーチによる自動探索：`PipelineAutoTuner`

精度測定ができるようになったら、次は「最も高いスコアを叩き出すパラメータの組み合わせ」の自動探索です。

`PipelineAutoTuner` は、検証用のデータセット（クエリ、ポジティブドキュメント、ネガティブドキュメントのセット）を入力として受け取り、指定されたハイパーパラメータの探索空間（Grid）を走査して、最適な `WarpPipeline` の構成を自動決定します。

### 1. メモリオーバーヘッドを抑えるストリーミング評価 (`O(1)` メモリ)

一般的な機械学習の評価では、データセット全体を一度に変換（エンコード）してメモリに保持しがちですが、これではサーバーのメモリを圧迫します。WarpVector では、 **1件ずつ推論（パイプライン実行）してスコアを加算するストリーミング方式** を採用することで、メモリオーバーヘッドを `O(1)` に抑えています。

```typescript
// PipelineAutoTuner.ts 内の評価ループの簡略版
private async evaluatePipeline(pipeline: WarpPipeline, metric: MetricType): Promise<number> {
  let scoreSum = 0;

  for (const ex of this.dataset) {
    // 1件ずつ推論を走らせ、メモリを解放しながらスコアを蓄積する
    const transformedQuery = await pipeline.run(ex.query);
    const transformedPositive = await pipeline.run(ex.positive);
    const transformedNegatives = ex.negatives
      ? await Promise.all(ex.negatives.map(neg => pipeline.run(neg)))
      : [];

    const rank = getPositiveRank(transformedQuery, transformedPositive, transformedNegatives);
    scoreSum += this.calculateScoreByMetric(rank, metric);
  }

  return scoreSum / this.dataset.length;
}
```

### 2. パラメータの直積グリッド生成と探索

探索空間（`searchSpace`）として渡された各パラメータの配列から、再帰を用いてすべての組み合わせ（直積）を動的に生成し、順次パイプラインを組み立てて評価します。

```typescript
// ハイパーパラメータの探索空間定義
const config: TuneConfig<MyParams> = {
  searchSpace: {
    learningRate: [0.005, 0.01, 0.02],
    numComponents: [1, 2, 4],
    clipThreshold: [50.0, 100.0]
  },
  pipelineBuilder: (params) => {
    return new WarpPipeline(1536)
      .addStep("whitening", new WhiteningAdapter(1536, {
        learningRate: params.learningRate,
        numComponents: params.numComponents
      }))
      .setFinalStage(new SafeQuantizationAdapter({
        type: "int8",
        dim: 1536,
        clipThreshold: params.clipThreshold
      }));
  },
  metric: "MRR"
};

const tuner = new PipelineAutoTuner(myDataset);
const result = await tuner.tuneGrid(config);

console.log("最良スコア (MRR):", result.bestScore);
console.log("最良パラメータ:", result.bestParams);
// result.bestPipeline をそのまま本番用パイプラインとして使用可能
```

これだけのコードで、Python なしで高精度なベクトル変換パイプラインの自動最適化が完了します。

---

## 🛡️ 実稼働を支える安全な量子化設計：`SafeQuantizationAdapter`

ベクトル検索のコストを下げるために「量子化（Quantization: Float32 から Int8 や Binary への変換）」は非常に強力ですが、実稼働（Production）環境では予期せぬ落とし穴があります。

それは、API から返される埋め込みベクトルに稀に混入する **NaN（Not a Number）や極端に巨大な値（Infinity、オーバーフロー）** です。これらがフィルタなしで量子化器に入ると、配列全体が破壊されたり、検索が全く機能しなくなるバグを引き起こします。

WarpVector v0.4.0 で導入された `SafeQuantizationAdapter` は、量子化を実行する直前に極めて厳密な **サニタイズ（クレンジング）とクリッピング** を施します。

### NaN/Infinity ガードとクリッピング

```typescript
export class SafeQuantizationAdapter implements FinalStageAdapter {
  public encode(vector: Float32Array): OutputVector {
    const len = vector.length;
    const safeVector = new Float32Array(len);
    
    // 量子化タイプ（int8等）に応じた適切なクリッピング値の決定
    const clipThreshold = this.options.clipThreshold ?? (this.options.type === "int8" ? 127.0 : 100.0);

    for (let i = 0; i < len; i++) {
      let val = vector[i];
      
      // 1. NaN や Infinity が入ってきた場合は 0.0 に安全にフォールバック
      if (Number.isNaN(val) || !Number.isFinite(val)) {
        val = 0.0;
      }
      
      // 2. 指定のしきい値でクランプ（クリッピング）し、値の破綻を防ぐ
      if (val > clipThreshold) {
        val = clipThreshold;
      } else if (val < -clipThreshold) {
        val = -clipThreshold;
      }
      
      safeVector[i] = val;
    }

    // クレンジング済みの安全なベクトルを本番の量子化ロジックに流す
    return this.baseAdapter.encode(safeVector);
  }
}
```

### 開発者体験（DX）へのこだわり：静的/動的なプロパティチェック

さらに、`SafeQuantizationAdapter` では「**壊れた設定のまま本番稼働して検索精度が下がるのを未然に防ぐ**」ために、古い API からのマイグレーションミスやパラメータ設計の誤りをコンストラクタで徹底的にバリデーションし、詳細なエラーメッセージを投げます。

```typescript
constructor(options: SafeQuantizationOptions) {
  if (arguments.length > 1) {
    throw new Error(
      "[WarpVector DX Error] SafeQuantizationAdapter のコンストラクタ引数が変更されました。\n" +
      "次元数 (dim) などを第1引数に渡す必要はありません。すべての設定は1つのオブジェクトで渡してください。"
    );
  }
  
  if (options && 'levels' in options) {
    throw new Error(
      "[WarpVector DX Error] SafeQuantizationAdapter のプロパティ 'levels' は廃止されました。\n" +
      "'type': 'int8' または 'binary' を使用してください。"
    );
  }
  
  if (options && 'adaptiveRange' in options) {
    throw new Error(
      "[WarpVector DX Error] SafeQuantizationAdapter のプロパティ 'adaptiveRange' は 'dynamic' に変更されました。"
    );
  }

  this.options = options;
  this.baseAdapter = new QuantizationAdapter(options);
}
```

コンパイルエラーだけでは防ぎきれない、ランタイム時の古い設定の混入を即座に通知するこの設計により、開発者はアップグレード時の設定ミスを数秒で発見できます。

---

## 📈 実際のベンチマーク検証（`evaluate.ts` より）

実際にこのオートチューンおよび意図の埋め込み（Intent Warping）を行った場合、検索精度はどれほど向上するでしょうか？

以下は、リポジトリ内の合成データセット（2つの異なるドメインに分かれた文書群とクエリ）に対して精度測定を走らせたベンチマークの結果です。

| 手法 (Method) | 次元数 (Dim) | NDCG@10 | MRR |
| :--- | :--- | :--- | :--- |
| **Vanilla (ベースライン)** | 384 | 72.1% | 76.5% |
| **Intent Warping (手動アフィン変換)** | 384 | 84.3% (+16.9%) | 88.2% (+15.2%) |
| **IntentMatrixFactory (自動チューン)** | 384 | **89.5% (+24.1%)** | **92.4% (+20.7%)** |

コサイン類似度だけに依存した単純な検索（Vanilla）に比べ、`IntentMatrixFactory` と `PipelineAutoTuner` を用いて自動最適化されたパイプラインでは、 **NDCG@10が 24% 以上改善** し、欲しい情報が的確にトップ10圏内に浮上するようになりました。

---

## まとめ

通常は Python や重厚な ML 用コンテナが必要だった「ベクトルの評価」「オートチューニング」「セーフガード付き量子化」ですが、TypeScript の表現力と WASM による計算加速の組み合わせにより、 **すべて JS / TS のライブラリ単体で完結** させることができます。

RAG やベクトル検索のシステム開発において、「動いた後」のチューニングや安定運用に課題を感じている方は、ぜひ WarpVector のオートチューンと安全設計アダプターを活用してみてください！

https://github.com/daiki-moritake/warpvector

---

### 📚 関連記事

- [RAGの検索精度が低い？ベクトル空間の「異方性」を3行で解決する方法](/daiki_moritake/articles/fix-rag-anisotropy)
- [Pineconeのコストを96%削減し、RAGの精度を劇的に向上させる方法](/daiki_moritake/articles/reduce-pinecone-costs)
- [Cloudflare Workersで「ベクトル推論」をサブミリ秒で動かす方法](/daiki_moritake/articles/edge-vector-inference)
- [LangChainの検索精度に不満？ミドルウェアを1つ挟むだけで劇的に改善する方法](/daiki_moritake/articles/langchain-search-improvement)
- [Pythonなしで検索のパーソナライズを実装する](/daiki_moritake/articles/ts-contrastive-learning)
