# オンライン学習エンジン (Trainers)

通常のベクトル検索エンジンでは、事前学習済みのモデル（OpenAI, Cohere 等）が吐き出したベクトルを「正解」として扱い、そのまま検索に使用します。しかし、実際のドメイン固有の検索（例えば自社のFAQ検索や特定業界の専門用語検索）では、そのベクトル空間が必ずしも最適とは限りません。

WarpVector は、PythonサーバーやGPUインスタンスを別途立ち上げることなく、**ユーザーのフィードバック（クリックログや「いいね/悪いね」）から直接、Edge/Node環境でリアルタイムにベクトル空間を再学習・最適化**できる「Trainer（学習エンジン）」を内蔵しています。

内部には **Adam オプティマイザー** が搭載されており、深層学習フレームワークと同等の勾配降下法を単独で実行します。

## 1. TripletTrainer (トリプレット学習)

最もシンプルな対照学習（Contrastive Learning）の形です。
「アンカー（検索クエリ）」、「ポジティブ（ユーザーがクリックした正解文書）」、「ネガティブ（表示されたがクリックされなかった無関係な文書）」の3つのベクトルを1セットとして学習します。

クエリをポジティブに近づけ、ネガティブから遠ざけるように、空間を歪める行列（$W$）とバイアス（$b$）を自動的に更新します。

### 使い方

```typescript
import { TripletTrainer } from 'warpvector';

const trainer = new TripletTrainer(1536);

// 現在の重み (初期状態や前回の状態)
let currentWeights = {
  matrix: new Float32Array(1536 * 1536), // 単位行列などで初期化
  bias: new Float32Array(1536)           // ゼロベクトルで初期化
};

// ユーザーが検索し、結果の1つをクリックしたタイミングで呼び出す
const updatedWeights = await trainer.updateOnline(
  currentWeights,
  {
    anchor: anchorVector,     // 検索クエリのベクトル
    positive: positiveVector, // クリックされた正解ドキュメントのベクトル
    negative: negativeVector,  // 無視されたドキュメントのベクトル
  },
  { learningRate: 0.001 }
);

// 更新された updatedWeights を IntentAdapter に渡して検索に使う！
```

## 2. InfoNCETrainer (複数 Negative に対応した対照学習)

TripletLoss をさらに発展させ、**1つの正解に対して複数の不正解（In-batch Negatives）を同時に遠ざける**ことができる強力な損失関数（InfoNCE Loss）を用いたトレーナーです。
現代の検索モデル（SimCSEやCLIP等）の学習で標準的に使われている手法をエッジ上で実行します。

より早く、より正確に空間を最適化できます。

### 使い方

```typescript
import { InfoNCETrainer } from 'warpvector';

const trainer = new InfoNCETrainer(1536);

// ... 検索とクリックイベントの取得 ...

// 1つの正解と、複数の不正解リストを渡して学習
const updatedWeights = await trainer.updateOnline(
  currentWeights,
  {
    anchor: anchorVector,
    positive: positiveVector,
    negatives: [negativeVector1, negativeVector2, negativeVector3],
  },
  { learningRate: 0.001, temperature: 0.1 }
);
```

## 3. Trainer のユースケース

- **検索結果のパーソナライズ**: ユーザーごとのクリック履歴から、そのユーザー専用の `IntentAdapter` の重みを学習。
- **ドメイン適応**: 業界特有の専門用語の検索精度が低い場合、運営側で「このクエリの正解はこの文書」というペアをいくつか作り、WarpVector に学習させて空間を補正する。
- **アクティブラーニング**: いいねボタン/低評価ボタンのログをそのまま学習パイプラインに流し込み、システムを自己進化させる。
