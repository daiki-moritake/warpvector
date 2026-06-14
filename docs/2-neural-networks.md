# ニューラルネットワーク (Neural Networks)

WarpVector は、単なる線形変換（行列の掛け算）だけでなく、**多層パーセプトロン (MLP: Multi-Layer Perceptron)** や **非線形活性化関数** による高度な空間変形をサポートしています。

## 1. MlpAdapter

`MlpAdapter` は、事前学習済みモデルの出力を、複数の層と非線形な関数を用いて複雑に曲げ込み、より高次な意味の切り分けを可能にします。
WASM (WebAssembly) に最適化されており、TensorFlow.js や PyTorch のような重い機械学習フレームワークを読み込むことなく、ブラウザやエッジ環境で爆速のニューラルネット推論を実行できます。

### 特徴
- **任意層のネットワーク構築**: 入力層から出力層まで、任意の次元数と活性化関数を組み合わせて定義可能。
- **シームレスな統合**: 他の `WarpAdapter` と同じ `tune()` インターフェースを持つため、PrismaやLangChainのプラグインとして透過的に動きます。
- **WASM 駆動**: ネットワーク全体がひとつの WASM モジュール内で完結して実行されるため、JSのガベージコレクションによる遅延が発生しません。

### 基本的な使い方

```typescript
import { MlpAdapter } from 'warpvector';

// 1536次元の入力を受け取り、128次元の中間層を経て、2次元（例：座標）に出力する2層ニューラルネットワーク
const mlp = new MlpAdapter([
  { inputDim: 1536, outputDim: 128, activation: "relu" },
  { inputDim: 128, outputDim: 2, activation: "linear" }
]);

// 第1層 (1536 -> 128) の重みを設定
mlp.setLayerWeights(0, matrixLayer1, biasLayer1);

// 第2層 (128 -> 2) の重みを設定
mlp.setLayerWeights(1, matrixLayer2, biasLayer2);

const baseVector = /* OpenAI等から取得した1536次元ベクトル */;

// 超高速非線形推論 (WASM内での一貫処理)
const outputVector = mlp.tune(baseVector); // Float32Array(2)
```

## 2. 非線形活性化関数 (Activation Functions)

線形変換だけでは表現しきれない「空間の歪み」を表現するために、各コアアダプタ（`IntentAdapter` や `MlpAdapter` 等）は変換後の非線形活性化関数の適用をサポートしています。

### サポートされている関数

- **`relu`**: 負の値を0にする（特徴のスパース化）
- **`sigmoid`**: 値を 0.0 〜 1.0 の範囲に滑らかに収める
- **`tanh`**: 値を -1.0 〜 1.0 の範囲に滑らかに収める
- **`linear`** (デフォルト): 何も変換しない

### 例: IntentAdapter での活性化関数の使用

`tune` や `tuneBatch` メソッドの第3引数として活性化関数を指定するだけで、自動的に適用されます。WASM利用時はWASM内部でインライン展開されて処理されるため、オーバーヘッドがゼロです。

```typescript
import { IntentAdapter } from 'warpvector';

const adapter = new IntentAdapter(myIntents);

// ReLU関数を通して負のノイズ成分をカットする
const activatedVector = adapter.tune(baseVector, "riskAnalysis", "relu");
```

### なぜ非線形変換が必要か？

単純な検索ではコサイン類似度が一般的ですが、文書が「肯定か否定か」といった鋭い境界線を持つ場合、線形な空間のままでは綺麗に分離できないことがあります。
MLPや非線形活性化関数を使うことで、空間を曲げて特定のクラスタ同士を引き離し、意図した検索結果だけを「手前に引き寄せる」高度な検索チューニングが可能になります。
