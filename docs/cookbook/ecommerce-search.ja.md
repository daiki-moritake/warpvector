# Cookbook: ECサイトのインテントベース検索

ECサイトでは、同じ「靴」という検索クエリでも、ユーザーの現在の閲覧コンテキストや過去の行動によって意味が全く異なる場合があります。たとえば、フォーマルな革靴が欲しいのか、ランニングシューズが欲しいのかという違いです。

別々のインデックスを維持したり、複雑なメタデータフィルタを渡したりする代わりに、`IntentAdapter` を使用してベクトル空間を動的にワープ（歪曲）させることができます。

## 実装

```typescript
import { IntentAdapter } from 'warpvector';

// 1. 埋め込みベクトルの次元数でアダプタを初期化（例: OpenAI 1536次元）
const adapter = new IntentAdapter(1536);

// 2. 変換行列とバイアスを提供してインテントを定義します。
// （これらは通常、事前に計算されるか、フィードバックループを通じて学習されます）
adapter.addIntent("formal", { matrix: formalMatrix, bias: formalBias });
adapter.addIntent("sports", { matrix: sportsMatrix, bias: sportsBias });

// 3. ユーザーが検索クエリを実行
const rawQueryVector = await getOpenAIEmbedding("shoes");

// 4. ユーザーのコンテキストを確認（セッション、過去のクリック、現在のカテゴリなど）
const userContext = getUserContext(req);

let warpedQueryVector = rawQueryVector;

if (userContext.preference === "formal_wear") {
  // フォーマルな概念に向けてベクトルをワープさせる
  warpedQueryVector = adapter.tune(rawQueryVector, "formal");
} else if (userContext.preference === "athletic") {
  // スポーツの概念に向けてベクトルをワープさせる
  warpedQueryVector = adapter.tune(rawQueryVector, "sports");
}

// 5. ワープされたベクトルでデータベースを検索
const results = await vectorDb.search(warpedQueryVector);
```

## 仕組み
`adapter.tune()` 関数は、WASMを使用した高度に最適化された行列乗算を実行します。これにかかる時間はわずか1マイクロ秒未満です。結果のベクトルは、ユーザーが意図したカテゴリに一致するデータベース内のアイテムに数学的に近づくため、コアとなる埋め込みモデルを再学習させることなく検索の関連性を向上させます。
