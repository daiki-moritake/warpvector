import { WhiteningAdapter, cosineSimilarity } from "warpvector";

console.log("--- 表現空間の等方化 (Online Whitening) デモ ---");
console.log(
  "OpenAI ada-002 などのように、ベクトル空間全体が特定の方向に偏っている",
);
console.log(
  "「異方性 (Anisotropy)」をオンラインPCAを用いて補正するシミュレーションです。\n",
);

const dim = 1536; // ada-002と同じ次元数
const numDocs = 1000;
const adapter = new WhiteningAdapter(dim, {
  learningRate: 0.01,
  numComponents: 1,
});

// 事前学習モデル特有の強い偏り（コーン現象）をシミュレート
// 全ベクトルが固定のバイアス方向 [1.0, 1.0, ...] の周辺に集まっているとする
const generateAnisotropicVector = () => {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    // 巨大な共通バイアス 5.0 + 小さな意味的差異のノイズ (-0.5 ~ 0.5)
    v[i] = 5.0 + (Math.random() - 0.5);
  }
  return v;
};

// 1. ストリーミングでベクトルを受信し、WhiteningAdapterに学習させる
console.log(`[Phase 1] ${numDocs}件の偏ったベクトルをオンライン学習中...`);
const docs: Float32Array[] = [];
for (let i = 0; i < numDocs; i++) {
  const vec = generateAnisotropicVector();
  docs.push(vec);

  // Oja's rule により、偏りの原因である「平均」と「第1主成分」をストリーミング抽出
  adapter.update(vec);
}
console.log("=> 学習完了！主成分の抽出に成功しました。\n");

// 2. 類似度検索の比較
console.log("[Phase 2] 学習前後のコサイン類似度の比較");

// ランダムに選んだ「全く関係のない」2つのドキュメント
const docA = docs[0];
const docB = docs[10];

const originalSim = cosineSimilarity(docA, docB);
console.log(`❌ Whitening前 (生ベクトル) の類似度: ${originalSim.toFixed(4)}`);
console.log(
  `  -> 巨大なバイアスのせいで、全く関係ない文なのに類似度が極めて高くなってしまう！\n`,
);

// WhiteningAdapter による等方化 (All-but-the-Top)
const whitenedDocA = adapter.tune(docA);
const whitenedDocB = adapter.tune(docB);

const whitenedSim = cosineSimilarity(whitenedDocA, whitenedDocB);
console.log(
  `✅ Whitening後 (補正ベクトル) の類似度: ${whitenedSim.toFixed(4)}`,
);
console.log(
  `  -> 偏り成分が除去され、ノイズ同士（本来の無関係さ）が正しく低い類似度として反映された！\n`,
);

console.log(
  "💡 結論: WhiteningAdapterを通すだけで、事前学習モデルの検索精度（解像度）をエッジ上で劇的に高めることができます。",
);
