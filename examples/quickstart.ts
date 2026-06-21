import { IntentAdapter, normalize, cosineSimilarity } from "warpvector";

async function run() {
  console.log("🚀 warpvector クイックスタート サンプル実行\n");

  // 1. サンプルデータとなる「ドキュメントのベクトル」を定義
  // ※ 実際のアプリケーションでは OpenAI の text-embedding-3-small などを利用して取得しますが、
  // ここでは理解しやすいようにシンプルな 3次元ベクトル を使います。
  const documents = [
    {
      id: "doc_weather",
      title: "☁️ 今日の天気と気温",
      vector: normalize([0.8, 0.6, 0.0]),
    },
    {
      id: "doc_economy",
      title: "📈 日経平均株価の推移",
      vector: normalize([0.0, 0.9, 0.5]),
    },
    {
      id: "doc_tech",
      title: "💻 最新のAIとWASM技術",
      vector: normalize([0.2, 0.1, 0.9]),
    },
  ];

  // 2. クエリとなる検索ベクトル
  // 例: ユーザーが検索窓に「最近のトレンド」と入力して得られたベクトルだと仮定します。
  const queryVector = normalize([0.3, 0.7, 0.5]);

  // 3. 意図（コンテキスト）に応じたベクトル空間の変形ルールを定義
  // 今回は「テクノロジー重視」という意図を定義します。
  // 第3の次元(Z軸)がTechに関連する特徴量であると仮定し、その軸を強調(拡大)させます。
  const adapter = new IntentAdapter({
    techFocus: {
      matrix: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.5], // Z軸(Techの特徴量)の重みを2.5倍に強調
      ],
      bias: [0.0, 0.0, 0.3], // さらに空間全体をZ軸方向にシフト（テック系の文書を全体的に底上げ）
    },
  });

  // 4. ベースとなる検索ベクトルを「テクノロジー重視」の意図空間にワープ（変換）させる
  const tunedQuery = adapter.tune(queryVector, "techFocus");

  // 比較のために正規化(ノルムを1に)しておく
  const normalizedTunedQuery = normalize(tunedQuery);

  // --- 結果の出力 ---

  console.log("=== 🔍 通常のベクトル検索 (意図なし) ===");
  // 普通の検索では、ベクトルが一番近い(0.0, 0.9, 0.5)「経済」のドキュメントが一番上に来ます。
  const normalResults = documents
    .map((doc) => ({
      title: doc.title,
      score: cosineSimilarity(queryVector, doc.vector),
    }))
    .sort((a, b) => b.score - a.score);

  normalResults.forEach((r) =>
    console.log(`- ${r.title.padEnd(20)} (スコア: ${r.score.toFixed(3)})`),
  );

  console.log("\n=== 🪄 warpvector 適用後 (テクノロジー重視) ===");
  // warpvectorを使うと、クエリベクトルが「テクノロジー空間」に歪められるため、
  // Tech関連のドキュメントがスコア上位に跳ね上がります！
  const tunedResults = documents
    .map((doc) => ({
      title: doc.title,
      score: cosineSimilarity(normalizedTunedQuery, doc.vector),
    }))
    .sort((a, b) => b.score - a.score);

  tunedResults.forEach((r) =>
    console.log(`- ${r.title.padEnd(20)} (スコア: ${r.score.toFixed(3)})`),
  );

  console.log(
    "\n✨ わずかな行数で、DBの中身を一切書き換えることなく検索結果のランキングを意図通りに操作できました！",
  );
}

run();
