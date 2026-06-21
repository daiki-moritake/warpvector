import { ColbertAdapter, initWasm } from "warpvector";

async function main() {
  console.log("--- Late Interaction (ColBERT) デモ ---");
  console.log(
    "通常の単一ベクトル(Dot Product)検索では欠落してしまう「細部の情報」を、",
  );
  console.log("MaxSim演算によって完全に比較する次世代の検索手法です。\n");

  // WASMの初期化
  await initWasm();

  const adapter = new ColbertAdapter();

  // 単語を1つの次元として単純化したベクトル(4次元)を用意
  // 例: [Apple, Orange, Banana, Grape] に対する重み
  const dim = 4;

  // クエリ: "Apple and Banana"
  // => Appleに0.9, Bananaに0.8 の特徴量を持つトークンの集まりと仮定
  const queryTokens = new Float32Array([
    // トークン1: Apple に強い
    0.9, 0.1, 0.0, 0.0,
    // トークン2: Banana に強い
    0.0, 0.0, 0.8, 0.1,
  ]);

  // ドキュメントA: "Apple is good. Orange is bad."
  const docATokens = new Float32Array([
    0.85,
    0.0,
    0.0,
    0.0, // Apple
    0.0,
    0.9,
    0.0,
    0.0, // Orange
    0.0,
    0.0,
    0.0,
    0.0, // (Bananaなし)
  ]);

  // ドキュメントB: "Banana is yellow. Apple is red."
  const docBTokens = new Float32Array([
    0.0,
    0.0,
    0.9,
    0.0, // Banana
    0.8,
    0.0,
    0.0,
    0.0, // Apple
    0.0,
    0.0,
    0.0,
    0.1, // Grape
  ]);

  // ドキュメントC: "Orange and Grape are sweet."
  const docCTokens = new Float32Array([
    0.0,
    0.8,
    0.0,
    0.0, // Orange
    0.0,
    0.0,
    0.0,
    0.9, // Grape
  ]);

  console.log("🔍 クエリ: 'Apple and Banana'");
  console.log("📄 ドキュメントA: 'Apple is good. Orange is bad.'");
  console.log("📄 ドキュメントB: 'Banana is yellow. Apple is red.'");
  console.log("📄 ドキュメントC: 'Orange and Grape are sweet.'");

  console.log("\n🚀 WASM MaxSim (Late Interaction) でスコア計算中...\n");

  const startTime = performance.now();

  const results = adapter.rank(
    queryTokens,
    [docATokens, docBTokens, docCTokens],
    dim,
  );

  const endTime = performance.now();

  const docNames = ["ドキュメントA", "ドキュメントB", "ドキュメントC"];

  console.log("🏆 検索結果ランキング:");
  for (const res of results) {
    console.log(`- ${docNames[res.index]}: Score = ${res.score.toFixed(4)}`);
  }

  console.log(
    `\n⏱️ 処理時間: ${(endTime - startTime).toFixed(3)} ms (超高速WASM処理)`,
  );
  console.log(
    "\n💡 結論: 単一のベクトルでは見逃されがちな「Bananaが存在するか」という細かい要素も、トークンごとのMaxSimによって正確に拾い上げ、ドキュメントBが1位にランクインしました！",
  );
}

main().catch(console.error);
