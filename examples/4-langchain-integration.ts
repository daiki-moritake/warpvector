import { Embeddings } from "@langchain/core/embeddings";
import { WarpEmbeddings } from "../src/integrations/langchain";
import { IntentAdapter } from "../src";

/**
 * LLM の呼び出し（例: OpenAIEmbeddings）をシミュレートするダミーの Embeddings クラス
 * 実際のアプリケーションでは `import { OpenAIEmbeddings } from "@langchain/openai";` などを利用します。
 */
class FakeOpenAIEmbeddings extends Embeddings {
  async embedDocuments(documents: string[]): Promise<number[][]> {
    // ドキュメント用のダミーベクトルを返す
    return documents.map(() => [0.1, 0.2, 0.3]);
  }

  async embedQuery(document: string): Promise<number[]> {
    console.log(`[FakeOpenAI] クエリのベースベクトルを生成中: "${document}"`);
    return [0.1, 0.2, 0.3];
  }
}

async function main() {
  console.log("=== LangChain 連携 (Integration) サンプル ===");

  // 1. ベースとなる embeddings (例: OpenAI) を初期化
  const baseEmbeddings = new FakeOpenAIEmbeddings({});

  // 2. 意図（インテント）を持つ WarpVector アダプターを初期化
  const adapter = new IntentAdapter({
    academic: {
      matrix: [
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
      ],
      bias: [0.1, 0.0, -0.1],
    },
    casual: {
      matrix: [
        [0.5, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      bias: [-0.1, 0.2, 0.0],
    }
  });

  // 3. embeddings をラップする！
  // この単一のオブジェクトをそのまま LangChain の VectorStore に渡すことができます
  // 例: `new MemoryVectorStore(warpEmbeddings);`
  const warpEmbeddings = new WarpEmbeddings({
    baseEmbeddings,
    adapter,
    intentName: "academic" // デフォルトの意図
  });

  console.log("\n--- 'academic' (学術的) な意図でクエリを実行 ---");
  let queryVector = await warpEmbeddings.embedQuery("コンピュータとは何ですか？");
  console.log("ワープ後のクエリベクトル:", queryVector);

  console.log("\n--- 'casual' (カジュアル) な意図に切り替え ---");
  warpEmbeddings.setIntent("casual");
  queryVector = await warpEmbeddings.embedQuery("コンピュータとは何ですか？");
  console.log("ワープ後のクエリベクトル:", queryVector);

  console.log("\n--- ドキュメントの埋め込み (インデックス保存時) ---");
  // ドキュメントの保存時にはワープ処理が行われず、純粋なベースベクトルが使用されることに注目してください。
  const docVectors = await warpEmbeddings.embedDocuments(["ドキュメント 1", "ドキュメント 2"]);
  console.log("ドキュメントベクトル (ワープなし):", docVectors);
  
  console.log("\n💡 成功です！ これで `warpEmbeddings` を LangChain のあらゆるワークフローに組み込めます。");
}

main().catch(console.error);
