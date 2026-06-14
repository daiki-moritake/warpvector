/**
 * 外部のベクトルデータベース（pgvector, Pinecone, Redisなど）へ
 * ワープ変換後のベクトルを保存・検索するためのアダプター/ユーティリティクラス
 */
export class VectorDBAdapter {
  /**
   * PostgreSQL (pgvector) 用のクエリ文字列表現を生成します。
   * INSERT や SELECT の際に使用できる形式です。
   *
   * @example
   * const sql = `SELECT * FROM items ORDER BY embedding <-> '${VectorDBAdapter.toPgvector(warpedVector)}' LIMIT 5`;
   *
   * @param {number[] | Float32Array} vector ワープ変換後のベクトル
   * @returns {string} `'[0.1, 0.2, 0.3]'` のような文字列表現
   */
  public static toPgvector(vector: number[] | Float32Array | Int8Array | Uint8Array): string {
    return `[${Array.from(vector).join(", ")}]`;
  }

  /**
   * Pinecone 用のクエリオブジェクトを生成します。
   * Pinecone クライアントに直接渡せる形式のオブジェクトを返します。
   *
   * @example
   * const query = VectorDBAdapter.toPineconeQuery(warpedVector, 10, { genre: "comedy" });
   * await index.query(query);
   *
   * @param {number[] | Float32Array} vector 検索クエリベクトル
   * @param {number} topK 取得する件数 (デフォルト: 10)
   * @param {Record<string, any>} [filter] メタデータフィルタ（オプション）
   * @returns {Record<string, any>} Pineconeのqueryメソッド用オブジェクト
   */
  public static toPineconeQuery(
    vector: number[] | Float32Array | Int8Array | Uint8Array,
    topK: number = 10,
    filter?: Record<string, unknown>,
  ): Record<string, unknown> {
    return {
      vector: Array.from(vector),
      topK,
      ...(filter ? { filter } : {}),
    };
  }

  /**
   * Redis (RediSearch) のベクトルフィールド用に、
   * Float32Array をバイナリ（Uint8Array）に変換します。
   * Node.js環境では Buffer.from() を使って Buffer に変換して渡してください。
   *
   * @example
   * const blob = Buffer.from(VectorDBAdapter.toRedis(warpedVector));
   * await redis.call('FT.SEARCH', 'idx', '*=>[KNN 5 @embedding $BLOB]', 'PARAMS', '2', 'BLOB', blob, 'DIALECT', '2');
   *
   * @param {number[] | Float32Array} vector ワープ変換後のベクトル
   * @returns {Uint8Array} バイナリデータ (Float32のバイト表現)
   */
  public static toRedis(vector: number[] | Float32Array | Int8Array | Uint8Array): Uint8Array {
    const f32Array =
      vector instanceof Float32Array ? vector : new Float32Array(Array.from(vector));
    // 新しいバッファのコピーを返すことで副作用を防ぐ
    return new Uint8Array(
      f32Array.buffer.slice(
        f32Array.byteOffset,
        f32Array.byteOffset + f32Array.byteLength,
      ),
    );
  }
}
