/**
 * ベクトルのバッチアフィン変換 (W * x + b) を行うWASMコア関数。
 * メモリレイアウトを直線的にアクセスすることで高速化を図る。
 * 
 * @param {usize} matrixPtr - 変換行列(W)が配置されているメモリの先頭ポインタ (Float32)
 * @param {usize} biasPtr - バイアス(b)が配置されているメモリの先頭ポインタ (Float32)
 * @param {usize} vectorsPtr - 入力ベクトル群(x)が配置されているメモリの先頭ポインタ (Float32)
 * @param {usize} resultsPtr - 結果を書き込むメモリの先頭ポインタ (Float32)
 * @param {i32} dim - ベクトルの次元数
 * @param {i32} batchSize - バッチ処理するベクトルの数
 * @returns {void}
 */
export function tuneBatchWasm(
  matrixPtr: usize,
  biasPtr: usize,
  vectorsPtr: usize,
  resultsPtr: usize,
  dim: i32,
  batchSize: i32
): void {
  // バッチごとのループ
  for (let k = 0; k < batchSize; k++) {
    // 出力ベクトルの次元ごとのループ
    for (let i = 0; i < dim; i++) {
      let sum: f32 = 0;
      let rowOffset = i * dim;
      let vectorOffset = k * dim;
      
      // 行列ベクトル積の計算ループ
      for (let j = 0; j < dim; j++) {
        // float32は4バイトなのでオフセットに4を掛ける
        let m = load<f32>(matrixPtr + (rowOffset + j) * 4);
        let v = load<f32>(vectorsPtr + (vectorOffset + j) * 4);
        sum += m * v;
      }
      
      // バイアスを加算して結果を保存
      let b = load<f32>(biasPtr + i * 4);
      store<f32>(resultsPtr + (k * dim + i) * 4, sum + b);
    }
  }
}
