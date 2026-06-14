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

/**
 * 1ステップの確率的勾配降下法 (SGD + Momentum) を実行し、W と b をインプレース更新する。
 * (Prediction => Error calculation => Backpropagation => Update)
 * 
 * @param {usize} matrixPtr - 変換行列(W) (Float32)
 * @param {usize} biasPtr - バイアス(b) (Float32)
 * @param {usize} vMatrixPtr - 行列用 Momentum (vMatrix) (Float32)
 * @param {usize} vBiasPtr - バイアス用 Momentum (vBias) (Float32)
 * @param {usize} xPtr - 入力ベクトル (x) (Float32)
 * @param {usize} yPtr - ターゲットベクトル (y) (Float32)
 * @param {f32} lr - 学習率
 * @param {f32} reg - L2正則化係数
 * @param {f32} momentum - Momentum係数
 * @param {i32} sDim - ソース次元数
 * @param {i32} tDim - ターゲット次元数
 * @param {usize} predPtr - 予測結果を一時保存するバッファ (Float32)
 * @returns {void}
 */
export function sgdMomentumStepWasm(
  matrixPtr: usize,
  biasPtr: usize,
  vMatrixPtr: usize,
  vBiasPtr: usize,
  xPtr: usize,
  yPtr: usize,
  lr: f32,
  reg: f32,
  momentum: f32,
  sDim: i32,
  tDim: i32,
  predPtr: usize
): void {
  // 順伝播: pred = Wx + b
  for (let i = 0; i < tDim; i++) {
    let sum: f32 = 0;
    let rowOffset = i * sDim;
    for (let j = 0; j < sDim; j++) {
      let m = load<f32>(matrixPtr + (rowOffset + j) * 4);
      let x_val = load<f32>(xPtr + j * 4);
      sum += m * x_val;
    }
    let b = load<f32>(biasPtr + i * 4);
    store<f32>(predPtr + i * 4, sum + b);
  }

  // 逆伝播 & パラメータ更新 (Momentum を考慮)
  for (let i = 0; i < tDim; i++) {
    let pred_val = load<f32>(predPtr + i * 4);
    let y_val = load<f32>(yPtr + i * 4);
    let error: f32 = pred_val - y_val;

    // バイアスの更新: v_b = momentum * v_b - lr * dL/db
    let v_b = load<f32>(vBiasPtr + i * 4);
    v_b = momentum * v_b - lr * error;
    store<f32>(vBiasPtr + i * 4, v_b);

    let b_val = load<f32>(biasPtr + i * 4);
    store<f32>(biasPtr + i * 4, b_val + v_b);

    let rowOffset = i * sDim;
    for (let j = 0; j < sDim; j++) {
      let wIdx = rowOffset + j;
      let m_val = load<f32>(matrixPtr + wIdx * 4);
      let x_val = load<f32>(xPtr + j * 4);
      
      let wGrad: f32 = error * x_val + reg * m_val;
      
      let v_w = load<f32>(vMatrixPtr + wIdx * 4);
      v_w = momentum * v_w - lr * wGrad;
      store<f32>(vMatrixPtr + wIdx * 4, v_w);
      
      store<f32>(matrixPtr + wIdx * 4, m_val + v_w);
    }
  }
}

