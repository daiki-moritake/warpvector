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

/**
 * MLP (多層パーセプトロン) の順伝播推論を行うWASMコア関数。
 * 
 * @param {usize} inputPtr - 入力ベクトル(x)のポインタ (f32)
 * @param {usize} outputPtr - 出力ベクトルを書き込むポインタ (f32)
 * @param {usize} weightsPtr - 全レイヤーの重み(W)とバイアス(b)が連結されたポインタ (f32)
 * @param {usize} layerDimsPtr - [入力次元, 隠れ層1次元, 隠れ層2次元, ..., 出力次元] の配列ポインタ (i32)
 * @param {usize} activationsPtr - 各層の活性化関数IDの配列ポインタ (i32) [0:Linear, 1:ReLU, 2:Sigmoid, 3:Tanh]
 * @param {i32} numLayers - レイヤー数 (重み行列の数)
 * @param {usize} bufferPtr - 中間計算用バッファAのポインタ (f32)
 * @param {usize} bufBPtr - 中間計算用バッファBのポインタ (f32)
 */
export function mlpInferenceWasm(
  inputPtr: usize,
  outputPtr: usize,
  weightsPtr: usize,
  layerDimsPtr: usize,
  activationsPtr: usize,
  numLayers: i32,
  bufferPtr: usize,
  bufBPtr: usize
): void {
  // バッファを2つの領域 (bufA, bufB) に分けて、層ごとに入出力をスワップしながら計算する
  let bufA = bufferPtr;
  let bufB = bufBPtr;

  // 初回は inputPtr から bufA にコピー
  let inputDim = load<i32>(layerDimsPtr);
  for (let i = 0; i < inputDim; i++) {
    let val = load<f32>(inputPtr + i * 4);
    store<f32>(bufA + i * 4, val);
  }

  let currentWeightsOffset = weightsPtr;

  for (let l = 0; l < numLayers; l++) {
    let sDim = load<i32>(layerDimsPtr + l * 4);
    let tDim = load<i32>(layerDimsPtr + (l + 1) * 4);
    let activation = load<i32>(activationsPtr + l * 4);

    let inBuf = (l % 2 == 0) ? bufA : bufB;
    let outBuf = (l % 2 == 0) ? bufB : bufA;

    // 行列ベクトル積とバイアス加算
    for (let i = 0; i < tDim; i++) {
      let sum: f32 = 0;
      for (let j = 0; j < sDim; j++) {
        let m = load<f32>(currentWeightsOffset);
        let x_val = load<f32>(inBuf + j * 4);
        sum += m * x_val;
        currentWeightsOffset += 4;
      }
      
      let b = load<f32>(currentWeightsOffset);
      sum += b;
      currentWeightsOffset += 4;

      // 活性化関数
      if (activation == 1) {
        // ReLU
        if (sum < 0.0) sum = 0.0;
      } else if (activation == 2) {
        // Sigmoid: 1 / (1 + exp(-x))
        sum = 1.0 / (1.0 + Math.exp(-sum) as f32);
      } else if (activation == 3) {
        // Tanh
        let exp2x = Math.exp(2.0 * sum) as f32;
        sum = (exp2x - 1.0) / (exp2x + 1.0);
      }
      // activation == 0 (Linear) の場合は何もしない

      store<f32>(outBuf + i * 4, sum);
    }
  }

  // 最終結果を outputPtr にコピー
  let finalOutBuf = (numLayers % 2 == 0) ? bufA : bufB;
  let outputDim = load<i32>(layerDimsPtr + numLayers * 4);
  for (let i = 0; i < outputDim; i++) {
    let val = load<f32>(finalOutBuf + i * 4);
    store<f32>(outputPtr + i * 4, val);
  }
}

/**
 * Late Interaction (ColBERT) の MaxSim スコアを計算するWASMコア関数。
 * クエリ行列 (N x d) とドキュメント行列 (M x d) の総当たり内積を計算し、
 * クエリの各トークンに対するドキュメントトークンの最大内積 (Max) を取得、
 * 最後にそれらを合計 (Sum) してスコアを返す。
 * 
 * @param {usize} queryPtr - クエリ行列のポインタ (f32)
 * @param {usize} docPtr - ドキュメント行列のポインタ (f32)
 * @param {i32} queryTokens - クエリのトークン数 (N)
 * @param {i32} docTokens - ドキュメントのトークン数 (M)
 * @param {i32} dim - ベクトルの次元数 (d)
 * @returns {f32} MaxSim スコア
 */
export function colbertMaxSimWasm(
  queryPtr: usize,
  docPtr: usize,
  queryTokens: i32,
  docTokens: i32,
  dim: i32
): f32 {
  if (queryTokens <= 0 || docTokens <= 0) return 0.0;

  let totalScore: f32 = 0.0;

  for (let i = 0; i < queryTokens; i++) {
    let qOffset = queryPtr + (i * dim) * 4;
    let maxSim: f32 = -1e30; // 負の無限大の代わり

    for (let j = 0; j < docTokens; j++) {
      let dOffset = docPtr + (j * dim) * 4;
      let sim: f32 = 0.0;

      // 内積の計算
      for (let k = 0; k < dim; k++) {
        let qVal = load<f32>(qOffset + k * 4);
        let dVal = load<f32>(dOffset + k * 4);
        sim += qVal * dVal;
      }

      if (sim > maxSim) {
        maxSim = sim;
      }
    }

    totalScore += maxSim;
  }

  return totalScore;
}

/**
 * プロジェクション（次元削減・拡張）を行うWASMコア関数。
 * 
 * @param {usize} matrixPtr - 変換行列(W)のポインタ (f32)
 * @param {usize} biasPtr - バイアス(b)のポインタ (f32, 0の場合はバイアスなし)
 * @param {usize} inputPtr - 入力ベクトルのポインタ (f32)
 * @param {usize} outputPtr - 出力ベクトルのポインタ (f32)
 * @param {i32} inDim - 入力次元数
 * @param {i32} outDim - 出力次元数
 */
export function projectWasm(
  matrixPtr: usize,
  biasPtr: usize,
  inputPtr: usize,
  outputPtr: usize,
  inDim: i32,
  outDim: i32
): void {
  for (let i = 0; i < outDim; i++) {
    let sum: f32 = 0;
    let rowOffset = i * inDim;
    for (let j = 0; j < inDim; j++) {
      let m = load<f32>(matrixPtr + (rowOffset + j) * 4);
      let v = load<f32>(inputPtr + j * 4);
      sum += m * v;
    }
    
    if (biasPtr != 0) {
      let b = load<f32>(biasPtr + i * 4);
      sum += b;
    }
    store<f32>(outputPtr + i * 4, sum);
  }
}

/**
 * Sanger's Rule (GHA) によるオンラインPCAを計算するWASMコア関数。
 * 複数の主成分をストリーミングで学習します。
 * 
 * @param {usize} componentsPtr - 主成分ベクトル群のポインタ (f32, サイズ: numComponents * dim)
 * @param {usize} xResidualPtr - ゼロセンタリングされた入力ベクトル(x)のポインタ (f32)。計算中に更新されます。
 * @param {i32} dim - ベクトルの次元数
 * @param {i32} numComponents - 抽出・更新する主成分の数
 * @param {f32} lr - 学習率 (learning rate)
 */
export function sangerUpdateWasm(
  componentsPtr: usize,
  xResidualPtr: usize,
  dim: i32,
  numComponents: i32,
  lr: f32
): void {
  for (let k = 0; k < numComponents; k++) {
    let wOffset = componentsPtr + k * dim * 4;
    
    // y = w^T * x_residual
    let y: f32 = 0;
    for (let i = 0; i < dim; i++) {
      let w_val = load<f32>(wOffset + i * 4);
      let x_val = load<f32>(xResidualPtr + i * 4);
      y += w_val * x_val;
    }
    
    // Oja's rule: w = w + lr * y * (x_residual - y * w)
    let normSq: f32 = 0;
    let lry = lr * y;
    for (let i = 0; i < dim; i++) {
      let w_val = load<f32>(wOffset + i * 4);
      let x_val = load<f32>(xResidualPtr + i * 4);
      
      let w_new = w_val + lry * (x_val - y * w_val);
      store<f32>(wOffset + i * 4, w_new);
      normSq += w_new * w_new;
    }
    
    // Normalize w
    let norm = Math.sqrt(normSq as f64) as f32;
    if (norm > 0) {
      for (let i = 0; i < dim; i++) {
        let w_val = load<f32>(wOffset + i * 4);
        store<f32>(wOffset + i * 4, w_val / norm);
      }
    }
    
    // x_residual = x_residual - y * w
    for (let i = 0; i < dim; i++) {
      let x_val = load<f32>(xResidualPtr + i * 4);
      let w_val = load<f32>(wOffset + i * 4);
      store<f32>(xResidualPtr + i * 4, x_val - y * w_val);
    }
  }
}

/**
 * Adamオプティマイザの1ステップパラメータ更新をWASMで高速実行する。
 * アフィン変換層の重み(matrix)とバイアス(bias)を更新します。
 */
export function adamUpdateWasm(
  matrixPtr: usize,
  biasPtr: usize,
  mMatrixPtr: usize,
  vMatrixPtr: usize,
  mBiasPtr: usize,
  vBiasPtr: usize,
  inputPtr: usize,
  outputGradientsPtr: usize,
  lr: f32,
  reg: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  t: i32,
  sDim: i32,
  tDim: i32
): void {
  let beta1_t = Mathf.pow(beta1, t as f32);
  let beta2_t = Mathf.pow(beta2, t as f32);
  let one_minus_beta1_t = (1.0 as f32) - beta1_t;
  let one_minus_beta2_t = (1.0 as f32) - beta2_t;

  for (let i = 0; i < tDim; i++) {
    let bGrad = load<f32>(outputGradientsPtr + i * 4);

    // mBias update
    let m_b = load<f32>(mBiasPtr + i * 4);
    m_b = beta1 * m_b + ((1.0 as f32) - beta1) * bGrad;
    store<f32>(mBiasPtr + i * 4, m_b);

    // vBias update
    let v_b = load<f32>(vBiasPtr + i * 4);
    v_b = beta2 * v_b + ((1.0 as f32) - beta2) * bGrad * bGrad;
    store<f32>(vBiasPtr + i * 4, v_b);

    let mHatB = m_b / one_minus_beta1_t;
    let vHatB = v_b / one_minus_beta2_t;

    let b_val = load<f32>(biasPtr + i * 4);
    b_val -= (lr * mHatB) / (Mathf.sqrt(vHatB) + epsilon);
    store<f32>(biasPtr + i * 4, b_val);

    let rowOffset = i * sDim;
    for (let j = 0; j < sDim; j++) {
      let wIdx = rowOffset + j;
      let input_j = load<f32>(inputPtr + j * 4);
      let w_val = load<f32>(matrixPtr + wIdx * 4);
      let wGrad = bGrad * input_j + reg * w_val;

      // mMatrix update
      let m_w = load<f32>(mMatrixPtr + wIdx * 4);
      m_w = beta1 * m_w + ((1.0 as f32) - beta1) * wGrad;
      store<f32>(mMatrixPtr + wIdx * 4, m_w);

      // vMatrix update
      let v_w = load<f32>(vMatrixPtr + wIdx * 4);
      v_w = beta2 * v_w + ((1.0 as f32) - beta2) * wGrad * wGrad;
      store<f32>(vMatrixPtr + wIdx * 4, v_w);

      let mHatW = m_w / one_minus_beta1_t;
      let vHatW = v_w / one_minus_beta2_t;

      w_val -= (lr * mHatW) / (Mathf.sqrt(vHatW) + epsilon);
      store<f32>(matrixPtr + wIdx * 4, w_val);
    }
  }
}
