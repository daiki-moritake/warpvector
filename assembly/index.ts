/**
 * 2つのベクトルの内積をSIMDを用いて高速に計算するインライン関数
 */
// @ts-ignore
@inline
function innerProductSimd(ptr1: usize, ptr2: usize, dim: i32): f32 {
  let sumVec0 = f32x4.splat(0.0);
  let sumVec1 = f32x4.splat(0.0);
  let sumVec2 = f32x4.splat(0.0);
  let sumVec3 = f32x4.splat(0.0);
  let j: i32 = 0;

  for (; j <= dim - 16; j += 16) {
    let v1_0 = v128.load(ptr1 + (j) * 4);
    let v2_0 = v128.load(ptr2 + (j) * 4);
    let v1_1 = v128.load(ptr1 + (j + 4) * 4);
    let v2_1 = v128.load(ptr2 + (j + 4) * 4);
    let v1_2 = v128.load(ptr1 + (j + 8) * 4);
    let v2_2 = v128.load(ptr2 + (j + 8) * 4);
    let v1_3 = v128.load(ptr1 + (j + 12) * 4);
    let v2_3 = v128.load(ptr2 + (j + 12) * 4);

    sumVec0 = f32x4.add(sumVec0, f32x4.mul(v1_0, v2_0));
    sumVec1 = f32x4.add(sumVec1, f32x4.mul(v1_1, v2_1));
    sumVec2 = f32x4.add(sumVec2, f32x4.mul(v1_2, v2_2));
    sumVec3 = f32x4.add(sumVec3, f32x4.mul(v1_3, v2_3));
  }

  let sumVec = f32x4.add(f32x4.add(sumVec0, sumVec1), f32x4.add(sumVec2, sumVec3));

  for (; j <= dim - 4; j += 4) {
    let v1 = v128.load(ptr1 + j * 4);
    let v2 = v128.load(ptr2 + j * 4);
    sumVec = f32x4.add(sumVec, f32x4.mul(v1, v2));
  }

  let sum: f32 = f32x4.extract_lane(sumVec, 0) +
                 f32x4.extract_lane(sumVec, 1) +
                 f32x4.extract_lane(sumVec, 2) +
                 f32x4.extract_lane(sumVec, 3);

  for (; j < dim; j++) {
    let v1 = load<f32>(ptr1 + j * 4);
    let v2 = load<f32>(ptr2 + j * 4);
    sum += v1 * v2;
  }
  return sum;
}

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
  // バッチごとのループを外側に、出力ベクトルの次元ごとのループを内側に
  for (let k = 0; k < batchSize; k++) {
    let vectorOffset = k * dim * 4;
    let resultOffset = k * dim * 4;
    for (let i = 0; i < dim; i++) {
      let rowOffset = i * dim * 4;
      let b = load<f32>(biasPtr + i * 4);
      let sum = innerProductSimd(matrixPtr + rowOffset, vectorsPtr + vectorOffset, dim);
      
      // バイアスを加算して結果を連続的に保存
      store<f32>(resultsPtr + resultOffset + i * 4, sum + b);
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
      let sum = innerProductSimd(currentWeightsOffset, inBuf, sDim);
      currentWeightsOffset += sDim * 4;
      
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
      // 内積の計算
      let sim = innerProductSimd(qOffset, dOffset, dim);

      if (sim > maxSim) {
        maxSim = sim;
      }
    }

    totalScore += maxSim;
  }

  return totalScore;
}

/**
 * Late Interaction (ColBERT) の MaxSim スコアを複数のドキュメントに対して一括計算するWASM関数（バッチ処理版）。
 * JS側とWASM側のコンテキストスイッチを最小化し、極限のパフォーマンスを引き出すために使用します。
 * 
 * @param {usize} queryPtr - クエリ行列のポインタ (f32)
 * @param {usize} docsPtr - ドキュメント群が連結された行列の先頭ポインタ (f32)
 * @param {usize} docTokensPtr - 各ドキュメントのトークン数(M_k)を格納した配列のポインタ (i32, サイズ: numDocs)
 * @param {usize} resultsPtr - 各ドキュメントに対する MaxSim スコアを格納する配列のポインタ (f32, サイズ: numDocs)
 * @param {i32} numDocs - ドキュメントの数
 * @param {i32} queryTokens - クエリのトークン数 (N)
 * @param {i32} dim - ベクトルの次元数 (d)
 */
export function colbertMaxSimBatchWasm(
  queryPtr: usize,
  docsPtr: usize,
  docTokensPtr: usize,
  resultsPtr: usize,
  numDocs: i32,
  queryTokens: i32,
  dim: i32
): void {
  let currentDocOffset = docsPtr;
  for (let d = 0; d < numDocs; d++) {
    let docTokens = load<i32>(docTokensPtr + d * 4);
    if (queryTokens <= 0 || docTokens <= 0) {
      store<f32>(resultsPtr + d * 4, 0.0);
    } else {
      let totalScore: f32 = 0.0;
      for (let i = 0; i < queryTokens; i++) {
        let qOffset = queryPtr + (i * dim) * 4;
        let maxSim: f32 = -1e30; // 負の無限大の代わり
        for (let j = 0; j < docTokens; j++) {
          let dOffset = currentDocOffset + (j * dim) * 4;
          let sim = innerProductSimd(qOffset, dOffset, dim);
          if (sim > maxSim) {
            maxSim = sim;
          }
        }
        totalScore += maxSim;
      }
      store<f32>(resultsPtr + d * 4, totalScore);
    }
    currentDocOffset += docTokens * dim * 4;
  }
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
    let rowOffset = i * inDim;
    let sum = innerProductSimd(matrixPtr + rowOffset * 4, inputPtr, inDim);
    
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
    let y = innerProductSimd(wOffset, xResidualPtr, dim);
    
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
    
    // SIMD最適化用のスプラット変数
    let v_bGrad = f32x4.splat(bGrad);
    let v_reg = f32x4.splat(reg);
    let v_beta1 = f32x4.splat(beta1);
    let v_one_minus_beta1 = f32x4.splat((1.0 as f32) - beta1);
    let v_beta2 = f32x4.splat(beta2);
    let v_one_minus_beta2 = f32x4.splat((1.0 as f32) - beta2);
    let v_one_minus_beta1_t = f32x4.splat(one_minus_beta1_t);
    let v_one_minus_beta2_t = f32x4.splat(one_minus_beta2_t);
    let v_lr = f32x4.splat(lr);
    let v_epsilon = f32x4.splat(epsilon);

    let j = 0;
    // SIMD loop (4要素同時処理)
    for (; j <= sDim - 4; j += 4) {
      let wIdx = rowOffset + j;
      let ptrOff = wIdx * 4;
      
      let v_input = v128.load(inputPtr + j * 4);
      let v_w_val = v128.load(matrixPtr + ptrOff);
      
      // wGrad = bGrad * input_j + reg * w_val
      let v_wGrad = f32x4.add(f32x4.mul(v_bGrad, v_input), f32x4.mul(v_reg, v_w_val));

      // mMatrix update
      let v_m_w = v128.load(mMatrixPtr + ptrOff);
      v_m_w = f32x4.add(f32x4.mul(v_beta1, v_m_w), f32x4.mul(v_one_minus_beta1, v_wGrad));
      v128.store(mMatrixPtr + ptrOff, v_m_w);

      // vMatrix update
      let v_v_w = v128.load(vMatrixPtr + ptrOff);
      v_v_w = f32x4.add(f32x4.mul(v_beta2, v_v_w), f32x4.mul(v_one_minus_beta2, f32x4.mul(v_wGrad, v_wGrad)));
      v128.store(vMatrixPtr + ptrOff, v_v_w);

      let v_mHatW = f32x4.div(v_m_w, v_one_minus_beta1_t);
      let v_vHatW = f32x4.div(v_v_w, v_one_minus_beta2_t);

      // w_val -= (lr * mHatW) / (sqrt(vHatW) + epsilon)
      let v_denom = f32x4.add(f32x4.sqrt(v_vHatW), v_epsilon);
      let v_update = f32x4.div(f32x4.mul(v_lr, v_mHatW), v_denom);
      v_w_val = f32x4.sub(v_w_val, v_update);
      v128.store(matrixPtr + ptrOff, v_w_val);
    }

    // Remainder loop (端数処理)
    for (; j < sDim; j++) {
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

/**
 * 候補ベクトル群から多重経路散乱用の遷移確率行列 (P) を構築するWASMコア関数。
 * 
 * @param {usize} vectorsPtr - 正規化済み候補ベクトル行列のポインタ (f32, サイズ: numDocs * dim)
 * @param {i32} numDocs - 候補ドキュメントの数 (N)
 * @param {i32} dim - ベクトルの次元数
 * @param {f32} threshold - 類似度のしきい値
 * @param {usize} pMatrixPtr - 出力される遷移確率行列のポインタ (f32, サイズ: N * N)
 */
export function buildMultipathTransitionMatrixWasm(
  vectorsPtr: usize,
  numDocs: i32,
  dim: i32,
  threshold: f32,
  pMatrixPtr: usize
): void {
  // まず類似度行列WをPのメモリ空間に一時的に作成し、行の和(D)を計算する
  let dArrayPtr = pMatrixPtr + numDocs * numDocs * 4; // 一時的なD配列をPの後に確保(TS側で確保済みの前提)
  
  // D配列を0で初期化
  for (let i = 0; i < numDocs; i++) {
    store<f32>(dArrayPtr + i * 4, 0.0);
  }

  // W(i, j) の計算 (対称性を利用)
  for (let i = 0; i < numDocs; i++) {
    let iOffset = vectorsPtr + i * dim * 4;
    for (let j = i + 1; j < numDocs; j++) {
      let jOffset = vectorsPtr + j * dim * 4;
      let sim = innerProductSimd(iOffset, jOffset, dim);
      let w = Math.max(0.0 as f64, (sim - threshold) as f64) as f32;

      if (w > 0) {
        // pMatrixPtr は N*N の1次元配列 (row-major)
        store<f32>(pMatrixPtr + (i * numDocs + j) * 4, w);
        store<f32>(pMatrixPtr + (j * numDocs + i) * 4, w);

        let d_i = load<f32>(dArrayPtr + i * 4);
        store<f32>(dArrayPtr + i * 4, d_i + w);
        
        let d_j = load<f32>(dArrayPtr + j * 4);
        store<f32>(dArrayPtr + j * 4, d_j + w);
      } else {
        store<f32>(pMatrixPtr + (i * numDocs + j) * 4, 0.0);
        store<f32>(pMatrixPtr + (j * numDocs + i) * 4, 0.0);
      }
    }
    // 対角成分 (自身への類似度) は0とする（自己ループは後で追加）
    store<f32>(pMatrixPtr + (i * numDocs + i) * 4, 0.0);
  }

  // W を D で割って P を構築
  // P_{ij} = j から i への遷移確率 = W_{ij} / D_j
  for (let j = 0; j < numDocs; j++) {
    let d_j = load<f32>(dArrayPtr + j * 4);
    if (d_j > 0) {
      for (let i = 0; i < numDocs; i++) {
        let w_ij = load<f32>(pMatrixPtr + (i * numDocs + j) * 4);
        store<f32>(pMatrixPtr + (i * numDocs + j) * 4, w_ij / d_j);
      }
    } else {
      // 孤立ノード：全て0にして、対角成分だけ1.0にする
      for (let i = 0; i < numDocs; i++) {
        store<f32>(pMatrixPtr + (i * numDocs + j) * 4, 0.0);
      }
      store<f32>(pMatrixPtr + (j * numDocs + j) * 4, 1.0);
    }
  }
}

/**
 * Random Walk with Restart (Power Iteration) を用いて多重経路散乱の定常場を計算するWASM関数。
 * 
 * @param {usize} pMatrixPtr - 遷移確率行列 (P) のポインタ (f32, サイズ: N * N)
 * @param {usize} s0Ptr - 初期状態ベクトル (S_0) のポインタ (f32, サイズ: N)
 * @param {usize} currentSPtr - 現在の状態ベクトル (S_t) のポインタ (f32, サイズ: N)
 * @param {usize} nextSPtr - 次の状態ベクトル (S_{t+1}) を一時保存するポインタ (f32, サイズ: N)
 * @param {i32} numDocs - ドキュメントの数 (N)
 * @param {f32} alpha - 多重散乱の減衰率 (0 ~ 1)
 * @param {i32} maxIterations - 最大ループ回数
 * @param {f32} tolerance - 収束判定の許容誤差
 */
export function multipathScatteringPowerIterationWasm(
  pMatrixPtr: usize,
  s0Ptr: usize,
  currentSPtr: usize,
  nextSPtr: usize,
  numDocs: i32,
  alpha: f32,
  maxIterations: i32,
  tolerance: f32
): void {
  let oneMinusAlpha = 1.0 as f32 - alpha;

  for (let iter = 0; iter < maxIterations; iter++) {
    let maxDiff: f32 = 0.0;

    for (let i = 0; i < numDocs; i++) {
      let pSum: f32 = 0.0;
      let rowOffset = pMatrixPtr + i * numDocs * 4;

      // Pのi行目とcurrentSの内積を計算 (SIMDを利用可能)
      // 注意: pMatrixPtr は row-major を前提としているため、P_{ij} = pMatrixPtr + i * numDocs + j となる。
      // これは i行目 が j番目のノードからの遷移確率であることを意味する。
      pSum = innerProductSimd(rowOffset, currentSPtr, numDocs);

      let s0_i = load<f32>(s0Ptr + i * 4);
      let nextS_i = alpha * pSum + oneMinusAlpha * s0_i;

      store<f32>(nextSPtr + i * 4, nextS_i);

      let currentS_i = load<f32>(currentSPtr + i * 4);
      let diff = Math.abs(nextS_i - currentS_i) as f32;
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }

    // currentS に nextS をコピー
    for (let i = 0; i < numDocs; i++) {
      let val = load<f32>(nextSPtr + i * 4);
      store<f32>(currentSPtr + i * 4, val);
    }

    if (maxDiff < tolerance) {
      break;
    }
  }
}

/**
 * TimeReversalReranker 用の類似度グラフ(W行列)と次数(D)を構築するWASM関数。
 * 
 * @param {usize} vectorsPtr - 候補ベクトル群 (N * dim)
 * @param {i32} numDocs - N
 * @param {i32} dim - 次元数
 * @param {f32} threshold - しきい値
 * @param {usize} wMatrixPtr - W行列の出力先 (N * N)
 * @param {usize} dArrayPtr - 次数配列(D)の出力先 (N)
 */
export function buildTimeReversalGraphWasm(
  vectorsPtr: usize,
  numDocs: i32,
  dim: i32,
  threshold: f32,
  wMatrixPtr: usize,
  dArrayPtr: usize
): void {
  // Dを0初期化
  for (let i = 0; i < numDocs; i++) {
    store<f32>(dArrayPtr + i * 4, 0.0);
  }

  for (let i = 0; i < numDocs; i++) {
    let iOffset = vectorsPtr + i * dim * 4;
    for (let j = i + 1; j < numDocs; j++) {
      let jOffset = vectorsPtr + j * dim * 4;
      let sim = innerProductSimd(iOffset, jOffset, dim);
      let w = Math.max(0.0 as f64, (sim - threshold) as f64) as f32;

      if (w > 0) {
        store<f32>(wMatrixPtr + (i * numDocs + j) * 4, w);
        store<f32>(wMatrixPtr + (j * numDocs + i) * 4, w);

        let d_i = load<f32>(dArrayPtr + i * 4);
        store<f32>(dArrayPtr + i * 4, d_i + w);

        let d_j = load<f32>(dArrayPtr + j * 4);
        store<f32>(dArrayPtr + j * 4, d_j + w);
      } else {
        store<f32>(wMatrixPtr + (i * numDocs + j) * 4, 0.0);
        store<f32>(wMatrixPtr + (j * numDocs + i) * 4, 0.0);
      }
    }
    // 対角成分
    store<f32>(wMatrixPtr + (i * numDocs + i) * 4, 0.0);
  }
}

/**
 * TimeReversalReranker 用のラプラシアン逆拡散ループを実行するWASM関数。
 * 
 * @param {usize} wMatrixPtr - 類似度行列 W (N * N)
 * @param {usize} dArrayPtr - 次数配列 D (N)
 * @param {usize} currentSPtr - 現在のスコア配列 S (N)
 * @param {usize} nextSPtr - 次のスコア配列を書き込むバッファ (N)
 * @param {i32} numDocs - N
 * @param {f32} tau - 時間反転の強さ（逆拡散パラメータ）
 * @param {i32} iterations - イテレーション数
 * @param {boolean} normalizeGraph - グラフ次数による正規化フラグ
 */
export function timeReversalIterationWasm(
  wMatrixPtr: usize,
  dArrayPtr: usize,
  currentSPtr: usize,
  nextSPtr: usize,
  numDocs: i32,
  tau: f32,
  iterations: i32,
  normalizeGraph: boolean
): void {
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < numDocs; i++) {
      let diffSum: f32 = 0.0;
      let s_i = load<f32>(currentSPtr + i * 4);
      let rowOffset = wMatrixPtr + i * numDocs * 4;

      for (let j = 0; j < numDocs; j++) {
        if (i === j) continue;
        let w_ij = load<f32>(rowOffset + j * 4);
        if (w_ij > 0) {
          let s_j = load<f32>(currentSPtr + j * 4);
          diffSum += w_ij * (s_i - s_j);
        }
      }

      if (normalizeGraph) {
        let d_i = load<f32>(dArrayPtr + i * 4);
        if (d_i > 0) {
          diffSum /= d_i;
        }
      }

      let next_s_i = Math.max(0.0 as f64, (s_i + tau * diffSum) as f64) as f32;
      store<f32>(nextSPtr + i * 4, next_s_i);
    }

    // copy nextS to currentS
    for (let i = 0; i < numDocs; i++) {
      let val = load<f32>(nextSPtr + i * 4);
      store<f32>(currentSPtr + i * 4, val);
    }
  }
}

/**
 * ベクトルを Int8 に量子化するWASM関数。
 * 
 * @param {usize} vectorPtr - 入力ベクトルのポインタ (f32, サイズ: dim)
 * @param {usize} outPtr - 出力先のポインタ (i8, サイズ: dynamic ? dim + 4 : dim)
 * @param {i32} dim - 次元数
 * @param {boolean} isDynamic - 動的スケールを使用するか
 */
export function quantizeToInt8Wasm(
  vectorPtr: usize,
  outPtr: usize,
  dim: i32,
  isDynamic: boolean
): void {
  let scale: f32 = 127.0 as f32;

  if (isDynamic) {
    let maxVal: f32 = 1e-8 as f32;
    let v_max = f32x4.splat(1e-8 as f32);
    let i = 0;
    
    // SIMD max loop
    for (; i <= dim - 4; i += 4) {
      let v = v128.load(vectorPtr + i * 4);
      let v_abs = f32x4.abs(v);
      v_max = f32x4.max(v_max, v_abs);
    }
    
    // Extract SIMD max
    maxVal = Math.max(f32x4.extract_lane(v_max, 0),
             Math.max(f32x4.extract_lane(v_max, 1),
             Math.max(f32x4.extract_lane(v_max, 2),
                      f32x4.extract_lane(v_max, 3)))) as f32;
                      
    // Remainder scalar max loop
    for (; i < dim; i++) {
      let val = Math.abs(load<f32>(vectorPtr + i * 4)) as f32;
      if (val > maxVal) {
        maxVal = val;
      }
    }

    scale = 127.0 as f32 / maxVal;
    
    // 最後に maxVal を f32 (リトルエンディアン) として dim バイト目から書き込む
    store<f32>(outPtr + dim, maxVal);
  }

  let v_scale = f32x4.splat(scale);
  let j = 0;

  // SIMD Quantization loop (16 elements per iteration)
  for (; j <= dim - 16; j += 16) {
    let v1 = v128.load(vectorPtr + j * 4);
    let v2 = v128.load(vectorPtr + (j + 4) * 4);
    let v3 = v128.load(vectorPtr + (j + 8) * 4);
    let v4 = v128.load(vectorPtr + (j + 12) * 4);

    let r1 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v1, v_scale)));
    let r2 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v2, v_scale)));
    let r3 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v3, v_scale)));
    let r4 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v4, v_scale)));

    let i16_1 = i16x8.narrow_i32x4_s(r1, r2);
    let i16_2 = i16x8.narrow_i32x4_s(r3, r4);

    let i8 = i8x16.narrow_i16x8_s(i16_1, i16_2);
    v128.store(outPtr + j, i8);
  }

  // Remainder loop
  for (; j < dim; j++) {
    let fval = load<f32>(vectorPtr + j * 4) * scale;
    let ival = Math.round(fval as f64) as i32;
    if (ival > 127) ival = 127;
    if (ival < -128) ival = -128;
    store<i8>(outPtr + j, ival as i8);
  }
}

/**
 * ベクトルを 1-bit バイナリに量子化（パッキング）するWASM関数。
 * 
 * @param {usize} vectorPtr - 入力ベクトルのポインタ (f32, サイズ: dim)
 * @param {usize} outPtr - 出力先のポインタ (u8, サイズ: dim / 8)
 * @param {i32} dim - 次元数 (8の倍数である必要がある)
 */
export function quantizeToBinaryWasm(
  vectorPtr: usize,
  outPtr: usize,
  dim: i32
): void {
  let zero = f32x4.splat(0.0 as f32);
  let byteIndex = 0;
  let i = 0;

  // SIMD Quantization loop (16 elements / 2 bytes per iteration)
  for (; i <= dim - 16; i += 16) {
    let v1 = v128.load(vectorPtr + i * 4);
    let v2 = v128.load(vectorPtr + (i + 4) * 4);
    let v3 = v128.load(vectorPtr + (i + 8) * 4);
    let v4 = v128.load(vectorPtr + (i + 12) * 4);

    let m1 = i32x4.bitmask(f32x4.gt(v1, zero));
    let m2 = i32x4.bitmask(f32x4.gt(v2, zero));
    let m3 = i32x4.bitmask(f32x4.gt(v3, zero));
    let m4 = i32x4.bitmask(f32x4.gt(v4, zero));

    let r1 = ((m1 & 1) << 3) | ((m1 & 2) << 1) | ((m1 & 4) >> 1) | ((m1 & 8) >> 3);
    let r2 = ((m2 & 1) << 3) | ((m2 & 2) << 1) | ((m2 & 4) >> 1) | ((m2 & 8) >> 3);
    let r3 = ((m3 & 1) << 3) | ((m3 & 2) << 1) | ((m3 & 4) >> 1) | ((m3 & 8) >> 3);
    let r4 = ((m4 & 1) << 3) | ((m4 & 2) << 1) | ((m4 & 4) >> 1) | ((m4 & 8) >> 3);

    let byte1 = (r1 << 4) | r2;
    let byte2 = (r3 << 4) | r4;

    store<u8>(outPtr + byteIndex, byte1 as u8);
    store<u8>(outPtr + byteIndex + 1, byte2 as u8);
    byteIndex += 2;
  }

  // Remainder loop (8 elements / 1 byte)
  for (; i < dim; i += 8) {
    let byte: u8 = 0;
    if (load<f32>(vectorPtr + i * 4) > 0) byte |= 128;
    if (load<f32>(vectorPtr + (i + 1) * 4) > 0) byte |= 64;
    if (load<f32>(vectorPtr + (i + 2) * 4) > 0) byte |= 32;
    if (load<f32>(vectorPtr + (i + 3) * 4) > 0) byte |= 16;
    if (load<f32>(vectorPtr + (i + 4) * 4) > 0) byte |= 8;
    if (load<f32>(vectorPtr + (i + 5) * 4) > 0) byte |= 4;
    if (load<f32>(vectorPtr + (i + 6) * 4) > 0) byte |= 2;
    if (load<f32>(vectorPtr + (i + 7) * 4) > 0) byte |= 1;
    
    store<u8>(outPtr + byteIndex, byte);
    byteIndex++;
  }
}


