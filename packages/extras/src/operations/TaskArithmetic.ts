import { IntentWeights } from "@warpvector/core";
import {
  flattenMatrix,
  assertDimension,
  addScaledVector,
} from "@warpvector/core";

export interface TaskConfig {
  /**
   * 追加するタスクの重み（学習済み IntentWeights）
   */
  weights: IntentWeights;
  /**
   * そのタスクを合成するスケール（重み付け）。
   * 1.0 が標準、0.5で半分、マイナスで逆効果。
   */
  scale: number;
}

/**
 * Task Arithmetic (モデルマージ) ユーティリティ
 *
 * 複数の学習済みアダプタの重み（IntentWeights）を「タスクベクトル」として扱い、
 * それらを合成（マージ）することで、推論時にオーバーヘッドのない新しい静的なアダプタ行列を生成します。
 */
export class TaskArithmetic {
  /**
   * 複数のタスク（IntentWeights）を合成し、新しい IntentWeights を作成します。
   * 数式: W_new = W_base + Σ scale_i * (W_i - W_base)
   *
   * @param tasks 合成するタスクのリスト（IntentWeights と スケールのペア）
   * @param baseIntent 基準となる IntentWeights。省略された場合は恒等行列（Identity）とゼロバイアスがベースになります。
   * @returns 完全にマージされた新しい IntentWeights
   */
  public static merge(
    tasks: TaskConfig[],
    baseIntent?: IntentWeights,
  ): IntentWeights {
    if (tasks.length === 0) {
      throw new Error("No tasks provided to merge.");
    }

    // 基準となる次元数を取得
    const dim = tasks[0].weights.bias.length;

    // ベース行列とベースバイアスを用意
    let baseMatrix: Float32Array;
    let baseBias: Float32Array;

    if (baseIntent) {
      assertDimension(baseIntent.bias, dim, "Base bias");
      baseBias = new Float32Array(baseIntent.bias);

      if (baseIntent.matrix instanceof Float32Array) {
        assertDimension(baseIntent.matrix, dim * dim, "Base matrix");
        baseMatrix = new Float32Array(baseIntent.matrix);
      } else {
        baseMatrix = flattenMatrix(baseIntent.matrix, dim, dim, "Base matrix");
      }
    } else {
      // ベースがない場合は、恒等行列（W=I）とゼロバイアス（b=0）をベースとする
      baseMatrix = new Float32Array(dim * dim);
      for (let i = 0; i < dim; i++) {
        baseMatrix[i * dim + i] = 1.0;
      }
      baseBias = new Float32Array(dim); // 初期値ゼロ
    }

    const newMatrix = new Float32Array(baseMatrix);
    const newBias = new Float32Array(baseBias);

    // 各タスクベクトルの差分を加算する
    for (let t = 0; t < tasks.length; t++) {
      const { weights, scale } = tasks[t];
      assertDimension(weights.bias, dim, `Task ${t} bias`);

      const taskBias = new Float32Array(weights.bias);
      let taskMatrix: Float32Array;
      if (weights.matrix instanceof Float32Array) {
        assertDimension(weights.matrix, dim * dim, `Task ${t} matrix`);
        taskMatrix = weights.matrix;
      } else {
        taskMatrix = flattenMatrix(
          weights.matrix,
          dim,
          dim,
          `Task ${t} matrix`,
        );
      }

      // ΔW = W_task - W_base
      // W_new = W_new + scale * ΔW
      addScaledVector(newMatrix, taskMatrix, scale);
      addScaledVector(newMatrix, baseMatrix, -scale);

      // Δb = b_task - b_base
      // b_new = b_new + scale * Δb
      addScaledVector(newBias, taskBias, scale);
      addScaledVector(newBias, baseBias, -scale);
    }

    return {
      matrix: newMatrix,
      bias: newBias,
    };
  }
}
