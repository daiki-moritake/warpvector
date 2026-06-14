import { ProjectionAdapter } from '../src';

/**
 * サンプル3: モデルの次元削減 (Migration)
 * 
 * 古い高次元モデル（例: 1536次元）で作られたベクトルDBを維持しつつ、
 * 新しい軽量モデル（例: 512次元）のクエリベクトルを使って検索できるように、
 * リアルタイムで次元数を投影（プロジェクション）します。
 * 
 * ※ここでは見やすくするため 4次元 -> 2次元 の投影を行います。
 */

// 1. プロジェクション（投影）用の重みを定義
// 実際には MigrationTrainer で事前に学習した重みを使用します
const projectionWeights = {
  // 2行(出力次元) x 4列(入力次元) の行列
  matrix: [
    [0.5, 0.2, 0.1, 0.0],
    [0.0, -0.3, 0.8, 0.4],
  ],
  // 2次元のバイアス
  bias: [0.1, -0.1],
};

// 2. アダプターの初期化 (入力: 4次元, 出力: 2次元)
const projector = new ProjectionAdapter(4, 2, { myProjection: projectionWeights });

// 3. 古いモデル（4次元）から出力されたベクトル
const oldDimVector = [1.0, 0.5, -0.2, 0.8];

console.log("=== 次元圧縮 / マイグレーション ===");
console.log(`元のベクトル (${oldDimVector.length}次元):`, oldDimVector);

// 4. 次元を圧縮して投影する
const newDimVector = projector.project(oldDimVector, "myProjection");

console.log(`投影後のベクトル (${newDimVector.length}次元):`, Array.from(newDimVector));

console.log("\n💡 これにより、巨大なベクトルDBを再構築することなく、新しいモデルの次元数と接続できます！");
