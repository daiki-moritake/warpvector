# 🌌 v0.1.7 — Adaptive Feedback Loop & Federated Aggregation / 適応型フィードバックループと連合集約

In this release (v0.1.7), we added the "Feedback Loop" feature, which collects implicit user feedback (clicks, skips, dwell time) to optimize search vectors in real-time.

本リリース（v0.1.7）では、ユーザーの検索行動（クリック、スキップ、滞在時間）から暗黙的なフィードバックを収集し、検索ベクトルをリアルタイムに最適化する「フィードバックループ機能」を追加しました。

---

## 🇬🇧 English Release Notes

### ✨ Highlights & New Features

- **`FeedbackCollector`**: Automatically converts impressions and user interaction histories into training datasets for Triplet and InfoNCE loss models.
- **`AdaptiveScheduler`**: Manages automatic decay of the learning rate and buffers incoming interactions for efficient scheduled batch training.
- **`FederatedAggregator`**: Aggregates weight updates learned across multiple edge clients/users using the Federated Averaging (FedAvg) algorithm on the server.

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト & 新機能

- **`FeedbackCollector`**: インプレッションとユーザーの操作履歴を Triplet / InfoNCE 用の学習データへ自動変換。
- **`AdaptiveScheduler`**: 自動的な学習率の減衰と、バッファリングによる効率的なバッチ学習のスケジュール管理。
- **`FederatedAggregator`**: エッジ・クライアント側で学習された複数ユーザーの重み差分を、FedAvgアルゴリズムでサーバーに集約。
