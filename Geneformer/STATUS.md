# 評価実行状況

## 現在の実行状況

### ✅ 実行中
- **Geneformer Fine-tuning (PBMC3k)**: 実行中
  - ログ: `logs/geneformer_finetune_v6.log`
  - プロセスID: 1420346
  - ステータス: トレーニング進行中（約1-2時間かかる見込み）

### ✅ 完了
- **Geneformer Frozen (PBMC3k)**: 完了
  - Accuracy: 0.613, Macro F1: 0.428
  - 結果: `results/metrics_geneformer_pbmc3k.csv`

- **scGPT Frozen (PBMC3k)**: 完了（既存結果）
  - Accuracy: 0.600, Macro F1: 0.294
  - 結果: `results/metrics_scgpt.csv`

### ⏳ 待機中
- **scGPT Fine-tuning (PBMC3k)**: スクリプト準備済み（torchtext問題の可能性あり）
- **Tabula Sapiens評価**: スクリプト準備済み（データセット未ダウンロード）
- **scFoundation評価**: スクリプト準備済み（モデル利用不可）

## ログファイル

すべてのログは `logs/` ディレクトリに保存されています：
- `logs/geneformer_finetune_v6.log` - Geneformer Fine-tuning（実行中）
- `logs/complete_pipeline.log` - 統合パイプラインログ

## 進行状況の確認

```bash
# 実行中のプロセス確認
ps aux | grep "python.*run_"

# 最新のログ確認
tail -f logs/geneformer_finetune_v6.log

# すべてのログ確認
ls -lht logs/*.log
```

## 次のステップ

1. Geneformer Fine-tuningの完了を待つ（約1-2時間）
2. 完了後、scGPT Fine-tuningを実行（torchtext問題がある可能性）
3. Tabula Sapiensデータセットをダウンロードして評価実行
4. すべての結果を統合して最終レポート生成
