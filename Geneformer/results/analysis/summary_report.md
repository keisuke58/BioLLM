
# Geneformer vs scGPT 評価結果レポート

## 概要
PBMC3kデータセットでの細胞タイプ分類タスクにおける、GeneformerとscGPTの性能比較結果。

## メトリクス比較

### 全体性能
- **Geneformer**
  - Accuracy: 0.570
  - Macro F1: 0.391
  - Macro F1 (Platelet除外): 0.469

- **scGPT**
  - Accuracy: 0.600
  - Macro F1: 0.294

### 主な所見
1. **Accuracy**: scGPTが0.600で、Geneformerの0.570より高い。

2. **Macro F1**: Geneformerが0.391で、scGPTの0.294より高い。

3. **Platelet除外**: GeneformerではPlateletを除外した場合のMacro F1が0.469と、全体より改善している。

## 可視化ファイル
- UMAP可視化: `results/umap_*.png`
- 混同行列: `results/confusion_*.png`
- メトリクス比較: `results/analysis/metrics_comparison.png`

## 次のステップ
1. クラスごとの詳細分析（Precision, Recall, F1）
2. 誤分類パターンの分析
3. ハイパーパラメータの調整
4. 他のデータセットでの評価
