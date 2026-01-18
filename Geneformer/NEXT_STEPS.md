# 次のステップ - 結果分析と改善提案

## 現在の状況
- ✅ GeneformerとscGPTの評価が完了
- ✅ メトリクスと可視化が生成済み
- ✅ 基本的な比較結果が確認済み

## 推奨される次のステップ

### 1. **詳細な結果分析** ⭐ 推奨
分析スクリプト `analyze_results.py` を作成しました。実行するには：
```bash
# 必要なパッケージをインストール（まだの場合）
pip install numpy pandas matplotlib seaborn scikit-learn

# 分析スクリプトを実行
python analyze_results.py
```

これにより以下が生成されます：
- モデル比較テーブル（CSV）
- メトリクス比較の可視化
- サマリーレポート（Markdown）

### 2. **クラスごとの詳細分析**
各細胞タイプごとの性能（Precision, Recall, F1）を分析：
- どのクラスで誤分類が多いか
- クラス間の混同パターン
- 少数クラス（Plateletなど）の扱い

### 3. **誤分類パターンの可視化**
- 誤分類された細胞のUMAP上での分布
- 混同行列の詳細分析
- エラーケースの特徴抽出

### 4. **ハイパーパラメータ調整**
現在の設定：
- **Geneformer**: XGBoost (n_estimators=400, max_depth=6, learning_rate=0.05)
- **scGPT**: LogisticRegression (max_iter=5000, solver='lbfgs')

改善の余地：
- グリッドサーチやランダムサーチで最適パラメータ探索
- 異なる分類器（Random Forest, SVMなど）の試行

### 5. **他のデータセットでの評価**
- より大きなデータセット（PBMC10kなど）
- 異なる組織・細胞タイプ
- 異なる実験条件

### 6. **埋め込み空間の分析**
- t-SNEやPCAでの可視化
- クラス間距離の定量化
- 埋め込み次元の削減分析

### 7. **モデルの統合・アンサンブル**
- GeneformerとscGPTの埋め込みを結合
- アンサンブル分類器の構築
- 重み付き投票

## 現在の結果サマリー

### Geneformer
- Accuracy: 0.57
- Macro F1: 0.39 (全体) / 0.47 (Platelet除外)

### scGPT
- Accuracy: 0.60
- Macro F1: 0.29

### 主な所見
1. scGPTはAccuracyが高いが、Macro F1は低い（クラス間の不均衡の影響）
2. GeneformerはMacro F1が高く、よりバランスの取れた性能
3. Platelet除外により、Geneformerの性能が改善

## すぐに実行できること

### オプションA: 詳細分析の実行
```bash
# 環境の確認とパッケージインストール
python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('All OK')" || pip install numpy pandas matplotlib seaborn scikit-learn

# 分析実行
python analyze_results.py
```

### オプションB: 混同行列の詳細確認
既存の可視化ファイルを確認：
- `results/confusion_geneformer_pbmc3k.png`
- `results/confusion_scgpt.png`
- `results/umap_*.png`

### オプションC: ハイパーパラメータ調整
`run_geneformer_pbmc3k.py` と `run_scgpt_pbmc3k.py` を修正して、異なるパラメータで再実行

## 質問・要望
どのステップから始めますか？特定の分析や改善に興味があれば教えてください。
