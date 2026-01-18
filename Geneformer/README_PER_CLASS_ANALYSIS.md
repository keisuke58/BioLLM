# クラス別詳細分析の実行方法

## 概要
`analyze_per_class.py` は、GeneformerとscGPTのクラス別性能を分析し、以下を生成します：

1. `results/analysis/per_class_geneformer.csv` - クラス別メトリクス（Geneformer）
2. `results/analysis/per_class_scgpt.csv` - クラス別メトリクス（scGPT）
3. `results/analysis/worst_classes.md` - F1下位5クラスと混同先Top3の分析

## 実行前の準備

### 必要なパッケージ
```bash
pip install numpy pandas scikit-learn xgboost scanpy
```

### 必要なファイル
以下のファイルが存在することを確認してください：
- `results/geneformer_emb/pbmc3k.csv` - Geneformer埋め込み
- `results/scgpt_emb.npy` - scGPT埋め込み
- `artifacts/h5ad/pbmc3k_for_geneformer.h5ad` - Geneformer用H5AD（または `pbmc3k_scgpt.h5ad`）

## 実行方法

```bash
cd /home/nishioka/LUH/BioLLM/Geneformer
python analyze_per_class.py
```

## 出力ファイル

### per_class_geneformer.csv / per_class_scgpt.csv
各クラスの詳細メトリクス：
- `class`: クラス名
- `support`: テストセット内のサンプル数
- `precision`: 適合率
- `recall`: 再現率
- `f1`: F1スコア

### worst_classes.md
F1下位5クラスの詳細分析：
- 各クラスのF1、Precision、Recall、Support
- 混同先Top3（どのクラスに誤分類されやすいか）

## 分析結果の活用

### 1. F1≈0のクラスを特定
`worst_classes.md` でF1が0または極端に低いクラスを確認

### 2. 改善策の選択

**特定クラスだけF1≈0（Platelet等）の場合：**
- そのクラスを重み付け
- 二段構え（rare vs others）の分類器を試す

**全体的に混ざっている場合：**
- 埋め込み品質の比較（距離/分離度）
- 分類器を強化（SVM、Random Forestなど）

**scGPTが多数派に吸われる場合：**
- `class_weight='balanced'` 付きLogisticRegressionを試す
- LinearSVMを試す

## トラブルシューティング

### ModuleNotFoundError
必要なパッケージがインストールされていない場合：
```bash
pip install numpy pandas scikit-learn xgboost scanpy
```

### FileNotFoundError
必要なファイルが存在しない場合：
1. `run_geneformer_pbmc3k.py` を実行してGeneformerの結果を生成
2. `run_scgpt_pbmc3k.py` を実行してscGPTの結果を生成

### 予測結果が一致しない
- 同じSEED（Geneformer: 42, scGPT: 0）で実行されていることを確認
- データの前処理が同じであることを確認
