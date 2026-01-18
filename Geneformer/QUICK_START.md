# Quick Start Guide

## エラー解決: ModuleNotFoundError

ターミナルで以下のエラーが発生している場合：

```
ModuleNotFoundError: No module named 'scanpy'
```

## 解決方法

### 方法1: requirements.txtからインストール（推奨）

```bash
cd /home/nishioka/LUH/BioLLM/Geneformer
pip install -r requirements.txt
```

### 方法2: インストールスクリプトを使用

```bash
cd /home/nishioka/LUH/BioLLM/Geneformer
bash install_dependencies.sh
```

### 方法3: 個別にインストール

```bash
# 基本的なパッケージ
pip install scanpy anndata numpy pandas scikit-learn matplotlib scipy

# 機械学習
pip install xgboost umap-learn

# 深層学習
pip install torch transformers datasets

# scGPT
pip install scgpt
```

### 方法4: condaを使用（推奨、特にscanpy）

```bash
conda install -c conda-forge scanpy anndata
pip install -r requirements.txt
```

## インストール確認

```bash
python -c "import scanpy, numpy, pandas, sklearn, xgboost, torch; print('All packages OK')"
```

## その後

依存関係をインストールした後、再度評価スクリプトを実行：

```bash
python run_all_evaluations.py
```
