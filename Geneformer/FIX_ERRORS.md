# エラー修正ガイド

## 現在のエラー

```
ModuleNotFoundError: No module named 'numpy'
```

または

```
エラー: 必要なパッケージがインストールされていません
不足しているパッケージ: numpy, pandas, scikit-learn, scanpy, xgboost
```

## 解決方法

### 方法1: pipでインストール

```bash
pip install numpy pandas scikit-learn scanpy xgboost
```

### 方法2: condaでインストール

```bash
conda install -c conda-forge numpy pandas scikit-learn scanpy xgboost
```

### 方法3: 仮想環境を使用している場合

元のスクリプト（`run_geneformer_pbmc3k.py`や`run_scgpt_pbmc3k.py`）が実行できている環境を確認：

```bash
# 元のスクリプトがどのPython環境で実行されているか確認
which python
python --version

# その環境で必要なパッケージをインストール
pip install numpy pandas scikit-learn scanpy xgboost
```

### 方法4: requirements.txtからインストール

```bash
pip install -r requirements.txt
```

## 確認方法

インストール後、以下で確認：

```bash
python -c "import numpy, pandas, sklearn, scanpy, xgboost; print('All packages OK')"
```

## その他のエラー

### FileNotFoundError

必要なファイルが存在しない場合：

1. `run_geneformer_pbmc3k.py` を実行してGeneformerの結果を生成
2. `run_scgpt_pbmc3k.py` を実行してscGPTの結果を生成

### 予測結果が一致しない

- 同じSEEDで実行されていることを確認（Geneformer: 42, scGPT: 0）
- データの前処理が同じであることを確認

## トラブルシューティング

### 環境の確認

```bash
# Pythonのパスを確認
which python
python --version

# インストール済みパッケージを確認
pip list | grep -E "numpy|pandas|scikit-learn|scanpy|xgboost"
```

### 別の環境を使用している場合

元のスクリプトが実行できている環境と同じ環境を使用してください。
