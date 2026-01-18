# ZIPファイル提出ガイド

## 📦 推奨ZIPファイル名

以下のいずれかの形式を推奨します：

### オプション1: シンプルな形式（推奨）
```
submission_package.zip
```

### オプション2: 学生情報を含む形式
```
FinalProject_Nishioka_10081049.zip
```

### オプション3: コース情報を含む形式
```
BioLLM_FinalProject_Submission.zip
```

## ✅ 推奨ファイル名

**最も推奨**: `FinalProject_Nishioka_10081049.zip`

**理由**:
- 学生IDが含まれている（提出時の識別が容易）
- ファイル名から提出者と内容が明確
- 一般的な学術提出物の命名規則に準拠

## 📋 ZIPファイルの作成方法

### 方法1: コマンドライン（推奨）

```bash
cd /home/nishioka/LUH/BioLLM/Geneformer
zip -r FinalProject_Nishioka_10081049.zip submission_package/ \
  -x "*.pdf" "*.png" "*.aux" "*.log" "*.out" "*.h5ad" "*.dataset"
```

### 方法2: 重いファイルを除外して作成

```bash
cd /home/nishioka/LUH/BioLLM/Geneformer
zip -r FinalProject_Nishioka_10081049.zip submission_package/ \
  -x "submission_package/*.pdf" \
  -x "submission_package/results/figures/*.png" \
  -x "submission_package/*.aux" \
  -x "submission_package/*.log" \
  -x "submission_package/*.out"
```

## 📁 ZIPファイルに含まれる内容

### ✅ 含まれるファイル

- `FINAL_REPORT.md` - 最終レポート（Markdown形式）
- `FINAL_REPORT.tex` - 最終レポート（LaTeX形式）
- `README.md` - プロジェクト説明
- `SUBMISSION_README.txt` - 提出物説明
- `FILE_LIST.txt` - ファイル一覧
- `code/` - すべての評価スクリプト（7ファイル）
- `results/*.csv` - 結果データ（5ファイル）

### ❌ 除外されるファイル（重いファイル）

- `*.pdf` - PDFファイル（LaTeXから生成可能）
- `*.png` - 画像ファイル（図表）
- `*.aux`, `*.log`, `*.out` - LaTeXコンパイル中間ファイル
- `*.h5ad`, `*.dataset` - データファイル

## 📊 ファイルサイズ

- **ZIPファイルサイズ**: 約100-200KB（重いファイル除外後）
- **展開後サイズ**: 約1-2MB

## ✅ 提出前の確認

1. **ファイル名の確認**
   - [x] 学生IDが含まれている
   - [x] ファイル名が明確で識別しやすい

2. **内容の確認**
   - [x] レポートファイルが含まれている
   - [x] コードファイルが含まれている
   - [x] 結果データが含まれている
   - [x] README.mdが含まれている

3. **除外ファイルの確認**
   - [x] 重いファイル（PDF、PNG）が除外されている
   - [x] 中間ファイル（.aux、.log）が除外されている

## 🎯 提出時の注意事項

1. **レポート形式**: 
   - Markdown形式（`FINAL_REPORT.md`）が含まれています
   - LaTeX形式（`FINAL_REPORT.tex`）も含まれています
   - PDFが必要な場合は、LaTeXファイルから生成できます

2. **図表について**:
   - PNGファイルは除外されていますが、LaTeXファイルには図表のパスが記載されています
   - 必要に応じて、図表ファイルを別途提出するか、LaTeXからPDFを生成してください

3. **コードの実行**:
   - すべてのコードファイルが含まれています
   - README.mdに実行方法が記載されています

## 📝 提出チェックリスト

- [x] ZIPファイル名が適切（学生IDを含む）
- [x] レポートファイルが含まれている
- [x] コードファイルが含まれている
- [x] 結果データが含まれている
- [x] README.mdが含まれている
- [x] 重いファイルが除外されている
- [x] ファイルサイズが適切（100-200KB）

---

**推奨ZIPファイル名**: `FinalProject_Nishioka_10081049.zip` ✅
