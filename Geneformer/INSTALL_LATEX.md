# LaTeX環境のインストール方法

## 方法1: apt-getでインストール（推奨）

```bash
# rootユーザーで実行
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# または、より完全なインストール（時間がかかりますが、より多くのパッケージが含まれます）
sudo apt-get install -y texlive-full
```

## 方法2: 最小限のインストール

```bash
# 基本的なLaTeX環境のみ
sudo apt-get install -y texlive-latex-base
```

## 方法3: Dockerを使用（推奨、システムに影響なし）

```bash
# Dockerイメージを使用してLaTeXをコンパイル
docker run --rm -v "$(pwd)/submission_package":/workdir texlive/texlive:latest \
  pdflatex -interaction=nonstopmode FINAL_REPORT.tex
```

## 方法4: オンラインサービスを使用

1. **Overleaf** (https://www.overleaf.com/)
   - `FINAL_REPORT.tex`をアップロード
   - ブラウザでPDFを生成・ダウンロード

2. **ShareLaTeX** (https://www.sharelatex.com/)
   - 同様にオンラインでコンパイル可能

## 現在の状況

- ✅ LaTeXファイル (`FINAL_REPORT.tex`) は作成済み
- ✅ 数式と図表も含まれている
- ⏳ PDF生成にはLaTeX環境が必要

## 推奨アクション

1. **apt-getでインストールを続行**（最も簡単）
   ```bash
   sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
   ```

2. **インストール後、PDFを生成**
   ```bash
   cd /home/nishioka/LUH/BioLLM/Geneformer
   ./compile_latex.sh
   ```
