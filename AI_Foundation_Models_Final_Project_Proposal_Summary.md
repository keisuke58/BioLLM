# AI Foundation Models in Biomedicine - Final Project Proposal 内容まとめ

**ファイル**: `AI_Foundation_Models_Final_Project_Proposal_Nishioka.pdf`  
**学生**: Keisuke Nishioka (Student ID: 10081049)  
**提出日**: January 18, 2026  
**コース**: AI Foundation Models in Biomedicine

---

## 📋 プロジェクト概要

### プロジェクトタイトル
**Understanding the Limits of Single-Cell Foundation Models on Downstream Tasks**

### 研究目的
単一細胞トランスクリプトミクスの代表的な基礎モデルを制御された解釈可能な方法で評価し、特にその限界に焦点を当てる。

---

## 🎯 動機と背景

### 背景
- トランスフォーマーアーキテクチャに基づく基礎モデルが単一細胞トランスクリプトミクスで強力なパラダイムとして登場
- **Geneformer**: 大規模な遺伝子発現シーケンスの事前学習により、多様な下流タスクへの転移可能な表現を実現
- **scGPT**: 生成的事前学習目標を導入し、複数の単一細胞アプリケーションで高い性能を示す
- **scFoundation**: 数千万のヒトトランスクリプトームで学習をスケールし、ベンチマークで競争力のある結果を報告

### 問題点
- 事前学習中にどの生物学的情報が堅牢にエンコードされるか不明
- どの条件下でこれらの表現が一般化に失敗するか不明
- 報告された性能は、異質なデータセット、前処理パイプライン、ファインチューニング戦略により比較が困難

### 仮説
基礎モデルは、下流データが事前学習分布と整合している場合に強い利点を提供するが、ドメインシフト下ではその利点が減少し、表現品質がタスク固有の最適化よりも決定的になる。

---

## 🔬 方法論

### 1. データセット

#### 主要評価データセット
- **10x Genomics PBMC 68k dataset**
  - 広く使用されている末梢血単核細胞ベンチマーク
  - キュレートされた細胞型アノテーション
  - 再現可能な比較とよく理解されたベースライン設定を提供

#### 一般化評価データセット
- **Tabula Sapiens human cell atlas**
  - 複数の臓器とドナーからの単一細胞トランスクリプトーム
  - クロスデータセット設定により、堅牢性とドメインシフト行動を明示的に調査

### 2. 評価モデル
- **Geneformer**
- **scGPT**
- **scFoundation**

### 3. 評価方法
- **統一された実験プロトコル**: すべてのモデルを同一の前処理、遺伝子選択、train/validation/test分割で評価
- **2つのレジームの比較**:
  1. **Frozen representations**: 軽量分類器を使用した凍結表現
  2. **Task-head fine-tuning**: 同一の最適化設定下でのタスクヘッドファインチューニング
- **設計の意図**: 表現品質の効果を下流最適化から分離

### 4. 評価指標
- **Accuracy** (精度)
- **Macro-F1 score** (マクロF1スコア)
- **層別分析**: 細胞型頻度による分析で、クラス不均衡に対する堅牢性を評価
- **埋め込み品質の検証**: 低次元可視化と混同行列分析により、系統的な失敗モードを特定

### 5. 一般化評価
- PBMCからTabula Sapiensへのクロスデータセット評価により、訓練分布を超えた一般化を明示的にテスト

---

## 📊 評価パイプライン

```
PBMC 68k / Tabula Sapiens
    ↓
統一前処理と固定分割
    ↓
Geneformer / scGPT / scFoundation
    ↓
埋め込み抽出
    ↓
細胞型分類器
    ↓
評価と分析
```

**重要なポイント**: 事前学習エンコーダー以外のすべてのコンポーネントを実験間で同一に保つ

---

## 🎯 期待される貢献

1. **再現可能で解釈可能な比較**: 代表的な単一細胞基礎モデルの比較
2. **標準化された評価**: 統一された評価プロトコルによる比較
3. **Frozen vs Fine-tuned**: 凍結表現とファインチューニングの比較
4. **ドメインシフト分析**: 分布外データでの性能評価
5. **実用的な洞察**: 
   - 事前学習が実質的な利点を提供する条件の明確化
   - 現在のモデルが不足している領域の特定
   - 実用的なモデル選択と将来の方法論開発への情報提供

---

## 📚 参考文献

1. **Geneformer**: C. V. Theodoris et al. Transfer learning enables predictions in network biology. *Nature*, 618(7965):616-624, 2023.

2. **scGPT**: H. Cui et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21(8):1470-1480, 2024.

3. **scFoundation**: M. Hao et al. Large-scale foundation model on single-cell transcriptomics. *Nature Methods*, 2024.

4. **PBMC 68k**: G. X. Y. Zheng et al. Massively parallel digital transcriptional profiling of single cells. *Nature Communications*, 8:14049, 2017.

5. **Tabula Sapiens**: Tabula Sapiens Consortium. The Tabula Sapiens: a multiple-organ, single-cell transcriptomic atlas of humans. *Science*, 376(6594):eabl4896, 2022.

---

## 📝 外部サポートとAIツール

コースポリシーに従い、執筆やコーディングに使用した外部サポートやAIベースのツールは最終レポートで明示的に開示される。

---

## 🔗 関連ファイル

- **プロポーザルPDF**: `AI_Foundation_Models_Final_Project_Proposal_Nishioka.pdf`
- **コース資料**: `Lectures/Final_Project.pdf` (一般的なプロジェクト要件)
- **試験対策**: `20_PROJECTS/ai-foundation-models-biomedicine-exam-preparation.md`

---

## ⚠️ 注意

このプロポーザルは**AI Foundation Models in Biomedicine**コースのFinal Project Proposalです。  
単一細胞トランスクリプトミクスの基礎モデル（Geneformer, scGPT, scFoundation）の評価に焦点を当てています。
