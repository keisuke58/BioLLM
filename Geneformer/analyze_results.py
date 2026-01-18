#!/usr/bin/env python3
"""
結果の詳細分析と比較レポート生成スクリプト
Geneformer vs scGPT の性能比較と詳細分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

# 設定
RES_DIR = Path("results")
OUTPUT_DIR = RES_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def load_metrics():
    """メトリクスCSVを読み込む"""
    geneformer_metrics = pd.read_csv(RES_DIR / "metrics_geneformer_pbmc3k.csv")
    scgpt_metrics = pd.read_csv(RES_DIR / "metrics_scgpt.csv")
    return geneformer_metrics, scgpt_metrics

def create_comparison_table(geneformer_metrics, scgpt_metrics):
    """比較テーブルを作成"""
    # Geneformerのメトリクスを安全に取得
    gf_acc = geneformer_metrics[geneformer_metrics["metric"] == "accuracy"]["value"].values[0]
    gf_f1 = geneformer_metrics[geneformer_metrics["metric"] == "macro_f1"]["value"].values[0]
    
    # macro_f1_no_plateletの取得（存在する場合）
    gf_f1_no_platelet = None
    if "macro_f1_no_platelet" in geneformer_metrics["metric"].values:
        gf_f1_no_platelet = geneformer_metrics[geneformer_metrics["metric"] == "macro_f1_no_platelet"]["value"].values[0]
    
    comparison = pd.DataFrame({
        "Model": ["Geneformer", "scGPT"],
        "Accuracy": [
            gf_acc,
            scgpt_metrics["accuracy"].values[0]
        ],
        "Macro F1": [
            gf_f1,
            scgpt_metrics["macro_f1"].values[0]
        ],
        "Macro F1 (no Platelet)": [
            gf_f1_no_platelet,
            None  # scGPTにはこの指標がない
        ]
    })
    
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n=== モデル比較 ===")
    print(comparison.to_string(index=False))
    return comparison

def plot_metrics_comparison(comparison):
    """メトリクス比較の可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics_to_plot = ["Accuracy", "Macro F1"]
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = comparison[metric].values
        models = comparison["Model"].values
        
        bars = ax.bar(models, values, color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(values) * 1.2])
        
        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'metrics_comparison.png'}")

def analyze_confusion_matrices():
    """混同行列から詳細分析を実行"""
    # 混同行列の画像からは直接データを取得できないため、
    # 実際の予測結果が必要な場合は、元のスクリプトを修正して予測結果を保存する必要がある
    print("\n=== 混同行列分析 ===")
    print("混同行列の画像は以下に保存されています:")
    print(f"  - {RES_DIR / 'confusion_geneformer_pbmc3k.png'}")
    print(f"  - {RES_DIR / 'confusion_scgpt.png'}")
    print("\n詳細なクラスごとの分析には、予測結果のCSVが必要です。")

def create_summary_report(geneformer_metrics, scgpt_metrics, comparison):
    """サマリーレポートを作成"""
    # 値を安全に取得
    gf_acc = comparison[comparison['Model']=='Geneformer']['Accuracy'].values[0]
    gf_f1 = comparison[comparison['Model']=='Geneformer']['Macro F1'].values[0]
    gf_f1_no_platelet_val = comparison[comparison['Model']=='Geneformer']['Macro F1 (no Platelet)'].values[0]
    gf_f1_no_platelet_str = f"{gf_f1_no_platelet_val:.3f}" if pd.notna(gf_f1_no_platelet_val) else 'N/A'
    
    scgpt_acc = comparison[comparison['Model']=='scGPT']['Accuracy'].values[0]
    scgpt_f1 = comparison[comparison['Model']=='scGPT']['Macro F1'].values[0]
    
    report = f"""
# Geneformer vs scGPT 評価結果レポート

## 概要
PBMC3kデータセットでの細胞タイプ分類タスクにおける、GeneformerとscGPTの性能比較結果。

## メトリクス比較

### 全体性能
- **Geneformer**
  - Accuracy: {gf_acc:.3f}
  - Macro F1: {gf_f1:.3f}
  - Macro F1 (Platelet除外): {gf_f1_no_platelet_str}

- **scGPT**
  - Accuracy: {scgpt_acc:.3f}
  - Macro F1: {scgpt_f1:.3f}

### 主な所見
1. **Accuracy**: scGPTが{scgpt_acc:.3f}で、Geneformerの{gf_acc:.3f}より{'高い' if scgpt_acc > gf_acc else '低い'}。

2. **Macro F1**: Geneformerが{gf_f1:.3f}で、scGPTの{scgpt_f1:.3f}より{'高い' if gf_f1 > scgpt_f1 else '低い'}。

3. **Platelet除外**: GeneformerではPlateletを除外した場合のMacro F1が{gf_f1_no_platelet_str}と、全体より改善している。

## 可視化ファイル
- UMAP可視化: `results/umap_*.png`
- 混同行列: `results/confusion_*.png`
- メトリクス比較: `results/analysis/metrics_comparison.png`

## 次のステップ
1. クラスごとの詳細分析（Precision, Recall, F1）
2. 誤分類パターンの分析
3. ハイパーパラメータの調整
4. 他のデータセットでの評価
"""
    
    with open(OUTPUT_DIR / "summary_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'summary_report.md'}")
    print("\n" + "="*60)
    print(report)
    print("="*60)

def main():
    print("="*60)
    print("結果分析スクリプト")
    print("="*60)
    
    try:
        # メトリクス読み込み
        geneformer_metrics, scgpt_metrics = load_metrics()
        
        # 比較テーブル作成
        comparison = create_comparison_table(geneformer_metrics, scgpt_metrics)
        
        # 可視化
        plot_metrics_comparison(comparison)
        
        # 混同行列分析
        analyze_confusion_matrices()
        
        # サマリーレポート作成
        create_summary_report(geneformer_metrics, scgpt_metrics, comparison)
        
        print(f"\n[完了] 分析結果は {OUTPUT_DIR} に保存されました。")
    except FileNotFoundError as e:
        print(f"\n[エラー] ファイルが見つかりません: {e}")
        print("結果ファイルが存在することを確認してください。")
    except Exception as e:
        print(f"\n[エラー] 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
