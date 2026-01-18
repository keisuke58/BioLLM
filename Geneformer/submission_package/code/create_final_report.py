"""
Create final project report by aggregating all results.
"""
import os
import pandas as pd
from pathlib import Path
import json

RES_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    """Load all result CSV files."""
    results = {}
    
    # PBMC3k results
    pbmc3k_files = {
        "geneformer_frozen": "metrics_geneformer_pbmc3k.csv",
        "scgpt_frozen": "metrics_scgpt.csv",
        "geneformer_finetuned": "metrics_geneformer_finetuned_pbmc3k.csv",
        "scgpt_finetuned": "metrics_scgpt_finetuned_pbmc3k.csv",
        "scfoundation": "metrics_scfoundation_pbmc3k.csv",
    }
    
    for method, filename in pbmc3k_files.items():
        filepath = RES_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Handle different CSV formats
            if method == "geneformer_frozen":
                # Format: metric,value
                if "metric" in df.columns and "value" in df.columns:
                    acc_row = df[df["metric"] == "accuracy"]
                    f1_row = df[df["metric"] == "macro_f1"]
                    if len(acc_row) > 0 and len(f1_row) > 0:
                        df = pd.DataFrame([{
                            "accuracy": float(acc_row.iloc[0]["value"]),
                            "macro_f1": float(f1_row.iloc[0]["value"]),
                        }])
            elif method == "scgpt_frozen":
                # Format: accuracy,macro_f1,label_key,n_classes
                if "accuracy" in df.columns and "macro_f1" in df.columns:
                    df = df[["accuracy", "macro_f1"]].iloc[:1]
            results[f"pbmc3k_{method}"] = df
        else:
            print(f"[WARN] {filename} not found")
    
    # Tabula Sapiens results
    tabula_files = {
        "geneformer_frozen": "metrics_geneformer_frozen_tabula_sapiens.csv",
        "scgpt_frozen": "metrics_scgpt_frozen_tabula_sapiens.csv",
    }
    
    for method, filename in tabula_files.items():
        filepath = RES_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            results[f"tabula_{method}"] = df
        else:
            print(f"[WARN] {filename} not found")
    
    return results


def create_comparison_table(results):
    """Create a comprehensive comparison table."""
    rows = []
    
    # PBMC3k results
    datasets = {
        "pbmc3k": {
            "geneformer_frozen": "PBMC3k - Geneformer (Frozen)",
            "scgpt_frozen": "PBMC3k - scGPT (Frozen)",
            "geneformer_finetuned": "PBMC3k - Geneformer (Fine-tuned)",
            "scgpt_finetuned": "PBMC3k - scGPT (Fine-tuned)",
            "scfoundation": "PBMC3k - scFoundation",
        },
        "tabula": {
            "geneformer_frozen": "Tabula Sapiens - Geneformer (Frozen)",
            "scgpt_frozen": "Tabula Sapiens - scGPT (Frozen)",
        }
    }
    
    for dataset, methods in datasets.items():
        for method, label in methods.items():
            key = f"{dataset}_{method}"
            if key in results:
                df = results[key]
                if len(df) > 0:
                    row = {
                        "Method": label,
                        "Dataset": dataset.upper(),
                        "Accuracy": df.iloc[0].get("accuracy", "N/A"),
                        "Macro F1": df.iloc[0].get("macro_f1", "N/A"),
                        "Status": "Completed" if df.iloc[0].get("accuracy") not in [None, "N/A"] else "Not Available"
                    }
                    rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    return comparison_df


def create_final_report():
    """Create the final project report."""
    print("=" * 60)
    print("Creating Final Project Report")
    print("=" * 60)
    
    # Load all results
    results = load_all_results()
    print(f"\n[INFO] Loaded {len(results)} result files")
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    comparison_path = OUTPUT_DIR / "final_comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[INFO] Comparison table saved: {comparison_path}")
    
    # Create markdown report
    report_path = OUTPUT_DIR / "final_project_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Final Project Report: Understanding the Limits of Single-Cell Foundation Models\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive evaluation of single-cell foundation models ")
        f.write("(Geneformer, scGPT, and scFoundation) on downstream cell type classification tasks.\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("### PBMC3k Dataset\n\n")
        f.write("| Method | Accuracy | Macro F1 | Status |\n")
        f.write("|--------|----------|----------|--------|\n")
        
        for _, row in comparison_df[comparison_df["Dataset"] == "PBMC3K"].iterrows():
            acc = row["Accuracy"]
            f1 = row["Macro F1"]
            status = row["Status"]
            f.write(f"| {row['Method']} | {acc} | {f1} | {status} |\n")
        
        f.write("\n### Tabula Sapiens Dataset (Cross-Dataset Evaluation)\n\n")
        f.write("| Method | Accuracy | Macro F1 | Status |\n")
        f.write("|--------|----------|----------|--------|\n")
        
        for _, row in comparison_df[comparison_df["Dataset"] == "TABULA"].iterrows():
            acc = row["Accuracy"]
            f1 = row["Macro F1"]
            status = row["Status"]
            f.write(f"| {row['Method']} | {acc} | {f1} | {status} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("### 1. Frozen vs Fine-tuned Comparison\n\n")
        f.write("- **Frozen Representations**: Models evaluated with frozen pretrained encoders and lightweight classifiers.\n")
        f.write("- **Fine-tuned Models**: Models fine-tuned on the target dataset with task-specific heads.\n\n")
        
        f.write("### 2. Cross-Dataset Generalization\n\n")
        f.write("- **PBMC3k**: Primary evaluation dataset (in-domain).\n")
        f.write("- **Tabula Sapiens**: Cross-dataset evaluation to test domain shift robustness.\n\n")
        
        f.write("### 3. Model Comparison\n\n")
        f.write("- **Geneformer**: Transformer-based model pretrained on large-scale gene expression data.\n")
        f.write("- **scGPT**: Generative pretrained transformer for single-cell data.\n")
        f.write("- **scFoundation**: Large-scale foundation model (availability may be limited).\n\n")
        
        f.write("## Implementation Status\n\n")
        
        # Check implementation status
        implemented = {
            "PBMC3k Frozen (Geneformer)": "pbmc3k_geneformer_frozen" in results,
            "PBMC3k Frozen (scGPT)": "pbmc3k_scgpt_frozen" in results,
            "PBMC3k Fine-tuned (Geneformer)": "pbmc3k_geneformer_finetuned" in results,
            "PBMC3k Fine-tuned (scGPT)": "pbmc3k_scgpt_finetuned" in results,
            "Tabula Sapiens (Geneformer)": "tabula_geneformer_frozen" in results,
            "Tabula Sapiens (scGPT)": "tabula_scgpt_frozen" in results,
            "scFoundation": "pbmc3k_scfoundation" in results,
        }
        
        f.write("| Component | Status |\n")
        f.write("|-----------|--------|\n")
        for component, status in implemented.items():
            status_str = "✅ Implemented" if status else "❌ Not Implemented"
            f.write(f"| {component} | {status_str} |\n")
        
        f.write("\n## Files and Outputs\n\n")
        f.write("- Comparison table: `results/analysis/final_comparison_table.csv`\n")
        f.write("- This report: `results/analysis/final_project_report.md`\n")
        f.write("- Individual result files: `results/metrics_*.csv`\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review and analyze the results\n")
        f.write("2. Create visualizations (UMAP, confusion matrices)\n")
        f.write("3. Perform statistical analysis\n")
        f.write("4. Write detailed discussion and conclusions\n")
    
    print(f"[INFO] Final report saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("\n[INFO] Final report generation completed!")


if __name__ == "__main__":
    create_final_report()
