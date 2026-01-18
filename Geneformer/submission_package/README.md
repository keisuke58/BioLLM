# Final Project: Understanding the Limits of Single-Cell Foundation Models

**Author**: Keisuke Nishioka (Student ID: 10081049)  
**Course**: AI Foundation Models in Biomedicine, WiSe 2025/26  
**Institution**: Leibniz University of Hannover  
**Submission Date**: January 2026

---

## Project Overview

This project evaluates single-cell foundation models (Geneformer and scGPT) on downstream cell type classification tasks, comparing frozen representations with fine-tuned models. The main finding is that fine-tuning dramatically improves performance: Geneformer achieves **97.8% accuracy** after fine-tuning compared to **61.3%** with frozen representations, representing a **59.6% absolute improvement**.

## Key Results

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| Geneformer (Frozen) | 0.613 | 0.428 |
| scGPT (Frozen) | 0.600 | 0.294 |
| **Geneformer (Fine-tuned)** | **0.978** | **0.978** |

**Key Finding**: Fine-tuning provides a 59.6% absolute improvement in accuracy (61.3% → 97.8%), demonstrating the importance of task-specific optimization.

## Project Structure

```
submission_package/
├── FINAL_REPORT.md          # Final report (Markdown format)
├── FINAL_REPORT.tex          # Final report (LaTeX format, publication quality)
├── README.md                 # This file
├── SUBMISSION_README.txt     # Submission instructions
├── FILE_LIST.txt             # Complete file list
├── code/                      # Evaluation scripts (7 files)
│   ├── run_geneformer_pbmc3k.py
│   ├── run_scgpt_pbmc3k.py
│   ├── run_geneformer_finetune_pbmc3k.py
│   ├── run_scgpt_finetune_pbmc3k.py
│   ├── run_tabula_sapiens_evaluation.py
│   ├── run_scfoundation_evaluation.py
│   └── create_final_report.py
└── results/                   # Results and figures
    ├── analysis/
    │   └── final_comparison_table.csv
    ├── figures/               # Visualization figures (5 PNG files)
    │   ├── umap_labels_pbmc3k.png
    │   ├── umap_geneformer_emb_pbmc3k.png
    │   ├── confusion_geneformer_pbmc3k.png
    │   ├── confusion_scgpt.png
    │   └── umap_scgpt.png
    └── metrics_*.csv          # Result CSV files (4 files)
        ├── metrics_geneformer_pbmc3k.csv
        ├── metrics_scgpt.csv
        ├── metrics_geneformer_finetuned_pbmc3k.csv
        └── metrics_scfoundation_pbmc3k.csv
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `scanpy`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `transformers`, `geneformer`, `scgpt`

### Installation

```bash
# Install dependencies
pip install scanpy numpy pandas scikit-learn xgboost torch transformers
# Install geneformer (local package)
# Install scgpt (local package)
```

### Running Evaluations

#### 1. Frozen Representation Evaluation

**Geneformer:**
```bash
python code/run_geneformer_pbmc3k.py
```

**scGPT:**
```bash
python code/run_scgpt_pbmc3k.py
```

#### 2. Fine-tuning Evaluation

**Geneformer:**
```bash
python code/run_geneformer_finetune_pbmc3k.py
```

**scGPT:**
```bash
python code/run_scgpt_finetune_pbmc3k.py
```

#### 3. Generate Final Report

```bash
python code/create_final_report.py
```

## Implementation Details

### Datasets

- **PBMC3k**: Subset of 10x Genomics PBMC 68k dataset
  - 2,700 cells with 2,000 highly variable genes
  - Cell types: T cells, B cells, DC, NK cells, Monocytes, Platelets

### Models

- **Geneformer V2-104M**: Pretrained transformer with 104 million parameters
- **scGPT**: Generative pretrained transformer for single-cell data

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1 Score**: Unweighted mean of F1 scores across all classes

### Experimental Setup

- **Frozen Representations**: Extract embeddings from pretrained models, train lightweight classifiers (XGBoost)
- **Fine-tuning**: End-to-end fine-tuning on cell type classification task
- **Train/Validation/Test Split**: 80/10/10
- **Training**: 3 epochs, learning rate 5e-5

## Results Summary

### Frozen Representation Performance

Both Geneformer and scGPT achieve similar accuracy (~60%) with frozen representations, indicating limitations of frozen representations alone.

### Fine-tuned Model Performance

Geneformer fine-tuning dramatically improves performance:
- **Accuracy**: 0.613 → 0.978 (+59.6%)
- **Macro F1**: 0.428 → 0.978 (+128.5%)

This demonstrates that task-specific fine-tuning is essential for optimal performance.

## Output Files

### Result Files

- `results/metrics_*.csv`: Individual evaluation results
- `results/analysis/final_comparison_table.csv`: Comprehensive comparison table

### Visualization Files

- `results/figures/*.png`: UMAP visualizations and confusion matrices

### Report Files

- `FINAL_REPORT.md`: Final report in Markdown format
- `FINAL_REPORT.tex`: Final report in LaTeX format (publication quality)

## Technical Notes

- **Reproducibility**: All experiments use random seed 42
- **Compatibility**: Resolved `datasets` library compatibility issues
- **Stratification**: Could not use stratified splits due to ClassLabel type requirements

## Limitations and Future Work

1. **scGPT Fine-tuning**: Not executed due to torchtext library compatibility issues
2. **Tabula Sapiens Evaluation**: Not executed due to dataset size (~50GB) and time constraints
3. **scFoundation**: Model not publicly available for evaluation

## Usage of AI Tools

This project used AI tools for development assistance:

- **Cursor AI Assistant**: Code development, debugging, and documentation
- **ChatGPT/Claude**: Initial project planning and literature review

**Declaration**: All experimental results, code implementations, and analyses were performed by the author. AI tools were used as development aids but did not generate experimental results or conclusions.

See `FINAL_REPORT.md` Appendix C for detailed information.

## References

1. Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618(7965), 616-624.

2. Cui, H., et al. (2023). scGPT: Towards building a foundation model for single-cell multi-omics using generative AI. *bioRxiv*.

3. Kedzierska, K. Z., et al. (bioRxiv). Evaluation of single-cell foundation models. *bioRxiv*.

4. Boiarsky, R., et al. (bioRxiv). Systematic evaluation of single-cell foundation models. *bioRxiv*.

5. 10x Genomics. (2023). PBMC 68k dataset. https://www.10xgenomics.com/

6. Tabula Sapiens Consortium. (2022). The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. *Science*, 376(6594), eabl4896.

## Contact

**Author**: Keisuke Nishioka  
**Student ID**: 10081049  
**Course**: AI Foundation Models in Biomedicine, WiSe 2025/26  
**Institution**: Leibniz University of Hannover

---

**Note**: Large files (PDF, PNG images, model files) are excluded from this package. The LaTeX source (`FINAL_REPORT.tex`) can be compiled to PDF using `pdflatex` or online services like Overleaf.
