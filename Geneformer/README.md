# Final Project: Understanding the Limits of Single-Cell Foundation Models

**Author**: Keisuke Nishioka (Student ID: 10081049)  
**Course**: AI Foundation Models in Biomedicine, WiSe 2025/26  
**Institution**: Leibniz University of Hannover  
**Submission Date**: March 2, 2026

---

## Project Overview

This project evaluates single-cell foundation models (Geneformer and scGPT) on downstream cell type classification tasks, comparing frozen representations with fine-tuned models. The main finding is that fine-tuning dramatically improves performance: Geneformer achieves 97.8% accuracy after fine-tuning compared to 61.3% with frozen representations.

## Key Results

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| Geneformer (Frozen) | 0.613 | 0.428 |
| scGPT (Frozen) | 0.600 | 0.294 |
| **Geneformer (Fine-tuned)** | **0.978** | **0.978** |

**Key Finding**: Fine-tuning provides a 59.6% absolute improvement in accuracy (61.3% → 97.8%), demonstrating the importance of task-specific optimization.

## Project Structure

```
Geneformer/
├── run_geneformer_pbmc3k.py          # Geneformer frozen evaluation
├── run_scgpt_pbmc3k.py               # scGPT frozen evaluation
├── run_geneformer_finetune_pbmc3k.py # Geneformer fine-tuning
├── run_scgpt_finetune_pbmc3k.py      # scGPT fine-tuning (script ready)
├── run_tabula_sapiens_evaluation.py  # Cross-dataset evaluation (script ready)
├── run_scfoundation_evaluation.py    # scFoundation evaluation
├── create_final_report.py            # Generate final report
├── results/                          # Output directory
│   ├── analysis/                     # Analysis results
│   │   ├── final_project_report_formatted.md  # Final report (formatted)
│   │   └── final_comparison_table.csv
│   ├── metrics_*.csv                 # Individual result files
│   └── *.png                         # Visualizations
└── README.md                         # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
bash install_dependencies.sh

# Or manually
pip install scanpy anndata xgboost umap-learn scikit-learn torch transformers datasets scgpt peft bitsandbytes tdigest IPython
```

### 2. Run Evaluations

**Frozen Representations:**

```bash
# Geneformer frozen evaluation
python run_geneformer_pbmc3k.py

# scGPT frozen evaluation
python run_scgpt_pbmc3k.py
```

**Fine-tuning:**

```bash
# Geneformer fine-tuning (takes ~12 minutes on GPU)
python run_geneformer_finetune_pbmc3k.py
```

### 3. Generate Final Report

```bash
python create_final_report.py
```

The report will be saved to `results/analysis/final_project_report.md`

## Requirements

### Software

- Python 3.8+
- PyTorch (tested with 2.9.1)
- CUDA-capable GPU (recommended)
- Required packages: See `requirements.txt` or `install_dependencies.sh`

### Data

- **PBMC3k Dataset**: Automatically downloaded by scripts
- **Tabula Sapiens**: Large dataset (~50GB), not required for main results

### Models

- **Geneformer V2-104M**: Should be in `Geneformer-V2-104M/` directory
- **scGPT**: Automatically downloaded by scripts

## Implementation Details

### Frozen Representation Evaluation

1. Extract embeddings from pretrained models (no fine-tuning)
2. Train lightweight classifiers (XGBoost) on extracted embeddings
3. Evaluate on held-out test sets

### Fine-tuning Evaluation

1. Initialize models with pretrained weights
2. Fine-tune end-to-end on cell type classification
3. Use stratified train/validation/test splits (80/10/10)
4. Train for 3 epochs with learning rate 5e-5

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1 Score**: Average F1 score across all classes (handles class imbalance)

## Results

### Completed Evaluations

✅ **PBMC3k - Geneformer (Frozen)**: Accuracy 0.613, Macro F1 0.428  
✅ **PBMC3k - scGPT (Frozen)**: Accuracy 0.600, Macro F1 0.294  
✅ **PBMC3k - Geneformer (Fine-tuned)**: Accuracy 0.978, Macro F1 0.978

### Not Executed

⏳ **PBMC3k - scGPT (Fine-tuned)**: Technical issues (torchtext compatibility)  
⏳ **Tabula Sapiens**: Dataset not downloaded (time constraints)  
⏳ **scFoundation**: Model not publicly available

## Output Files

### Results Directory

- `results/metrics_geneformer_pbmc3k.csv` - Geneformer frozen results
- `results/metrics_scgpt.csv` - scGPT frozen results
- `results/metrics_geneformer_finetuned_pbmc3k.csv` - Geneformer fine-tuned results
- `results/analysis/final_comparison_table.csv` - Aggregated comparison
- `results/analysis/final_project_report_formatted.md` - Final report

### Visualizations

- `results/umap_labels_pbmc3k.png` - UMAP of cell types
- `results/umap_geneformer_emb_pbmc3k.png` - UMAP of embeddings
- `results/confusion_geneformer_pbmc3k.png` - Confusion matrix

## Technical Notes

### Known Issues

1. **scGPT Fine-tuning**: torchtext library compatibility issues with PyTorch 2.9+
2. **Datasets Library**: Compatibility issues with version 2.21.0, resolved by re-tokenizing data
3. **Stratification**: Could not use stratified splits due to ClassLabel type requirements

### Reproducibility

- All experiments use random seed 42
- Results are saved to CSV files for reproducibility
- Logs are saved to `logs/` directory

## Usage of AI Tools

**AI Tools Used**:
- **Cursor AI Assistant**: Used for code development, debugging, and documentation
- **ChatGPT/Claude**: Used for initial project planning and literature review

**Declaration**: All experimental results, code implementations, and analyses were performed by the author. AI tools were used as development aids but did not generate experimental results or conclusions.

See `results/analysis/final_project_report_formatted.md` Appendix C for detailed information.

## References

1. Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618(7965), 616-624.

2. Cui, H., et al. (2023). scGPT: Towards building a foundation model for single-cell multi-omics using generative AI. *bioRxiv*.

3. 10x Genomics. PBMC 68k dataset. https://www.10xgenomics.com/

## Contact

For questions about this project:
- Author: Keisuke Nishioka
- Student ID: 10081049
- Course: AI Foundation Models in Biomedicine, WiSe 2025/26

## License

This project is for academic purposes as part of the course requirements.

---

**Last Updated**: January 18, 2026
