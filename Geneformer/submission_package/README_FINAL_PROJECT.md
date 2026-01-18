# Final Project: Understanding the Limits of Single-Cell Foundation Models

## Overview

This project implements a comprehensive evaluation of single-cell foundation models (Geneformer, scGPT, and scFoundation) on downstream cell type classification tasks, as specified in the proposal.

## Project Structure

```
Geneformer/
├── run_geneformer_pbmc3k.py          # Geneformer frozen evaluation on PBMC3k
├── run_scgpt_pbmc3k.py                # scGPT frozen evaluation on PBMC3k
├── run_geneformer_finetune_pbmc3k.py  # Geneformer fine-tuning on PBMC3k
├── run_scgpt_finetune_pbmc3k.py       # scGPT fine-tuning on PBMC3k
├── run_tabula_sapiens_evaluation.py   # Cross-dataset evaluation on Tabula Sapiens
├── run_scfoundation_evaluation.py     # scFoundation evaluation (if available)
├── create_final_report.py             # Generate final report
├── run_all_evaluations.py             # Run all evaluations
└── results/                            # Output directory
    ├── analysis/                       # Analysis results
    └── *.csv                          # Individual result files
```

## Implementation Status

### ✅ Implemented

1. **PBMC3k Dataset Evaluation**
   - Geneformer (Frozen representations)
   - scGPT (Frozen representations)
   - Geneformer (Fine-tuning) - *Script created*
   - scGPT (Fine-tuning) - *Script created*

2. **Tabula Sapiens Cross-Dataset Evaluation**
   - Geneformer (Frozen) - *Script created*
   - scGPT (Frozen) - *Script created*

3. **Result Aggregation**
   - Final report generation script

### ⚠️ Partially Implemented

1. **scFoundation**
   - Evaluation script created
   - May require special access or installation

## Usage

### Quick Start

Run all evaluations:

```bash
python run_all_evaluations.py
```

### Individual Scripts

#### 1. PBMC3k - Frozen Representations

**Geneformer:**
```bash
python run_geneformer_pbmc3k.py
```

**scGPT:**
```bash
python run_scgpt_pbmc3k.py
```

#### 2. PBMC3k - Fine-tuning

**Geneformer:**
```bash
python run_geneformer_finetune_pbmc3k.py
```

**scGPT:**
```bash
python run_scgpt_finetune_pbmc3k.py
```

#### 3. Tabula Sapiens Evaluation

```bash
python run_tabula_sapiens_evaluation.py
```

**Note:** This requires downloading the Tabula Sapiens dataset (~50GB). The script will attempt to download it, or you can download manually from:
- https://tabula-sapiens-portal.ds.czbiohub.org/

#### 4. scFoundation Evaluation

```bash
python run_scfoundation_evaluation.py
```

**Note:** scFoundation may not be publicly available. The script will check availability and create placeholder results if unavailable.

#### 5. Generate Final Report

```bash
python create_final_report.py
```

This will:
- Aggregate all result files
- Create a comparison table
- Generate a markdown report

## Requirements

### Dependencies

- Python 3.8+
- PyTorch
- scanpy
- scgpt
- geneformer
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- umap-learn

### Data Requirements

1. **PBMC3k Dataset**
   - Automatically downloaded by scripts
   - Or use existing preprocessed data

2. **Tabula Sapiens Dataset**
   - Large dataset (~50GB)
   - Download from: https://tabula-sapiens-portal.ds.czbiohub.org/
   - Or use: `wget https://covid19.cog.sanger.ac.uk/tabula-sapiens.h5ad`

3. **Model Checkpoints**
   - Geneformer: Should be in `artifacts/fine_tuned_geneformer/`
   - scGPT: Automatically downloaded to `models/`
   - scFoundation: May require special access

## Output Files

### Results Directory Structure

```
results/
├── analysis/
│   ├── final_comparison_table.csv
│   ├── final_project_report.md
│   └── [other analysis files]
├── metrics_geneformer_pbmc3k.csv
├── metrics_scgpt.csv
├── metrics_geneformer_finetuned_pbmc3k.csv
├── metrics_scgpt_finetuned_pbmc3k.csv
├── metrics_geneformer_frozen_tabula_sapiens.csv
├── metrics_scgpt_frozen_tabula_sapiens.csv
└── [other result files]
```

### Key Outputs

1. **Metrics CSV Files**: Individual results for each model/dataset combination
2. **Comparison Table**: Aggregated results in CSV format
3. **Final Report**: Markdown report with summary and findings
4. **Visualizations**: UMAP plots, confusion matrices (in results/)

## Evaluation Metrics

All evaluations report:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Macro-averaged F1 score (handles class imbalance)

Additional metrics may be available in per-class analysis files.

## Notes

### Fine-tuning

- Fine-tuning scripts are created but may require:
  - Longer execution time
  - More GPU memory
  - Hyperparameter tuning for optimal results

### Tabula Sapiens

- Very large dataset - consider subsampling for faster processing
- Script includes subsampling option (default: 10k cells)
- Full dataset evaluation will take significantly longer

### scFoundation

- May not be publicly available
- Script will create placeholder results if unavailable
- Check scFoundation GitHub for latest availability

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure model checkpoints are in correct directories
   - Check `artifacts/` and `models/` directories

2. **Out of memory**
   - Reduce batch size in scripts
   - Use data subsampling for Tabula Sapiens

3. **Missing dependencies**
   - Install required packages: `pip install -r requirements.txt`
   - Check scGPT and Geneformer installation

4. **Data download fails**
   - Check internet connection
   - Download datasets manually if needed

## Next Steps

1. Run all evaluation scripts
2. Review results in `results/analysis/`
3. Generate visualizations
4. Perform statistical analysis
5. Write detailed discussion and conclusions

## Contact

For questions or issues, refer to:
- Geneformer: https://github.com/ctheodoris/Geneformer
- scGPT: https://github.com/bowang-lab/scGPT
- scFoundation: Check for availability and documentation
