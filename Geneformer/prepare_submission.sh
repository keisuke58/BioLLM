#!/bin/bash
# Prepare submission package for final project

cd /home/nishioka/LUH/BioLLM/Geneformer

SUBMISSION_DIR="submission_package"
rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

echo "============================================================"
echo "Preparing Submission Package"
echo "============================================================"
echo ""

# 1. Create directory structure
mkdir -p "$SUBMISSION_DIR/code"
mkdir -p "$SUBMISSION_DIR/results"
mkdir -p "$SUBMISSION_DIR/results/analysis"
mkdir -p "$SUBMISSION_DIR/results/figures"

echo "[1/5] Creating directory structure..."

# 2. Copy final report
echo "[2/5] Copying final report..."
cp results/analysis/final_project_report_formatted.md "$SUBMISSION_DIR/FINAL_REPORT.md"
cp results/analysis/final_comparison_table.csv "$SUBMISSION_DIR/results/analysis/"

# 3. Copy essential code files
echo "[3/5] Copying code files..."
cp run_geneformer_pbmc3k.py "$SUBMISSION_DIR/code/"
cp run_scgpt_pbmc3k.py "$SUBMISSION_DIR/code/"
cp run_geneformer_finetune_pbmc3k.py "$SUBMISSION_DIR/code/"
cp run_scgpt_finetune_pbmc3k.py "$SUBMISSION_DIR/code/"
cp run_tabula_sapiens_evaluation.py "$SUBMISSION_DIR/code/"
cp run_scfoundation_evaluation.py "$SUBMISSION_DIR/code/"
cp create_final_report.py "$SUBMISSION_DIR/code/"

# 4. Copy result files
echo "[4/5] Copying result files..."
cp results/metrics_geneformer_pbmc3k.csv "$SUBMISSION_DIR/results/"
cp results/metrics_scgpt.csv "$SUBMISSION_DIR/results/"
cp results/metrics_geneformer_finetuned_pbmc3k.csv "$SUBMISSION_DIR/results/"
cp results/metrics_scfoundation_pbmc3k.csv "$SUBMISSION_DIR/results/" 2>/dev/null || true

# 5. Copy essential figures
echo "[5/5] Copying figures..."
cp results/umap_labels_pbmc3k.png "$SUBMISSION_DIR/results/figures/" 2>/dev/null || true
cp results/umap_geneformer_emb_pbmc3k.png "$SUBMISSION_DIR/results/figures/" 2>/dev/null || true
cp results/confusion_geneformer_pbmc3k.png "$SUBMISSION_DIR/results/figures/" 2>/dev/null || true
cp results/confusion_scgpt.png "$SUBMISSION_DIR/results/figures/" 2>/dev/null || true

# 6. Copy README and documentation
cp README.md "$SUBMISSION_DIR/"
cp README_FINAL_PROJECT.md "$SUBMISSION_DIR/" 2>/dev/null || true

# 7. Create submission README
cat > "$SUBMISSION_DIR/SUBMISSION_README.txt" << 'EOF'
Final Project Submission Package
================================

Author: Keisuke Nishioka (Student ID: 10081049)
Course: AI Foundation Models in Biomedicine, WiSe 2025/26
Submission Date: March 2, 2026

Package Contents:
-----------------
1. FINAL_REPORT.md - Final project report (formatted, 6-8 pages)
2. code/ - All evaluation scripts
3. results/ - Result files and figures
4. README.md - Project documentation

Key Results:
------------
- Geneformer (Frozen): Accuracy 0.613, Macro F1 0.428
- scGPT (Frozen): Accuracy 0.600, Macro F1 0.294
- Geneformer (Fine-tuned): Accuracy 0.978, Macro F1 0.978

Main Finding: Fine-tuning improves accuracy by 59.6% (61.3% → 97.8%)

AI Tools Used:
--------------
- Cursor AI Assistant: Code development and debugging
- ChatGPT/Claude: Initial project planning

See FINAL_REPORT.md Appendix C for detailed information.

EOF

# 8. Create file list
echo ""
echo "Creating file list..."
find "$SUBMISSION_DIR" -type f | sort > "$SUBMISSION_DIR/FILE_LIST.txt"

# Summary
echo ""
echo "============================================================"
echo "Submission Package Created"
echo "============================================================"
echo "Location: $SUBMISSION_DIR"
echo ""
echo "Package contents:"
du -sh "$SUBMISSION_DIR"/*
echo ""
echo "Total files:"
find "$SUBMISSION_DIR" -type f | wc -l
echo ""
echo "File list saved to: $SUBMISSION_DIR/FILE_LIST.txt"
echo ""
echo "Next steps:"
echo "1. Review FINAL_REPORT.md"
echo "2. Verify all code files are included"
echo "3. Check result files"
echo "4. Create ZIP archive if needed: zip -r submission.zip $SUBMISSION_DIR"
