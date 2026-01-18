#!/bin/bash
# Run all evaluations sequentially with logging

cd /home/nishioka/LUH/BioLLM/Geneformer
mkdir -p logs

echo "============================================================"
echo "Complete Evaluation Pipeline - Sequential Execution"
echo "============================================================"
echo ""
echo "This will run ALL evaluations sequentially."
echo "Logs will be saved to logs/ directory"
echo ""

# Function to run script with logging
run_with_log() {
    local script=$1
    local name=$2
    local logfile=$3
    
    echo ""
    echo "============================================================"
    echo "Running: $name"
    echo "============================================================"
    echo "Log file: $logfile"
    echo ""
    
    if [ ! -f "$script" ]; then
        echo "[ERROR] Script not found: $script"
        return 1
    fi
    
    python "$script" 2>&1 | tee "$logfile"
    return ${PIPESTATUS[0]}
}

# Phase 1: Required evaluations
echo "Phase 1: Required Evaluations (Frozen Representations)"
echo "============================================================"

run_with_log "run_geneformer_pbmc3k.py" \
    "PBMC3k - Geneformer (Frozen)" \
    "logs/geneformer_frozen.log"

run_with_log "run_scgpt_pbmc3k.py" \
    "PBMC3k - scGPT (Frozen)" \
    "logs/scgpt_frozen.log"

# Phase 2: Fine-tuning
echo ""
echo "Phase 2: Fine-tuning Evaluations"
echo "============================================================"

run_with_log "run_geneformer_finetune_pbmc3k.py" \
    "PBMC3k - Geneformer (Fine-tuned)" \
    "logs/geneformer_finetune.log"

# Note: scGPT fine-tuning may fail due to torchtext issues
run_with_log "run_scgpt_finetune_pbmc3k.py" \
    "PBMC3k - scGPT (Fine-tuned)" \
    "logs/scgpt_finetune.log" || echo "[WARN] scGPT fine-tuning failed (torchtext issue)"

# Phase 3: Cross-dataset evaluation
echo ""
echo "Phase 3: Cross-Dataset Evaluation (Tabula Sapiens)"
echo "============================================================"

if [ -f "artifacts/h5ad/tabula_sapiens.h5ad" ]; then
    run_with_log "run_tabula_sapiens_evaluation.py" \
        "Tabula Sapiens - Cross-dataset Evaluation" \
        "logs/tabula_sapiens.log"
else
    echo "[SKIP] Tabula Sapiens dataset not found. Skipping cross-dataset evaluation."
    echo "[INFO] To run this evaluation, download Tabula Sapiens dataset first."
fi

# Phase 4: scFoundation
echo ""
echo "Phase 4: scFoundation Evaluation"
echo "============================================================"

run_with_log "run_scfoundation_evaluation.py" \
    "scFoundation Evaluation" \
    "logs/scfoundation.log"

# Generate final report
echo ""
echo "============================================================"
echo "Generating Final Report"
echo "============================================================"

python create_final_report.py 2>&1 | tee logs/final_report.log

echo ""
echo "============================================================"
echo "All Evaluations Completed!"
echo "============================================================"
echo "Check logs/ directory for detailed logs"
echo "Check results/ directory for output files"
