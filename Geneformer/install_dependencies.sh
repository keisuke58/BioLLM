#!/bin/bash
# Install dependencies for Final Project evaluation scripts

echo "============================================================"
echo "Installing Dependencies for Final Project"
echo "============================================================"

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "[INFO] Using conda environment: $CONDA_DEFAULT_ENV"
    INSTALL_CMD="conda install -y -c conda-forge"
    PIP_CMD="pip install"
else
    echo "[INFO] Using pip"
    INSTALL_CMD="pip install"
    PIP_CMD="pip install"
fi

# Core dependencies
echo ""
echo "Installing core dependencies..."
$PIP_CMD numpy pandas scikit-learn matplotlib scipy

# Single-cell analysis
echo ""
echo "Installing single-cell analysis packages..."
$PIP_CMD scanpy anndata

# Machine learning
echo ""
echo "Installing machine learning packages..."
$PIP_CMD xgboost umap-learn

# Deep learning
echo ""
echo "Installing deep learning packages..."
$PIP_CMD torch transformers datasets

# scGPT (if available)
echo ""
echo "Installing scGPT..."
$PIP_CMD scgpt || echo "[WARN] scGPT installation failed. You may need to install manually."

# Geneformer dependencies (from requirements.txt)
echo ""
echo "Installing Geneformer dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD -r requirements.txt
else
    echo "[WARN] requirements.txt not found"
fi

# Verify installation
echo ""
echo "============================================================"
echo "Verifying Installation"
echo "============================================================"

python -c "
import sys
packages = [
    'numpy', 'pandas', 'sklearn', 'scanpy', 'xgboost', 
    'matplotlib', 'umap', 'torch', 'transformers'
]
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - MISSING')
        missing.append(pkg)

if missing:
    print(f'\n[ERROR] Missing packages: {missing}')
    sys.exit(1)
else:
    print('\n[SUCCESS] All core packages installed!')
"

echo ""
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "You can now run the evaluation scripts:"
echo "  python run_all_evaluations.py"
echo ""
