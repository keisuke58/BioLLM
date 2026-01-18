#!/bin/bash
# Compile LaTeX to PDF
cd "$(dirname "$0")/submission_package"

if ! command -v pdflatex &> /dev/null; then
    echo "pdflatex not found. Please install:"
    echo "  sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
    exit 1
fi

echo "Compiling LaTeX to PDF..."
pdflatex -interaction=nonstopmode FINAL_REPORT.tex
pdflatex -interaction=nonstopmode FINAL_REPORT.tex  # Run twice for references

if [ -f FINAL_REPORT.pdf ]; then
    echo "✓ PDF created: FINAL_REPORT.pdf"
    ls -lh FINAL_REPORT.pdf
else
    echo "✗ PDF compilation failed. Check FINAL_REPORT.log for errors."
    exit 1
fi
