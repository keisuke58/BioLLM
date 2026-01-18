#!/bin/bash
# Compile LaTeX using Docker (no local installation needed)
cd "$(dirname "$0")/submission_package"

if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker or use apt-get to install LaTeX."
    echo "  sudo apt-get install docker.io"
    exit 1
fi

echo "Compiling LaTeX to PDF using Docker..."
docker run --rm -v "$(pwd)":/workdir -w /workdir texlive/texlive:latest \
    pdflatex -interaction=nonstopmode FINAL_REPORT.tex

docker run --rm -v "$(pwd)":/workdir -w /workdir texlive/texlive:latest \
    pdflatex -interaction=nonstopmode FINAL_REPORT.tex  # Run twice for references

if [ -f FINAL_REPORT.pdf ]; then
    echo "✓ PDF created: FINAL_REPORT.pdf"
    ls -lh FINAL_REPORT.pdf
else
    echo "✗ PDF compilation failed. Check FINAL_REPORT.log for errors."
    exit 1
fi
