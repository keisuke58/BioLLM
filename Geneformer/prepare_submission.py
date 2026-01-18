#!/usr/bin/env python3
"""
Prepare submission package for final project.
This script creates a clean submission directory with all required files.
"""
import os
import shutil
from pathlib import Path

WORKDIR = Path(__file__).parent
SUBMISSION_DIR = WORKDIR / "submission_package"

# Remove existing submission directory
if SUBMISSION_DIR.exists():
    shutil.rmtree(SUBMISSION_DIR)

# Create directory structure
print("=" * 60)
print("Preparing Submission Package")
print("=" * 60)
print()

print("[1/6] Creating directory structure...")
SUBMISSION_DIR.mkdir(exist_ok=True)
(SUBMISSION_DIR / "code").mkdir(exist_ok=True)
(SUBMISSION_DIR / "results").mkdir(exist_ok=True)
(SUBMISSION_DIR / "results" / "analysis").mkdir(exist_ok=True)
(SUBMISSION_DIR / "results" / "figures").mkdir(exist_ok=True)

# Copy final report
print("[2/6] Copying final report...")
report_src = WORKDIR / "results" / "analysis" / "final_project_report_formatted.md"
if report_src.exists():
    shutil.copy2(report_src, SUBMISSION_DIR / "FINAL_REPORT.md")
    print(f"  ✓ Copied: FINAL_REPORT.md")
else:
    print(f"  ✗ Not found: {report_src}")

# Copy comparison table
table_src = WORKDIR / "results" / "analysis" / "final_comparison_table.csv"
if table_src.exists():
    shutil.copy2(table_src, SUBMISSION_DIR / "results" / "analysis" / "final_comparison_table.csv")
    print(f"  ✓ Copied: final_comparison_table.csv")

# Copy code files
print("[3/6] Copying code files...")
code_files = [
    "run_geneformer_pbmc3k.py",
    "run_scgpt_pbmc3k.py",
    "run_geneformer_finetune_pbmc3k.py",
    "run_scgpt_finetune_pbmc3k.py",
    "run_tabula_sapiens_evaluation.py",
    "run_scfoundation_evaluation.py",
    "create_final_report.py",
]

for code_file in code_files:
    src = WORKDIR / code_file
    if src.exists():
        shutil.copy2(src, SUBMISSION_DIR / "code" / code_file)
        print(f"  ✓ Copied: {code_file}")
    else:
        print(f"  ✗ Not found: {code_file}")

# Copy result files
print("[4/6] Copying result files...")
result_files = [
    "metrics_geneformer_pbmc3k.csv",
    "metrics_scgpt.csv",
    "metrics_geneformer_finetuned_pbmc3k.csv",
    "metrics_scfoundation_pbmc3k.csv",
]

for result_file in result_files:
    src = WORKDIR / "results" / result_file
    if src.exists():
        shutil.copy2(src, SUBMISSION_DIR / "results" / result_file)
        print(f"  ✓ Copied: {result_file}")
    else:
        print(f"  - Not found (optional): {result_file}")

# Copy figures
print("[5/6] Copying figures...")
figure_files = [
    "umap_labels_pbmc3k.png",
    "umap_geneformer_emb_pbmc3k.png",
    "confusion_geneformer_pbmc3k.png",
    "confusion_scgpt.png",
    "umap_scgpt.png",
]

for fig_file in figure_files:
    src = WORKDIR / "results" / fig_file
    if src.exists():
        shutil.copy2(src, SUBMISSION_DIR / "results" / "figures" / fig_file)
        print(f"  ✓ Copied: {fig_file}")
    else:
        print(f"  - Not found (optional): {fig_file}")

# Copy README
print("[6/6] Copying documentation...")
readme_src = WORKDIR / "README.md"
if readme_src.exists():
    shutil.copy2(readme_src, SUBMISSION_DIR / "README.md")
    print(f"  ✓ Copied: README.md")

# Create submission README
submission_readme = SUBMISSION_DIR / "SUBMISSION_README.txt"
with open(submission_readme, "w") as f:
    f.write("""Final Project Submission Package
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

""")

# Create file list
file_list = []
for root, dirs, files in os.walk(SUBMISSION_DIR):
    for file in files:
        rel_path = os.path.relpath(os.path.join(root, file), SUBMISSION_DIR)
        file_list.append(rel_path)

file_list.sort()
with open(SUBMISSION_DIR / "FILE_LIST.txt", "w") as f:
    f.write("Submission Package File List\n")
    f.write("=" * 40 + "\n\n")
    for file_path in file_list:
        f.write(f"{file_path}\n")

# Summary
print()
print("=" * 60)
print("Submission Package Created Successfully")
print("=" * 60)
print(f"Location: {SUBMISSION_DIR}")
print(f"Total files: {len(file_list)}")
print()
print("Package structure:")
for file_path in file_list[:20]:  # Show first 20 files
    print(f"  {file_path}")
if len(file_list) > 20:
    print(f"  ... and {len(file_list) - 20} more files")
print()
print("Next steps:")
print("1. Review FINAL_REPORT.md")
print("2. Verify all code files are included")
print("3. Check result files")
print("4. Create ZIP archive: zip -r submission.zip submission_package/")
