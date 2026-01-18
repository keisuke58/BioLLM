"""
Main script to run all evaluations for the final project.
This script orchestrates the execution of all evaluation pipelines.
"""
import os
import sys
import subprocess
from pathlib import Path

WORKDIR = Path(__file__).parent
os.chdir(WORKDIR)

print("=" * 60)
print("Final Project: Complete Evaluation Pipeline")
print("=" * 60)

# List of evaluation scripts to run
evaluation_scripts = [
    {
        "name": "PBMC3k - Geneformer (Frozen)",
        "script": "run_geneformer_pbmc3k.py",
        "required": True,
    },
    {
        "name": "PBMC3k - scGPT (Frozen)",
        "script": "run_scgpt_pbmc3k.py",
        "required": True,
    },
    {
        "name": "PBMC3k - Geneformer (Fine-tuned)",
        "script": "run_geneformer_finetune_pbmc3k.py",
        "required": False,  # Fine-tuning may take longer
    },
    {
        "name": "PBMC3k - scGPT (Fine-tuned)",
        "script": "run_scgpt_finetune_pbmc3k.py",
        "required": False,
    },
    {
        "name": "Tabula Sapiens - Cross-dataset Evaluation",
        "script": "run_tabula_sapiens_evaluation.py",
        "required": False,  # Requires large dataset download
    },
    {
        "name": "scFoundation Evaluation",
        "script": "run_scfoundation_evaluation.py",
        "required": False,  # May not be available
    },
]


def run_script(script_path, script_name):
    """Run a single evaluation script."""
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=WORKDIR,
            check=False,
            capture_output=False
        )
        if result.returncode == 0:
            print(f"[SUCCESS] {script_name} completed successfully")
            return True
        else:
            print(f"[WARNING] {script_name} exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False


def main():
    """Run all evaluation scripts."""
    print("\nThis script will run all evaluation pipelines.")
    print("Some scripts may take a long time to execute.")
    print("Press Ctrl+C to cancel.\n")
    
    # Ask for confirmation
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    results = {}
    
    # Run required scripts first
    print("\n" + "=" * 60)
    print("Phase 1: Required Evaluations (Frozen Representations)")
    print("=" * 60)
    
    for eval_info in evaluation_scripts:
        if eval_info["required"]:
            script_path = WORKDIR / eval_info["script"]
            success = run_script(script_path, eval_info["name"])
            results[eval_info["name"]] = success
    
    # Run optional scripts
    print("\n" + "=" * 60)
    print("Phase 2: Optional Evaluations (Fine-tuning, Cross-dataset)")
    print("=" * 60)
    
    for eval_info in evaluation_scripts:
        if not eval_info["required"]:
            script_path = WORKDIR / eval_info["script"]
            print(f"\n[INFO] Optional script: {eval_info['name']}")
            response = input(f"Run {eval_info['name']}? (y/n): ")
            if response.lower() == 'y':
                success = run_script(script_path, eval_info["name"])
                results[eval_info["name"]] = success
            else:
                print(f"[SKIP] Skipping {eval_info['name']}")
                results[eval_info["name"]] = None
    
    # Generate final report
    print("\n" + "=" * 60)
    print("Generating Final Report")
    print("=" * 60)
    
    report_script = WORKDIR / "create_final_report.py"
    if report_script.exists():
        run_script(report_script, "Final Report Generation")
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    for name, success in results.items():
        if success is True:
            status = "✅ Completed"
        elif success is False:
            status = "❌ Failed"
        else:
            status = "⏭️  Skipped"
        print(f"{status}: {name}")
    
    print("\n[INFO] All evaluations completed!")
    print("[INFO] Check results/ directory for output files.")


if __name__ == "__main__":
    main()
