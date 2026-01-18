"""
Complete evaluation pipeline - runs all evaluations sequentially with logging.
This script runs all evaluations and waits for completion.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

WORKDIR = Path(__file__).parent
os.chdir(WORKDIR)

# Create logs directory
os.makedirs("logs", exist_ok=True)

print("=" * 60)
print("Complete Evaluation Pipeline - All Evaluations")
print("=" * 60)
print("\nThis will run ALL evaluations sequentially.")
print("Estimated time: Several hours (depending on GPU and dataset size)")
print("\nPress Ctrl+C to cancel.\n")

# List of evaluation scripts to run
evaluation_scripts = [
    {
        "name": "PBMC3k - Geneformer (Frozen)",
        "script": "run_geneformer_pbmc3k.py",
        "log": "logs/geneformer_frozen.log",
        "required": True,
    },
    {
        "name": "PBMC3k - scGPT (Frozen)",
        "script": "run_scgpt_pbmc3k.py",
        "log": "logs/scgpt_frozen.log",
        "required": True,
    },
    {
        "name": "PBMC3k - Geneformer (Fine-tuned)",
        "script": "run_geneformer_finetune_pbmc3k.py",
        "log": "logs/geneformer_finetune.log",
        "required": False,
    },
    {
        "name": "PBMC3k - scGPT (Fine-tuned)",
        "script": "run_scgpt_finetune_pbmc3k.py",
        "log": "logs/scgpt_finetune.log",
        "required": False,
    },
    {
        "name": "Tabula Sapiens - Cross-dataset Evaluation",
        "script": "run_tabula_sapiens_evaluation.py",
        "log": "logs/tabula_sapiens.log",
        "required": False,
    },
    {
        "name": "scFoundation Evaluation",
        "script": "run_scfoundation_evaluation.py",
        "log": "logs/scfoundation.log",
        "required": False,
    },
]


def run_script_with_logging(script_path, script_name, log_path):
    """Run a single evaluation script with logging."""
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")
    print(f"Log file: {log_path}")
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=WORKDIR,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        elapsed_time = time.time() - start_time
        
        # Also print to terminal
        print(f"\n[INFO] Output (last 20 lines):")
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] {script_name} completed successfully")
            print(f"[INFO] Time elapsed: {elapsed_time/60:.1f} minutes")
            return True
        else:
            print(f"\n[WARNING] {script_name} exited with code {result.returncode}")
            print(f"[INFO] Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"[INFO] Check log file for details: {log_path}")
            return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {script_name} was interrupted by user")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False


def main():
    """Run all evaluation scripts."""
    results = {}
    total_start_time = time.time()
    
    # Run required scripts first
    print("\n" + "=" * 60)
    print("Phase 1: Required Evaluations (Frozen Representations)")
    print("=" * 60)
    
    for eval_info in evaluation_scripts:
        if eval_info["required"]:
            script_path = WORKDIR / eval_info["script"]
            success = run_script_with_logging(
                script_path, 
                eval_info["name"],
                eval_info["log"]
            )
            results[eval_info["name"]] = success
    
    # Run optional scripts
    print("\n" + "=" * 60)
    print("Phase 2: Optional Evaluations (Fine-tuning, Cross-dataset)")
    print("=" * 60)
    
    for eval_info in evaluation_scripts:
        if not eval_info["required"]:
            script_path = WORKDIR / eval_info["script"]
            print(f"\n[INFO] Optional script: {eval_info['name']}")
            print(f"[INFO] This may take a long time. Continue? (y/n): ", end="")
            
            # Auto-approve for non-interactive mode, or ask user
            response = "y"  # Auto-approve for complete run
            if response.lower() == 'y':
                success = run_script_with_logging(
                    script_path,
                    eval_info["name"],
                    eval_info["log"]
                )
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
        run_script_with_logging(
            report_script,
            "Final Report Generation",
            "logs/final_report.log"
        )
    
    # Summary
    total_elapsed = time.time() - total_start_time
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
    
    print(f"\n[INFO] Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print("[INFO] All evaluations completed!")
    print("[INFO] Check logs/ directory for detailed logs")
    print("[INFO] Check results/ directory for output files")


if __name__ == "__main__":
    main()
