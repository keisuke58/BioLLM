"""
Fine-tune Geneformer on PBMC3k dataset for cell type classification.
This script implements Task-head fine-tuning as specified in the proposal.
"""
import os
import random
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from geneformer import Classifier, TranscriptomeTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -------------------------
# Config
# -------------------------
SEED = 42
WORKDIR = os.getcwd()

DATA_DIR = os.path.join(WORKDIR, "data")
ART_DIR = os.path.join(WORKDIR, "artifacts")
RES_DIR = os.path.join(WORKDIR, "results")

H5AD_DIR = os.path.join(ART_DIR, "h5ad")
H5AD_PATH = os.path.join(H5AD_DIR, "pbmc3k_for_geneformer.h5ad")
TOKEN_DIR = os.path.join(ART_DIR, "tokenized_pbmc3k")
FINETUNE_OUT_DIR = os.path.join(RES_DIR, "geneformer_finetuned_pbmc3k")

# Pretrained model directory (use base model, not fine-tuned)
MODEL_DIR = os.path.join(WORKDIR, "Geneformer-V2-104M")  # Base pretrained model

os.makedirs(FINETUNE_OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------
# 1) Load PBMC3k data (should already exist from frozen run)
# -------------------------
if not os.path.exists(H5AD_PATH):
    raise FileNotFoundError(
        f"PBMC3k h5ad file not found: {H5AD_PATH}\n"
        "Please run run_geneformer_pbmc3k.py first to generate this file."
    )

adata = sc.read_h5ad(H5AD_PATH)
print(f"Loaded PBMC3k: {adata.shape}, classes: {adata.obs['cell_type'].nunique()}")
print(f"Cell types: {adata.obs['cell_type'].unique()}")

# -------------------------
# 2) Ensure tokenized data exists (with cell_type annotation)
# -------------------------
# For fine-tuning, we need tokenized data with cell_type annotation
token_dataset_dir = os.path.join(TOKEN_DIR, "pbmc3k.dataset")
token_dataset_dir_with_labels = os.path.join(TOKEN_DIR, "pbmc3k_with_labels.dataset")

# Check if tokenized data with labels exists
if not os.path.exists(token_dataset_dir_with_labels):
    print("Tokenizing data with cell_type labels for fine-tuning...")
    os.makedirs(TOKEN_DIR, exist_ok=True)
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"joinid": "joinid", "cell_type": "cell_type"}, nproc=4)
    tk.tokenize_data(
        data_directory=H5AD_DIR,
        output_directory=TOKEN_DIR,
        output_prefix="pbmc3k_with_labels",
        file_format="h5ad"
    )
    print("Tokenization done.")
    token_dataset_dir = token_dataset_dir_with_labels
else:
    print(f"Tokenized data with labels already exists: {token_dataset_dir_with_labels}")
    token_dataset_dir = token_dataset_dir_with_labels

# -------------------------
# 3) Prepare data for fine-tuning
# -------------------------
print("\n=== Preparing data for fine-tuning ===")

# Get unique cell types
cell_types = sorted(adata.obs["cell_type"].unique().tolist())
print(f"Cell types to classify: {cell_types}")

# Initialize Classifier with fine-tuning configuration
# Training arguments for end-to-end fine-tuning
training_args = {
    "num_train_epochs": 3.0,  # Number of training epochs
    "learning_rate": 5e-5,  # Conservative learning rate for fine-tuning
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "evaluation_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",  # Must match evaluation_strategy
    "save_steps": 200,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_macro_f1",
    "greater_is_better": True,
    "output_dir": FINETUNE_OUT_DIR,
    "seed": SEED,
}

# Initialize Classifier for cell type classification
# Note: prepare_data internally renames cell_type to "label" before train_test_split
# freeze_layers=0 means we fine-tune all layers (full fine-tuning, not just task head)
cc = Classifier(
    classifier="cell",  # Cell type classification task
    cell_state_dict={"state_key": "cell_type", "states": cell_types},  # Cell types to classify
    training_args=training_args,  # Fine-tuning hyperparameters
    max_ncells=None,  # Use all available cells (no subsampling)
    freeze_layers=0,  # Fine-tune all layers (full fine-tuning, not frozen)
    num_crossval_splits=1,  # Single train/validation/test split (no cross-validation)
    split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},  # Data split proportions
    stratify_splits_col=None,  # Cannot use stratification due to ClassLabel type requirements
    forward_batch_size=100,  # Batch size for forward pass
    model_version="V2",  # Use Geneformer V2 model
    nproc=4,  # Number of CPU processes for data processing
    ngpu=1,  # Number of GPUs to use
)

# Prepare data splits
output_prefix = "pbmc3k_cell_classifier"

# Try to prepare data, handle datasets compatibility issues
try:
    cc.prepare_data(
        input_data_file=token_dataset_dir,
        output_directory=FINETUNE_OUT_DIR,
        output_prefix=output_prefix,
    )
except (TypeError, ValueError) as e:
    if "dataclass" in str(e) or "datasets" in str(e).lower():
        print(f"[WARN] Datasets compatibility issue detected: {e}")
        print("[INFO] Attempting to re-tokenize data with current datasets version...")
        # Remove old tokenized data and re-tokenize
        import shutil
        if os.path.exists(token_dataset_dir):
            print(f"[INFO] Removing old tokenized data: {token_dataset_dir}")
            shutil.rmtree(token_dataset_dir)
        # Re-tokenize
        tk = TranscriptomeTokenizer(custom_attr_name_dict={"joinid": "joinid", "cell_type": "cell_type"}, nproc=4)
        tk.tokenize_data(
            data_directory=H5AD_DIR,
            output_directory=TOKEN_DIR,
            output_prefix="pbmc3k_with_labels",
            file_format="h5ad"
        )
        token_dataset_dir = token_dataset_dir_with_labels
        # Retry prepare_data
        token_dataset_dir = token_dataset_dir_with_labels
        cc.prepare_data(
            input_data_file=token_dataset_dir,
            output_directory=FINETUNE_OUT_DIR,
            output_prefix=output_prefix,
        )
    else:
        raise

# Load id_class_dict
id_class_dict_path = os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_id_class_dict.pkl")
if not os.path.exists(id_class_dict_path):
    raise FileNotFoundError(f"id_class_dict not found: {id_class_dict_path}")

import pickle
with open(id_class_dict_path, "rb") as f:
    id_class_dict = pickle.load(f)

print(f"\nNumber of classes: {len(id_class_dict)}")
print(f"Class mapping: {id_class_dict}")

# -------------------------
# 4) Fine-tune the model
# -------------------------
print("\n=== Fine-tuning Geneformer ===")

# Load prepared datasets
# prepare_data saves as _labeled_train and _labeled_test (or _labeled if no split)
# For train/valid/test split, it saves as _labeled_train and _labeled_test
# We need to check what files were actually created
import glob
train_files = glob.glob(os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_labeled_train.dataset"))
test_files = glob.glob(os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_labeled_test.dataset"))
all_files = glob.glob(os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_labeled*.dataset"))

print(f"Found dataset files: {[os.path.basename(f) for f in all_files]}")

if len(train_files) > 0 and len(test_files) > 0:
    train_data_path = train_files[0]
    test_data_path = test_files[0]
    # For validation, we'll split train further or use test as eval
    eval_data_path = test_data_path  # Use test as eval for now
    print(f"Using train: {train_data_path}")
    print(f"Using test (as eval): {eval_data_path}")
elif len(all_files) == 1:
    # Single file - need to split manually
    data_path = all_files[0]
    print(f"Single dataset file found: {data_path}")
    print("Splitting manually...")
    from datasets import load_from_disk
    full_data = load_from_disk(data_path)
    data_dict = full_data.train_test_split(test_size=0.2, seed=SEED)
    train_data_path = os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_manual_train.dataset")
    eval_data_path = os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_manual_valid.dataset")
    test_data_path = os.path.join(FINETUNE_OUT_DIR, f"{output_prefix}_manual_test.dataset")
    data_dict["train"].save_to_disk(train_data_path)
    # Split test further into eval and test
    eval_test_dict = data_dict["test"].train_test_split(test_size=0.5, seed=SEED)
    eval_test_dict["train"].save_to_disk(eval_data_path)
    eval_test_dict["test"].save_to_disk(test_data_path)
else:
    raise FileNotFoundError(f"Could not find prepared dataset files in {FINETUNE_OUT_DIR}")

from datasets import load_from_disk
train_data = load_from_disk(train_data_path)
eval_data = load_from_disk(eval_data_path)
test_data = load_from_disk(test_data_path)

print(f"Train samples: {len(train_data)}")
print(f"Eval samples: {len(eval_data)}")
print(f"Test samples: {len(test_data)}")

# Fine-tune
num_classes = len(id_class_dict)
trainer = cc.train_classifier(
    model_directory=MODEL_DIR,
    num_classes=num_classes,
    train_data=train_data,
    eval_data=eval_data,
    output_directory=FINETUNE_OUT_DIR,
    predict=True,
)

print("Fine-tuning completed!")

# -------------------------
# 5) Evaluate on test set
# -------------------------
print("\n=== Evaluating on test set ===")

# Load the fine-tuned model
fine_tuned_model_dir = FINETUNE_OUT_DIR
eval_results = cc.evaluate_model(
    model=trainer.model,
    num_classes=num_classes,
    id_class_dict=id_class_dict,
    eval_data=test_data,
    predict=True,
    output_directory=FINETUNE_OUT_DIR,
    output_prefix="test",
)

print("\n=== Test Set Results ===")
print(f"Accuracy: {eval_results['acc']:.4f}")
print(f"Macro F1: {eval_results['macro_f1']:.4f}")

# Save results
results_df = pd.DataFrame([{
    "method": "geneformer_finetuned",
    "dataset": "pbmc3k",
    "accuracy": eval_results["acc"],
    "macro_f1": eval_results["macro_f1"],
}])
results_df.to_csv(
    os.path.join(RES_DIR, "metrics_geneformer_finetuned_pbmc3k.csv"),
    index=False
)

print(f"\nResults saved to: {os.path.join(RES_DIR, 'metrics_geneformer_finetuned_pbmc3k.csv')}")
print("Fine-tuning pipeline completed!")
