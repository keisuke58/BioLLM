"""
Evaluate models on Tabula Sapiens dataset for cross-dataset evaluation.
This implements the domain shift analysis as specified in the proposal.
"""
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import umap

from geneformer import TranscriptomeTokenizer, EmbExtractor
# Try to import scGPT, handle torchtext compatibility issues
try:
    import scgpt as scg
except OSError as e:
    if "torchtext" in str(e) or "libtorchtext" in str(e):
        print("[ERROR] scGPT requires torchtext, but there's a compatibility issue.")
        print("[ERROR] This is a known issue with PyTorch 2.9+ and torchtext.")
        print("[INFO] Attempting workaround...")
        import os
        os.environ['TORCHTEXT_DISABLE_EXTENSION'] = '1'
        try:
            import scgpt as scg
            print("[INFO] Workaround successful!")
        except Exception as e2:
            print(f"[ERROR] Workaround failed: {e2}")
            print("[ERROR] scGPT evaluation will be skipped.")
            scg = None
    else:
        raise

# -------------------------
# Config
# -------------------------
SEED = 42
WORKDIR = os.getcwd()

DATA_DIR = os.path.join(WORKDIR, "data")
ART_DIR = os.path.join(WORKDIR, "artifacts")
RES_DIR = os.path.join(WORKDIR, "results")

TABULA_DIR = os.path.join(DATA_DIR, "tabula_sapiens")
TABULA_H5AD = os.path.join(ART_DIR, "h5ad", "tabula_sapiens.h5ad")
TOKEN_DIR = os.path.join(ART_DIR, "tokenized_tabula_sapiens")
EMB_OUT = os.path.join(RES_DIR, "tabula_sapiens_emb")

# Model directories
GENEFORMER_MODEL_DIR = os.path.join(ART_DIR, "fine_tuned_geneformer")
SCGPT_MODEL_DIR = os.path.join(WORKDIR, "models")

os.makedirs(TABULA_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(EMB_OUT, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def download_tabula_sapiens():
    """
    Download Tabula Sapiens dataset.
    Tabula Sapiens is available from: https://tabula-sapiens-portal.ds.czbiohub.org/
    """
    if os.path.exists(TABULA_H5AD):
        print(f"Tabula Sapiens already exists: {TABULA_H5AD}")
        return TABULA_H5AD

    print("Downloading Tabula Sapiens dataset...")
    print("Note: Tabula Sapiens is a large dataset (~50GB).")
    print("Please download manually from: https://tabula-sapiens-portal.ds.czbiohub.org/")
    print("Or use the following command:")
    print("  wget https://covid19.cog.sanger.ac.uk/tabula-sapiens.h5ad -O tabula_sapiens.h5ad")

    # Try to download a subset or use a smaller version if available
    # For now, we'll create a placeholder that expects the file to exist
    if not os.path.exists(TABULA_H5AD):
        raise FileNotFoundError(
            f"Tabula Sapiens file not found: {TABULA_H5AD}\n"
            "Please download from: https://tabula-sapiens-portal.ds.czbiohub.org/\n"
            "Or use: wget https://covid19.cog.sanger.ac.uk/tabula-sapiens.h5ad"
        )

    return TABULA_H5AD


def preprocess_tabula_sapiens(adata, subsample_n=None):
    """
    Preprocess Tabula Sapiens dataset similar to PBMC3k preprocessing.
    """
    print(f"Original shape: {adata.shape}")

    # Subsample if requested (Tabula Sapiens is very large)
    if subsample_n is not None and adata.n_obs > subsample_n:
        print(f"Subsampling to {subsample_n} cells...")
        sc.pp.subsample(adata, n_obs=subsample_n, random_state=SEED)

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.raw = adata.copy()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)

    # Ensure cell_type annotation exists
    if "cell_type" not in adata.obs.columns:
        # Try alternative column names
        for col in ["cell_ontology_class", "free_annotation", "cell_annotation"]:
            if col in adata.obs.columns:
                adata.obs["cell_type"] = adata.obs[col].astype(str)
                print(f"Using {col} as cell_type")
                break
        else:
            # Create from leiden clustering
            print("Creating cell_type from leiden clustering...")
            sc.tl.pca(adata, svd_solver="arpack")
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
            sc.tl.leiden(adata, resolution=0.6, key_added="leiden")
            adata.obs["cell_type"] = adata.obs["leiden"].astype(str)

    # Add joinid for compatibility
    if "joinid" not in adata.obs.columns:
        adata.obs["joinid"] = np.arange(adata.n_obs).astype(str)

    print(f"Processed shape: {adata.shape}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"Cell type distribution:\n{adata.obs['cell_type'].value_counts()}")

    return adata


def evaluate_geneformer_frozen(adata, output_prefix="tabula_sapiens"):
    """
    Evaluate Geneformer with frozen representations on Tabula Sapiens.
    """
    print("\n=== Evaluating Geneformer (Frozen) on Tabula Sapiens ===")

    # Tokenize
    token_dataset_dir = os.path.join(TOKEN_DIR, f"{output_prefix}.dataset")
    os.makedirs(TOKEN_DIR, exist_ok=True)

    if not os.path.exists(token_dataset_dir):
        print("Tokenizing Tabula Sapiens...")
        h5ad_dir = os.path.dirname(TABULA_H5AD)
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"joinid": "joinid", "cell_type": "cell_type"},
            nproc=4
        )
        tk.tokenize_data(
            data_directory=h5ad_dir,
            output_directory=TOKEN_DIR,
            output_prefix=output_prefix,
            file_format="h5ad"
        )
    else:
        print(f"Tokenized data already exists: {token_dataset_dir}")

    # Extract embeddings
    emb_csv = os.path.join(EMB_OUT, f"{output_prefix}_geneformer.csv")
    if not os.path.exists(emb_csv):
        print("Extracting Geneformer embeddings...")
        if not os.path.exists(GENEFORMER_MODEL_DIR):
            raise FileNotFoundError(f"Geneformer model not found: {GENEFORMER_MODEL_DIR}")

        import json
        with open(os.path.join(GENEFORMER_MODEL_DIR, "config.json"), "r") as f:
            cfg = json.load(f)
        num_classes = int(cfg.get("num_labels", 88))

        embex = EmbExtractor(
            model_type="CellClassifier",
            num_classes=num_classes,
            emb_mode="cell",
            max_ncells=None,
            emb_layer=-1,
            emb_label=["joinid"]
        )
        embs_df, embs = embex.extract_embs(
            model_directory=GENEFORMER_MODEL_DIR,
            input_data_file=token_dataset_dir,
            output_directory=EMB_OUT,
            output_prefix=f"{output_prefix}_geneformer",
            output_torch_embs=True
        )
        print(f"Embeddings extracted: {embs_df.shape}")
    else:
        print(f"Embeddings already exist: {emb_csv}")
        embs_df = pd.read_csv(emb_csv)

    # Prepare data for classification
    emb_cols = [c for c in embs_df.columns if c.startswith("emb_") or c.isdigit()]
    if len(emb_cols) == 0:
        emb_cols = [c for c in embs_df.columns if c != "joinid"]

    embs_df["joinid"] = embs_df["joinid"].astype(str)
    meta = adata.obs[["joinid", "cell_type"]].reset_index(drop=True)
    meta["joinid"] = meta["joinid"].astype(str)
    df = embs_df.merge(meta, on="joinid", how="inner").dropna()

    X = df[emb_cols].values.astype(np.float32)
    y_cat = df["cell_type"].astype("category")
    y = y_cat.cat.codes.values
    label_names = list(y_cat.cat.categories)

    # Train/test split
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

    # Train classifier
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {mf1:.4f}")

    # Save results
    results_df = pd.DataFrame([{
        "method": "geneformer_frozen",
        "dataset": "tabula_sapiens",
        "accuracy": acc,
        "macro_f1": mf1,
    }])
    results_df.to_csv(
        os.path.join(RES_DIR, "metrics_geneformer_frozen_tabula_sapiens.csv"),
        index=False
    )

    return acc, mf1


def evaluate_scgpt_frozen(adata, output_prefix="tabula_sapiens"):
    """
    Evaluate scGPT with frozen representations on Tabula Sapiens.
    """
    print("\n=== Evaluating scGPT (Frozen) on Tabula Sapiens ===")

    # Preprocess for scGPT
    ad = adata.copy()
    if "feature_name" not in ad.var.columns:
        ad.var["feature_name"] = ad.var_names.astype(str)

    # Normalize and select HVGs
    sc.pp.normalize_total(ad, target_sum=1e4, inplace=True)
    sc.pp.log1p(ad, inplace=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=3000, flavor="seurat_v3", inplace=True)
    ad_hvg = ad[:, ad.var["highly_variable"]].copy()

    # Load pretrained scGPT model
    model_dir = Path(SCGPT_MODEL_DIR)
    if not (model_dir / "best_model.pt").exists():
        print("Downloading scGPT checkpoint...")
        # Download logic here (similar to run_scgpt_pbmc3k.py)
        raise FileNotFoundError(f"scGPT model not found: {SCGPT_MODEL_DIR}")

    # Extract embeddings
    print("Extracting scGPT embeddings...")
    ad_emb = scg.tasks.embed_data(
        ad_hvg,
        model_dir,
        gene_col="feature_name",
        obs_to_save="cell_type",
        batch_size=64,
        return_new_adata=True,
    )

    emb = np.asarray(ad_emb.X)
    y = ad_emb.obs["cell_type"].astype(str).values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            emb, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            emb, y_enc, test_size=0.2, random_state=SEED
        )

    # Train classifier
    clf = LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        multi_class="auto",
        solver="lbfgs",
        random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {mf1:.4f}")

    # Save results
    results_df = pd.DataFrame([{
        "method": "scgpt_frozen",
        "dataset": "tabula_sapiens",
        "accuracy": acc,
        "macro_f1": mf1,
    }])
    results_df.to_csv(
        os.path.join(RES_DIR, "metrics_scgpt_frozen_tabula_sapiens.csv"),
        index=False
    )

    return acc, mf1


def main():
    print("=" * 60)
    print("Tabula Sapiens Cross-Dataset Evaluation")
    print("=" * 60)

    # Download/prepare Tabula Sapiens
    if not os.path.exists(TABULA_H5AD):
        print("Tabula Sapiens file not found. Attempting to download...")
        download_tabula_sapiens()

    # Load and preprocess
    print("\n=== Loading Tabula Sapiens ===")
    adata = sc.read_h5ad(TABULA_H5AD)
    print(f"Loaded: {adata.shape}")

    # Subsample for faster processing (Tabula Sapiens is very large)
    # Remove subsample_n=None to use full dataset
    subsample_n = 10000  # Use 10k cells for faster processing
    adata = preprocess_tabula_sapiens(adata, subsample_n=subsample_n)

    # Save preprocessed data
    os.makedirs(os.path.dirname(TABULA_H5AD), exist_ok=True)
    adata.write_h5ad(TABULA_H5AD)
    print(f"Saved preprocessed data: {TABULA_H5AD}")

    # Evaluate models
    print("\n" + "=" * 60)
    print("Evaluating Models on Tabula Sapiens")
    print("=" * 60)

    # Geneformer (Frozen)
    try:
        gf_acc, gf_f1 = evaluate_geneformer_frozen(adata)
    except Exception as e:
        print(f"Error evaluating Geneformer: {e}")
        gf_acc, gf_f1 = None, None

    # scGPT (Frozen)
    try:
        sc_acc, sc_f1 = evaluate_scgpt_frozen(adata)
    except Exception as e:
        print(f"Error evaluating scGPT: {e}")
        sc_acc, sc_f1 = None, None

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Tabula Sapiens Evaluation")
    print("=" * 60)
    print(f"Geneformer (Frozen): Accuracy={gf_acc:.4f}, Macro F1={gf_f1:.4f}" if gf_acc else "Geneformer: Failed")
    print(f"scGPT (Frozen): Accuracy={sc_acc:.4f}, Macro F1={sc_f1:.4f}" if sc_acc else "scGPT: Failed")

    print("\n[DONE] Tabula Sapiens evaluation completed!")


if __name__ == "__main__":
    main()
