"""
Evaluate Geneformer with frozen representations on PBMC3k dataset.

This script:
1. Downloads and preprocesses PBMC3k dataset
2. Tokenizes data for Geneformer
3. Extracts embeddings from frozen (pretrained) Geneformer model
4. Trains XGBoost classifier on extracted embeddings
5. Evaluates performance and saves results

Author: Keisuke Nishioka
Course: AI Foundation Models in Biomedicine, WiSe 2025/26
"""
import os, random
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import umap

from geneformer import TranscriptomeTokenizer, EmbExtractor

import glob
import json

# -------------------------
# Configuration
# -------------------------
SEED = 42
DATA_URL = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
WORKDIR = os.getcwd()

DATA_DIR = os.path.join(WORKDIR, "data")
ART_DIR  = os.path.join(WORKDIR, "artifacts")
RES_DIR  = os.path.join(WORKDIR, "results")

PBMC_TAR = os.path.join(DATA_DIR, "pbmc3k.tar.gz")
PBMC_EXTRACT = os.path.join(DATA_DIR, "pbmc3k")
MTX_DIR = os.path.join(PBMC_EXTRACT, "filtered_gene_bc_matrices", "hg19")

H5AD_DIR  = os.path.join(ART_DIR, "h5ad")
H5AD_PATH = os.path.join(H5AD_DIR, "pbmc3k_for_geneformer.h5ad")  # write用に残してOK
TOKEN_DIR = os.path.join(ART_DIR, "tokenized_pbmc3k")

MODEL_DIR = os.path.join(ART_DIR, "fine_tuned_geneformer")  # you already synced from S3 earlier
EMB_OUT   = os.path.join(RES_DIR, "geneformer_emb")

TOKEN_DATASET_DIR = os.path.join(TOKEN_DIR, "pbmc3k")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(EMB_OUT, exist_ok=True)

# ---- find actual HF dataset dir (robust) ----
candidates = glob.glob(os.path.join(TOKEN_DIR, "**", "dataset_info.json"), recursive=True)
if len(candidates) == 0:
    raise FileNotFoundError(f"No dataset_info.json found under {TOKEN_DIR}")

TOKEN_DATASET_DIR = os.path.dirname(candidates[0])
print("Detected TOKEN_DATASET_DIR =", TOKEN_DATASET_DIR)


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
# 1) Download + load PBMC 3k
# -------------------------
if not os.path.exists(PBMC_TAR):
    os.system(f"wget -nv -O {PBMC_TAR} {DATA_URL}")
if not os.path.exists(PBMC_EXTRACT):
    os.system(f"mkdir -p {PBMC_EXTRACT}")
    os.system(f"tar -xzf {PBMC_TAR} -C {PBMC_EXTRACT}")

adata = sc.read_10x_mtx(MTX_DIR, var_names="gene_symbols", make_unique=True)
genes_tsv = pd.read_csv(os.path.join(MTX_DIR, "genes.tsv"), sep="\t", header=None)
adata.var["ensembl_id"] = genes_tsv[0].values
adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
adata.obs["joinid"] = np.arange(adata.n_obs).astype(str)

# -------------------------
# Step 2: Preprocess data and assign cell type labels
# -------------------------
# Performs standard single-cell preprocessing (filtering, normalization, scaling)
# and assigns cell type labels using Leiden clustering and marker gene scoring
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.raw = adata.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.leiden(adata, resolution=0.6, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(adata)

marker_sets = {
    "T": ["CD3D","CD3E","TRAC"],
    "NK": ["NKG7","GNLY"],
    "B": ["MS4A1","CD79A"],
    "Mono": ["LYZ","S100A8","S100A9","FCGR3A"],
    "DC": ["FCER1A","CST3"],
    "Platelet": ["PPBP","PF4"]
}

for name, genes in marker_sets.items():
    present = [g for g in genes if g in adata.var_names]
    if len(present) == 0:
        adata.obs[f"score_{name}"] = 0.0
    else:
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}")

cluster_labels = {}
for cl in adata.obs["leiden"].unique():
    idx = (adata.obs["leiden"] == cl)
    means = {k: float(adata.obs.loc[idx, f"score_{k}"].mean()) for k in marker_sets.keys()}
    cluster_labels[cl] = max(means, key=means.get)

adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).astype("category")

# save label UMAP
sc.pl.umap(adata, color=["leiden","cell_type"], wspace=0.4, show=False)
plt.savefig(os.path.join(RES_DIR, "umap_labels_pbmc3k.png"), dpi=200, bbox_inches="tight")
plt.close()

adata.write_h5ad(H5AD_PATH)

print("PBMC loaded:", adata.shape, "classes:", adata.obs["cell_type"].nunique())

# -------------------------
# 3) Tokenize
# -------------------------
os.makedirs(TOKEN_DIR, exist_ok=True)
tk = TranscriptomeTokenizer(custom_attr_name_dict={"joinid":"joinid"}, nproc=4)

# If token dir exists from previous run, skip tokenization
if len(os.listdir(TOKEN_DIR)) == 0:
    tk.tokenize_data(data_directory=H5AD_DIR, output_directory=TOKEN_DIR, output_prefix="pbmc3k",file_format="h5ad")
    print("Tokenization done.")
else:
    print("Token dir not empty -> skip tokenization:", TOKEN_DIR)

# -------------------------
# Step 4: Extract embeddings from frozen (pretrained) Geneformer model
# -------------------------
# Extracts cell embeddings from the pretrained model without fine-tuning
# These frozen embeddings will be used to train a downstream classifier
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"MODEL_DIR not found: {MODEL_DIR}\n"
        "Please sync the public geneformer fine-tuned model into artifacts/fine_tuned_geneformer first."
    )

emb_csv = os.path.join(EMB_OUT, "pbmc3k.csv")
need_extract = not os.path.exists(emb_csv)
if not need_extract:
    # EmbExtractor default is max_ncells=1000; if a previous run produced a 1000-row CSV,
    # automatically back it up and regenerate full embeddings for all cells.
    try:
        prev_n = len(pd.read_csv(emb_csv))
        if prev_n < adata.n_obs:
            backup_csv = os.path.join(EMB_OUT, f"pbmc3k_{prev_n}.csv")
            print(f"[INFO] Existing embeddings look downsampled: {prev_n} < {adata.n_obs}")
            print(f"[INFO] Backing up: {emb_csv} -> {backup_csv}")
            os.replace(emb_csv, backup_csv)
            need_extract = True
    except Exception as e:
        print(f"[WARN] Could not inspect existing {emb_csv}: {e}")
        print("[WARN] Re-extracting embeddings to be safe.")
        need_extract = True

if need_extract:
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    num_classes = int(cfg.get("num_labels", 88))
    print("Detected num_classes from config:", num_classes)

    embex = EmbExtractor(
        model_type="CellClassifier",
        num_classes=num_classes,
        emb_mode="cell",
        # IMPORTANT: extract embeddings for ALL cells (not the default 1000 sampled)
        max_ncells=None,
        emb_layer=-1,
        emb_label=["joinid"]
    )
    embs_df, embs = embex.extract_embs(
        model_directory=MODEL_DIR,
        input_data_file=TOKEN_DATASET_DIR,
        output_directory=EMB_OUT,
        output_prefix="pbmc3k",
        output_torch_embs=True
    )
    print("Embeddings extracted:", embs_df.shape)
else:
    print("Embeddings already exist:", emb_csv)

# -------------------------
# 5) Train classifier + metrics
# -------------------------
embs = pd.read_csv(emb_csv)

emb_cols = [c for c in embs.columns if c.startswith("emb_") or c.isdigit()]
if len(emb_cols) == 0:
    emb_cols = [c for c in embs.columns if c != "joinid"]

# Ensure joinid is string type in both DataFrames for merging
embs["joinid"] = embs["joinid"].astype(str)
meta = adata.obs[["joinid", "cell_type"]].reset_index(drop=True)
meta["joinid"] = meta["joinid"].astype(str)
df = embs.merge(meta, on="joinid", how="inner").dropna()

X = df[emb_cols].values.astype(np.float32)
y_cat = df["cell_type"].astype("category")
y = y_cat.cat.codes.values
label_names = list(y_cat.cat.categories)

# Stratify split (fallback if any class has < 2 samples)
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
except ValueError:
    # If stratify fails (e.g., Platelet has only 1 sample), use non-stratified split
    print("Warning: Stratify failed (likely due to class with < 2 samples), using non-stratified split")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist"
)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

acc = accuracy_score(y_te, pred)
mf1 = f1_score(y_te, pred, average="macro")

print("Accuracy:", acc)
print("Macro-F1:", mf1)
print(classification_report(y_te, pred, target_names=label_names, zero_division=0))

# Platelet除外版の評価（少数クラス除外で現実的なMacro-F1を計算）
acc_no_platelet = None
mf1_no_platelet = None
platelet_idx = label_names.index("Platelet") if "Platelet" in label_names else None
if platelet_idx is not None:
    mask = y_te != platelet_idx
    if mask.sum() > 0:  # Platelet以外のサンプルが存在する場合
        acc_no_platelet = accuracy_score(y_te[mask], pred[mask])
        mf1_no_platelet = f1_score(y_te[mask], pred[mask], average="macro")
        print("\n--- Platelet除外版 ---")
        print("Accuracy (no Platelet):", acc_no_platelet)
        print("Macro-F1 (no Platelet):", mf1_no_platelet)
        print(classification_report(y_te[mask], pred[mask], 
                                    target_names=[l for l in label_names if l != "Platelet"], 
                                    zero_division=0))

pd.DataFrame(
    {"metric":["accuracy","macro_f1","accuracy_no_platelet","macro_f1_no_platelet"], 
     "value":[acc, mf1, acc_no_platelet, mf1_no_platelet]}
).to_csv(os.path.join(RES_DIR, "metrics_geneformer_pbmc3k.csv"), index=False)

# confusion matrix
cm = confusion_matrix(y_te, pred)
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix (Geneformer emb + LogReg) - PBMC3k")
plt.xlabel("Pred"); plt.ylabel("True")
plt.xticks(range(len(label_names)), label_names, rotation=45, ha="right")
plt.yticks(range(len(label_names)), label_names)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "confusion_geneformer_pbmc3k.png"), dpi=200)
plt.close()

# -------------------------
# Step 6: Visualize embeddings using UMAP
# -------------------------
# Creates UMAP visualizations of embeddings colored by:
# - Cell type labels
# - Classification errors (for test set)
Z = umap.UMAP().fit_transform(X)  # random_state removed for parallelism

plt.figure(figsize=(6,5))
plt.scatter(Z[:,0], Z[:,1], s=6, c=y, alpha=0.85)
plt.title("UMAP of Geneformer Cell Embeddings - PBMC3k")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "umap_geneformer_emb_pbmc3k.png"), dpi=200)
plt.close()

# UMAP colored by errors
err = (pred != y_te).astype(int)
Z_te = umap.UMAP().fit_transform(X_te)  # random_state removed for parallelism
plt.figure(figsize=(6,5))
plt.scatter(Z_te[:,0], Z_te[:,1], s=10, c=err, alpha=0.85)
plt.title("UMAP (Test) Error Map - PBMC3k (0=correct,1=wrong)")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "umap_test_error_pbmc3k.png"), dpi=200)
plt.close()

print("DONE. Outputs in:", RES_DIR)
