import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Try to import scGPT, handle torchtext compatibility issues
try:
    import scgpt as scg
    from pathlib import Path as _Path
except OSError as e:
    if "torchtext" in str(e) or "libtorchtext" in str(e):
        print("[ERROR] scGPT requires torchtext, but there's a compatibility issue.")
        print("[ERROR] This is a known issue with PyTorch 2.9+ and torchtext.")
        print("[INFO] Attempting workaround...")
        # Try to work around by setting environment variable
        import os
        os.environ['TORCHTEXT_DISABLE_EXTENSION'] = '1'
        try:
            import scgpt as scg
            from pathlib import Path as _Path
            print("[INFO] Workaround successful!")
        except Exception as e2:
            print(f"[ERROR] Workaround failed: {e2}")
            print("[ERROR] Please install compatible torchtext version or use scGPT in a different environment.")
            raise
    else:
        raise


def ensure_dirs(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def download_scgpt_checkpoint(model_root: Path):
    """
    gdown --folder は環境によって「フォルダ名」を保持しないことがある。
    なので "models/" 直下に必要3点 (best_model.pt, vocab.json, args.json) があればOKにする。
    """
    model_root.mkdir(parents=True, exist_ok=True)

    # 直下に必要ファイルがあるか？
    need = ["best_model.pt", "vocab.json", "args.json"]
    if all((model_root / n).exists() for n in need):
        return model_root

    folder_id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    cmd = f"gdown --folder {folder_id} -O {model_root}"
    print(f"[INFO] Downloading scGPT checkpoint: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("gdown download failed. Check network / gdown install.")

    # 再チェック（直下 or サブフォルダのどこか）
    if all((model_root / n).exists() for n in need):
        return model_root

    # サブフォルダに入った場合も拾う
    for sub in model_root.glob("**/*"):
        if sub.is_dir() and all((sub / n).exists() for n in need):
            return sub

    raise RuntimeError(f"Checkpoint files not found under: {model_root}")



def pick_label_key(adata) -> str:
    # まず “真のラベル”がありそうなものを優先
    priority = ["cell_type", "celltype", "CellType", "labels", "label", "truth", "y", "annot", "annotation"]
    for k in priority:
        if k in adata.obs.columns:
            return k

    # 無ければLeidenで擬似ラベル（分類はできるが「精度」は参考値）
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    adata2 = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.pca(adata2, n_comps=50)
    sc.pp.neighbors(adata2)
    sc.tl.leiden(adata2, resolution=0.6, key_added="leiden_scgpt_tmp")
    adata.obs["leiden_scgpt_tmp"] = adata2.obs["leiden_scgpt_tmp"].astype(str).values
    return "leiden_scgpt_tmp"


def plot_umap(adata, out_png: Path, color_key: str):
    sc.pp.neighbors(adata, use_rep="X_scgpt")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[color_key], show=False)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, classes, out_png: Path, title="Confusion Matrix"):
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", type=str, default="", help="既にあるPBMC h5ad。無ければscanpyのpbmc3kを使う")
    ap.add_argument("--out_h5ad", type=str, default="pbmc3k_scgpt.h5ad")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    # --- Step1: PBMC load ---
    if args.input_h5ad and Path(args.input_h5ad).exists():
        adata = sc.read_h5ad(args.input_h5ad)
        print(f"[INFO] Loaded input: {args.input_h5ad}  shape={adata.shape}")
    else:
        adata = sc.datasets.pbmc3k()
        print(f"[INFO] Loaded scanpy pbmc3k()  shape={adata.shape}")

    # 保存（要求どおり）
    out_h5ad = Path(args.out_h5ad)
    adata.write(out_h5ad)
    print(f"[OK] Saved: {out_h5ad}")

    # --- label key ---
    label_key = pick_label_key(adata)
    print(f"[INFO] label_key = {label_key}")

    # --- Step2: scGPT Frozen embedding ---
    # scGPT quickstart: HVG 3000 & embed_data
    # pbmc3kはgene symbol indexなので gene_col は var_names を使う形に寄せる
    ad = adata.copy()

    # HVGを3000に揃える（推奨）
    # --- HVG: skip (input is not raw counts; HVG seurat_v3 can crash) ---
    ad_hvg = ad.copy()  # pbmc3k_for_geneformer is already 2000 genes
    ad_hvg.var["feature_name"] = ad_hvg.var_names.astype(str)
    print(f"[INFO] Skip HVG. Using all genes: {ad_hvg.shape[1]}")


    model_root = Path("./models")
    ensure_dirs(model_root)
    model_dir = download_scgpt_checkpoint(model_root)

    # gene_col: quickstartでは feature_name を指定（CZ dataset用）
    # pbmc3kは var_names が gene symbol なので、いったん var に入れて渡す
    if "feature_name" not in ad_hvg.var.columns:
        ad_hvg.var["feature_name"] = ad_hvg.var_names.astype(str)

    ref_embed_adata = scg.tasks.embed_data(
        ad_hvg,
        _Path(model_dir),
        gene_col="feature_name",
        obs_to_save=label_key,
        batch_size=args.batch_size,
        return_new_adata=True,
    )

    emb = np.asarray(ref_embed_adata.X)  # (N, 512) 想定
    np.save(outdir / "scgpt_emb.npy", emb)
    print(f"[OK] Saved: {outdir / 'scgpt_emb.npy'}  shape={emb.shape}")

    # UMAP用に格納
    adata.obsm["X_scgpt"] = emb

    # --- Step3: LogisticRegression classification ---
    y = adata.obs[label_key].astype(str).values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        emb, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
    )

    clf = LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")

    metrics = pd.DataFrame([{"accuracy": acc, "macro_f1": f1m, "label_key": label_key, "n_classes": len(le.classes_)}])
    metrics.to_csv(outdir / "metrics_scgpt.csv", index=False)
    print(f"[OK] Saved: {outdir / 'metrics_scgpt.csv'}")

    # --- Step4: UMAP + Confusion ---
    plot_umap(adata, outdir / "umap_scgpt.png", color_key=label_key)

    cm = confusion_matrix(y_test, pred, labels=np.arange(len(le.classes_)))
    plot_confusion(cm, le.classes_, outdir / "confusion_scgpt.png")

    print("[DONE] UMAP/Confusion saved in results/")

if __name__ == "__main__":
    main()
