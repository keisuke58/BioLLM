#!/usr/bin/env python3
"""
クラス別詳細分析スクリプト
per-class F1, Precision, Recallを計算し、F1=0/極端に低いクラスを特定
"""

import os
import sys
from pathlib import Path

# 必要なパッケージのチェック
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'scanpy': 'scanpy',
    'xgboost': 'xgboost'
}

missing_packages = []
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("="*60)
    print("エラー: 必要なパッケージがインストールされていません")
    print("="*60)
    print(f"不足しているパッケージ: {', '.join(missing_packages)}")
    print("\n以下のコマンドでインストールしてください:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\nまたは conda を使用する場合:")
    print(f"conda install -c conda-forge {' '.join(missing_packages)}")
    sys.exit(1)

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# 設定
RES_DIR = Path("results")
ANALYSIS_DIR = RES_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def _print_pred_distribution(y_pred, label_names, top_k=10, title="Pred distribution"):
    """予測ラベル分布を簡易表示（偏り検知用）"""
    vc = pd.Series(y_pred).value_counts().sort_values(ascending=False)
    lines = [f"{title} (top {top_k}):"]
    for cls_id, cnt in vc.head(top_k).items():
        name = label_names[int(cls_id)] if int(cls_id) < len(label_names) else str(cls_id)
        lines.append(f"  - {name}: {int(cnt)}")
    print("\n".join(lines))

def _set_thread_env_defaults():
    # 並列スレッド過多による極端な遅延/ハングを避ける（環境によってxgboostが遅くなることがある）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _fit_xgb_with_fallback(clf, X, y):
    """Try fit; if GPU tree_method unsupported, fall back to hist."""
    try:
        clf.fit(X, y)
        return clf
    except Exception as e:
        # Common when xgboost isn't built with CUDA
        from xgboost import XGBClassifier

        params = clf.get_params()
        params["tree_method"] = "hist"
        params.pop("predictor", None)
        fallback = XGBClassifier(**params)
        fallback.fit(X, y)
        return fallback

def load_geneformer_table():
    """Geneformer 埋め込み + cell_type を joinid で結合して返す（モデル比較用にrow単位で揃える）"""
    import scanpy as sc

    emb_csv = RES_DIR / "geneformer_emb" / "pbmc3k.csv"
    if not emb_csv.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_csv}")

    embs = pd.read_csv(emb_csv)
    if "joinid" not in embs.columns:
        raise ValueError(f"'joinid' column not found in {emb_csv}")

    emb_cols = [c for c in embs.columns if c.startswith("emb_") or (c.isdigit() and c != "joinid")]
    if len(emb_cols) == 0:
        emb_cols = [c for c in embs.columns if c != "joinid"]
    if len(emb_cols) == 0:
        raise ValueError(f"Embedding columns not found in {emb_csv}")

    h5ad_path = Path("artifacts/h5ad/pbmc3k_for_geneformer.h5ad")
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}\nPlease run run_geneformer_pbmc3k.py first.")

    adata = sc.read_h5ad(h5ad_path)
    if "joinid" not in adata.obs.columns or "cell_type" not in adata.obs.columns:
        raise ValueError("Required keys not found in adata.obs. Need: joinid, cell_type")

    meta = adata.obs[["joinid", "cell_type"]].reset_index(drop=True)

    embs["joinid"] = embs["joinid"].astype(str)
    meta["joinid"] = meta["joinid"].astype(str)

    df = embs.merge(meta, on="joinid", how="inner").dropna()
    if len(df) == 0:
        raise ValueError("No matching rows found after merging Geneformer embeddings and metadata")

    return df[["joinid", "cell_type"] + emb_cols], emb_cols

def load_scgpt_table():
    """scGPT 埋め込み + cell_type + joinid を返す（Geneformer側のsubsetに合わせてフィルタ可能にする前提）"""
    import scanpy as sc

    emb_path = RES_DIR / "scgpt_emb.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}\nPlease run run_scgpt_pbmc3k.py first.")

    emb = np.load(emb_path)

    # scGPT実行時に保存されたh5adを優先（順序一致が前提）
    h5ad_path = Path("pbmc3k_scgpt.h5ad")
    if not h5ad_path.exists():
        # 互換h5adがある場合はこちらを使う
        compat = Path("artifacts/h5ad/pbmc3k_for_scgpt_compat.h5ad")
        if compat.exists():
            h5ad_path = compat
        else:
            raise FileNotFoundError(
                "H5AD for scGPT not found. Need pbmc3k_scgpt.h5ad (recommended) "
                "or artifacts/h5ad/pbmc3k_for_scgpt_compat.h5ad."
            )

    adata = sc.read_h5ad(h5ad_path)
    if "joinid" not in adata.obs.columns:
        raise ValueError("Required key 'joinid' not found in scGPT adata.obs")

    # ラベルキー（基本はcell_type）
    label_key = None
    for k in ["cell_type", "celltype", "CellType", "labels", "label"]:
        if k in adata.obs.columns:
            label_key = k
            break
    if label_key is None:
        raise ValueError("Label key not found in adata.obs. Available columns: " + ", ".join(adata.obs.columns))

    if emb.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding rows ({emb.shape[0]}) doesn't match adata.n_obs ({adata.n_obs})")

    joinid = adata.obs["joinid"].astype(str).values
    y = adata.obs[label_key].astype(str).values

    df = pd.DataFrame({"joinid": joinid, "cell_type": y})
    return df, emb.astype(np.float32)

def prepare_aligned_dataset():
    """
    Geneformer/scGPT を joinid で揃えた学習用配列を一度だけ作る（seed sweep高速化用）。
    Returns:
      dict with keys:
        X_gf_all, X_scgpt_all, y_all, label_names, n_common
    """
    from sklearn.preprocessing import LabelEncoder

    _set_thread_env_defaults()

    gf_df, gf_emb_cols = load_geneformer_table()
    scgpt_meta_full, scgpt_emb_full = load_scgpt_table()

    gf_sub = gf_df[["joinid", "cell_type"] + gf_emb_cols].copy()
    gf_sub["joinid"] = gf_sub["joinid"].astype(str)

    scgpt_meta_full = scgpt_meta_full.copy()
    scgpt_meta_full["joinid"] = scgpt_meta_full["joinid"].astype(str)

    gf_ids = set(gf_sub["joinid"].tolist())
    scgpt_ids = set(scgpt_meta_full["joinid"].tolist())
    common = sorted(gf_ids.intersection(scgpt_ids))
    if len(common) == 0:
        raise ValueError("No overlapping joinid between Geneformer embeddings and scGPT h5ad")

    gf_sub = gf_sub[gf_sub["joinid"].isin(common)].sort_values("joinid").reset_index(drop=True)

    scgpt_joinids = scgpt_meta_full["joinid"].tolist()
    if len(scgpt_joinids) != len(set(scgpt_joinids)):
        raise ValueError("Duplicate joinid found in scGPT h5ad; cannot build a stable mapping")
    joinid_to_row = {jid: i for i, jid in enumerate(scgpt_joinids)}
    rows = np.array([joinid_to_row[jid] for jid in gf_sub["joinid"].tolist()], dtype=int)

    X_scgpt_all = scgpt_emb_full[rows].astype(np.float32)
    scgpt_labels_all = scgpt_meta_full.iloc[rows]["cell_type"].astype(str).values

    y_str = gf_sub["cell_type"].astype(str).values
    if not np.array_equal(y_str, scgpt_labels_all):
        raise ValueError("cell_type mismatch after joinid alignment. Check source h5ad consistency.")

    le = LabelEncoder()
    y_all = le.fit_transform(y_str)
    label_names = list(le.classes_)

    X_gf_all = gf_sub[gf_emb_cols].values.astype(np.float32)

    return {
        "X_gf_all": X_gf_all,
        "X_scgpt_all": X_scgpt_all,
        "y_all": y_all,
        "label_names": label_names,
        "n_common": int(len(gf_sub)),
    }

def run_models_on_prepared(
    prepared: dict,
    *,
    seed: int,
    test_size: float,
    gf_head: str,
    scgpt_head: str,
    gf_n_estimators: int,
    scgpt_n_estimators: int,
):
    """prepare_aligned_dataset() の戻り値を使って1回分の学習/評価を回す。"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier

    _set_thread_env_defaults()

    X_gf_all = prepared["X_gf_all"]
    X_scgpt_all = prepared["X_scgpt_all"]
    y_all = prepared["y_all"]
    label_names = prepared["label_names"]

    idx = np.arange(len(y_all))
    try:
        tr_idx, te_idx = train_test_split(idx, test_size=float(test_size), random_state=int(seed), stratify=y_all)
    except ValueError:
        tr_idx, te_idx = train_test_split(idx, test_size=float(test_size), random_state=int(seed))

    # Geneformer head
    if gf_head == "xgb":
        # Prefer GPU if available; fall back to CPU hist if unsupported
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        try:
            import torch

            if not torch.cuda.is_available():
                tree_method = "hist"
                predictor = None
        except Exception:
            tree_method = "hist"
            predictor = None

        gf_clf = XGBClassifier(
            n_estimators=int(gf_n_estimators),
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method=tree_method,
            random_state=int(seed),
            n_jobs=1,
            predictor=predictor,
        )
        gf_clf = _fit_xgb_with_fallback(gf_clf, X_gf_all[tr_idx], y_all[tr_idx])
        y_pred_gf = gf_clf.predict(X_gf_all[te_idx])
    elif gf_head == "sgd":
        gf_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=1e-4,
                        max_iter=2000,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        gf_clf.fit(X_gf_all[tr_idx], y_all[tr_idx])
        y_pred_gf = gf_clf.predict(X_gf_all[te_idx])
    else:
        raise ValueError(f"Unknown gf_head: {gf_head}")

    # scGPT head
    if scgpt_head == "xgb":
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        try:
            import torch

            if not torch.cuda.is_available():
                tree_method = "hist"
                predictor = None
        except Exception:
            tree_method = "hist"
            predictor = None

        scgpt_clf = XGBClassifier(
            n_estimators=int(scgpt_n_estimators),
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method=tree_method,
            random_state=int(seed),
            n_jobs=1,
            predictor=predictor,
        )
        scgpt_clf = _fit_xgb_with_fallback(scgpt_clf, X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    elif scgpt_head == "logreg":
        logreg = LogisticRegression(
            max_iter=2000,
            tol=1e-2,
            multi_class="multinomial",
            solver="saga",
            class_weight="balanced",
            random_state=int(seed),
        )
        scgpt_clf = Pipeline([("scaler", StandardScaler()), ("clf", logreg)])
        scgpt_clf.fit(X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    elif scgpt_head == "sgd":
        sgd = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
        )
        scgpt_clf = Pipeline([("scaler", StandardScaler()), ("clf", sgd)])
        scgpt_clf.fit(X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    else:
        raise ValueError(f"Unknown scgpt_head: {scgpt_head}")

    y_true = y_all[te_idx]

    gf_acc = float(accuracy_score(y_true, y_pred_gf))
    gf_f1m = float(f1_score(y_true, y_pred_gf, average="macro", zero_division=0))
    sc_acc = float(accuracy_score(y_true, y_pred_scgpt))
    sc_f1m = float(f1_score(y_true, y_pred_scgpt, average="macro", zero_division=0))

    gf_f1m_no_platelet = None
    sc_f1m_no_platelet = None
    if "Platelet" in label_names:
        platelet_id = label_names.index("Platelet")
        labels_no_platelet = [i for i in range(len(label_names)) if i != platelet_id]
        gf_f1m_no_platelet = float(
            f1_score(y_true, y_pred_gf, labels=labels_no_platelet, average="macro", zero_division=0)
        )
        sc_f1m_no_platelet = float(
            f1_score(y_true, y_pred_scgpt, labels=labels_no_platelet, average="macro", zero_division=0)
        )

    gf_per_class = calculate_per_class_metrics(y_true, y_pred_gf, label_names)
    scgpt_per_class = calculate_per_class_metrics(y_true, y_pred_scgpt, label_names)

    overall = {
        "seed": int(seed),
        "test_size": float(test_size),
        "n_common": int(prepared["n_common"]),
        "n_test": int(len(te_idx)),
        "gf_head": gf_head,
        "scgpt_head": scgpt_head,
        "gf_accuracy": gf_acc,
        "gf_macro_f1": gf_f1m,
        "gf_macro_f1_no_platelet": gf_f1m_no_platelet,
        "scgpt_accuracy": sc_acc,
        "scgpt_macro_f1": sc_f1m,
        "scgpt_macro_f1_no_platelet": sc_f1m_no_platelet,
    }

    return overall, gf_per_class, scgpt_per_class

def run_analysis(
    *,
    scgpt_head: str = "xgb",
    gf_head: str = "xgb",
    seed: int = 42,
    test_size: float = 0.2,
    gf_n_estimators: int = 300,
    scgpt_n_estimators: int = 300,
    save_suffix: str | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
):
    """
    1回分の評価を実行し、(overall metrics, per-class metrics) を返す。
    save_outputs=True のときは results/analysis 以下にCSV/MDを保存する。
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier

    _set_thread_env_defaults()

    SEED = int(seed)
    TEST_SIZE = float(test_size)
    if save_suffix is None:
        save_suffix = f"gf-{gf_head}_scgpt-{scgpt_head}_seed-{SEED}"

    # --- load / align ---
    if verbose:
        print("\n[1/4] 埋め込み・ラベルを読み込み中（Geneformer / scGPT）...")
    gf_df, gf_emb_cols = load_geneformer_table()
    scgpt_meta_full, scgpt_emb_full = load_scgpt_table()

    gf_sub = gf_df[["joinid", "cell_type"] + gf_emb_cols].copy()
    gf_sub["joinid"] = gf_sub["joinid"].astype(str)
    scgpt_meta_full = scgpt_meta_full.copy()
    scgpt_meta_full["joinid"] = scgpt_meta_full["joinid"].astype(str)

    gf_ids = set(gf_sub["joinid"].tolist())
    scgpt_ids = set(scgpt_meta_full["joinid"].tolist())
    common = sorted(gf_ids.intersection(scgpt_ids))
    if len(common) == 0:
        raise ValueError("No overlapping joinid between Geneformer embeddings and scGPT h5ad")

    gf_sub = gf_sub[gf_sub["joinid"].isin(common)].sort_values("joinid").reset_index(drop=True)

    scgpt_joinids = scgpt_meta_full["joinid"].tolist()
    if len(scgpt_joinids) != len(set(scgpt_joinids)):
        raise ValueError("Duplicate joinid found in scGPT h5ad; cannot build a stable mapping")
    joinid_to_row = {jid: i for i, jid in enumerate(scgpt_joinids)}
    rows = np.array([joinid_to_row[jid] for jid in gf_sub["joinid"].tolist()], dtype=int)
    X_scgpt_all = scgpt_emb_full[rows].astype(np.float32)
    scgpt_labels_all = scgpt_meta_full.iloc[rows]["cell_type"].astype(str).values

    y_str = gf_sub["cell_type"].astype(str).values
    if not np.array_equal(y_str, scgpt_labels_all):
        raise ValueError("cell_type mismatch after joinid alignment. Check source h5ad consistency.")

    le = LabelEncoder()
    y_all = le.fit_transform(y_str)
    label_names = list(le.classes_)

    idx = np.arange(len(gf_sub))
    try:
        tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y_all)
    except ValueError:
        tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)

    X_gf_all = gf_sub[gf_emb_cols].values.astype(np.float32)

    if verbose:
        print(f"  - 共通サンプル数: {len(gf_sub)}  (Geneformer subsetに合わせてscGPTもフィルタ)")
        print(f"  - テストサンプル数: {len(te_idx)}")
        print(f"  - クラス数: {len(label_names)}")

    # --- Geneformer head ---
    if verbose:
        print("[2/4] Geneformer分類器を学習・予測中...")
    if gf_head == "xgb":
        gf_clf = XGBClassifier(
            n_estimators=int(gf_n_estimators),
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        )
        gf_clf.fit(X_gf_all[tr_idx], y_all[tr_idx])
        y_pred_gf = gf_clf.predict(X_gf_all[te_idx])
    elif gf_head == "sgd":
        gf_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=1e-4,
                        max_iter=2000,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        gf_clf.fit(X_gf_all[tr_idx], y_all[tr_idx])
        y_pred_gf = gf_clf.predict(X_gf_all[te_idx])
    else:
        raise ValueError(f"Unknown gf_head: {gf_head}")

    if verbose:
        _print_pred_distribution(y_pred_gf, label_names, title="Geneformer y_pred distribution")

    # --- scGPT head ---
    if verbose:
        print(f"[3/4] scGPT分類器を学習・予測中...  (head={scgpt_head})")

    if scgpt_head == "xgb":
        scgpt_clf = XGBClassifier(
            n_estimators=int(scgpt_n_estimators),
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        )
        scgpt_clf.fit(X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    elif scgpt_head == "logreg":
        # 環境によって収束が重いことがある（必要なら sgd を推奨）
        logreg = LogisticRegression(
            max_iter=2000,
            tol=1e-2,
            multi_class="multinomial",
            solver="saga",
            class_weight="balanced",
            random_state=SEED,
        )
        scgpt_clf = Pipeline([("scaler", StandardScaler()), ("clf", logreg)])
        scgpt_clf.fit(X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    elif scgpt_head == "sgd":
        sgd = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            random_state=SEED,
        )
        scgpt_clf = Pipeline([("scaler", StandardScaler()), ("clf", sgd)])
        scgpt_clf.fit(X_scgpt_all[tr_idx], y_all[tr_idx])
        y_pred_scgpt = scgpt_clf.predict(X_scgpt_all[te_idx])
    else:
        raise ValueError(f"Unknown scgpt_head: {scgpt_head}")

    if verbose:
        _print_pred_distribution(y_pred_scgpt, label_names, title="scGPT y_pred distribution")

    y_true = y_all[te_idx]

    # --- overall metrics ---
    gf_acc = float(accuracy_score(y_true, y_pred_gf))
    gf_f1m = float(f1_score(y_true, y_pred_gf, average="macro", zero_division=0))
    sc_acc = float(accuracy_score(y_true, y_pred_scgpt))
    sc_f1m = float(f1_score(y_true, y_pred_scgpt, average="macro", zero_division=0))

    # --- macro F1 excluding Platelet (if present) ---
    gf_f1m_no_platelet = None
    sc_f1m_no_platelet = None
    if "Platelet" in label_names:
        platelet_id = label_names.index("Platelet")
        labels_no_platelet = [i for i in range(len(label_names)) if i != platelet_id]
        # f1_score(..., labels=...) computes over those labels only (macro across them)
        gf_f1m_no_platelet = float(
            f1_score(y_true, y_pred_gf, labels=labels_no_platelet, average="macro", zero_division=0)
        )
        sc_f1m_no_platelet = float(
            f1_score(y_true, y_pred_scgpt, labels=labels_no_platelet, average="macro", zero_division=0)
        )

    # --- per-class + confusion report ---
    if verbose:
        print("[4/4] クラス別メトリクスと混同行列を計算中...")
    gf_per_class = calculate_per_class_metrics(y_true, y_pred_gf, label_names)
    scgpt_per_class = calculate_per_class_metrics(y_true, y_pred_scgpt, label_names)
    gf_confusion = analyze_confusion_top3(y_true, y_pred_gf, label_names)
    scgpt_confusion = analyze_confusion_top3(y_true, y_pred_scgpt, label_names)

    report = create_worst_classes_report(gf_per_class, scgpt_per_class, gf_confusion, scgpt_confusion)

    if save_outputs:
        gf_per_class.to_csv(ANALYSIS_DIR / "per_class_geneformer.csv", index=False)
        scgpt_per_class.to_csv(ANALYSIS_DIR / "per_class_scgpt.csv", index=False)
        scgpt_per_class.to_csv(ANALYSIS_DIR / f"per_class_scgpt_{scgpt_head}.csv", index=False)
        with open(ANALYSIS_DIR / "worst_classes.md", "w", encoding="utf-8") as f:
            f.write(report)
        with open(ANALYSIS_DIR / f"worst_classes_{scgpt_head}.md", "w", encoding="utf-8") as f:
            f.write(report)

    overall = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_common": int(len(gf_sub)),
        "n_test": int(len(te_idx)),
        "gf_head": gf_head,
        "scgpt_head": scgpt_head,
        "gf_accuracy": gf_acc,
        "gf_macro_f1": gf_f1m,
        "gf_macro_f1_no_platelet": gf_f1m_no_platelet,
        "scgpt_accuracy": sc_acc,
        "scgpt_macro_f1": sc_f1m,
        "scgpt_macro_f1_no_platelet": sc_f1m_no_platelet,
    }

    return overall, gf_per_class, scgpt_per_class

def calculate_per_class_metrics(y_true, y_pred, label_names):
    """クラス別メトリクスを計算"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )
    
    df = pd.DataFrame({
        "class": label_names,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    
    return df

def analyze_confusion_top3(y_true, y_pred, label_names):
    """混同行列から混同先Top3を分析"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    
    confusion_analysis = []
    for i, true_class in enumerate(label_names):
        row = cm[i, :]
        # 自分自身を除く
        row_no_self = row.copy()
        row_no_self[i] = 0
        
        # Top3の混同先
        top3_indices = np.argsort(row_no_self)[-3:][::-1]
        top3_confusions = []
        for idx in top3_indices:
            if row_no_self[idx] > 0:
                top3_confusions.append({
                    "predicted_class": label_names[idx],
                    "count": int(row_no_self[idx])
                })
        
        confusion_analysis.append({
            "true_class": true_class,
            "correct": int(row[i]),
            "total": int(row.sum()),
            "top3_confusions": top3_confusions
        })
    
    return confusion_analysis

def create_worst_classes_report(gf_df, scgpt_df, gf_confusion, scgpt_confusion):
    """F1下位5と混同先Top3のレポートを作成"""
    report_lines = [
        "# 最悪パフォーマンスクラス分析",
        "",
        "## Geneformer",
        "",
        "### F1下位5クラス",
        ""
    ]
    
    # Geneformer F1下位5
    gf_sorted = gf_df.sort_values("f1").head(5)
    for idx, row in gf_sorted.iterrows():
        report_lines.append(f"- **{row['class']}**: F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}, Support={int(row['support'])}")
    
    report_lines.extend([
        "",
        "### 混同先Top3（F1下位5クラス）",
        ""
    ])
    
    worst_classes = gf_sorted['class'].tolist()
    for conf in gf_confusion:
        if conf['true_class'] in worst_classes:
            report_lines.append(f"#### {conf['true_class']}")
            report_lines.append(f"- 正解数: {conf['correct']}/{conf['total']}")
            if conf['top3_confusions']:
                report_lines.append("- 主な混同先:")
                for c in conf['top3_confusions']:
                    report_lines.append(f"  - {c['predicted_class']}: {c['count']}回")
            report_lines.append("")
    
    report_lines.extend([
        "## scGPT",
        "",
        "### F1下位5クラス",
        ""
    ])
    
    # scGPT F1下位5
    scgpt_sorted = scgpt_df.sort_values("f1").head(5)
    for idx, row in scgpt_sorted.iterrows():
        report_lines.append(f"- **{row['class']}**: F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}, Support={int(row['support'])}")
    
    report_lines.extend([
        "",
        "### 混同先Top3（F1下位5クラス）",
        ""
    ])
    
    worst_classes_scgpt = scgpt_sorted['class'].tolist()
    for conf in scgpt_confusion:
        if conf['true_class'] in worst_classes_scgpt:
            report_lines.append(f"#### {conf['true_class']}")
            report_lines.append(f"- 正解数: {conf['correct']}/{conf['total']}")
            if conf['top3_confusions']:
                report_lines.append("- 主な混同先:")
                for c in conf['top3_confusions']:
                    report_lines.append(f"  - {c['predicted_class']}: {c['count']}回")
            report_lines.append("")
    
    return "\n".join(report_lines)

def main():
    print("="*60)
    print("クラス別詳細分析")
    print("="*60)
    
    try:
        import argparse

        ap = argparse.ArgumentParser(description="Per-class analysis with aligned split (Geneformer vs scGPT)")
        ap.add_argument(
            "--scgpt_head",
            type=str,
            default="xgb",
            choices=["xgb", "logreg", "sgd"],
            help="Classifier head for scGPT embeddings. Default: xgb (same as Geneformer).",
        )
        ap.add_argument(
            "--gf_head",
            type=str,
            default="xgb",
            choices=["xgb", "sgd"],
            help="Classifier head for Geneformer embeddings. Default: xgb.",
        )
        ap.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for shared train/test split.",
        )
        ap.add_argument(
            "--test_size",
            type=float,
            default=0.2,
            help="Test size ratio for shared split.",
        )
        ap.add_argument("--gf_n_estimators", type=int, default=300, help="n_estimators for Geneformer XGB head.")
        ap.add_argument("--scgpt_n_estimators", type=int, default=300, help="n_estimators for scGPT XGB head.")
        args = ap.parse_args()

        overall, gf_per_class, scgpt_per_class = run_analysis(
            scgpt_head=args.scgpt_head,
            gf_head=args.gf_head,
            seed=args.seed,
            test_size=args.test_size,
            gf_n_estimators=args.gf_n_estimators,
            scgpt_n_estimators=args.scgpt_n_estimators,
            save_suffix=None,
            save_outputs=True,
            verbose=True,
        )

        print(f"[OK] Saved: {ANALYSIS_DIR / 'per_class_geneformer.csv'}")
        print(f"[OK] Saved: {ANALYSIS_DIR / 'per_class_scgpt.csv'}")
        print(f"[OK] Saved: {ANALYSIS_DIR / f'per_class_scgpt_{args.scgpt_head}.csv'}")
        print(f"[OK] Saved: {ANALYSIS_DIR / 'worst_classes.md'}")
        print(f"[OK] Saved: {ANALYSIS_DIR / f'worst_classes_{args.scgpt_head}.md'}")
        
        # 結果を表示
        print("\n" + "="*60)
        print("Geneformer - F1下位5クラス:")
        print(gf_per_class.sort_values("f1").head(5)[["class", "f1", "precision", "recall", "support"]].to_string(index=False))
        
        print("\n" + "="*60)
        print("scGPT - F1下位5クラス:")
        print(scgpt_per_class.sort_values("f1").head(5)[["class", "f1", "precision", "recall", "support"]].to_string(index=False))
        
        print("\n" + "="*60)
        print("Overall metrics (same split):")
        print(
            pd.DataFrame([overall])[["gf_accuracy", "gf_macro_f1", "scgpt_accuracy", "scgpt_macro_f1"]]
            .to_string(index=False)
        )

        print(f"\n[完了] 分析結果は {ANALYSIS_DIR} に保存されました。")
        
    except FileNotFoundError as e:
        print(f"\n[エラー] ファイルが見つかりません: {e}")
        print("\n必要なファイル:")
        print("  - results/geneformer_emb/pbmc3k.csv")
        print("  - results/scgpt_emb.npy")
        print("  - artifacts/h5ad/pbmc3k_for_geneformer.h5ad (または pbmc3k_scgpt.h5ad)")
        print("\n先に run_geneformer_pbmc3k.py と run_scgpt_pbmc3k.py を実行してください。")
    except Exception as e:
        print(f"\n[エラー] 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("\nトラブルシューティング:")
        print("1. 必要なパッケージがインストールされているか確認")
        print("2. 必要なファイルが存在するか確認")
        print("3. Python環境が正しいか確認")

if __name__ == "__main__":
    main()
