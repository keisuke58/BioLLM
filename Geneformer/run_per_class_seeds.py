#!/usr/bin/env python3
"""
Seed sweep runner for analyze_per_class.py

- Runs multiple seeds with aligned split.
- Saves a summary CSV with overall metrics and per-class F1 columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_per_class import prepare_aligned_dataset, run_models_on_prepared, run_analysis


def _build_f1_table(mean_row: pd.Series, std_row: pd.Series, prefix: str) -> pd.DataFrame:
    cols = [c for c in mean_row.index if c.startswith(prefix)]
    rows = []
    for c in cols:
        cls = c.split("__", 1)[1] if "__" in c else c
        rows.append(
            {
                "class": cls,
                "f1_mean": float(mean_row[c]),
                "f1_std": float(std_row[c]),
            }
        )
    return pd.DataFrame(rows).sort_values(["f1_std", "f1_mean"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=5, help="Number of runs (seeds).")
    ap.add_argument("--seed_start", type=int, default=0, help="Start seed (inclusive).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Shared test size.")

    ap.add_argument("--scgpt_head", type=str, default="sgd", choices=["xgb", "sgd", "logreg"])
    ap.add_argument("--gf_head", type=str, default="xgb", choices=["xgb", "sgd"])

    # speed knobs (use smaller defaults for sweep)
    ap.add_argument("--gf_n_estimators", type=int, default=200)
    ap.add_argument("--scgpt_n_estimators", type=int, default=200)

    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/analysis/seed_sweep_summary.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to avoid overwriting (e.g., 'n2700_r30').",
    )
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    if args.tag:
        out_csv = out_csv.with_name(f"{out_csv.stem}_{args.tag}{out_csv.suffix}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    f1_cols: set[str] = set()

    # load/align once (big speedup for n_runs=30+)
    prepared = prepare_aligned_dataset()

    for i in range(int(args.n_runs)):
        seed = int(args.seed_start) + i
        if i % 5 == 0:
            print(f"[INFO] run {i+1}/{int(args.n_runs)}  seed={seed}")
        overall, gf_pc, sc_pc = run_models_on_prepared(
            prepared,
            seed=seed,
            test_size=args.test_size,
            gf_head=args.gf_head,
            scgpt_head=args.scgpt_head,
            gf_n_estimators=args.gf_n_estimators,
            scgpt_n_estimators=args.scgpt_n_estimators,
        )

        r = dict(overall)
        # per-class F1 を列に展開（Geneformer / scGPT）
        for _idx, row in gf_pc.iterrows():
            col = f"gf_f1__{row['class']}"
            r[col] = float(row["f1"])
            f1_cols.add(col)
        for _idx, row in sc_pc.iterrows():
            col = f"scgpt_f1__{row['class']}"
            r[col] = float(row["f1"])
            f1_cols.add(col)

        rows.append(r)

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

    # mean/std rows (seed=-1: mean, seed=-2: std)
    numeric_cols = [
        "gf_accuracy",
        "gf_macro_f1",
        "gf_macro_f1_no_platelet",
        "scgpt_accuracy",
        "scgpt_macro_f1",
        "scgpt_macro_f1_no_platelet",
        *sorted(f1_cols),
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    mean_row = {k: df[k].iloc[0] if k in df.columns else None for k in df.columns}
    std_row = {k: df[k].iloc[0] if k in df.columns else None for k in df.columns}

    mean_row["seed"] = -1
    std_row["seed"] = -2

    for c in numeric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0

    # keep metadata consistent
    for k in ["test_size", "n_common", "n_test", "gf_head", "scgpt_head"]:
        if k in df.columns:
            mean_row[k] = df[k].iloc[0]
            std_row[k] = df[k].iloc[0]

    df2 = pd.concat([pd.DataFrame([mean_row, std_row]), df], ignore_index=True)

    df2.to_csv(out_csv, index=False)
    print(f"[OK] Saved: {out_csv}")
    print("Summary (mean/std):")
    show_cols = [
        c
        for c in [
            "gf_accuracy",
            "gf_macro_f1",
            "gf_macro_f1_no_platelet",
            "scgpt_accuracy",
            "scgpt_macro_f1",
            "scgpt_macro_f1_no_platelet",
        ]
        if c in df2.columns
    ]
    print(df2[df2["seed"].isin([-1, -2])][["seed", *show_cols]].to_string(index=False))

    # --- markdown report ---
    mean_row = df2[df2["seed"] == -1].iloc[0]
    std_row = df2[df2["seed"] == -2].iloc[0]

    gf_tbl = _build_f1_table(mean_row, std_row, "gf_f1__")
    sc_tbl = _build_f1_table(mean_row, std_row, "scgpt_f1__")

    # unstable classes: scGPT std high
    unstable_top = sc_tbl.head(10).copy()

    report_path = out_csv.parent / "seed_sweep_report.md"
    lines = []
    lines.append("# Seed sweep report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- n_runs: {int(args.n_runs)} (seed_start={int(args.seed_start)})")
    lines.append(f"- test_size: {float(args.test_size)}")
    lines.append(f"- heads: gf={args.gf_head}, scgpt={args.scgpt_head}")
    lines.append("")
    lines.append("## Overall (mean ± std)")
    lines.append("")
    lines.append("| metric | Geneformer | scGPT |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Accuracy | {float(mean_row['gf_accuracy']):.3f} ± {float(std_row['gf_accuracy']):.3f} | {float(mean_row['scgpt_accuracy']):.3f} ± {float(std_row['scgpt_accuracy']):.3f} |"
    )
    lines.append(
        f"| Macro F1 | {float(mean_row['gf_macro_f1']):.3f} ± {float(std_row['gf_macro_f1']):.3f} | {float(mean_row['scgpt_macro_f1']):.3f} ± {float(std_row['scgpt_macro_f1']):.3f} |"
    )
    if "gf_macro_f1_no_platelet" in mean_row.index and pd.notna(mean_row["gf_macro_f1_no_platelet"]):
        lines.append(
            f"| Macro F1 (no Platelet) | {float(mean_row['gf_macro_f1_no_platelet']):.3f} ± {float(std_row['gf_macro_f1_no_platelet']):.3f} | {float(mean_row['scgpt_macro_f1_no_platelet']):.3f} ± {float(std_row['scgpt_macro_f1_no_platelet']):.3f} |"
        )
    lines.append("")
    lines.append("## Per-class F1 (mean ± std)")
    lines.append("")
    lines.append("### scGPT (sorted by std desc)")
    lines.append("")
    lines.append("| class | F1 mean | F1 std |")
    lines.append("|---|---:|---:|")
    for _, r in sc_tbl.iterrows():
        lines.append(f"| {r['class']} | {r['f1_mean']:.3f} | {r['f1_std']:.3f} |")
    lines.append("")
    lines.append("### Geneformer (sorted by std desc)")
    lines.append("")
    lines.append("| class | F1 mean | F1 std |")
    lines.append("|---|---:|---:|")
    for _, r in gf_tbl.iterrows():
        lines.append(f"| {r['class']} | {r['f1_mean']:.3f} | {r['f1_std']:.3f} |")
    lines.append("")

    # --- delta table (GF - scGPT) ---
    lines.append("## Per-class ΔF1 (Geneformer − scGPT)")
    lines.append("")
    # merge by class
    delta = pd.merge(
        gf_tbl[["class", "f1_mean", "f1_std"]].rename(columns={"f1_mean": "gf_f1_mean", "f1_std": "gf_f1_std"}),
        sc_tbl[["class", "f1_mean", "f1_std"]].rename(columns={"f1_mean": "scgpt_f1_mean", "f1_std": "scgpt_f1_std"}),
        on="class",
        how="outer",
    )
    delta["delta_f1_mean"] = delta["gf_f1_mean"] - delta["scgpt_f1_mean"]
    delta = delta.sort_values("delta_f1_mean", ascending=True).reset_index(drop=True)

    lines.append("### Worst classes for Geneformer (most negative ΔF1)")
    lines.append("")
    lines.append("| rank | class | GF F1 mean | scGPT F1 mean | ΔF1 mean |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, (_, r) in enumerate(delta.head(10).iterrows(), start=1):
        lines.append(f"| {i} | {r['class']} | {r['gf_f1_mean']:.3f} | {r['scgpt_f1_mean']:.3f} | {r['delta_f1_mean']:.3f} |")
    lines.append("")
    lines.append("### Best classes for Geneformer (most positive ΔF1)")
    lines.append("")
    lines.append("| rank | class | GF F1 mean | scGPT F1 mean | ΔF1 mean |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, (_, r) in enumerate(delta.tail(10).iloc[::-1].iterrows(), start=1):
        lines.append(f"| {i} | {r['class']} | {r['gf_f1_mean']:.3f} | {r['scgpt_f1_mean']:.3f} | {r['delta_f1_mean']:.3f} |")
    lines.append("")
    lines.append("## Most unstable classes (scGPT std top10)")
    lines.append("")
    lines.append("| rank | class | F1 mean | F1 std |")
    lines.append("|---:|---|---:|---:|")
    for i, (_, r) in enumerate(unstable_top.iterrows(), start=1):
        lines.append(f"| {i} | {r['class']} | {r['f1_mean']:.3f} | {r['f1_std']:.3f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `Platelet` は support が非常に小さく、seed/splitで F1 が大きくブレやすいです。")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved: {report_path}")


if __name__ == "__main__":
    main()

