"""
Phase 4: Benchmarking Missing-Modality Imputation

Equivalent to 4_Benchmark_Missing_Modalities.ipynb but runnable as a plain script.
Compares MIMIR against two baselines on leave-one-out test scenarios from Phase 3:
  - TOBMI*: kNN donor-based translation (cosine distance, k=sqrt(N_train))
  - MOFA+:  global latent factor model trained on the training set

MIMIR predictions must already exist (produced by Phase 3 / impute_missing_modality.py).
TOBMI and MOFA+ are run here from scratch.

Usage:
    python benchmark_missing_modality.py
    python benchmark_missing_modality.py \\
        --data data/tcga_redo_mlomicZ.pkl \\
        --splits data/splits.json \\
        --mimir_pkl results/imputation_modality/imputations_leave_one_out_test.pkl \\
        --scenarios_dir results/imputation_modality/scenarios_leave_one_out_test \\
        --out results/benchmark_modality
    python benchmark_missing_modality.py --skip_mofa
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.others.tobmi as tobmi
import src.others.mofa_imputer as mofa_imputer

from src.data_utils import (
    MultiOmicDataset,
    load_shared_splits_from_json,
    compute_shared_splits,
)
from src.evaluation import evaluate_imputations, compare_methods_per_feature


# ─── Display config ───────────────────────────────────────────────────────────

DISPLAY_NAME = {
    "rna": "mRNA",
    "miRNA": "miRNA",
    "cnv": "CNV",
    "methylation": "Methylation",
}

METHOD_COLORS = {
    "MIMIR":  "tab:red",
    "TOBMI*": "tab:orange",
    "MOFA+":  "tab:green",
}

COMPARE_COLORS = {
    "vs TOBMI*": "tab:green",
    "vs MOFA+":  "tab:orange",
}

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def metrics_dict_to_df(metrics: dict, method_name: str) -> pd.DataFrame:
    rows = []
    for (present_mods, target), d in metrics.items():
        rows.append({
            "method":         method_name,
            "target":         target,
            "target_display": DISPLAY_NAME.get(target, target),
            "present_mods":   tuple(present_mods),
            "n_present":      len(present_mods),
            "mse":            d.get("mse"),
            "pearson":        d.get("pearson"),
            "spearman":       d.get("spearman"),
            "n_points":       d.get("n_points"),
        })
    return pd.DataFrame(rows)


def compare_dict_to_df(compare_dict: dict, method_label: str) -> pd.DataFrame:
    rows = []
    for (present_mods, target), d in compare_dict.items():
        rows.append({
            "method":         method_label,
            "target":         target,
            "target_display": DISPLAY_NAME.get(target, target),
            "frac_better":    float(d["n_better_MIMR"]),
            "n_features":     int(d.get("n_features", 0)),
            "present_mods":   tuple(present_mods),
        })
    return pd.DataFrame(rows)


def print_metrics(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
    header = f"{'Present':40s} {'Target':12s} {'MSE':>8s} {'r':>7s} {'rho':>7s} {'N pts':>10s}"
    print(header)
    print("-" * len(header))
    for (present_mods, target_mod), m in metrics.items():
        print(
            f"{', '.join(present_mods):40s} {target_mod:12s} "
            f"{m['mse']:8.4f} {m['pearson']:7.4f} {m['spearman']:7.4f} {m['n_points']:10,d}"
        )


def grouped_barplot(df_agg, metric_key="pearson", method_order=None, target_order=None,
                    save_path=None):
    if method_order is None:
        method_order = list(df_agg["method"].unique())
    if target_order is None:
        target_order = sorted(df_agg["target_display"].unique())

    mat = (
        df_agg.pivot(index="target_display", columns="method", values=metric_key)
              .reindex(index=target_order, columns=method_order)
    )

    x = np.arange(len(mat.index))
    n_methods = len(mat.columns)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 1.3), 4))
    for i, method in enumerate(mat.columns):
        ax.bar(
            x + (i - (n_methods - 1) / 2) * width,
            mat[method].values,
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, None),
        )

    ymax = mat.max().max()
    offset = 0.015 * ymax
    for i, method in enumerate(mat.columns):
        xs = x + (i - (n_methods - 1) / 2) * width
        ys = mat[method].values
        for xi, yi in zip(xs, ys):
            if np.isnan(yi):
                continue
            ax.text(xi, yi + offset, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(mat.index, rotation=0)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} by target modality")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_frac_better(df_cmp, method_order=None, target_order=None, save_path=None):
    if method_order is None:
        method_order = ["vs TOBMI*", "vs MOFA+"]
    if target_order is None:
        target_order = ["CNV", "Methylation", "mRNA", "miRNA"]

    mat = (
        df_cmp.pivot(index="target_display", columns="method", values="frac_better")
              .reindex(index=target_order, columns=method_order)
    )

    x = np.arange(len(mat.index))
    n_methods = len(mat.columns)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, method in enumerate(mat.columns):
        ax.bar(
            x + (i - (n_methods - 1) / 2) * width,
            mat[method].values,
            width=width,
            label=method,
            color=COMPARE_COLORS.get(method, None),
        )

    ymax = mat.max().max()
    offset = 0.015 * ymax
    for i, method in enumerate(mat.columns):
        xs = x + (i - (n_methods - 1) / 2) * width
        ys = mat[method].values
        for xi, yi in zip(xs, ys):
            if np.isnan(yi):
                continue
            ax.text(xi, yi + offset, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(mat.index)
    ax.set_ylabel("% features")
    ax.set_title("Fraction of features where MIMIR improves vs baselines")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Benchmark missing-modality imputation")
    p.add_argument("--data",         default="data/tcga_redo_mlomicZ.pkl",
                   help="Path to multi-omic pickle")
    p.add_argument("--splits",       default="data/splits.json",
                   help="Path to splits JSON")
    p.add_argument("--mimir_pkl",    default="results/imputation_modality/imputations_leave_one_out_test.pkl",
                   help="MIMIR LOO predictions pickle (from Phase 3)")
    p.add_argument("--scenarios_dir", default="results/imputation_modality/scenarios_leave_one_out_test",
                   help="Leave-one-out scenario directory (from Phase 3)")
    p.add_argument("--out",          default="results/benchmark_modality",
                   help="Output directory for plots and prediction pickles")
    p.add_argument("--mofa_hdf5",    default=None,
                   help="Path to pre-trained MOFA HDF5 (skip training if provided)")
    p.add_argument("--mofa_factors", type=int,   default=256,
                   help="Number of MOFA factors (default: 256, matches MIMIR shared_dim)")
    p.add_argument("--skip_mofa",    action="store_true",
                   help="Skip MOFA+ baseline entirely")
    p.add_argument("--tobmi_k",      type=int,   default=None,
                   help="k for TOBMI kNN (default: sqrt(N_train))")
    p.add_argument("--tobmi_metric", default="cosine",
                   help="Distance metric for TOBMI (default: cosine)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Data & splits ──────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} …")
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)
    print(f"Modalities: {list(multi_omic_data.keys())}")

    if os.path.exists(args.splits):
        print(f"Loading splits from {args.splits} …")
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        print("splits.json not found – computing splits (70/10/20) …")
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )
    print(
        f"Samples: total={len(common_samples)} | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    multi_ds = MultiOmicDataset({m: df for m, df in multi_omic_data.items()})
    train_samples = [multi_ds.common_samples[i] for i in train_idx]

    # ── MIMIR predictions (Phase 3) ────────────────────────────────────────────
    print(f"\nLoading MIMIR predictions from {args.mimir_pkl} …")
    with open(args.mimir_pkl, "rb") as f:
        mimir_preds = pickle.load(f)
    mimir_metrics = evaluate_imputations(mimir_preds, multi_omic_data, plot_scatter=False)
    print_metrics(mimir_metrics, label="MIMIR Metrics")

    # ── TOBMI* baseline ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TOBMI* baseline (kNN donor-based translation)")
    print("="*65)
    tobmi_pkl = os.path.join(args.out, "tobmi_leave_one_out_test.pkl")

    tobmi_preds = tobmi.translate_from_scenario_dir(
        scenarios_dir=args.scenarios_dir,
        multi_omic_data=multi_omic_data,
        train_samples=train_samples,
        k=args.tobmi_k,
        metric=args.tobmi_metric,
        scale=False,   # inputs already z-scored
        save_pred_pickle_path=tobmi_pkl,
    )
    tobmi_metrics = evaluate_imputations(tobmi_preds, multi_omic_data, plot_scatter=False)
    print_metrics(tobmi_metrics, label="TOBMI* Metrics")

    # ── MOFA+ baseline ─────────────────────────────────────────────────────────
    mofa_metrics = None
    compare_mofa = None

    if not args.skip_mofa:
        print("\n" + "="*65)
        print("  MOFA+ baseline (global latent factor model)")
        print("="*65)

        mofa_hdf5 = args.mofa_hdf5
        if mofa_hdf5 is None:
            mofa_hdf5 = os.path.join(args.out, "mofa_global_train.hdf5")

        if not os.path.exists(mofa_hdf5):
            print(f"Training global MOFA+ on {len(train_samples)} train samples …")
            mofa_imputer.train_global_mofa(
                multi_omic_data=multi_omic_data,
                train_samples=train_samples,
                out_hdf5_path=mofa_hdf5,
                n_factors=args.mofa_factors,
                train_iter=None,
                seed=1,
                verbose=True,
            )
        else:
            print(f"Reusing existing MOFA+ model: {mofa_hdf5}")

        mofa_pkl = os.path.join(args.out, "mofa_leave_one_out_test.pkl")
        mofa_preds = mofa_imputer.translate_from_scenario_dir(
            scenarios_dir=args.scenarios_dir,
            mofa_hdf5_path=mofa_hdf5,
            multi_omic_data=multi_omic_data,
            projection_view=None,
            use_multi_view_projection=True,
            verbose=True,
            save_pred_pickle_path=mofa_pkl,
        )
        mofa_metrics = evaluate_imputations(mofa_preds, multi_omic_data, plot_scatter=False)
        print_metrics(mofa_metrics, label="MOFA+ Metrics")

        compare_mofa = compare_methods_per_feature(
            method1=mimir_preds,
            method2=mofa_preds,
            multi_omic_data=multi_omic_data,
            plot_scatter=False,
            m1_name="MIMR",
            m2_name="MOFA",
        )

    # ── Feature-wise wins: MIMIR vs TOBMI ─────────────────────────────────────
    compare_tobmi = compare_methods_per_feature(
        method1=mimir_preds,
        method2=tobmi_preds,
        multi_omic_data=multi_omic_data,
        plot_scatter=False,
        m1_name="MIMR",
        m2_name="TOBMI",
    )

    # ── Summary plots ──────────────────────────────────────────────────────────
    print("\nGenerating summary plots …")

    # Build tidy DataFrame of global metrics
    dfs = [
        metrics_dict_to_df(mimir_metrics, "MIMIR"),
        metrics_dict_to_df(tobmi_metrics, "TOBMI*"),
    ]
    method_order = ["MIMIR", "TOBMI*"]
    if mofa_metrics is not None:
        dfs.append(metrics_dict_to_df(mofa_metrics, "MOFA+"))
        method_order.append("MOFA+")

    df_all = pd.concat(dfs, ignore_index=True)

    for metric_key in ("pearson", "spearman", "mse"):
        df_agg = (
            df_all.groupby(["method", "target", "target_display"], as_index=False)[metric_key]
                  .max()
        )
        save_path = os.path.join(plots_dir, f"barplot_{metric_key}.png")
        grouped_barplot(
            df_agg,
            metric_key=metric_key,
            method_order=method_order,
            save_path=save_path,
        )

    # Fraction of features improved vs baselines
    cmp_dfs = [compare_dict_to_df(compare_tobmi, "vs TOBMI*")]
    cmp_method_order = ["vs TOBMI*"]
    if compare_mofa is not None:
        cmp_dfs.append(compare_dict_to_df(compare_mofa, "vs MOFA+"))
        cmp_method_order.append("vs MOFA+")

    df_cmp = pd.concat(cmp_dfs, ignore_index=True)
    save_path = os.path.join(plots_dir, "frac_features_better.png")
    plot_frac_better(df_cmp, method_order=cmp_method_order, save_path=save_path)

    # Save tidy metrics table as CSV
    csv_path = os.path.join(args.out, "metrics_summary.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nMetrics CSV saved → {csv_path}")

    print(f"\nDone. Results saved to {args.out}/")


if __name__ == "__main__":
    main()
