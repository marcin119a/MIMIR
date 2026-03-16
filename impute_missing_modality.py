"""
Phase 3: Imputation with Missing Modalities

Equivalent to 3_Imputation_Missing_Modality.ipynb but runnable as a plain script.
Loads pretrained AE + shared model checkpoints and evaluates imputation under
two missingness scenarios:
  1. Leave-one-modality-out (LOO)
  2. All possible missingness patterns

Usage:
    python impute_missing_modality.py
    python impute_missing_modality.py --data data/tcga_redo_mlomicZ.pkl --splits data/splits.json
    python impute_missing_modality.py --checkpoint checkpoints/finetuned/shared_model_ep200.pt
    python impute_missing_modality.py --skip_all_possible
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data_utils import load_shared_splits_from_json, compute_shared_splits
from src.mae_masked import (
    MultiModalWithSharedSpace,
    load_modality_with_config,
    extract_encoder_decoder_from_pretrained,
)
from src.shared_finetune import load_shared_model, make_loaders_from_splits
from src.translation import leave_one_out_imputation, all_possible_imputation
from src.evaluation import evaluate_imputations
from src.data_utils import MultiOmicDataset


# ─── Helpers ──────────────────────────────────────────────────────────────────

DISPLAY_NAME = {
    "rna": "mRNA",
}


def plot_upset_for_target(df, target, all_modalities, score_label="Correlation r", save_path=None):
    """UpSet-style bar + dot-matrix plot for imputing one target modality."""
    sub = df[df["target"] == target].copy()
    sub = sub.sort_values(by="score", ascending=False).reset_index(drop=True)

    n = len(sub)
    if n == 0:
        print(f"[WARN] No rows for target={target}; skipping plot.")
        return

    mods = [m for m in all_modalities if m != target]

    fig = plt.figure(figsize=(max(6, n * 0.6), 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_bar = fig.add_subplot(gs[0])
    ax_bar.bar(range(n), sub["score"], color="tab:orange")
    ymax = sub["score"].max()
    ax_bar.set_ylim(0, ymax * 1.1)
    ax_bar.set_ylabel(score_label)
    display_target = DISPLAY_NAME.get(target, target)
    ax_bar.set_title(f"Imputing {display_target}")
    for i, k in enumerate(sub["n_present"]):
        ax_bar.text(i, sub["score"].iloc[i] + 0.01, str(k),
                    ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks([])

    ax_mat = fig.add_subplot(gs[1], sharex=ax_bar)
    for i, present in enumerate(sub["present"]):
        for j, mod in enumerate(mods):
            filled = mod in present
            ax_mat.scatter(
                i, j, s=50,
                color="black" if filled else "white",
                edgecolor="black",
                zorder=3,
            )
    ax_mat.set_yticks(range(len(mods)))
    ax_mat.set_yticklabels([DISPLAY_NAME.get(m, m) for m in mods])
    ax_mat.set_xlabel("Available modalities")
    ax_mat.set_xlim(-0.5, n - 0.5)
    ax_mat.set_ylim(-0.5, len(mods) - 0.5)
    for spine in ["top", "right", "left"]:
        ax_mat.spines[spine].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def metrics_to_upset_df(metrics, score_key="pearson"):
    rows = []
    for (present_mods, target_mod), vals in metrics.items():
        rows.append({
            "target": target_mod,
            "present": set(present_mods),
            "n_present": len(present_mods),
            "score": vals[score_key],
        })
    return pd.DataFrame(rows)


def print_metrics(metrics, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    header = f"{'Present':40s} {'Target':12s} {'MSE':>8s} {'r':>7s} {'rho':>7s} {'N pts':>10s}"
    print(header)
    print("-" * len(header))
    for (present_mods, target_mod), m in metrics.items():
        present_str = ", ".join(present_mods)
        print(
            f"{present_str:40s} {target_mod:12s} "
            f"{m['mse']:8.4f} {m['pearson']:7.4f} {m['spearman']:7.4f} {m['n_points']:10,d}"
        )


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Impute missing modalities")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl", help="Path to multi-omic pickle")
    p.add_argument("--splits",     default="data/splits.json",           help="Path to splits JSON")
    p.add_argument("--ae_dir",     default="aes_redo_z",                 help="Directory with Phase-1 AE checkpoints")
    p.add_argument("--checkpoint", default="checkpoints/finetuned/shared_model_ep200.pt",
                                                                          help="Shared model checkpoint (.pt)")
    p.add_argument("--out",        default="results/imputation_modality", help="Output directory")
    p.add_argument("--device",     default=None,                         help="cuda / mps / cpu (auto-detected if omitted)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--shared_dim", type=int, default=256)
    p.add_argument("--skip_all_possible", action="store_true",
                   help="Skip the all-possible-missingness evaluation (faster)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── 0.1  Data & splits ─────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data} …")
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

    # Filter to only the two modalities used in this pipeline
    ACTIVE_MODALITIES = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE_MODALITIES}
    print(f"Active modalities: {list(multi_omic_data.keys())}")

    # ── 0.2  Load shared model ─────────────────────────────────────────────────
    # Map full modality names → short AE checkpoint names
    name_map = {"rna": "rna", "methylation": "mth"}

    encoders, decoders, hidden_dims, mask_values = {}, {}, {}, {}
    for mod in multi_omic_data.keys():
        short = name_map.get(mod, mod)
        ae_path = os.path.join(args.ae_dir, f"{short}_ae.pt")
        ae_m, hidden_dim_m, cfg_m = load_modality_with_config(ae_path, map_location=device)
        ae_m = ae_m.to(device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m
        mask_values[mod] = cfg_m.get("mask_value", 0.0)

    print(f"\nmask_values: {mask_values}")

    model = load_shared_model(
        MultiModalWithSharedSpace,
        encoders,
        decoders,
        hidden_dims,
        shared_dim=args.shared_dim,
        proj_depth=1,
        checkpoint_path=args.checkpoint,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    print(model)

    # Resolve test samples
    multi_ds = MultiOmicDataset({m: df for m, df in multi_omic_data.items()})
    test_samples = [multi_ds.common_samples[i] for i in test_idx]

    # ── 0.3  Leave-one-modality-out imputation ─────────────────────────────────
    print("\n" + "="*60)
    print("  Leave-one-modality-out imputation")
    print("="*60)

    loo_scenarios_dir = os.path.join(args.out, "scenarios_leave_one_out_test")
    loo_pred_path     = os.path.join(args.out, "imputations_leave_one_out_test.pkl")

    pred_dict_loo = leave_one_out_imputation(
        model=model,
        mask_values=mask_values,
        multi_omic_data=multi_omic_data,
        common_samples=test_samples,
        batch_size=args.batch_size,
        device=device,
        scenarios_dir=loo_scenarios_dir,
        save_pred_pickle_path=loo_pred_path,
    )

    metrics_loo = evaluate_imputations(pred_dict_loo, multi_omic_data, plot_scatter=False)
    print_metrics(metrics_loo, label="LOO Imputation Metrics")

    # ── 0.4  All-possible-missingness imputation ───────────────────────────────
    if not args.skip_all_possible:
        print("\n" + "="*60)
        print("  All-possible-missingness imputation")
        print("="*60)

        ap_scenarios_dir = os.path.join(args.out, "scenarios_all_possible_test")
        ap_pred_path     = os.path.join(args.out, "imputations_all_possible_test.pkl")

        pred_dict_ap = all_possible_imputation(
            model=model,
            mask_values=mask_values,
            multi_omic_data=multi_omic_data,
            common_samples=test_samples,
            batch_size=args.batch_size,
            device=device,
            scenarios_dir=ap_scenarios_dir,
            save_pred_pickle_path=ap_pred_path,
        )

        metrics_ap = evaluate_imputations(pred_dict_ap, multi_omic_data, plot_scatter=False)
        print_metrics(metrics_ap, label="All-Possible Imputation Metrics")

        # UpSet-style plots for each target modality
        df_upset = metrics_to_upset_df(metrics_ap, score_key="pearson")
        all_modalities = list(multi_omic_data.keys())

        plots_dir = os.path.join(args.out, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for target in all_modalities:
            save_path = os.path.join(plots_dir, f"upset_{target}.png")
            plot_upset_for_target(
                df_upset,
                target=target,
                all_modalities=all_modalities,
                score_label="Pearson r",
                save_path=save_path,
            )

    print(f"\nDone. Results saved to {args.out}/")


if __name__ == "__main__":
    main()
