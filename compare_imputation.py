"""
compare_imputation.py
=====================
Porównuje trzy modele na zadaniu imputacji brakujących wartości:

  1. DNA2RNAVAE  (methylation → RNA)       – train_dna2rna.py
  2. RNA2DNAVAE  (RNA → methylation)       – train_rna2dna.py
  3. SharedVAE   (shared latent space)     – train_shared.py

Dla każdego modelu:
  - maskuje losowo ~20% wartości w zbiorze testowym
  - rekonstruuje i liczy MSE / Pearson r / Spearman ρ
  - generuje wykres porównawczy

Wynik: tabela w konsoli + compare_imputation_results.png

Usage:
    python compare_imputation.py
    python compare_imputation.py --device cuda --masking_fraction 0.2
    python compare_imputation.py --dna2rna_id 20260313_145754 --rna2dna_id 20260313_150141
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
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data_utils import (
    MultiOmicDataset,
    compute_shared_splits,
    load_shared_splits_from_json,
)
from src.models import DNA2RNAVAE, RNA2DNAVAE
from src.mae_masked import MultiModalWithSharedSpace
from src.shared_finetune import load_shared_model, load_modality_with_config, extract_encoder_decoder_from_pretrained


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_random_mask(X: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    eligible = ~np.isnan(X)
    return (rng.random(X.shape) < frac) & eligible


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return dict(mse=float("nan"), pearson=float("nan"), spearman=float("nan"), n=0)
    mse = float(np.mean((yt - yp) ** 2))
    s_t = pd.Series(yt)
    s_p = pd.Series(yp)
    return dict(
        mse=mse,
        pearson=float(s_t.corr(s_p, method="pearson")),
        spearman=float(s_t.corr(s_p, method="spearman")),
        n=int(mask.sum()),
    )


# ─── Model loaders ────────────────────────────────────────────────────────────

def load_dna2rna(run_id: str, rna_dim: int, meth_dim: int, device: torch.device) -> DNA2RNAVAE:
    path = os.path.join(Config.CHECKPOINT_DIR, f"best_dna2rna_{run_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DNA2RNAVAE checkpoint not found: {path}")
    model = DNA2RNAVAE(rna_dim, meth_dim, n_sites=1, latent_dim=Config.LATENT_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[OK] DNA2RNAVAE loaded from {path}")
    return model


def load_rna2dna(run_id: str, rna_dim: int, meth_dim: int, device: torch.device) -> RNA2DNAVAE:
    path = os.path.join(Config.CHECKPOINT_DIR, f"best_rna2dna_{run_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RNA2DNAVAE checkpoint not found: {path}")
    model = RNA2DNAVAE(rna_dim, meth_dim, n_sites=1, latent_dim=Config.LATENT_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[OK] RNA2DNAVAE loaded from {path}")
    return model


def load_shared(ae_dir: str, shared_ckpt: str, multi_omic_data: dict, device: torch.device):
    """Load shared MultiModalWithSharedSpace model."""
    name_map = {"rna": "rna", "methylation": "mth"}
    encoders, decoders, hidden_dims = {}, {}, {}
    for mod in multi_omic_data:
        short = name_map.get(mod, mod)
        path = os.path.join(ae_dir, f"{short}_ae.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"AE checkpoint missing: {path}")
        ae_m, hidden_dim_m, _ = load_modality_with_config(path, map_location=device)
        ae_m = ae_m.to(device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m

    model = load_shared_model(
        model_class=MultiModalWithSharedSpace,
        encoders=encoders,
        decoders=decoders,
        hidden_dims=hidden_dims,
        shared_dim=256,
        proj_depth=1,
        checkpoint_path=shared_ckpt,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model


# ─── Imputation functions ─────────────────────────────────────────────────────

@torch.no_grad()
def impute_dna2rna(
    model: DNA2RNAVAE,
    meth_masked: np.ndarray,
    rna_orig: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
):
    """DNA2RNAVAE: methylation → RNA. Evaluate on masked RNA positions."""
    N = meth_masked.shape[0]
    all_preds = []

    # zero-fill NaN sentinel for masked positions
    meth_input = np.where(np.isnan(meth_masked), 0.0, meth_masked).astype(np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        meth_b = torch.tensor(meth_input[start:end], device=device)
        recon_rna, _, _ = model(dna=meth_b, site=None)
        all_preds.append(recon_rna.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    # evaluate only on masked positions
    y_true = rna_orig[mask]
    y_pred = preds[mask]
    return compute_metrics(y_true, y_pred), preds


@torch.no_grad()
def impute_rna2dna(
    model: RNA2DNAVAE,
    rna_orig: np.ndarray,
    meth_orig: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
):
    """RNA2DNAVAE: RNA → methylation. Evaluate on masked methylation positions."""
    N = rna_orig.shape[0]
    all_preds = []

    rna_input = np.where(np.isnan(rna_orig), 0.0, rna_orig).astype(np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        rna_b = torch.tensor(rna_input[start:end], device=device)
        recon_dna, _, _ = model(rna=rna_b, site=None)
        all_preds.append(recon_dna.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    y_true = meth_orig[mask]
    y_pred = preds[mask]
    return compute_metrics(y_true, y_pred), preds


@torch.no_grad()
def impute_shared_crossmodal(
    model: MultiModalWithSharedSpace,
    source_mod: str,
    target_mod: str,
    source_data: np.ndarray,
    target_orig: np.ndarray,
    target_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
):
    """
    Fair cross-modal imputation: encode source_mod only → decode target_mod.
    Mirrors exactly what DNA2RNAVAE / RNA2DNAVAE do.
    """
    from src.data_utils import get_dataloader

    N = source_data.shape[0]
    src_input = np.where(np.isnan(source_data), 0.0, source_data).astype(np.float32)
    all_preds = []

    model.eval()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.tensor(src_input[start:end], device=device)
        h = model.encoders[source_mod](xb)
        z = model.projections[source_mod](h)
        h_hat = model.rev_projections[target_mod](z)
        x_imp = model.decoders[target_mod](h_hat)
        all_preds.append(x_imp.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    y_true = target_orig[target_mask]
    y_pred = preds[target_mask]
    return compute_metrics(y_true, y_pred), preds


@torch.no_grad()
def impute_shared(
    model: MultiModalWithSharedSpace,
    data_corrupted: dict,
    data_orig: dict,
    mask_by_mod: dict,
    device: torch.device,
    batch_size: int = 128,
    self_weight: float = 10.0,
):
    """SharedVAE: both modalities together. Evaluate per-modality masked positions."""
    from src.data_utils import get_dataloader

    modalities = list(data_corrupted.keys())
    ds = MultiOmicDataset(data_corrupted)
    loader = get_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=None)

    preds_by_mod = {m: [] for m in modalities}
    model.eval()

    for batch in loader:
        batch = {m: x.to(device) for m, x in batch.items()}
        batch_clean = {}
        for mod, xb in batch.items():
            xb_c = xb.clone()
            xb_c[torch.isnan(xb_c)] = 0.0
            batch_clean[mod] = xb_c

        shared_all = {}
        for mod, xb_c in batch_clean.items():
            h = model.encoders[mod](xb_c)
            z = model.projections[mod](h)
            shared_all[mod] = z

        for target_mod in modalities:
            weights = []
            z_list = []
            for m in list(shared_all.keys()):
                weights.append(self_weight if m == target_mod else 1.0)
                z_list.append(shared_all[m])
            w = torch.tensor(weights, device=device, dtype=z_list[0].dtype)
            w = w / w.sum()
            z_stack = torch.stack(z_list, dim=0)
            z_w = (w.view(-1, 1, 1) * z_stack).sum(dim=0)
            h_hat = model.rev_projections[target_mod](z_w)
            x_imp = model.decoders[target_mod](h_hat)
            preds_by_mod[target_mod].append(x_imp.cpu().numpy())

    metrics_by_mod = {}
    preds_full = {}
    for mod in modalities:
        preds = np.concatenate(preds_by_mod[mod], axis=0)
        preds_full[mod] = preds
        mask = mask_by_mod[mod]
        y_true = data_orig[mod][mask]
        y_pred = preds[mask]
        metrics_by_mod[mod] = compute_metrics(y_true, y_pred)

    return metrics_by_mod, preds_full


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, save_path: str):
    """
    results = {
      "rna":   {"DNA2RNAVAE": metrics, "SharedVAE": metrics},
      "methylation": {"RNA2DNAVAE": metrics, "SharedVAE": metrics},
    }
    """
    metrics_order = ["mse", "pearson", "spearman"]
    metric_labels = {"mse": "MSE ↓", "pearson": "Pearson r ↑", "spearman": "Spearman ρ ↑"}
    colors = {"DNA2RNAVAE": "#e07b54", "RNA2DNAVAE": "#4c8bb5", "SharedVAE": "#4caf7d"}

    n_mods = len(results)
    fig, axes = plt.subplots(n_mods, 3, figsize=(14, 4.5 * n_mods))
    if n_mods == 1:
        axes = [axes]

    for row, (mod, method_metrics) in enumerate(results.items()):
        for col, metric in enumerate(metrics_order):
            ax = axes[row][col]
            methods = list(method_metrics.keys())
            values = [method_metrics[m][metric] for m in methods]
            bar_colors = [colors.get(m, "#888888") for m in methods]

            bars = ax.bar(methods, values, color=bar_colors, width=0.5, edgecolor="white")
            for bar, val in zip(bars, values):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9)

            ax.set_title(f"{mod.upper()} — {metric_labels[metric]}", fontsize=10)
            ax.set_ylabel(metric, fontsize=9)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)

    plt.suptitle("Imputation comparison: DNA2RNAVAE vs RNA2DNAVAE vs SharedVAE", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {save_path}")


# ─── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare imputation models")
    p.add_argument("--data",              default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",            default="data/splits.json")
    p.add_argument("--device",            default=None)
    p.add_argument("--batch_size",        type=int,   default=128)
    p.add_argument("--masking_fraction",  type=float, default=0.2)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--dna2rna_id",        default=None,
                   help="run_id for DNA2RNAVAE (reads from latest_dna2rna_run_id.txt if omitted)")
    p.add_argument("--rna2dna_id",        default=None,
                   help="run_id for RNA2DNAVAE (reads from latest_rna2dna_run_id.txt if omitted)")
    p.add_argument("--ae_dir",            default="aes_redo_z",
                   help="Directory with Phase-1 AE checkpoints for SharedVAE")
    p.add_argument("--shared_ckpt",       default="checkpoints/finetuned/shared_model_ep200.pt",
                   help="Path to shared model checkpoint")
    p.add_argument("--out",               default="compare_imputation_results.png")
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load run IDs
    dna2rna_id = args.dna2rna_id
    if dna2rna_id is None:
        with open("latest_dna2rna_run_id.txt") as f:
            dna2rna_id = f.read().strip()
    rna2dna_id = args.rna2dna_id
    if rna2dna_id is None:
        with open("latest_rna2dna_run_id.txt") as f:
            rna2dna_id = f.read().strip()

    print(f"DNA2RNAVAE run: {dna2rna_id}")
    print(f"RNA2DNAVAE run: {rna2dna_id}")

    # Load data
    print(f"\nLoading data from {args.data} ...")
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)

    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ["rna", "methylation"]}
    print(f"Modalities: {list(multi_omic_data.keys())}")

    rna_dim  = multi_omic_data["rna"].shape[1]
    meth_dim = multi_omic_data["methylation"].shape[1]
    print(f"RNA dim: {rna_dim}, Methylation dim: {meth_dim}")

    # Splits
    if os.path.exists(args.splits):
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )

    test_samples = [common_samples[i] for i in test_idx]
    print(f"Test samples: {len(test_samples)}")

    # Extract test arrays
    rna_test  = multi_omic_data["rna"].loc[test_samples].values.astype(np.float32)
    meth_test = multi_omic_data["methylation"].loc[test_samples].values.astype(np.float32)

    # Create masks
    rng = np.random.default_rng(args.seed)
    rna_mask  = make_random_mask(rna_test,  args.masking_fraction, rng)
    meth_mask = make_random_mask(meth_test, args.masking_fraction, rng)

    print(f"Masked: RNA={rna_mask.sum()} positions, Methylation={meth_mask.sum()} positions")

    # Corrupted arrays
    rna_corrupted  = rna_test.copy();  rna_corrupted[rna_mask]   = np.nan
    meth_corrupted = meth_test.copy(); meth_corrupted[meth_mask]  = np.nan

    # ── Load models ──────────────────────────────────────────────────────────
    model_dna2rna = load_dna2rna(dna2rna_id, rna_dim, meth_dim, device)
    model_rna2dna = load_rna2dna(rna2dna_id, rna_dim, meth_dim, device)

    shared_available = os.path.exists(args.shared_ckpt)
    if shared_available:
        model_shared = load_shared(args.ae_dir, args.shared_ckpt, multi_omic_data, device)
    else:
        print(f"[WARN] Shared model checkpoint not found: {args.shared_ckpt}  — skipping SharedVAE")
        model_shared = None

    # ── Run imputation ───────────────────────────────────────────────────────

    print("\n--- DNA2RNAVAE: methylation → RNA ---")
    metrics_d2r, _ = impute_dna2rna(
        model_dna2rna, meth_corrupted, rna_test, rna_mask, device, args.batch_size
    )
    print(f"  MSE={metrics_d2r['mse']:.4f}  r={metrics_d2r['pearson']:.4f}  ρ={metrics_d2r['spearman']:.4f}  n={metrics_d2r['n']}")

    print("\n--- RNA2DNAVAE: RNA → methylation ---")
    metrics_r2d, _ = impute_rna2dna(
        model_rna2dna, rna_corrupted, meth_test, meth_mask, device, args.batch_size
    )
    print(f"  MSE={metrics_r2d['mse']:.4f}  r={metrics_r2d['pearson']:.4f}  ρ={metrics_r2d['spearman']:.4f}  n={metrics_r2d['n']}")

    if model_shared is not None:
        print("\n--- SharedVAE: methylation → RNA (fair cross-modal) ---")
        metrics_shared_rna, _ = impute_shared_crossmodal(
            model_shared,
            source_mod="methylation", target_mod="rna",
            source_data=meth_corrupted, target_orig=rna_test, target_mask=rna_mask,
            device=device, batch_size=args.batch_size,
        )
        print(f"  MSE={metrics_shared_rna['mse']:.4f}  r={metrics_shared_rna['pearson']:.4f}  ρ={metrics_shared_rna['spearman']:.4f}  n={metrics_shared_rna['n']}")

        print("\n--- SharedVAE: RNA → methylation (fair cross-modal) ---")
        metrics_shared_meth, _ = impute_shared_crossmodal(
            model_shared,
            source_mod="rna", target_mod="methylation",
            source_data=rna_corrupted, target_orig=meth_test, target_mask=meth_mask,
            device=device, batch_size=args.batch_size,
        )
        print(f"  MSE={metrics_shared_meth['mse']:.4f}  r={metrics_shared_meth['pearson']:.4f}  ρ={metrics_shared_meth['spearman']:.4f}  n={metrics_shared_meth['n']}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  IMPUTATION BENCHMARK  (masking fraction={args.masking_fraction}, seed={args.seed})")
    print(f"{'='*72}")
    print(f"{'Modality':<14} {'Model':<14} {'MSE':>10} {'Pearson r':>10} {'Spearman ρ':>11} {'n_points':>9}")
    print("─" * 72)

    rows = [
        ("rna",         "DNA2RNAVAE",  metrics_d2r),
        ("methylation", "RNA2DNAVAE",  metrics_r2d),
    ]
    if model_shared is not None:
        rows += [
            ("rna",         "SharedVAE",   metrics_shared_rna),
            ("methylation", "SharedVAE",   metrics_shared_meth),
        ]

    for mod, model_name, m in rows:
        print(f"{mod:<14} {model_name:<14} {m['mse']:>10.4f} {m['pearson']:>10.4f} {m['spearman']:>11.4f} {m['n']:>9}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_data = {
        "rna":         {"DNA2RNAVAE": metrics_d2r},
        "methylation": {"RNA2DNAVAE": metrics_r2d},
    }
    if model_shared is not None:
        plot_data["rna"]["SharedVAE"]         = metrics_shared_rna
        plot_data["methylation"]["SharedVAE"] = metrics_shared_meth

    plot_comparison(plot_data, args.out)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
