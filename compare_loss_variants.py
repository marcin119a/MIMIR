"""
compare_loss_variants.py
========================
Benchmarking script to compare two Phase 2 shared models (Baseline vs Modified Loss).
It generates:
  1. Comparative latent space embeddings contrasting modality alignment (UMAP/PCA).
  2. Comparative performance benchmarks (bar charts) on missing-modality and missing-value imputation.

Usage:
    python compare_loss_variants.py \
        --baseline_ckpt checkpoints/shared_model_baseline.pt \
        --modified_ckpt checkpoints/shared_model_modified.pt \
        --out results/compare_loss
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_utils import load_shared_splits_from_json, compute_shared_splits
from src.mae_masked import MultiModalWithSharedSpace
from src.shared_finetune import load_shared_model, load_modality_with_config, extract_encoder_decoder_from_pretrained
from compare_imputation import make_random_mask, compute_metrics, impute_shared_crossmodal

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

def load_shared(ae_dir: str, shared_ckpt: str, multi_omic_data: dict, device: torch.device, shared_dim: int):
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
        shared_dim=shared_dim,
        proj_depth=1,
        checkpoint_path=shared_ckpt,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_shared_embeddings(model, data_dict, device, batch_size=128):
    """ Extract embeddings for given data dictionary (numpy arrays). """
    embeddings = {m: [] for m in data_dict}
    model.eval()
    
    mods = list(data_dict.keys())
    N = data_dict[mods[0]].shape[0]
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        for mod in mods:
            xb = torch.tensor(data_dict[mod][start:end], device=device)
            xb_c = xb.clone()
            xb_c[torch.isnan(xb_c)] = 0.0
            h = model.encoders[mod](xb_c)
            z = model.projections[mod](h)
            embeddings[mod].append(z.cpu().numpy())
            
    for m in embeddings:
        embeddings[m] = np.concatenate(embeddings[m], axis=0)
    return embeddings

@torch.no_grad()
def impute_shared_values(model, data_corrupted: dict, data_orig: dict, mask_by_mod: dict, device: torch.device, batch_size: int = 128, self_weight: float = 10.0):
    """ Impute missing values jointly from all available modalities. 
        Replaces the buggy compare_imputation.impute_shared implementation. """
    modalities = list(data_corrupted.keys())
    N = data_corrupted[modalities[0]].shape[0]
    preds_by_mod = {m: [] for m in modalities}
    model.eval()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_clean = {}
        for mod in modalities:
            xb = torch.tensor(data_corrupted[mod][start:end], device=device)
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
            for m in modalities:
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
    for mod in modalities:
        preds = np.concatenate(preds_by_mod[mod], axis=0)
        mask = mask_by_mod[mod]
        y_true = data_orig[mod][mask]
        y_pred = preds[mask]
        metrics_by_mod[mod] = compute_metrics(y_true, y_pred)

    return metrics_by_mod

def plot_latent_space(emb_base, emb_mod, save_path):
    mods = list(emb_base.keys())
    assert len(mods) == 2, "Plotting assumes 2 modalities (e.g. RNA and Methylation)"
    mod1, mod2 = mods[0], mods[1]
    
    base_all = np.vstack([emb_base[mod1], emb_base[mod2]])
    mod_all = np.vstack([emb_mod[mod1], emb_mod[mod2]])
    
    if HAS_UMAP:
        proj_base = umap.UMAP(n_components=2, random_state=42).fit_transform(base_all)
        proj_mod = umap.UMAP(n_components=2, random_state=42).fit_transform(mod_all)
        method_name = "UMAP"
    else:
        proj_base = TSNE(n_components=2, random_state=42).fit_transform(base_all)
        proj_mod = TSNE(n_components=2, random_state=42).fit_transform(mod_all)
        method_name = "t-SNE"
        
    N = emb_base[mod1].shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(proj_base[:N, 0], proj_base[:N, 1], s=10, alpha=0.6, label=mod1, color='tab:blue')
    axes[0].scatter(proj_base[N:, 0], proj_base[N:, 1], s=10, alpha=0.6, label=mod2, color='tab:orange')
    axes[0].set_title(f"Baseline Loss - Latent Space ({method_name})")
    axes[0].legend()
    
    axes[1].scatter(proj_mod[:N, 0], proj_mod[:N, 1], s=10, alpha=0.6, label=mod1, color='tab:blue')
    axes[1].scatter(proj_mod[N:, 0], proj_mod[N:, 1], s=10, alpha=0.6, label=mod2, color='tab:orange')
    axes[1].set_title(f"Modified Loss - Latent Space ({method_name})")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved latent space plot -> {save_path}")

def plot_benchmark_bar(results, title, save_path):
    metrics = list(results['Baseline'].keys())
    methods = list(results.keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
        
    colors = {"Baseline": "#e07b54", "Modified": "#4caf7d"}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = [results[m][metric] for m in methods]
        bars = ax.bar(methods, vals, color=[colors[m] for m in methods], width=0.5, edgecolor="white")
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved bar chart -> {save_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ae_dir", default="aes_redo_z")
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--modified_ckpt", required=True)
    p.add_argument("--shared_dim", type=int, default=128)
    p.add_argument("--out", default="results/compare_loss")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.out, exist_ok=True)
    
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    
    data = {k: v for k, v in data.items() if k in ["rna", "methylation"]}
    if os.path.exists(args.splits):
        common_samples, _, _, test_idx = load_shared_splits_from_json(data, args.splits)
    else:
        common_samples, _, _, test_idx = compute_shared_splits(data, val_size=0.1, test_size=0.2, seed=42)
        
    test_samples = [common_samples[i] for i in test_idx]
    
    test_data = {
        "rna": data["rna"].loc[test_samples].values.astype(np.float32),
        "methylation": data["methylation"].loc[test_samples].values.astype(np.float32)
    }
    
    print("Loading Baseline model...")
    model_base = load_shared(args.ae_dir, args.baseline_ckpt, data, device, args.shared_dim)
    print("Loading Modified model...")
    model_mod = load_shared(args.ae_dir, args.modified_ckpt, data, device, args.shared_dim)
    
    print("Extracting embeddings for Latent Space Analysis...")
    emb_base = extract_shared_embeddings(model_base, test_data, device, args.batch_size)
    emb_mod = extract_shared_embeddings(model_mod, test_data, device, args.batch_size)
    
    plot_latent_space(emb_base, emb_mod, os.path.join(args.out, "latent_space_comparison.png"))
    
    print("Evaluating Missing-Modality Imputation...")
    meth_mask_ones = np.ones_like(test_data["methylation"], dtype=bool)
    rna_mask_ones = np.ones_like(test_data["rna"], dtype=bool)
    
    metrics_base_r2m, _ = impute_shared_crossmodal(model_base, "rna", "methylation", test_data["rna"], test_data["methylation"], meth_mask_ones, device, args.batch_size)
    metrics_mod_r2m, _ = impute_shared_crossmodal(model_mod, "rna", "methylation", test_data["rna"], test_data["methylation"], meth_mask_ones, device, args.batch_size)
    
    metrics_base_m2r, _ = impute_shared_crossmodal(model_base, "methylation", "rna", test_data["methylation"], test_data["rna"], rna_mask_ones, device, args.batch_size)
    metrics_mod_m2r, _ = impute_shared_crossmodal(model_mod, "methylation", "rna", test_data["methylation"], test_data["rna"], rna_mask_ones, device, args.batch_size)
    
    res_modality = {
        "Baseline": {
            "MSE (R->M)": metrics_base_r2m["mse"],
            "MSE (M->R)": metrics_base_m2r["mse"],
            "Pearson (R->M)": metrics_base_r2m["pearson"],
            "Pearson (M->R)": metrics_base_m2r["pearson"],
        },
        "Modified": {
            "MSE (R->M)": metrics_mod_r2m["mse"],
            "MSE (M->R)": metrics_mod_m2r["mse"],
            "Pearson (R->M)": metrics_mod_r2m["pearson"],
            "Pearson (M->R)": metrics_mod_m2r["pearson"],
        }
    }
    plot_benchmark_bar(res_modality, "Missing-Modality Imputation", os.path.join(args.out, "missing_modality_benchmark.png"))
    
    print("Evaluating Missing-Value Imputation...")
    rng = np.random.default_rng(42)
    rna_missing_mask = make_random_mask(test_data["rna"], 0.2, rng)
    meth_missing_mask = make_random_mask(test_data["methylation"], 0.2, rng)
    
    rna_corrupted = test_data["rna"].copy()
    rna_corrupted[rna_missing_mask] = np.nan
    meth_corrupted = test_data["methylation"].copy()
    meth_corrupted[meth_missing_mask] = np.nan
    
    corrupted_data = {"rna": rna_corrupted, "methylation": meth_corrupted}
    masks = {"rna": rna_missing_mask, "methylation": meth_missing_mask}
    
    metrics_base_val = impute_shared_values(model_base, corrupted_data, test_data, masks, device, batch_size=args.batch_size)
    metrics_mod_val = impute_shared_values(model_mod, corrupted_data, test_data, masks, device, batch_size=args.batch_size)
    
    res_value = {
        "Baseline": {
            "MSE (RNA)": metrics_base_val["rna"]["mse"],
            "MSE (Meth)": metrics_base_val["methylation"]["mse"],
            "Pearson (RNA)": metrics_base_val["rna"]["pearson"],
            "Pearson (Meth)": metrics_base_val["methylation"]["pearson"],
        },
        "Modified": {
            "MSE (RNA)": metrics_mod_val["rna"]["mse"],
            "MSE (Meth)": metrics_mod_val["methylation"]["mse"],
            "Pearson (RNA)": metrics_mod_val["rna"]["pearson"],
            "Pearson (Meth)": metrics_mod_val["methylation"]["pearson"],
        }
    }
    plot_benchmark_bar(res_value, "Missing-Value Imputation", os.path.join(args.out, "missing_value_benchmark.png"))
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
