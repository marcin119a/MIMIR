"""
compare_all_variants.py
========================
Benchmarking script to evaluate 6 models: 
(AE, VAE, CVAE) x (Baseline, Modified Loss).

Generates:
  1. Loss Curves (Validation Imputation & Contrastive Loss).
  2. Latent Space Grid (3x2 UMAP plots for each variant).
  3. Imputation Performance Bar Charts (Missing Modality & Missing Value).
"""
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

from src.data_utils import load_shared_splits_from_json, compute_shared_splits
from src.mae_masked import MultiModalWithSharedSpace, MultiModalWithSharedVAE
from src.shared_finetune import extract_encoder_decoder_from_pretrained, load_modality_with_config
# CVAE imports
from src.cvae import load_conditions_from_json, extract_encoder_decoder_from_cvae, load_cvae_with_config
from src.cvae_phase2 import ConditionalMultiModalWithSharedSpace
from compare_imputation import make_random_mask, compute_metrics, impute_shared_crossmodal

MODELS_CONFIG = {
    "AE": {
        "class": MultiModalWithSharedSpace,
        "base_dir": "checkpoints/baseline_ae",
        "mod_dir": "checkpoints/modified_ae",
        "ckpt": "shared_model_ep100.pt",
        "hist": "loss_history_ep100.json",
        "needs_c": False
    },
    "VAE": {
        "class": MultiModalWithSharedVAE,
        "base_dir": "checkpoints/baseline_vae",
        "mod_dir": "checkpoints/modified_vae",
        "ckpt": "shared_model_ep100.pt",
        "hist": "loss_history_ep100.json",
        "needs_c": False
    },
    "CVAE": {
        "class": ConditionalMultiModalWithSharedSpace,
        "base_dir": "checkpoints/baseline_cvae/exp_baseline",
        "mod_dir": "checkpoints/modified_cvae/exp_baseline",
        "ckpt": "model_best.pt",
        "hist": "loss_history.json",
        "needs_c": True
    }
}

def load_ae_encoders_decoders(ae_dir, multi_omic_data, device):
    name_map = {"rna": "rna", "methylation": "mth"}
    encoders, decoders, hidden_dims = {}, {}, {}
    for mod in multi_omic_data:
        short = name_map.get(mod, mod)
        path = os.path.join(ae_dir, f"{short}_ae.pt")
        ae_m, hidden_dim_m, _ = load_modality_with_config(path, map_location=device)
        ae_m = ae_m.to(device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m
    return encoders, decoders, hidden_dims

def load_cvae_encoders_decoders(cvae_dir, multi_omic_data, device):
    name_map = {"rna": "rna", "methylation": "mth"}
    encoders, decoders, hidden_dims = {}, {}, {}
    for mod in multi_omic_data:
        short = name_map.get(mod, mod)
        path = os.path.join(cvae_dir, f"{short}_cvae.pt")
        cvae_m, hidden_dim_m, _ = load_cvae_with_config(path, map_location=device)
        cvae_m = cvae_m.to(device)
        enc, dec = extract_encoder_decoder_from_cvae(cvae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m
    return encoders, decoders, hidden_dims

def construct_model(arch_name, ckpt_path, ae_dir, cvae_dir, data, device):
    cfg = MODELS_CONFIG[arch_name]
    if cfg["needs_c"]:
        encoders, decoders, hidden_dims = load_cvae_encoders_decoders(cvae_dir, data, device)
        model = cfg["class"](
            encoders=encoders, decoders=decoders, hidden_dims=hidden_dims, shared_dim=128
        )
    else:
        encoders, decoders, hidden_dims = load_ae_encoders_decoders(ae_dir, data, device)
        model = cfg["class"](
            encoders=encoders, decoders=decoders, hidden_dims=hidden_dims, shared_dim=128, proj_depth=1
        )
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_shared_embeddings(model, data_dict, cond_matrix, device, batch_size=128, needs_c=False):
    embeddings = {m: [] for m in data_dict}
    model.eval()
    
    mods = list(data_dict.keys())
    N = data_dict[mods[0]].shape[0]
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb_clean = {}
        for mod in mods:
            xb = torch.tensor(data_dict[mod][start:end], device=device)
            xb_c = xb.clone()
            xb_c[torch.isnan(xb_c)] = 0.0
            xb_clean[mod] = xb_c
            
        if needs_c:
            c_batch = torch.tensor(cond_matrix[start:end], device=device)
            shared, _, _ = model(xb_clean, c_batch)
        else:
            shared_all = {}
            for mod, x in xb_clean.items():
                if isinstance(model, MultiModalWithSharedVAE):
                    h = model.encoders[mod](x)
                    mu = model.proj_mu[mod](h)
                    shared_all[mod] = mu
                else:
                    h = model.encoders[mod](x)
                    z = model.projections[mod](h)
                    shared_all[mod] = z
            shared = shared_all
            
        for mod in mods:
            embeddings[mod].append(shared[mod].cpu().numpy())
            
    for m in embeddings:
        embeddings[m] = np.concatenate(embeddings[m], axis=0)
    return embeddings

@torch.no_grad()
def impute_crossmodal_cvae(model, source_mod, target_mod, source_data, target_orig, target_mask, cond_matrix, device, batch_size=128):
    model.eval()
    N = source_data.shape[0]
    preds = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.tensor(source_data[start:end], device=device)
        xb[torch.isnan(xb)] = 0.0
        c_batch = torch.tensor(cond_matrix[start:end], device=device)
        h = model.encoders[source_mod](xb, c_batch)
        z = model.projections[source_mod](h)
        h_hat = model.rev_projections[target_mod](z)
        x_imp = model.decoders[target_mod](h_hat, c_batch)
        preds.append(x_imp.cpu().numpy())
        
    preds = np.concatenate(preds, axis=0)
    y_true = target_orig[target_mask]
    y_pred = preds[target_mask]
    return compute_metrics(y_true, y_pred)
    
@torch.no_grad()
def impute_shared_values(model, data_corrupted, data_orig, mask_by_mod, cond_matrix, device, batch_size=128, needs_c=False, self_weight=10.0):
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
        if needs_c:
            c_batch = torch.tensor(cond_matrix[start:end], device=device)
            for mod, xb_c in batch_clean.items():
                h = model.encoders[mod](xb_c, c_batch)
                z = model.projections[mod](h)
                shared_all[mod] = z
        else:
            for mod, xb_c in batch_clean.items():
                h = model.encoders[mod](xb_c)
                if isinstance(model, MultiModalWithSharedVAE):
                    shared_all[mod] = model.proj_mu[mod](h)
                else:
                    shared_all[mod] = model.projections[mod](h)

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
            
            if needs_c:
                h_hat = model.rev_projections[target_mod](z_w)
                x_imp = model.decoders[target_mod](h_hat, c_batch)
            else:
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

def plot_loss_curves(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mathematical description of the Hybrid Loss
    # L_total = L_recon + lambda_c * L_contrast + lambda_i * L_impute
    # L_contrast = L_NT-Xent + w_hard * L_Triplet
    
    axes[0].set_title("Validation Alignment Loss\n(NT-Xent + Hard Negative Triplet)")
    axes[1].set_title("Validation Imputation Loss\n(RNA <-> DNA Methylation)")
    
    colors = {"AE": "tab:blue", "VAE": "tab:orange", "CVAE": "tab:green"}
    
    for arch, cfg in MODELS_CONFIG.items():
        base_hist = os.path.join(cfg["base_dir"], cfg["hist"])
        mod_hist = os.path.join(cfg["mod_dir"], cfg["hist"])
        
        if os.path.exists(base_hist) and os.path.exists(mod_hist):
            with open(base_hist, "r") as f: hb = json.load(f)["val"]
            with open(mod_hist, "r") as f: hm = json.load(f)["val"]
            
            axes[0].plot(hb["contrast"], linestyle="--", color=colors[arch], label=f"{arch} Baseline", alpha=0.5)
            axes[0].plot(hm["contrast"], linestyle="-", color=colors[arch], label=f"{arch} Hybrid Loss", linewidth=2)
            
            axes[1].plot(hb["impute"], linestyle="--", color=colors[arch], label=f"{arch} Baseline", alpha=0.5)
            axes[1].plot(hm["impute"], linestyle="-", color=colors[arch], label=f"{arch} Hybrid Loss", linewidth=2)
            
    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value")
        ax.legend()
    
    # Add formula text to the plot
    formula_text = (
        r"$\mathcal{L}_{contrast} = \mathcal{L}_{NT-Xent} + \lambda_{hard} \cdot \mathcal{L}_{Triplet}$" + "\n" +
        r"$\mathcal{L}_{Triplet} = \max(0, d_{pos} - d_{neg} + margin)$"
    )
    fig.text(0.5, 0.01, formula_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(out_dir, "loss_curves_comparison.png"), dpi=150)
    plt.close()

def plot_benchmark_bar(results, title, save_path):
    metrics = list(results[list(results.keys())[0]]['Baseline'].keys())
    arches = list(results.keys())
    
    # We want one subplot per metric (e.g., MSE and Pearson)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1: axes = [axes]
    
    x = np.arange(len(arches))
    width = 0.3
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals_base = []
        vals_mod = []
        for arch in arches:
            vals_base.append(results[arch]['Baseline'].get(metric, 0))
            vals_mod.append(results[arch]['Modified'].get(metric, 0))
            
        ax.bar(x - width/2, vals_base, width, label='Baseline', color='lightgray', edgecolor='black')
        ax.bar(x + width/2, vals_mod, width, label='Hybrid Loss', color='tab:green', edgecolor='black')
        
        ax.set_title(f"Metric: {metric}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(arches)
        ax.set_ylabel("Score")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tcga_redo_mlomicZ.pkl")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--ae_dir", default="aes_redo_z")
    parser.add_argument("--cvae_dir", default="cvae_phase1")
    parser.add_argument("--primary_sites", default="data/primary_sites.json")
    parser.add_argument("--out", default="results/compare_all")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    
    # Plot loss curves
    print("Generating loss curves...")
    plot_loss_curves(args.out)
    
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
    
    # Load condition matrix for CVAE
    cond_matrix_all, _ = load_conditions_from_json(args.primary_sites, common_samples)
    cond_matrix = cond_matrix_all[test_idx].astype(np.float32)
    
    fig_umap, axes_umap = plt.subplots(3, 2, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.3)
    
    rng = np.random.default_rng(42)
    rna_missing_mask = make_random_mask(test_data["rna"], 0.2, rng)
    meth_missing_mask = make_random_mask(test_data["methylation"], 0.2, rng)
    corrupted_data = {
        "rna": np.where(rna_missing_mask, np.nan, test_data["rna"]),
        "methylation": np.where(meth_missing_mask, np.nan, test_data["methylation"])
    }
    masks = {"rna": rna_missing_mask, "methylation": meth_missing_mask}
    meth_mask_ones = np.ones_like(test_data["methylation"], dtype=bool)
    rna_mask_ones = np.ones_like(test_data["rna"], dtype=bool)
    
    results_modality = {}
    results_value = {}
    
    for row, (arch, cfg) in enumerate(MODELS_CONFIG.items()):
        print(f"\nProcessing {arch}...")
        results_modality[arch] = {}
        results_value[arch] = {}
        
        for col, variant in enumerate(["Baseline", "Modified"]):
            dir_path = cfg["base_dir"] if variant == "Baseline" else cfg["mod_dir"]
            ckpt_path = os.path.join(dir_path, cfg["ckpt"])
            
            if not os.path.exists(ckpt_path):
                print(f"Skipping {arch} {variant} - checkpoint missing: {ckpt_path}")
                continue
                
            model = construct_model(arch, ckpt_path, args.ae_dir, args.cvae_dir, data, device)
            
            emb = extract_shared_embeddings(model, test_data, cond_matrix, device, needs_c=cfg["needs_c"])
            
            # UMAP & Silhouette
            mods = ["rna", "methylation"]
            stacked = np.vstack([emb[mods[0]], emb[mods[1]]])
            labels = np.array([0]*emb[mods[0]].shape[0] + [1]*emb[mods[1]].shape[0])
            
            if HAS_UMAP:
                proj = umap.UMAP(n_components=2, random_state=42).fit_transform(stacked)
            else:
                proj = TSNE(n_components=2, random_state=42).fit_transform(stacked)
                
            # Silhouette Score on the projected 2D space
            score = silhouette_score(proj, labels)
            
            N = emb[mods[0]].shape[0]
            axes_umap[row, col].scatter(proj[:N, 0], proj[:N, 1], s=5, alpha=0.5, label="RNA")
            axes_umap[row, col].scatter(proj[N:, 0], proj[N:, 1], s=5, alpha=0.5, label="DNA Methylation")
            axes_umap[row, col].set_title(f"{arch} {variant}\nSilhouette: {score:.3f}")
            if row == 0 and col == 0: axes_umap[row, col].legend()
            
            # Missing Modality
            if cfg["needs_c"]:
                m_r2m = impute_crossmodal_cvae(model, "rna", "methylation", test_data["rna"], test_data["methylation"], meth_mask_ones, cond_matrix, device)
                m_m2r = impute_crossmodal_cvae(model, "methylation", "rna", test_data["methylation"], test_data["rna"], rna_mask_ones, cond_matrix, device)
            else:
                m_r2m, _ = impute_shared_crossmodal(model, "rna", "methylation", test_data["rna"], test_data["methylation"], meth_mask_ones, device, 128)
                m_m2r, _ = impute_shared_crossmodal(model, "methylation", "rna", test_data["methylation"], test_data["rna"], rna_mask_ones, device, 128)
                
            results_modality[arch][variant] = {
                "MSE (RNA to DNA methylation)": m_r2m["mse"],
                "Pearson (RNA to DNA methylation)": m_r2m["pearson"],
                "MSE (DNA methylation to RNA)": m_m2r["mse"],
                "Pearson (DNA methylation to RNA)": m_m2r["pearson"]
            }
            
            # Missing Value
            m_val = impute_shared_values(model, corrupted_data, test_data, masks, cond_matrix, device, needs_c=cfg["needs_c"])
            results_value[arch][variant] = {
                "Pearson (RNA)": m_val["rna"]["pearson"],
                "Pearson (DNA methylation)": m_val["methylation"]["pearson"]
            }
            
    fig_umap.suptitle("Latent Space Modality Alignment Grid")
    fig_umap.savefig(os.path.join(args.out, "latent_grid_comparison.png"), dpi=150)
    
    print("\nGenerating bar charts...")
    plot_benchmark_bar(results_modality, "Missing-Modality Architecture Comparison", os.path.join(args.out, "missing_modality_arch.png"))
    plot_benchmark_bar(results_value, "Missing-Value Architecture Comparison", os.path.join(args.out, "missing_value_arch.png"))
    print("Done! Check results/compare_all/")

if __name__ == "__main__":
    main()
