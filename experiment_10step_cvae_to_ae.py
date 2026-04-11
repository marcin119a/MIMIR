import argparse
import json
import os
import pickle
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_utils import (
    compute_shared_splits,
    load_shared_splits_from_json,
)
from src.cvae import (
    ConditionalSingleModalityDataset,
    get_conditional_dataloader,
    load_conditions_from_json
)
from src.cvae_experiment import ModalityCVAEExperiment

# ─── Configuration for the 10 steps ──────────────────────────────────────────

STEPS = [
    {
        "name": "1. Marcin-CVAE (Base)",
        "use_batchnorm": False, "add_final_activation": True, "kl_reduction": "mean",
        "beta": 1.0, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "2. + LayerNorm",
        "use_batchnorm": True, "add_final_activation": True, "kl_reduction": "mean",
        "beta": 1.0, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "3. - Final ReLU",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "mean",
        "beta": 1.0, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "4. Fix KL (Mean->Sum)",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 1.0, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "5. Low Beta (1.0->0.1)",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.1, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "6. Zero Beta",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.0, "deterministic": False, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "7. Deterministic",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.0, "deterministic": True, "condition_on_site": True, "l1_alpha": 0.0
    },
    {
        "name": "8. - Conditioning",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.0, "deterministic": True, "condition_on_site": False, "l1_alpha": 0.0
    },
    {
        "name": "9. + L1 Latent",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.0, "deterministic": True, "condition_on_site": False, "l1_alpha": 1e-4
    },
    {
        "name": "10. Mimir-AE (Target)",
        "use_batchnorm": True, "add_final_activation": False, "kl_reduction": "sum",
        "beta": 0.0, "deterministic": True, "condition_on_site": False, "l1_alpha": 1e-4
    },
]

# ─── Training / Evaluation ───────────────────────────────────────────────────

def train_experiment_epoch(model, loader, opt, device, l1_alpha, alpha_mask, beta, grad_clip):
    model.train()
    total_loss = total_overall = total_masked = 0.0
    n = 0
    for xb, cb in loader:
        xb, cb = xb.to(device), cb.to(device)
        orig_missing = torch.isnan(xb)
        xb_in = xb.clone()
        xb_in[orig_missing] = model.mask_value
        
        opt.zero_grad()
        mu, recon = model(xb_in, cb)
        
        diff_sq = (recon - xb_in) ** 2
        valid = ~orig_missing
        overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()
        
        if model.denoising and model.loss_on_masked:
            mask_art = model._last_mask.to(device)
            combined = mask_art & valid
            masked_mse = diff_sq[combined].mean() if combined.any() else overall_mse
        else:
            masked_mse = overall_mse
            
        recon_loss = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse
        loss = recon_loss + beta * model._last_kl
        
        if l1_alpha > 0:
            loss = loss + l1_alpha * mu.abs().mean()
            
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        
        total_loss += loss.item()
        total_overall += overall_mse.item()
        total_masked += masked_mse.item()
        n += 1
    return total_loss / n, total_overall / n, total_masked / n

def eval_experiment_distribution(model, loader, device):
    """Returns (mean_overall, mean_masked, all_masked_errors)"""
    model.eval()
    total_overall = total_masked = 0.0
    all_masked_errors = []
    n = 0
    with torch.no_grad():
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            orig_missing = torch.isnan(xb)
            xb_in = xb.clone()
            xb_in[orig_missing] = model.mask_value
            
            # Use deterministic masking for eval stability
            if model.denoising and model.mask_p > 0.0:
                torch.manual_seed(42) # fix mask for fair comparison
                art_mask = torch.rand_like(xb_in) < model.mask_p
                xb_noisy = xb_in.clone()
                xb_noisy[art_mask] = model.mask_value
            else:
                xb_noisy = xb_in
                art_mask = torch.zeros_like(xb_in, dtype=torch.bool)
                
            _, recon = model(xb_noisy, cb)
            
            diff_sq = (recon - xb_in) ** 2
            valid = ~orig_missing
            overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()
            
            combined = art_mask & valid
            if combined.any():
                masked_errors = diff_sq[combined].cpu().numpy()
                all_masked_errors.extend(masked_errors)
                masked_mse = masked_errors.mean()
            else:
                masked_mse = overall_mse

            total_overall += overall_mse.item()
            total_masked += masked_mse.item()
            n += 1
    return total_overall / n, total_masked / n, np.array(all_masked_errors)

# ─── Unified Plotting ───────────────────────────────────────────────────────

def plot_unified_comparison(results, out_path):
    fig, ax = plt.subplots(figsize=(15, 9))
    
    # Classy palette for 9 lines + 1 reference
    palette = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", 
               "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#000000"] # Step 10 is Black

    # Legend strings summarizing the parameters
    step_descriptions = [
        "Base (Stoch, Cond, NoLN, ReLU, MeanKL)",
        "+ LayerNorm (Normalization)",
        "- Final ReLU (Linear Head)",
        "Fix KL (Standard Sum Scaling)",
        "Low Beta (0.1, Relaxed Bottleneck)",
        "Zero Beta (No KL Regularization)",
        "Deterministic (No Reparam Sampling)",
        "- Conditioning (Unconditional AE)",
        "+ L1 Latent (Mimir Regularization)",
        "Mimir-AE (Target - Reference)"
    ]

    for i, res in enumerate(results):
        h = res["history"]
        is_ref = (i == 9)
        color = palette[i]
        
        label = f"Step {i+1}: {step_descriptions[i]}"
        
        ax.plot(h["val_masked"], label=label, color=color, 
                linewidth=3 if is_ref else 1.5,
                linestyle="--" if is_ref else "-",
                alpha=1.0 if is_ref else 0.7)
        
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Validation Masked MSE", fontsize=12)
    ax.set_title("Architectural Convergence to Mimir-AE (100 Epochs)", fontsize=16, pad=20)
    
    # Summary of flags/parameters for the plot description
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    textstr = "\n".join([
        "Parameter Flags:",
        "LN: LayerNorm",
        "ReLU: Encoder Final Activation",
        "KL-Sum: Sum-based KL Scaling",
        "Beta: KL weight",
        "Det: No Reparameterization",
        "Cond: Conditioning on Primary Site"
    ])
    ax.text(1.02, 0.4, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.5)
    
    # Legend outside to the right
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tcga_redo_mlomicZ.pkl")
    parser.add_argument("--primary_sites", default="data/primary_sites.json")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    # Apple Silicon / CUDA / CPU detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    rna_df = data["rna"]
    
    if os.path.exists(args.splits):
        samples, train_idx, val_idx, _ = load_shared_splits_from_json({"rna": rna_df}, args.splits)
    else:
        samples, train_idx, val_idx, _ = compute_shared_splits({"rna": rna_df})
        
    cond_matrix, class_names = load_conditions_from_json(args.primary_sites, samples)
    num_classes = len(class_names)
    
    ds = ConditionalSingleModalityDataset(rna_df, samples, cond_matrix)
    train_loader = get_conditional_dataloader(ds, batch_size=128, shuffle=True, split_idx=train_idx)
    val_loader = get_conditional_dataloader(ds, batch_size=128, shuffle=False, split_idx=val_idx)
    
    results = []
    for step_cfg in STEPS:
        print(f"\n>>> Running Step: {step_cfg['name']}")
        model = ModalityCVAEExperiment(
            input_dim=rna_df.shape[1],
            num_classes=num_classes,
            hidden_layers=[512],
            activation_dropout=0.05,
            denoising=True,
            mask_p=0.3,
            mask_value=0.0,
            loss_on_masked=True,
            beta=step_cfg["beta"],
            use_batchnorm=step_cfg["use_batchnorm"],
            add_final_activation=step_cfg["add_final_activation"],
            kl_reduction=step_cfg["kl_reduction"],
            deterministic=step_cfg["deterministic"],
            condition_on_site=step_cfg["condition_on_site"]
        ).to(device)
        
        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        
        history = {"train_masked": [], "val_masked": []}
        best_val_masked = float("inf")
        best_val_overall = float("inf")
        
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_overall, tr_masked = train_experiment_epoch(
                model, train_loader, opt, device, 
                l1_alpha=step_cfg["l1_alpha"], alpha_mask=0.5, 
                beta=step_cfg["beta"], grad_clip=1.0
            )
            va_overall, va_masked, _ = eval_experiment_distribution(model, val_loader, device)
            scheduler.step()
            
            history["train_masked"].append(tr_masked)
            history["val_masked"].append(va_masked)
            
            if va_masked < best_val_masked:
                best_val_masked = va_masked
                best_val_overall = va_overall
            
            if ep % 20 == 0:
                print(f"  Ep {ep}/{args.epochs} | Val Masked MSE: {va_masked:.4f}")
                
        results.append({
            "name": step_cfg["name"],
            "config": step_cfg,
            "best_val_masked": best_val_masked,
            "best_val_overall": best_val_overall,
            "history": history
        })

    # Unified Comparison Plot
    os.makedirs("results/experiment_10step", exist_ok=True)
    plot_unified_comparison(results, "results/experiment_10step/unified_comparison.png")
    
    # Save metadata
    with open("results/experiment_10step/results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in results], f, indent=2)
    
    print("\nExperiment Complete. Unified comparison saved to results/experiment_10step/")

if __name__ == "__main__":
    main()
