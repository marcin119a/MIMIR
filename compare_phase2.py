"""
compare_phase2.py
=================
Porównuje proces uczenia modeli w Fazie 2:
  1. Shared AE (deterministyczna wspólna przestrzeń latenta)
  2. Shared VAE (wspólna przestrzeń generatywna z KL loss)

Dla obu wariantów:
  - korzysta z pretrained AE z Fazy 1 (flaga --ae_dir)
  - trenuje nowy Shared AE / Shared VAE na tych samych parametrach
  - generuje czytelny wykres uczący (krzywe loss: Total, Recon, Contrast, Impute, KL)
  - drukuje w konsoli podsumowanie najlepszych straty walidacyjnych

Wynik: tabela w konsoli + compare_phase2_results.png

Usage:
    python compare_phase2.py
    python compare_phase2.py --epochs 200 --device cuda
"""

import argparse
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data_utils import (
    compute_shared_splits,
    load_shared_splits_from_json,
)
from src.shared_finetune import run_shared_finetune, run_shared_vae_finetune


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_compare_phase2(hist_ae_train, hist_ae_val, hist_vae_train, hist_vae_val, save_path):
    loss_types = ["total", "recon", "contrast", "impute", "kl"]
    
    fig, axes = plt.subplots(1, len(loss_types), figsize=(4 * len(loss_types), 4.5))
    
    for i, lt in enumerate(loss_types):
        ax = axes[i]
        
        # AE lines
        if lt in hist_ae_train and hist_ae_train[lt]:
            ax.plot(hist_ae_train[lt], label="AE Train", color="#e07b54", linestyle="-")
            ax.plot(hist_ae_val[lt],   label="AE Val",   color="#e07b54", linestyle="--")
            
        # VAE lines
        if lt in hist_vae_train and hist_vae_train[lt]:
            ax.plot(hist_vae_train[lt], label="VAE Train", color="#4c8bb5", linestyle="-")
            ax.plot(hist_vae_val[lt],   label="VAE Val",   color="#4c8bb5", linestyle="--")

        ax.set_title(f"{lt.capitalize()} Loss", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle("Phase 2 finetuning: Shared AE vs Shared VAE", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {save_path}")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 Comparison: Shared AE vs Shared VAE")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl", help="Path to multi-omic pickle")
    p.add_argument("--splits",     default="data/splits.json",           help="Path to splits JSON (optional)")
    p.add_argument("--ae_dir",     default="aes_redo_z",                 help="Directory with Phase-1 AE checkpoints")
    p.add_argument("--device",     default=None,                         help="cuda / cpu (auto-detected if omitted)")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--shared_dim", type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_contrast", type=float, default=1.0)
    p.add_argument("--lambda_impute",   type=float, default=1.0)
    p.add_argument("--beta_max",          type=float, default=1e-3, help="(VAE) Max KL weight after annealing")
    p.add_argument("--beta_warmup_epochs", type=int,  default=50, help="(VAE) Epochs to linearly anneal beta")
    p.add_argument("--modality_dropout_prob", type=float, default=0.4)
    p.add_argument("--feature_mask_p",        type=float, default=0.15)
    p.add_argument("--alpha_mask_recon",      type=float, default=0.5)
    p.add_argument("--freeze_encoders_decoders", action="store_true",
                   help="Freeze encoder/decoder weights; train only projection heads")
    p.add_argument("--out",        default="compare_phase2_results.png", help="Output plot path")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    # Load data
    print(f"Loading data from {args.data} …")
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)
        
    # Splits
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

    # Filter to only the two modalities used in this pipeline
    ACTIVE_MODALITIES = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE_MODALITIES}
    print(f"Active modalities: {list(multi_omic_data.keys())}")
    
    print(
        f"Samples: total={len(common_samples)} | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}\n"
    )

    # Map modality names to Phase-1 checkpoint paths
    name_map = {"rna": "rna", "methylation": "mth"}
    model_paths = {}
    for mod in multi_omic_data.keys():
        short = name_map.get(mod, mod)
        model_paths[mod] = os.path.join(args.ae_dir, f"{short}_ae.pt")

    print("AE checkpoint paths:")
    for mod, path in model_paths.items():
        exists = "OK" if os.path.exists(path) else "MISSING"
        if exists == "MISSING":
            raise FileNotFoundError(f"AE checkpoint missing, aborting: {path}")
        print(f"  [{exists}] {mod}: {path}")

    # =========================================================================
    # 1. RUN SHARED AE
    # =========================================================================
    print(f"\n{'='*78}")
    print("  [1/2] RUNNING SHARED AE FINETUNING")
    print(f"{'='*78}")
    t0_ae = time.time()
    _, train_hist_ae, val_hist_ae, _, _, _, _ = run_shared_finetune(
        multi_omic_data=multi_omic_data,
        common_samples=common_samples,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_paths=model_paths,
        device=device,
        shared_dim=args.shared_dim,
        proj_depth=1,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        lambda_contrast=args.lambda_contrast,
        lambda_impute=args.lambda_impute,
        modality_dropout_prob=args.modality_dropout_prob,
        feature_mask_p_train=args.feature_mask_p,
        feature_mask_p_val=args.feature_mask_p,
        alpha_mask_recon=args.alpha_mask_recon,
        two_path_clean_for_contrast=False,
        freeze_encoders_decoders=args.freeze_encoders_decoders,
        verbose=False,
    )
    time_ae = time.time() - t0_ae
    
    # =========================================================================
    # 2. RUN SHARED VAE
    # =========================================================================
    print(f"\n{'='*78}")
    print("  [2/2] RUNNING SHARED VAE FINETUNING")
    print(f"{'='*78}")
    t0_vae = time.time()
    _, train_hist_vae, val_hist_vae, _, _, _, _ = run_shared_vae_finetune(
        multi_omic_data=multi_omic_data,
        common_samples=common_samples,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_paths=model_paths,
        device=device,
        shared_dim=args.shared_dim,
        proj_depth=1,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        lambda_contrast=args.lambda_contrast,
        lambda_impute=args.lambda_impute,
        beta_max=args.beta_max,
        beta_warmup_epochs=args.beta_warmup_epochs,
        modality_dropout_prob=args.modality_dropout_prob,
        feature_mask_p_train=args.feature_mask_p,
        feature_mask_p_val=args.feature_mask_p,
        alpha_mask_recon=args.alpha_mask_recon,
        freeze_encoders_decoders=args.freeze_encoders_decoders,
        verbose=False,
    )
    time_vae = time.time() - t0_vae

    # =========================================================================
    # 3. PLOT AND SUMMARY
    # =========================================================================
    
    plot_compare_phase2(
        hist_ae_train=train_hist_ae,
        hist_ae_val=val_hist_ae,
        hist_vae_train=train_hist_vae,
        hist_vae_val=val_hist_vae,
        save_path=args.out
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    def get_best(hist_dict, key):
        if key in hist_dict and len(hist_dict[key]) > 0:
            return min(hist_dict[key])
        return float('nan')

    print(f"\n{'='*78}")
    print(f"{'Variant':<14} {'best_val_total':>16} {'best_val_recon':>16} {'best_val_impute':>16} {'epochs':>7} {'time(s)':>8}")
    print("─" * 78)

    print(f"{'Shared AE':<14} {get_best(val_hist_ae, 'total'):>16.4f} {get_best(val_hist_ae, 'recon'):>16.4f} {get_best(val_hist_ae, 'impute'):>16.4f} {args.epochs:>7} {time_ae:>8.1f}")
    print(f"{'Shared VAE':<14} {get_best(val_hist_vae, 'total'):>16.4f} {get_best(val_hist_vae, 'recon'):>16.4f} {get_best(val_hist_vae, 'impute'):>16.4f} {args.epochs:>7} {time_vae:>8.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
