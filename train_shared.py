"""
Phase 2: Fine-tune modality-specific autoencoders into a shared latent space.

Equivalent to 2_Phase2_Train_MAE.ipynb but runnable as a plain script.
Loads pretrained AE checkpoints from --ae_dir, saves the shared model to --out.

Usage:
    python train_shared.py
    python train_shared.py --data data/tcga_redo_mlomicZ.pkl --splits data/splits.json
    python train_shared.py --ae_dir aes_redo_z --out checkpoints/finetuned --epochs 200
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data_utils import (
    load_shared_splits_from_json,
    compute_shared_splits,
)
from src.shared_finetune import run_shared_finetune, save_shared_model


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_loss_curves(train_hist, val_hist, save_path):
    loss_types = ["total", "recon", "contrast", "impute"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, lt in enumerate(loss_types):
        axs[i].plot(train_hist[lt], label=f"Train")
        axs[i].plot(val_hist[lt],   label=f"Val")
        axs[i].set_title(f"{lt.capitalize()} Loss")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Loss curves saved → {save_path}")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Train shared latent space")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl", help="Path to multi-omic pickle")
    p.add_argument("--splits",     default="data/splits.json",           help="Path to splits JSON (optional)")
    p.add_argument("--ae_dir",     default="aes_redo_z",                 help="Directory with Phase-1 AE checkpoints")
    p.add_argument("--out",        default="checkpoints/finetuned",      help="Output directory for shared model")
    p.add_argument("--device",     default=None,                         help="cuda / cpu (auto-detected if omitted)")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--shared_dim", type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_contrast", type=float, default=1.0)
    p.add_argument("--lambda_impute",   type=float, default=1.0)
    p.add_argument("--modality_dropout_prob", type=float, default=0.4)
    p.add_argument("--feature_mask_p",        type=float, default=0.15)
    p.add_argument("--alpha_mask_recon",      type=float, default=0.5)
    p.add_argument("--freeze_encoders_decoders", action="store_true",
                   help="Freeze encoder/decoder weights; train only projection heads")
    p.add_argument("--two_path", action="store_true",
                   help="Use two-path (clean for contrastive, noisy for recon)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data} …")
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)
    print(f"Modalities: {list(multi_omic_data.keys())}")

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

    print(
        f"Samples: total={len(common_samples)} | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    # Filter to only the two modalities used in this pipeline
    ACTIVE_MODALITIES = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE_MODALITIES}
    print(f"Active modalities: {list(multi_omic_data.keys())}")

    # Map modality names to Phase-1 checkpoint paths
    name_map = {"rna": "rna", "methylation": "mth"}
    model_paths = {}
    for mod in multi_omic_data.keys():
        short = name_map.get(mod, mod)
        model_paths[mod] = os.path.join(args.ae_dir, f"{short}_ae.pt")

    print("AE checkpoint paths:")
    for mod, path in model_paths.items():
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{exists}] {mod}: {path}")

    # Run Phase 2 finetuning
    model, train_hist, val_hist, _, _, _, _ = run_shared_finetune(
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
        two_path_clean_for_contrast=args.two_path,
        freeze_encoders_decoders=args.freeze_encoders_decoders,
        verbose=True,
    )

    # Save model checkpoint + loss history
    save_shared_model(
        model,
        save_dir=args.out,
        epoch=args.epochs,
        train_loss_hist=train_hist,
        val_loss_hist=val_hist,
    )

    # Plot loss curves
    plot_loss_curves(
        train_hist, val_hist,
        save_path=os.path.join(args.out, "loss_curves.png"),
    )

    print(f"\nDone. Shared model saved to {args.out}/")


if __name__ == "__main__":
    main()
