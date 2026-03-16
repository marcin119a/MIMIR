"""
Phase 1: Train per-modality Conditional VAEs (CVAE) conditioned on primary site.

Architecture per modality:
    Encoder: [x; c] → backbone → mu_head / logvar_head
    Decoder: [z; c] → x_recon
where c is a one-hot vector over primary sites from data/primary_sites.json.

Usage:
    python train_cvae_autoencoders.py
    python train_cvae_autoencoders.py --data data/tcga_redo_mlomicZ.pkl \\
        --primary_sites data/primary_sites.json --out cvae_phase1
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW

from src.data_utils import compute_shared_splits, load_shared_splits_from_json
from src.cvae import (
    ConditionalSingleModalityDataset,
    build_pretrain_cvae_for_modality,
    eval_cvae_epoch_masked,
    get_conditional_dataloader,
    load_conditions_from_json,
    pretrain_cvae_epoch,
    save_cvae_with_config,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_curves(train_overall, train_masked, val_overall, val_masked, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_overall, label="Train overall MSE")
    plt.plot(train_masked,  label="Train masked MSE")
    plt.plot(val_overall,   label="Val overall MSE")
    plt.plot(val_masked,    label="Val masked MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f"  Curve saved → {save_path}")


def train_one_modality_cvae(
    name, data_df, common_samples, condition_matrix,
    train_idx, val_idx,
    hidden_layers, n_epochs, mask_value, num_classes,
    device, out_dir, plot_dir,
    batch_size=128, lr=1e-3, weight_decay=1e-4,
    activation_dropout=0.05, mask_p=0.3,
    l1_alpha=1e-4, alpha_mask=0.5, beta=1.0,
    grad_clip=1.0, patience=15,
):
    print(f"\n{'='*60}")
    print(f"  Training {name} CVAE  |  epochs={n_epochs}  |  hidden={hidden_layers}")
    print(f"  num_classes={num_classes}  beta={beta}  mask_p={mask_p}")
    print(f"{'='*60}")

    ds = ConditionalSingleModalityDataset(data_df, common_samples, condition_matrix)
    train_loader = get_conditional_dataloader(ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
    val_loader   = get_conditional_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)

    input_dim = data_df.shape[1]

    config = {
        "input_dim":          input_dim,
        "num_classes":        num_classes,
        "hidden_layers":      hidden_layers,
        "activation_dropout": activation_dropout,
        "denoising":          True,
        "mask_p":             mask_p,
        "mask_value":         mask_value,
        "loss_on_masked":     True,
        "beta":               beta,
    }

    cvae, _ = build_pretrain_cvae_for_modality(
        input_dim, num_classes, hidden_layers,
        activation_dropout=activation_dropout,
        denoising=True, mask_p=mask_p, mask_value=mask_value,
        loss_on_masked=True, beta=beta,
    )
    cvae = cvae.to(device)
    opt = AdamW(cvae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    tr_overall_h, tr_masked_h, va_overall_h, va_masked_h = [], [], [], []
    best_val_masked = float("inf")
    best_state = None
    epochs_no_improve = 0

    for ep in range(1, n_epochs + 1):
        tr_loss, tr_overall, tr_masked = pretrain_cvae_epoch(
            cvae, train_loader, opt, device,
            l1_alpha=l1_alpha, alpha_mask=alpha_mask,
            beta=beta, grad_clip=grad_clip,
        )
        va_overall, va_masked = eval_cvae_epoch_masked(cvae, val_loader, device)
        scheduler.step()

        tr_overall_h.append(tr_overall)
        tr_masked_h.append(tr_masked)
        va_overall_h.append(va_overall)
        va_masked_h.append(va_masked)

        if va_masked < best_val_masked:
            best_val_masked = va_masked
            best_state = {k: v.cpu().clone() for k, v in cvae.state_dict().items()}
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"  [{name}] ep {ep:03d} | loss {tr_loss:.4f} | "
            f"overall {tr_overall:.4f} | masked {tr_masked:.4f} | "
            f"val_overall {va_overall:.4f} | val_masked {va_masked:.4f}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"  [Early stop] No improvement for {patience} epochs.")
            break

    if best_state is not None:
        cvae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model (val_masked={best_val_masked:.4f})")

    ckpt_path = os.path.join(out_dir, f"{name}_cvae")
    save_cvae_with_config(cvae, config, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}.pt")

    plot_curves(
        tr_overall_h, tr_masked_h, va_overall_h, va_masked_h,
        title=f"{name} CVAE – loss curves",
        save_path=os.path.join(plot_dir, f"{name}_cvae_curves.png"),
    )
    return cvae


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Train per-modality CVAEs conditioned on primary site")
    p.add_argument("--data",          default="data/tcga_redo_mlomicZ.pkl", help="Multi-omic pickle")
    p.add_argument("--splits",        default="data/splits.json",           help="Train/val/test splits JSON")
    p.add_argument("--primary_sites", default="data/primary_sites.json",    help="Primary sites JSON")
    p.add_argument("--out",           default="cvae_phase1",                help="Output directory")
    p.add_argument("--device",        default=None,                         help="cuda / cpu (auto-detected)")
    return p.parse_args()


def main():
    args = parse_args()

    plot_dir = os.path.join(args.out, "plots")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

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

    # Load primary site conditions aligned with common_samples
    condition_matrix, class_names = load_conditions_from_json(args.primary_sites, common_samples)
    num_classes = len(class_names)
    preview = class_names[:5]
    print(f"Primary sites: {num_classes} classes → {preview}{'...' if num_classes > 5 else ''}")

    # Save class names for reference in Phase 2
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved → {args.out}/class_names.json")

    # Per-modality CVAE configs (mirror train_autoencoders.py)
    modality_configs = [
        dict(name="rna", key="rna",         hidden_layers=[1024, 512], n_epochs=100,
             mask_value=0.0, patience=15),
        dict(name="mth", key="methylation", hidden_layers=[512, 256],  n_epochs=100,
             mask_value=0.0, patience=15),
    ]

    for cfg in modality_configs:
        modality_key = cfg.pop("key")
        if modality_key not in multi_omic_data:
            print(f"\n[SKIP] '{modality_key}' not found in data keys: {list(multi_omic_data.keys())}")
            continue
        train_one_modality_cvae(
            data_df=multi_omic_data[modality_key],
            common_samples=common_samples,
            condition_matrix=condition_matrix,
            train_idx=train_idx,
            val_idx=val_idx,
            num_classes=num_classes,
            device=device,
            out_dir=args.out,
            plot_dir=plot_dir,
            **cfg,
        )

    print(f"\nDone. All CVAE checkpoints saved to {args.out}")


if __name__ == "__main__":
    main()