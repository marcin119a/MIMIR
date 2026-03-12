"""
Phase 1: Train modality-specific denoising autoencoders.

Equivalent to 1_Phase1_Train_Autoencoders.ipynb but runnable as a plain script.
Saves checkpoints to aes_redo_z/ and loss curves to aes_redo_z/plots/.

Usage:
    python train_autoencoders.py
    python train_autoencoders.py --data tcga_redo_mlomicZ.pkl --splits splits.json
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from src.data_utils import (
    SingleModalityDatasetAligned,
    compute_shared_splits,
    get_dataloader,
    load_shared_splits_from_json,
)
from src.mae_masked import (
    build_pretrain_ae_for_modality,
    eval_modality_epoch_masked,
    pretrain_modality_epoch,
    save_modality_with_config,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_curves(train_overall, train_masked, val_overall, val_masked, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_overall, label="Train overall MSE")
    plt.plot(train_masked,  label="Train masked MSE")
    plt.plot(val_overall,   label="Val overall MSE")
    plt.plot(val_masked,    label="Val masked MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Curve saved → {save_path}")


def plot_scatter(ae, val_loader, device, title, save_path):
    ae.eval()
    true_all, pred_all = [], []
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            _, recon = ae(xb)
            true_all.append(xb.cpu())
            pred_all.append(recon.cpu())
    true = torch.cat(true_all).flatten().numpy()
    pred = torch.cat(pred_all).flatten().numpy()
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    r = np.corrcoef(true, pred)[0, 1]
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=2, alpha=0.25)
    plt.plot(lims, lims, linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True"); plt.ylabel("Reconstructed")
    plt.title(f"{title} (r = {r:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Scatter saved → {save_path}  (r = {r:.3f})")


def train_one_modality(
    name, data_df, common_samples, train_idx, val_idx,
    hidden_layers, n_epochs, mask_value,
    device, out_dir, plot_dir,
    batch_size=128, lr=1e-3, weight_decay=1e-4,
    activation_dropout=0.05, mask_p=0.3,
    l1_alpha=1e-4, alpha_mask=0.5,
    use_batchnorm=True, grad_clip=1.0,
    patience=15,
):
    print(f"\n{'='*60}")
    print(f"  Training {name} autoencoder  |  epochs={n_epochs}  |  hidden={hidden_layers}")
    print(f"  batchnorm={use_batchnorm}  grad_clip={grad_clip}  patience={patience}")
    print(f"{'='*60}")

    ds = SingleModalityDatasetAligned(data_df, common_samples)
    train_loader = get_dataloader(ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
    val_loader   = get_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)

    input_dim = data_df.shape[1]

    config = {
        "input_dim":          input_dim,
        "hidden_layers":      hidden_layers,
        "activation_dropout": activation_dropout,
        "denoising":          True,
        "mask_p":             mask_p,
        "tied":               False,
        "mask_value":         mask_value,
        "loss_on_masked":     True,
        "use_batchnorm":      use_batchnorm,
    }

    ae, _ = build_pretrain_ae_for_modality(
        input_dim, hidden_layers,
        activation_dropout=activation_dropout,
        denoising=True, mask_p=mask_p, tied=False,
        mask_value=mask_value, loss_on_masked=True,
        use_batchnorm=use_batchnorm,
    )
    ae = ae.to(device)
    opt = AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    tr_overall_h, tr_masked_h, va_overall_h, va_masked_h = [], [], [], []

    best_val_masked = float("inf")
    best_state = None
    epochs_no_improve = 0

    for ep in range(1, n_epochs + 1):
        tr_loss, tr_overall, tr_masked = pretrain_modality_epoch(
            ae, train_loader, opt, device,
            l1_alpha=l1_alpha, alpha_mask=alpha_mask,
            grad_clip=grad_clip,
        )
        va_overall, va_masked = eval_modality_epoch_masked(ae, val_loader, device)
        scheduler.step()

        tr_overall_h.append(tr_overall)
        tr_masked_h.append(tr_masked)
        va_overall_h.append(va_overall)
        va_masked_h.append(va_masked)

        # Early stopping based on val masked MSE
        if va_masked < best_val_masked:
            best_val_masked = va_masked
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"  [{name}] ep {ep:03d} | loss {tr_loss:.4f} | "
            f"overall {tr_overall:.4f} | masked {tr_masked:.4f} | "
            f"val_overall {va_overall:.4f} | val_masked {va_masked:.4f}"
            f"{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"  [Early stop] No improvement for {patience} epochs.")
            break

    # Restore best weights before saving
    if best_state is not None:
        ae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model (val_masked={best_val_masked:.4f})")

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, f"{name}_ae")
    save_modality_with_config(ae, config, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}.pt")

    # Save plots
    plot_curves(
        tr_overall_h, tr_masked_h, va_overall_h, va_masked_h,
        title=f"{name} Autoencoder – loss curves",
        save_path=os.path.join(plot_dir, f"{name}_curves.png"),
    )
    plot_scatter(
        ae, val_loader, device,
        title=f"{name} AE (val) – Predicted vs True",
        save_path=os.path.join(plot_dir, f"{name}_scatter.png"),
    )

    return ae


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Train per-modality autoencoders")
    p.add_argument("--data",   default="data/tcga_redo_mlomicZ.pkl", help="Path to multi-omic pickle")
    p.add_argument("--splits", default="data/splits.json",           help="Path to splits JSON (optional)")
    p.add_argument("--out",    default="aes_redo_z",            help="Output directory for checkpoints")
    p.add_argument("--device", default=None,                    help="cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    # Output dirs
    plot_dir = os.path.join(args.out, "plots")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
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

    # ── Per-modality configs ─────────────────────────────────────────────────
    modality_configs = [
        dict(name="rna",  key="rna",         hidden_layers=[1024, 512], n_epochs=100, mask_value=0.0,
             use_batchnorm=True, grad_clip=1.0, patience=15),
        dict(name="mth",  key="methylation", hidden_layers=[512, 256],  n_epochs=100, mask_value=0.0,
             use_batchnorm=True, grad_clip=1.0, patience=15),
    ]

    for cfg in modality_configs:
        modality_key = cfg.pop("key")
        if modality_key not in multi_omic_data:
            print(f"\n[SKIP] '{modality_key}' not found in data keys: {list(multi_omic_data.keys())}")
            continue
        train_one_modality(
            data_df=multi_omic_data[modality_key],
            common_samples=common_samples,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            out_dir=args.out,
            plot_dir=plot_dir,
            **cfg,
        )

    print("\nDone. All checkpoints saved to", args.out)


if __name__ == "__main__":
    main()