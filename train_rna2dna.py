"""
Training script for RNA2DNAVAE.

Loads data from tcga_redo_mlomicZ.pkl / splits.json (same as train_shared.py)
so results are directly comparable on imputation benchmarks.

Usage:
    python train_rna2dna.py
    python train_rna2dna.py --data data/tcga_redo_mlomicZ.pkl --splits data/splits.json
    python train_rna2dna.py --epochs 200 --latent_dim 64
"""
import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import RNA2DNAVAE
from src.data_utils import (
    MultiOmicDataset,
    load_shared_splits_from_json,
    compute_shared_splits,
)
from src.utils.directional_losses import rna2dna_loss


def setup_directories():
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def load_data_and_splits(data_path, splits_path):
    print(f"Loading data from {data_path} ...")
    with open(data_path, "rb") as f:
        multi_omic_data = pickle.load(f)

    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ["rna", "methylation"]}
    print(f"Active modalities: {list(multi_omic_data.keys())}")

    if os.path.exists(splits_path):
        print(f"Loading splits from {splits_path} ...")
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, splits_path
        )
    else:
        print("splits.json not found – computing splits (70/10/20) ...")
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )

    print(
        f"Samples: total={len(common_samples)} | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )
    return multi_omic_data, common_samples, train_idx, val_idx, test_idx


def prepare_dataloaders(multi_omic_data, common_samples, train_idx, val_idx, batch_size):
    dataset = MultiOmicDataset({m: df.loc[common_samples] for m, df in multi_omic_data.items()})
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _compute_loss(recon_dna, meth, mu, logvar, beta):
    """MSE loss excluding true-NaN positions in the methylation target.
    Returns (total_loss, recon_loss) where recon_loss is used for model selection."""
    nan_mask = torch.isnan(meth)
    meth_clean = meth.clone()
    meth_clean[nan_mask] = 0.0

    diff_sq = (recon_dna - meth_clean) ** 2
    valid = ~nan_mask
    recon_loss = diff_sq[valid].mean() if valid.any() else diff_sq.mean()

    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kld, recon_loss, kld


def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_total, running_recon, running_kld = 0.0, 0.0, 0.0
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * (Config.BETA_START * 0.1)

    for batch in dataloader:
        rna  = batch["rna"].to(Config.DEVICE)
        meth = batch["methylation"].to(Config.DEVICE)

        recon_dna, mu, logvar = model(rna=rna, site=None)
        loss, recon, kld = _compute_loss(recon_dna, meth, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_total += loss.item()
        running_recon += recon.item()
        running_kld   += kld.item()

    n = len(dataloader)
    return running_total / n, running_recon / n, running_kld / n, beta


def validate(model, dataloader, epoch):
    model.eval()
    running_total, running_recon, running_kld = 0.0, 0.0, 0.0
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * (Config.BETA_START * 0.1)

    with torch.no_grad():
        for batch in dataloader:
            rna  = batch["rna"].to(Config.DEVICE)
            meth = batch["methylation"].to(Config.DEVICE)

            recon_dna, mu, logvar = model(rna=rna, site=None)
            loss, recon, kld = _compute_loss(recon_dna, meth, mu, logvar, beta)
            running_total += loss.item()
            running_recon += recon.item()
            running_kld   += kld.item()

    n = len(dataloader)
    return running_total / n, running_recon / n, running_kld / n


def plot_losses(train_losses, val_losses, run_id):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss for RNA2DNAVAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    filename = f'plots/training_losses_rna2dna_{run_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {filename}")


def parse_args():
    p = argparse.ArgumentParser(description="Train RNA2DNAVAE")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",     default="data/splits.json")
    p.add_argument("--epochs",     type=int,   default=Config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int,   default=Config.LATENT_DIM)
    return p.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting RNA2DNAVAE training run: {run_id}")

    setup_directories()

    multi_omic_data, common_samples, train_idx, val_idx, _ = load_data_and_splits(
        args.data, args.splits
    )

    train_loader, val_loader = prepare_dataloaders(
        multi_omic_data, common_samples, train_idx, val_idx, args.batch_size
    )

    rna_dim  = multi_omic_data["rna"].shape[1]
    meth_dim = multi_omic_data["methylation"].shape[1]
    print(f"RNA dim: {rna_dim}, Methylation dim: {meth_dim}")

    print(f"Initializing RNA2DNAVAE on {Config.DEVICE} ...")
    model = RNA2DNAVAE(
        rna_dim,
        meth_dim,
        n_sites=1,          # site conditioning not used
        latent_dim=args.latent_dim,
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.LR_SCHEDULER_FACTOR, patience=Config.LR_SCHEDULER_PATIENCE
    )

    best_val_recon = np.inf
    trigger = 0
    train_losses, val_losses, train_recons, val_recons = [], [], [], []

    print(f"Starting training for {args.epochs} epochs ...")
    for epoch in range(args.epochs):
        avg_train_loss, avg_train_recon, avg_train_kld, beta = train_epoch(model, train_loader, optimizer, epoch)
        avg_val_loss, avg_val_recon, avg_val_kld = validate(model, val_loader, epoch)
        scheduler.step(avg_val_recon)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recons.append(avg_train_recon)
        val_recons.append(avg_val_recon)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train recon: {avg_train_recon:.4f} | Train KLD: {avg_train_kld:.4f} | "
            f"Val recon: {avg_val_recon:.4f} | Val KLD: {avg_val_kld:.4f} | "
            f"Val total: {avg_val_loss:.4f} | β={beta:.5f}"
        )

        if avg_val_recon < best_val_recon:
            best_val_recon = avg_val_recon
            trigger = 0
            model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_rna2dna_{run_id}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  Best model saved (val_recon: {avg_val_recon:.4f})")
        else:
            if epoch >= Config.BETA_WARMUP_EPOCHS:
                trigger += 1
                if trigger >= Config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                trigger = 0  # don't penalise during β warmup

    plot_losses(train_recons, val_recons, run_id)

    with open('latest_rna2dna_run_id.txt', 'w') as f:
        f.write(run_id)

    print(f"\nDone. Best val recon: {best_val_recon:.4f}")
    print(f"Model: {os.path.join(Config.CHECKPOINT_DIR, f'best_rna2dna_{run_id}.pt')}")


if __name__ == "__main__":
    main()
