"""
Training script for DNA2RNAVAE.

Loads data from tcga_redo_mlomicZ.pkl / splits.json (same as train_shared.py)
so results are directly comparable on imputation benchmarks.

Usage:
    python train_dna2rna.py
    python train_dna2rna.py --data data/tcga_redo_mlomicZ.pkl --splits data/splits.json
    python train_dna2rna.py --epochs 200 --latent_dim 64
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
from src.models import DNA2RNAVAE
from src.data_utils import (
    MultiOmicDataset,
    load_shared_splits_from_json,
    compute_shared_splits,
)
from src.utils.directional_losses import dna2rna_loss


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


def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_total, running_recon = 0.0, 0.0
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START

    for batch in dataloader:
        rna  = batch["rna"].to(Config.DEVICE)
        meth = batch["methylation"].to(Config.DEVICE)

        nan_mask = torch.isnan(meth)
        meth_clean = meth.clone()
        meth_clean[nan_mask] = 0.0

        recon_rna, mu, logvar = model(dna=meth_clean, site=None)
        loss, recon, _ = dna2rna_loss(recon_rna, rna, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_total += loss.item()
        running_recon += recon

    n = len(dataloader)
    return running_total / n, running_recon / n, beta


def validate(model, dataloader, epoch):
    model.eval()
    running_total, running_recon = 0.0, 0.0
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START

    with torch.no_grad():
        for batch in dataloader:
            rna  = batch["rna"].to(Config.DEVICE)
            meth = batch["methylation"].to(Config.DEVICE)

            nan_mask = torch.isnan(meth)
            meth_clean = meth.clone()
            meth_clean[nan_mask] = 0.0

            recon_rna, mu, logvar = model(dna=meth_clean, site=None)
            loss, recon, _ = dna2rna_loss(recon_rna, rna, mu, logvar, beta=beta)
            running_total += loss.item()
            running_recon += recon

    n = len(dataloader)
    return running_total / n, running_recon / n


def plot_losses(train_losses, val_losses, run_id):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss for DNA2RNAVAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    filename = f'plots/training_losses_dna2rna_{run_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {filename}")


def parse_args():
    p = argparse.ArgumentParser(description="Train DNA2RNAVAE")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",     default="data/splits.json")
    p.add_argument("--epochs",     type=int,   default=Config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=Config.LEARNING_RATE)
    p.add_argument("--latent_dim", type=int,   default=Config.LATENT_DIM)
    return p.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting DNA2RNAVAE training run: {run_id}")

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

    print(f"Initializing DNA2RNAVAE on {Config.DEVICE} ...")
    model = DNA2RNAVAE(
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
        avg_train_loss, avg_train_recon, beta = train_epoch(model, train_loader, optimizer, epoch)
        avg_val_loss, avg_val_recon = validate(model, val_loader, epoch)
        scheduler.step(avg_val_recon)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recons.append(avg_train_recon)
        val_recons.append(avg_val_recon)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train recon: {avg_train_recon:.4f} | Val recon: {avg_val_recon:.4f} | "
            f"Val total: {avg_val_loss:.4f} | β={beta:.5f}"
        )

        if avg_val_recon < best_val_recon:
            best_val_recon = avg_val_recon
            trigger = 0
            model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_dna2rna_{run_id}.pt')
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

    with open('latest_dna2rna_run_id.txt', 'w') as f:
        f.write(run_id)

    print(f"\nDone. Best val recon: {best_val_recon:.4f}")
    print(f"Model: {os.path.join(Config.CHECKPOINT_DIR, f'best_dna2rna_{run_id}.pt')}")


if __name__ == "__main__":
    main()
