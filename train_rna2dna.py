"""
Training script for DNA2RNAVAE
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import DNA2RNAVAE
from src.data import MultiModalDataset
from src.utils.directional_losses import dna2rna_loss


def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def load_data():
    """Load processed data"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Data shape: {merged_df.shape}")
    print(f"Number of primary sites: {len(label_encoder.classes_)}")
    
    return merged_df, label_encoder


def prepare_dataloaders(merged_df):
    """Split data and create dataloaders"""
    print("\nSplitting data into train/validation sets...")
    train_df, val_df = train_test_split(
        merged_df, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Create datasets
    train_dataset = MultiModalDataset(train_df)
    val_dataset = MultiModalDataset(val_df)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_dataloader, val_dataloader


def train_epoch(model, dataloader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_train_loss = 0.0
    recon_component = 0.0
    kl_component = 0.0
    
    # Beta warmup: gradually increase KL weight
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START
    
    for tpm, beta_data, site in dataloader:
        tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
        
        # Forward pass: predict RNA from DNA + site
        recon_rna, mu, logvar = model(dna=beta_data, site=site)
        
        # Compute loss
        loss, recon_loss, kld_loss = dna2rna_loss(
            recon_rna, tpm, mu, logvar, beta=beta
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_train_loss += loss.item()
        recon_component += recon_loss
        kl_component += kld_loss
    
    avg_train_loss = running_train_loss / len(dataloader)
    
    return avg_train_loss, beta


def validate(model, dataloader, epoch):
    """Validate the model"""
    model.eval()
    running_val_loss = 0.0
    
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            
            # Forward pass
            recon_rna, mu, logvar = model(dna=beta_data, site=site)
            
            # Compute loss
            loss, _, _ = dna2rna_loss(
                recon_rna, tpm, mu, logvar, beta=beta
            )
            
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(dataloader)
    
    return avg_val_loss


def plot_losses(train_losses, val_losses, run_id):
    """Plot training and validation losses"""
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


def main():
    """Main training loop"""
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting DNA2RNAVAE training run: {run_id}")
    
    # Setup
    setup_directories()
    
    # Load data
    merged_df, label_encoder = load_data()
    n_sites = len(label_encoder.classes_)
    
    # Prepare dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(merged_df)
    
    # Initialize model (allow env overrides, e.g. on Kaggle)
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None and "DEVICE" not in os.environ:
        Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif "DEVICE" in os.environ:
        Config.DEVICE = torch.device(os.environ["DEVICE"])

    Config.INPUT_DIM_A = int(os.getenv("INPUT_DIM_A", Config.INPUT_DIM_A))
    Config.INPUT_DIM_B = int(os.getenv("INPUT_DIM_B", Config.INPUT_DIM_B))
    Config.LATENT_DIM = int(os.getenv("LATENT_DIM", Config.LATENT_DIM))

    print(f"\nInitializing DNA2RNAVAE model on {Config.DEVICE}...")
    model = DNA2RNAVAE(
        Config.INPUT_DIM_A,  # RNA dimension
        Config.INPUT_DIM_B,   # DNA dimension
        n_sites, 
        Config.LATENT_DIM
    ).to(Config.DEVICE)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.LR_SCHEDULER_FACTOR, 
        patience=Config.LR_SCHEDULER_PATIENCE
    )
    
    # Training loop
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {Config.PATIENCE}")
    
    best_val_loss = np.inf
    trigger = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        avg_train_loss, beta = train_epoch(model, train_dataloader, optimizer, epoch)
        train_losses.append(avg_train_loss)
        
        # Validate
        avg_val_loss = validate(model, val_dataloader, epoch)
        val_losses.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.2f} | "
              f"Val Loss: {avg_val_loss:.2f} | "
              f"β={beta:.5f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger = 0
            
            # Save best model
            model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_dna2rna_{run_id}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best model saved (val_loss: {avg_val_loss:.2f})")
        else:
            trigger += 1
            if trigger >= Config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
    
    # Plot losses
    print("\nGenerating loss plots...")
    plot_losses(train_losses, val_losses, run_id)
    
    # Save final model path for evaluation
    with open('latest_dna2rna_run_id.txt', 'w') as f:
        f.write(run_id)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Best model saved to: {os.path.join(Config.CHECKPOINT_DIR, f'best_dna2rna_{run_id}.pt')}")
    print("="*50)


if __name__ == "__main__":
    main()