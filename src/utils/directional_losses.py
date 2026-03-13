"""
Loss functions for directional VAE models
"""
import torch
import torch.nn.functional as F


def rna2dna_loss(recon_dna, dna, mu, logvar, beta=1e-3):
    """
    Compute loss for RNA2DNAVAE
    
    Args:
        recon_dna: Reconstructed DNA methylation data
        dna: Original DNA methylation data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (MSE for Z-score normalised methylation)
    recon_loss = F.mse_loss(recon_dna, dna, reduction='mean')

    # KL Divergence: mean over batch, sum over latent dim
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss.item(), kld.item()


def dna2rna_loss(recon_rna, rna, mu, logvar, beta=1e-3):
    """
    Compute loss for DNA2RNAVAE
    
    Args:
        recon_rna: Reconstructed RNA expression data
        rna: Original RNA expression data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (MSE for RNA expression)
    recon_loss = F.mse_loss(recon_rna, rna, reduction='mean')

    # KL Divergence: mean over batch, sum over latent dim
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss.item(), kld.item()