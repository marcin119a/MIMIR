"""
Multi-Modal VAE model
"""
import torch
import torch.nn as nn

from .encoders import EncoderA, EncoderB, EncoderC
from .decoders import DecoderA, DecoderB, DecoderC


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class MultiModalVAE(nn.Module):
    """
    Multi-Modal Variational Autoencoder
    
    Supports three modalities:
    - A: RNA expression data
    - B: DNA methylation data
    - C: Primary site labels
    """
    def __init__(self, input_dim_a, input_dim_b, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.encoder_a = EncoderA(input_dim_a, latent_dim)
        self.encoder_b = EncoderB(input_dim_b, latent_dim)
        self.encoder_c = EncoderC(n_sites, latent_dim, embed_dim=embed_dim)

        self.decoder_a = DecoderA(latent_dim, input_dim_a)
        self.decoder_b = DecoderB(latent_dim, input_dim_b)
        self.decoder_c = DecoderC(latent_dim, n_sites)

    def forward(self, a=None, b=None, site=None):
        """
        Forward pass through the VAE
        
        Args:
            a: RNA expression data (optional)
            b: DNA methylation data (optional)
            site: Primary site labels (optional)
            
        Returns:
            Tuple of (out_a, out_b, out_c, mu, logvar)
        """
        mu_list, logvar_list = [], []

        if a is not None:
            mu_a, logvar_a = self.encoder_a(a)
            mu_list.append(mu_a)
            logvar_list.append(logvar_a)
        if b is not None:
            mu_b, logvar_b = self.encoder_b(b)
            mu_list.append(mu_b)
            logvar_list.append(logvar_b)
        if site is not None:
            mu_c, logvar_c = self.encoder_c(site)
            mu_list.append(mu_c)
            logvar_list.append(logvar_c)

        # Aggregate latent representations
        if len(mu_list) == 0:
            return None, None, None, None, None
        elif len(mu_list) == 1:
            mu, logvar = mu_list[0], logvar_list[0]
        else:
            mu = torch.stack(mu_list).mean(0)
            logvar = torch.stack(logvar_list).mean(0)

        z = reparameterize(mu, logvar)

        out_a = self.decoder_a(z)
        out_b = self.decoder_b(z)
        out_c = self.decoder_c(z)

        return out_a, out_b, out_c, mu, logvar