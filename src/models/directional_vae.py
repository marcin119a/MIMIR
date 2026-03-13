"""
Directional VAE models for imputation
"""
import torch
import torch.nn as nn

from .encoders import EncoderA, EncoderB, EncoderC
from .decoders import DecoderA, DecoderB
from .vae import reparameterize


class RNA2DNAVAE(nn.Module):
    """
    VAE that predicts DNA methylation from RNA expression and primary site
    
    Encodes: RNA + Primary Site
    Decodes: DNA methylation
    """
    def __init__(self, rna_dim, dna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.encoder_rna = EncoderA(rna_dim, latent_dim)
        self.encoder_site = EncoderC(n_sites, latent_dim, embed_dim=embed_dim)
        self.decoder_dna = DecoderB(latent_dim, dna_dim)

    def forward(self, rna=None, site=None):
        """
        Forward pass
        
        Args:
            rna: RNA expression data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_dna, mu, logvar)
        """
        mu_list = []
        logvar_list = []

        if rna is not None:
            mu_rna, logvar_rna = self.encoder_rna(rna)
            mu_list.append(mu_rna)
            logvar_list.append(logvar_rna)
        if site is not None:
            mu_site, logvar_site = self.encoder_site(site)
            mu_list.append(mu_site)
            logvar_list.append(logvar_site)

        # Aggregate latent representations
        if len(mu_list) == 0:
            return None, None, None
        elif len(mu_list) == 1:
            mu, logvar = mu_list[0], logvar_list[0]
        else:
            mu = torch.stack(mu_list).mean(0)
            logvar = torch.stack(logvar_list).mean(0)

        z = reparameterize(mu, logvar)
        recon_dna = self.decoder_dna(z)

        return recon_dna, mu, logvar


class DNA2RNAVAE(nn.Module):
    """
    VAE that predicts RNA expression from DNA methylation and primary site
    
    Encodes: DNA + Primary Site
    Decodes: RNA expression
    """
    def __init__(self, rna_dim, dna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.encoder_dna = EncoderB(dna_dim, latent_dim)
        self.encoder_site = EncoderC(n_sites, latent_dim, embed_dim=embed_dim)
        self.decoder_rna = DecoderA(latent_dim, rna_dim)

    def forward(self, dna=None, site=None):
        """
        Forward pass
        
        Args:
            dna: DNA methylation data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_rna, mu, logvar)
        """
        mu_list = []
        logvar_list = []

        if dna is not None:
            mu_dna, logvar_dna = self.encoder_dna(dna)
            mu_list.append(mu_dna)
            logvar_list.append(logvar_dna)
        if site is not None:
            mu_site, logvar_site = self.encoder_site(site)
            mu_list.append(mu_site)
            logvar_list.append(logvar_site)

        # Aggregate latent representations
        if len(mu_list) == 0:
            return None, None, None
        elif len(mu_list) == 1:
            mu, logvar = mu_list[0], logvar_list[0]
        else:
            mu = torch.stack(mu_list).mean(0)
            logvar = torch.stack(logvar_list).mean(0)

        z = reparameterize(mu, logvar)
        recon_rna = self.decoder_rna(z)

        return recon_rna, mu, logvar