"""
Decoder modules for MultiModal VAE
"""
import torch
import torch.nn as nn


class DecoderA(nn.Module):
    """Decoder for RNA expression data (modality A)"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


class DecoderB(nn.Module):
    """Decoder for DNA methylation data (modality B)"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.fc(z)


class DecoderC(nn.Module):
    """Decoder for primary site classification (modality C)"""
    def __init__(self, latent_dim, n_sites):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_sites)
        )

    def forward(self, z):
        return self.fc(z)