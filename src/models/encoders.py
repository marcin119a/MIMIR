"""
Encoder modules for MultiModal VAE
"""
import torch
import torch.nn as nn


class EncoderA(nn.Module):
    """Encoder for RNA expression data (modality A)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


class EncoderB(nn.Module):
    """Encoder for DNA methylation data (modality B)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


class EncoderC(nn.Module):
    """Encoder for primary site labels (modality C)"""
    def __init__(self, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_sites, embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        h = self.embedding(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar