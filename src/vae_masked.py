"""
VAE (Variational Autoencoder) versions of the Phase-1 modality-specific autoencoders.

Drop-in replacements for the relevant parts of mae_masked.py:
  - ModalityVAE                      <- ModalityAutoencoder
  - pretrain_vae_epoch               <- pretrain_modality_epoch
  - eval_vae_epoch_masked            <- eval_modality_epoch_masked
  - build_pretrain_vae_for_modality  <- build_pretrain_ae_for_modality
  - save_vae_with_config             <- save_modality_with_config
  - load_vae_with_config             <- load_modality_with_config
  - extract_encoder_decoder_from_vae <- extract_encoder_decoder_from_pretrained

Architecture (per modality):
    backbone  : input_dim → hidden_layers[:-1]   (empty if len==1)
    mu_head   : intermediate_dim → latent_dim
    logvar_head: intermediate_dim → latent_dim
    decoder   : latent_dim → reversed(hidden_layers[:-1]) → input_dim

latent_dim = hidden_layers[-1]  (same as AE hidden_dim for Phase-2 compatibility).
At eval time z = mu (deterministic, no sampling).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from .mae_masked import build_mlp, ModalityDecoder


# ─── Core VAE Module ──────────────────────────────────────────────────────────

class ModalityVAE(nn.Module):
    """
    Denoising Variational Autoencoder for a single omics modality.

    forward(x) -> (mu, x_recon)
      - mu is the encoder mean (used as "h" for L1 / Phase-2 compatibility)
      - x_recon is reconstructed from z = reparameterise(mu, logvar) (train)
                                    or z = mu                         (eval)
    KL divergence is stored in self._last_kl after each forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation_dropout: float = 0.0,
        denoising: bool = False,
        mask_p: float = 0.0,
        mask_value: float = 0.0,
        loss_on_masked: bool = True,
        beta: float = 1.0,
    ):
        super().__init__()
        assert len(hidden_layers) >= 1, "hidden_layers must have at least one element"

        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.latent_dim = hidden_layers[-1]
        self.denoising = denoising
        self.mask_p = mask_p
        self.mask_value = mask_value
        self.loss_on_masked = loss_on_masked
        self.beta = beta

        # Internal bookkeeping (set during forward)
        self._last_mask: torch.Tensor = None
        self._last_mu: torch.Tensor = None
        self._last_logvar: torch.Tensor = None
        self._last_kl: torch.Tensor = None

        # Encoder backbone: input → hidden_layers[:-1] (with ReLU activations)
        if len(hidden_layers) > 1:
            backbone_dims = [input_dim] + hidden_layers[:-1]
            self.backbone = build_mlp(
                backbone_dims,
                add_final_activation=True,
                activation_dropout=activation_dropout,
            )
            intermediate_dim = hidden_layers[-2]
        else:
            self.backbone = nn.Identity()
            intermediate_dim = input_dim

        # Latent heads
        self.mu_head = nn.Linear(intermediate_dim, self.latent_dim)
        self.logvar_head = nn.Linear(intermediate_dim, self.latent_dim)

        # Decoder: latent_dim → reversed backbone dims → input_dim
        dec_dims = [self.latent_dim] + list(reversed(hidden_layers[:-1])) + [input_dim]
        self.decoder = ModalityDecoder(dec_dims, activation_dropout=activation_dropout)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_mask_noise(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.denoising or self.mask_p <= 0.0:
            self._last_mask = torch.zeros_like(x, dtype=torch.bool)
            return x
        mask = torch.rand_like(x) < self.mask_p
        x_noisy = x.clone()
        x_noisy[mask] = self.mask_value
        self._last_mask = mask
        return x_noisy

    @staticmethod
    def _reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_in = self._add_mask_noise(x)

        h = self.backbone(x_in)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        self._last_mu = mu
        self._last_logvar = logvar
        # KL( q || p ) = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        self._last_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

        z = self._reparameterise(mu, logvar) if self.training else mu
        x_recon = self.decoder(z)
        return mu, x_recon


# ─── Training / Eval Loops ────────────────────────────────────────────────────

def pretrain_vae_epoch(
    vae: ModalityVAE,
    dataloader: DataLoader,
    optimizer,
    device,
    l1_alpha: float = 0.0,
    alpha_mask: float = 1.0,
    beta: float = None,
) -> Tuple[float, float, float]:
    """
    One training epoch for ModalityVAE.

    Loss = alpha_mask * masked_mse + (1 - alpha_mask) * overall_mse + beta * KL

    Returns (avg_total_loss, avg_overall_mse, avg_masked_mse).
    """
    if beta is None:
        beta = vae.beta

    vae.train()
    total_loss = total_overall = total_masked = 0.0
    n = 0

    for xb in dataloader:
        xb = xb.to(device)

        orig_missing = torch.isnan(xb)
        xb_in = xb.clone()
        xb_in[orig_missing] = vae.mask_value

        optimizer.zero_grad()
        mu, recon = vae(xb_in)

        diff_sq = (recon - xb_in) ** 2
        valid = ~orig_missing

        overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()

        if vae.denoising and vae.loss_on_masked:
            mask_art = vae._last_mask.to(device)
            combined = mask_art & valid
            masked_mse = diff_sq[combined].mean() if combined.any() else overall_mse
        else:
            masked_mse = overall_mse

        recon_loss = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse
        loss = recon_loss + beta * vae._last_kl

        if l1_alpha > 0:
            loss = loss + l1_alpha * mu.abs().mean()

        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_overall += overall_mse.item()
        total_masked  += masked_mse.item()
        n += 1

    avg = max(n, 1)
    return total_loss / avg, total_overall / avg, total_masked / avg


def eval_vae_epoch_masked(
    vae: ModalityVAE,
    dataloader: DataLoader,
    device,
) -> Tuple[float, float]:
    """
    Eval with artificial masking enabled (mirrors eval_modality_epoch_masked).
    Uses mu (deterministic) for reconstruction — no sampling noise in metrics.
    Returns (overall_mse, masked_mse).
    """
    was_training = vae.training
    vae.eval()  # deterministic mu, but we still want masking…
    # Re-enable masking by temporarily switching training flag only for _add_mask_noise
    # We do this by eval-ing the vae but calling a separate masking routine.
    total_overall = total_masked = 0.0
    n = 0

    with torch.no_grad():
        for xb in dataloader:
            xb = xb.to(device)
            orig_missing = torch.isnan(xb)
            xb_in = xb.clone()
            xb_in[orig_missing] = vae.mask_value

            # Apply artificial masking manually (same logic as _add_mask_noise in train mode)
            if vae.denoising and vae.mask_p > 0.0:
                art_mask = torch.rand_like(xb_in) < vae.mask_p
                xb_noisy = xb_in.clone()
                xb_noisy[art_mask] = vae.mask_value
            else:
                xb_noisy = xb_in
                art_mask = torch.zeros_like(xb_in, dtype=torch.bool)

            # Forward in eval mode: z = mu, no sampling
            h = vae.backbone(xb_noisy)
            mu = vae.mu_head(h)
            recon = vae.decoder(mu)

            diff_sq = (recon - xb_in) ** 2
            valid = ~orig_missing

            overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()

            combined = art_mask & valid
            masked_mse = diff_sq[combined].mean() if combined.any() else overall_mse

            total_overall += overall_mse.item()
            total_masked  += masked_mse.item()
            n += 1

    if was_training:
        vae.train()

    avg = max(n, 1)
    return total_overall / avg, total_masked / avg


# ─── Deterministic Encoder Wrapper (for Phase 2) ──────────────────────────────

class VAEDeterministicEncoder(nn.Module):
    """
    Wraps a trained VAE's backbone + mu_head.
    Returns mu (deterministic) — compatible with ModalityEncoder interface.
    """
    def __init__(self, backbone: nn.Module, mu_head: nn.Linear):
        super().__init__()
        self.backbone = backbone
        self.mu_head = mu_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu_head(self.backbone(x))


def extract_encoder_decoder_from_vae(vae: ModalityVAE):
    """
    Return (deterministic_encoder, decoder) from a trained VAE.
    Drop-in for extract_encoder_decoder_from_pretrained.
    """
    enc = VAEDeterministicEncoder(vae.backbone, vae.mu_head)
    dec = vae.decoder
    return enc, dec


# ─── Build / Save / Load ──────────────────────────────────────────────────────

def build_pretrain_vae_for_modality(
    input_dim: int,
    hidden_layers: List[int],
    activation_dropout: float = 0.0,
    denoising: bool = False,
    mask_p: float = 0.0,
    mask_value: float = 0.0,
    loss_on_masked: bool = True,
    beta: float = 1.0,
) -> Tuple[ModalityVAE, int]:
    """Returns (vae, latent_dim). Drop-in for build_pretrain_ae_for_modality."""
    vae = ModalityVAE(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation_dropout=activation_dropout,
        denoising=denoising,
        mask_p=mask_p,
        mask_value=mask_value,
        loss_on_masked=loss_on_masked,
        beta=beta,
    )
    return vae, hidden_layers[-1]


def save_vae_with_config(vae: ModalityVAE, config: dict, path_prefix: str):
    torch.save(
        {'state_dict': vae.state_dict(), 'config': config, 'model_type': 'vae'},
        f"{path_prefix}.pt",
    )


def load_vae_with_config(
    path: str, map_location=None
) -> Tuple[ModalityVAE, int, dict]:
    data = torch.load(path, map_location=map_location)
    config = data['config']
    vae, hidden_dim = build_pretrain_vae_for_modality(
        config['input_dim'],
        config['hidden_layers'],
        activation_dropout=config.get('activation_dropout', 0.0),
        denoising=config.get('denoising', False),
        mask_p=config.get('mask_p', 0.0),
        mask_value=config.get('mask_value', 0.0),
        loss_on_masked=config.get('loss_on_masked', True),
        beta=config.get('beta', 1.0),
    )
    vae.load_state_dict(data['state_dict'])
    return vae, hidden_dim, config
