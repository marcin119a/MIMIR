"""
Conditional VAE (CVAE) for Phase 1.
Primary site is used as condition c (one-hot encoded).

Phase 1: ModalityCVAE — per-modality denoising CVAE conditioned on primary site.
Condition c is concatenated to encoder input and to latent z before decoding.

Compatible with Phase 2 via extract_encoder_decoder_from_cvae which returns
CVAEConditionedEncoder and CVAEConditionedDecoder wrappers.
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .mae_masked import ModalityDecoder, build_mlp


# ─── Condition helpers ────────────────────────────────────────────────────────

def load_conditions_from_json(
    primary_sites_json: str,
    sample_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Load primary site labels and return one-hot encoded conditions.

    Args:
        primary_sites_json: path to JSON {case_id (first 12 chars): primary_site_str}
        sample_ids: list of sample barcodes

    Returns:
        condition_matrix: np.ndarray [N, num_classes] float32 (one-hot)
        class_names: sorted list of primary site names
    """
    with open(primary_sites_json, "r") as f:
        ps_dict = json.load(f)

    labels = []
    for sid in sample_ids:
        case_id = sid[:12]
        site = ps_dict.get(case_id)
        labels.append(site if site else "Unknown")

    unique_classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    indices = np.array([class_to_idx[l] for l in labels], dtype=np.int64)
    n = len(sample_ids)
    num_classes = len(unique_classes)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), indices] = 1.0

    return one_hot, unique_classes


# ─── Datasets ─────────────────────────────────────────────────────────────────

class ConditionalSingleModalityDataset(Dataset):
    """
    Single-modality dataset returning (x, c) pairs.
    condition_matrix rows are aligned with common_samples order.
    """
    def __init__(self, df: pd.DataFrame, common_samples: List[str],
                 condition_matrix: np.ndarray):
        X = df.loc[common_samples].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.C = torch.tensor(condition_matrix, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx]


class ConditionalMultiOmicDataset(Dataset):
    """
    Multi-modal dataset returning ({mod: tensor}, c) per sample.
    condition_matrix rows are aligned with common_samples order.
    """
    def __init__(self, data_dict: Dict[str, pd.DataFrame],
                 condition_matrix: np.ndarray):
        self.modalities = list(data_dict.keys())
        sample_sets = [set(df.index) for df in data_dict.values()]
        self.common_samples = sorted(set.intersection(*sample_sets))
        self.data = {
            mod: torch.tensor(df.loc[self.common_samples].values, dtype=torch.float32)
            for mod, df in data_dict.items()
        }
        self.conditions = torch.tensor(condition_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.common_samples)

    def __getitem__(self, idx):
        x = {mod: self.data[mod][idx] for mod in self.modalities}
        return x, self.conditions[idx]


def get_conditional_dataloader(dataset: Dataset, batch_size: int = 64,
                                shuffle: bool = True, split_idx=None) -> DataLoader:
    ds = dataset if split_idx is None else Subset(dataset, split_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─── Phase 1 CVAE ─────────────────────────────────────────────────────────────

class ModalityCVAE(nn.Module):
    """
    Per-modality Conditional Variational Autoencoder.

    Condition c (one-hot, shape [B, num_classes]) is concatenated to:
      - Encoder input:  [x; c]  → backbone → mu_head, logvar_head
      - Decoder input:  [z; c]  → x_recon

    forward(x, c) → (mu, x_recon)
    At eval time z = mu (deterministic).
    KL divergence is stored in self._last_kl after each forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
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
        self.num_classes = num_classes
        self.latent_dim = hidden_layers[-1]
        self.denoising = denoising
        self.mask_p = mask_p
        self.mask_value = mask_value
        self.loss_on_masked = loss_on_masked
        self.beta = beta

        self._last_mask: Optional[torch.Tensor] = None
        self._last_kl: Optional[torch.Tensor] = None

        # Encoder backbone: [x; c] → hidden_layers[:-1]
        enc_in = input_dim + num_classes
        if len(hidden_layers) > 1:
            backbone_dims = [enc_in] + hidden_layers[:-1]
            self.backbone = build_mlp(backbone_dims, add_final_activation=True,
                                      activation_dropout=activation_dropout)
            intermediate_dim = hidden_layers[-2]
        else:
            self.backbone = nn.Identity()
            intermediate_dim = enc_in

        self.mu_head = nn.Linear(intermediate_dim, self.latent_dim)
        self.logvar_head = nn.Linear(intermediate_dim, self.latent_dim)

        # Decoder: [z; c] → input_dim
        dec_in = self.latent_dim + num_classes
        dec_dims = [dec_in] + list(reversed(hidden_layers[:-1])) + [input_dim]
        self.decoder = ModalityDecoder(dec_dims, activation_dropout=activation_dropout)

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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_in = self._add_mask_noise(x)
        xc = torch.cat([x_in, c], dim=-1)
        h = self.backbone(xc)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        self._last_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        z = self._reparameterise(mu, logvar) if self.training else mu
        zc = torch.cat([z, c], dim=-1)
        x_recon = self.decoder(zc)
        return mu, x_recon


# ─── Training / Eval Loops (Phase 1) ─────────────────────────────────────────

def pretrain_cvae_epoch(
    cvae: ModalityCVAE,
    dataloader: DataLoader,
    optimizer,
    device,
    l1_alpha: float = 0.0,
    alpha_mask: float = 1.0,
    beta: float = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float, float]:
    """
    One training epoch for ModalityCVAE.
    Loss = alpha_mask * masked_mse + (1 - alpha_mask) * overall_mse + beta * KL
    Dataloader must yield (x, c) batches (ConditionalSingleModalityDataset).
    Returns (avg_total_loss, avg_overall_mse, avg_masked_mse).
    """
    if beta is None:
        beta = cvae.beta

    cvae.train()
    total_loss = total_overall = total_masked = 0.0
    n = 0

    for xb, cb in dataloader:
        xb, cb = xb.to(device), cb.to(device)

        orig_missing = torch.isnan(xb)
        xb_in = xb.clone()
        xb_in[orig_missing] = cvae.mask_value

        optimizer.zero_grad()
        mu, recon = cvae(xb_in, cb)

        diff_sq = (recon - xb_in) ** 2
        valid = ~orig_missing
        overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()

        if cvae.denoising and cvae.loss_on_masked:
            mask_art = cvae._last_mask.to(device)
            combined = mask_art & valid
            masked_mse = diff_sq[combined].mean() if combined.any() else overall_mse
        else:
            masked_mse = overall_mse

        recon_loss = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse
        loss = recon_loss + beta * cvae._last_kl

        if l1_alpha > 0:
            loss = loss + l1_alpha * mu.abs().mean()

        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), grad_clip)
        optimizer.step()

        total_loss    += loss.item()
        total_overall += overall_mse.item()
        total_masked  += masked_mse.item()
        n += 1

    avg = max(n, 1)
    return total_loss / avg, total_overall / avg, total_masked / avg


def eval_cvae_epoch_masked(
    cvae: ModalityCVAE,
    dataloader: DataLoader,
    device,
) -> Tuple[float, float]:
    """
    Eval with deterministic z = mu and artificial masking.
    Returns (overall_mse, masked_mse).
    Dataloader must yield (x, c) batches.
    """
    was_training = cvae.training
    cvae.eval()
    total_overall = total_masked = 0.0
    n = 0

    with torch.no_grad():
        for xb, cb in dataloader:
            xb, cb = xb.to(device), cb.to(device)
            orig_missing = torch.isnan(xb)
            xb_in = xb.clone()
            xb_in[orig_missing] = cvae.mask_value

            if cvae.denoising and cvae.mask_p > 0.0:
                art_mask = torch.rand_like(xb_in) < cvae.mask_p
                xb_noisy = xb_in.clone()
                xb_noisy[art_mask] = cvae.mask_value
            else:
                xb_noisy = xb_in
                art_mask = torch.zeros_like(xb_in, dtype=torch.bool)

            # Deterministic forward: z = mu
            xc = torch.cat([xb_noisy, cb], dim=-1)
            h = cvae.backbone(xc)
            mu = cvae.mu_head(h)
            zc = torch.cat([mu, cb], dim=-1)
            recon = cvae.decoder(zc)

            diff_sq = (recon - xb_in) ** 2
            valid = ~orig_missing
            overall_mse = diff_sq[valid].mean() if valid.any() else diff_sq.mean()

            combined = art_mask & valid
            masked_mse = diff_sq[combined].mean() if combined.any() else overall_mse

            total_overall += overall_mse.item()
            total_masked  += masked_mse.item()
            n += 1

    if was_training:
        cvae.train()

    avg = max(n, 1)
    return total_overall / avg, total_masked / avg


# ─── Wrappers for Phase 2 compatibility ──────────────────────────────────────

class CVAEConditionedEncoder(nn.Module):
    """
    Wraps trained CVAE backbone + mu_head (+ optional logvar_head).
    forward(x, c=None) → z
      - training + logvar_head present: reparameterised sample
      - otherwise: mu (deterministic)
    encode_params(x, c=None) → (mu, logvar | None)  — for KL computation
    If c is None, a zero-condition vector is used as fallback.
    """
    def __init__(self, backbone: nn.Module, mu_head: nn.Linear, num_classes: int,
                 logvar_head: Optional[nn.Linear] = None):
        super().__init__()
        self.backbone = backbone
        self.mu_head = mu_head
        self.logvar_head = logvar_head
        self.num_classes = num_classes

    def _backbone_out(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        if c is None:
            c = torch.zeros(x.size(0), self.num_classes, device=x.device, dtype=x.dtype)
        return self.backbone(torch.cat([x, c], dim=-1))

    def encode_params(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (mu, logvar). logvar is None when logvar_head is not available."""
        h = self._backbone_out(x, c)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h) if self.logvar_head is not None else None
        return mu, logvar

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu, logvar = self.encode_params(x, c)
        if self.training and logvar is not None:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu


class CVAEConditionedDecoder(nn.Module):
    """
    Wraps trained CVAE decoder.
    forward(z, c=None) → x_recon
    If c is None, a zero-condition vector is used as fallback.
    """
    def __init__(self, decoder: nn.Module, num_classes: int):
        super().__init__()
        self.decoder = decoder
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is None:
            c = torch.zeros(z.size(0), self.num_classes, device=z.device, dtype=z.dtype)
        zc = torch.cat([z, c], dim=-1)
        return self.decoder(zc)


def extract_encoder_decoder_from_cvae(
    cvae: ModalityCVAE,
) -> Tuple[CVAEConditionedEncoder, CVAEConditionedDecoder]:
    """
    Extract (encoder, decoder) wrappers from a trained ModalityCVAE.
    Drop-in for use in ConditionalMultiModalWithSharedSpace.
    """
    enc = CVAEConditionedEncoder(cvae.backbone, cvae.mu_head, cvae.num_classes,
                                  logvar_head=cvae.logvar_head)
    dec = CVAEConditionedDecoder(cvae.decoder, cvae.num_classes)
    return enc, dec


# ─── Build / Save / Load ─────────────────────────────────────────────────────

def build_pretrain_cvae_for_modality(
    input_dim: int,
    num_classes: int,
    hidden_layers: List[int],
    activation_dropout: float = 0.0,
    denoising: bool = False,
    mask_p: float = 0.0,
    mask_value: float = 0.0,
    loss_on_masked: bool = True,
    beta: float = 1.0,
) -> Tuple[ModalityCVAE, int]:
    """Returns (cvae, latent_dim)."""
    cvae = ModalityCVAE(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        activation_dropout=activation_dropout,
        denoising=denoising,
        mask_p=mask_p,
        mask_value=mask_value,
        loss_on_masked=loss_on_masked,
        beta=beta,
    )
    return cvae, hidden_layers[-1]


def save_cvae_with_config(cvae: ModalityCVAE, config: dict, path_prefix: str):
    torch.save(
        {'state_dict': cvae.state_dict(), 'config': config, 'model_type': 'cvae'},
        f"{path_prefix}.pt",
    )


def load_cvae_with_config(path: str, map_location=None) -> Tuple[ModalityCVAE, int, dict]:
    data = torch.load(path, map_location=map_location)
    config = data['config']
    cvae, hidden_dim = build_pretrain_cvae_for_modality(
        config['input_dim'],
        config['num_classes'],
        config['hidden_layers'],
        activation_dropout=config.get('activation_dropout', 0.0),
        denoising=config.get('denoising', False),
        mask_p=config.get('mask_p', 0.0),
        mask_value=config.get('mask_value', 0.0),
        loss_on_masked=config.get('loss_on_masked', True),
        beta=config.get('beta', 1.0),
    )
    cvae.load_state_dict(data['state_dict'])
    return cvae, hidden_dim, config
