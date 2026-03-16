"""
Phase 2: Conditional Shared-Space Model using CVAE encoders/decoders.

ConditionalMultiModalWithSharedSpace extends MultiModalWithSharedSpace by
threading the primary-site condition c through encoders and decoders.

Pipeline per modality:
    x -(encoder(x,c))-> mu -(projection(h,c))-> z [shared_dim]
                                                  |
                              (average/impute combined z)
                                                  |
    x_recon <-(decoder(h_hat,c))- h_hat <-(rev_projection(z,c))- z_combined
"""
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cvae import CVAEConditionedDecoder, CVAEConditionedEncoder
from .mae_masked import build_mlp, apply_modality_dropout


# ─── Conditional projection heads ────────────────────────────────────────────

class ConditionalProjectionHead(nn.Module):
    """hidden_dim -> shared_dim, conditioned on c via concatenation: [h; c]."""

    def __init__(self, hidden_dim: int, num_classes: int, shared_dim: int,
                 depth: int = 1, activation_dropout: float = 0.0):
        super().__init__()
        in_dim = hidden_dim + num_classes
        if depth == 1:
            self.proj = nn.Linear(in_dim, shared_dim)
        else:
            dims = [in_dim] + [in_dim] * (depth - 2) + [shared_dim]
            self.proj = build_mlp(dims, add_final_activation=False,
                                  activation_dropout=activation_dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h, c], dim=-1))


class ConditionalReverseProjectionHead(nn.Module):
    """shared_dim -> hidden_dim, conditioned on c via concatenation: [z; c]."""

    def __init__(self, shared_dim: int, num_classes: int, hidden_dim: int,
                 depth: int = 1, activation_dropout: float = 0.0):
        super().__init__()
        in_dim = shared_dim + num_classes
        if depth == 1:
            self.rproj = nn.Linear(in_dim, hidden_dim)
        else:
            dims = [in_dim] + [in_dim] * (depth - 2) + [hidden_dim]
            self.rproj = build_mlp(dims, add_final_activation=False,
                                   activation_dropout=activation_dropout)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.rproj(torch.cat([z, c], dim=-1))


# ─── Model ────────────────────────────────────────────────────────────────────

class ConditionalMultiModalWithSharedSpace(nn.Module):
    """
    Like MultiModalWithSharedSpace but passes condition c through the full
    pipeline: encoders, projections, reverse-projections, and decoders.
    Encoders:         CVAEConditionedEncoder       — forward(x, c) → mu
    Projections:      ConditionalProjectionHead    — forward(h, c) → z
    Rev-projections:  ConditionalReverseProjectionHead — forward(z, c) → h_hat
    Decoders:         CVAEConditionedDecoder        — forward(h_hat, c) → x_recon
    """

    def __init__(
        self,
        encoders: Dict[str, CVAEConditionedEncoder],
        decoders: Dict[str, CVAEConditionedDecoder],
        hidden_dims: Dict[str, int],
        shared_dim: int,
        num_classes: int,
        proj_depth: int = 1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.modalities = list(encoders.keys())
        self.shared_dim = shared_dim
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

        self.projections = nn.ModuleDict({
            m: ConditionalProjectionHead(hidden_dims[m], num_classes, shared_dim,
                                         proj_depth, activation_dropout)
            for m in self.modalities
        })
        self.rev_projections = nn.ModuleDict({
            m: ConditionalReverseProjectionHead(shared_dim, num_classes, hidden_dims[m],
                                                proj_depth, activation_dropout)
            for m in self.modalities
        })

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        c: torch.Tensor,
        return_kl_params: bool = False,
    ):
        """
        Args:
            batch: {mod: [B, input_dim]}
            c:     [B, num_classes]
            return_kl_params: if True, also return (mu_dict, logvar_dict) for KL loss
        Returns:
            shared_embeddings, reconstructions, hidden_states
            [, mu_dict, logvar_dict]  — only when return_kl_params=True
        """
        shared_embeddings, reconstructions, hidden_states = {}, {}, {}
        mu_dict: Dict[str, torch.Tensor] = {}
        logvar_dict: Dict[str, Optional[torch.Tensor]] = {}

        for mod, xb in batch.items():
            enc = self.encoders[mod]
            if return_kl_params and hasattr(enc, "encode_params"):
                mu, logvar = enc.encode_params(xb, c)
                # reparameterise when training and logvar available
                if self.training and logvar is not None:
                    std = torch.exp(0.5 * logvar)
                    h = mu + std * torch.randn_like(std)
                else:
                    h = mu
                mu_dict[mod] = mu
                logvar_dict[mod] = logvar
            else:
                h = enc(xb, c)

            z = self.projections[mod](h, c)
            h_hat = self.rev_projections[mod](z, c)
            x_recon = self.decoders[mod](h_hat, c)
            hidden_states[mod] = h
            shared_embeddings[mod] = z
            reconstructions[mod] = x_recon

        if return_kl_params:
            return shared_embeddings, reconstructions, hidden_states, mu_dict, logvar_dict
        return shared_embeddings, reconstructions, hidden_states


# ─── Loss helpers ─────────────────────────────────────────────────────────────

def _prepare_clean_batch(
    batch: Dict[str, torch.Tensor],
    mask_values: Dict[str, float],
) -> Tuple[Dict, Dict]:
    batch_clean, missing_masks = {}, {}
    for mod, xb in batch.items():
        missing = torch.isnan(xb)
        xb_clean = xb.clone()
        xb_clean[missing] = mask_values.get(mod, 0.0)
        batch_clean[mod] = xb_clean
        missing_masks[mod] = missing
    return batch_clean, missing_masks


def _apply_feature_mask_sentinels(
    batch: Dict[str, torch.Tensor],
    mask_values: Dict[str, float],
    p: float,
) -> Tuple[Dict, Dict]:
    if p <= 0.0:
        return batch, {m: torch.zeros_like(x, dtype=torch.bool) for m, x in batch.items()}
    out, masks = {}, {}
    for mod, x in batch.items():
        art_mask = torch.rand_like(x) < p
        x_noisy = x.clone()
        x_noisy[art_mask] = mask_values.get(mod, 0.0)
        out[mod] = x_noisy
        masks[mod] = art_mask
    return out, masks


def _recon_loss(
    target: Dict[str, torch.Tensor],
    recons: Dict[str, torch.Tensor],
    orig_missing: Dict[str, torch.Tensor],
    art_masks: Optional[Dict[str, torch.Tensor]] = None,
    alpha_mask: float = 0.5,
) -> Tuple[torch.Tensor, Dict]:
    total = 0.0
    per_mod = {}
    for mod, tgt in target.items():
        diff_sq = (recons[mod] - tgt) ** 2
        valid = ~orig_missing[mod]
        overall = diff_sq[valid].mean() if valid.any() else diff_sq.mean()
        if art_masks is not None:
            combined = art_masks[mod] & valid
            masked = diff_sq[combined].mean() if combined.any() else overall
            loss_m = alpha_mask * masked + (1.0 - alpha_mask) * overall
        else:
            loss_m = overall
        per_mod[mod] = loss_m
        total = total + loss_m
    return total, per_mod


def _contrastive_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.1,
) -> torch.Tensor:
    mods = list(embeddings.keys())
    if len(mods) < 2:
        return next(iter(embeddings.values())).new_tensor(0.0)
    loss, count = 0.0, 0
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            z1, z2 = embeddings[mods[i]], embeddings[mods[j]]
            labels = torch.arange(z1.size(0), device=z1.device)
            sim_ab = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / temperature
            loss += F.cross_entropy(sim_ab, labels)
            sim_ba = F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=-1) / temperature
            loss += F.cross_entropy(sim_ba, labels)
            count += 2
    return loss / max(count, 1)


def _imputation_loss(
    target: Dict[str, torch.Tensor],
    embeddings: Dict[str, torch.Tensor],
    model: ConditionalMultiModalWithSharedSpace,
    orig_missing: Dict[str, torch.Tensor],
    c: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    losses, per_mod = [], {}
    for tgt_mod in embeddings:
        if tgt_mod not in target:
            continue
        other = [m for m in embeddings if m != tgt_mod and m in target]
        if not other:
            continue
        z_mean = torch.stack([embeddings[m] for m in other]).mean(0)
        h_hat = model.rev_projections[tgt_mod](z_mean, c)
        x_imp = model.decoders[tgt_mod](h_hat, c)
        diff_sq = (x_imp - target[tgt_mod]) ** 2
        valid = ~orig_missing[tgt_mod]
        loss_t = diff_sq[valid].mean() if valid.any() else diff_sq.mean()
        per_mod[tgt_mod] = loss_t
        losses.append(loss_t)
    if not losses:
        return next(iter(embeddings.values())).new_tensor(0.0), per_mod
    return torch.stack(losses).mean(), per_mod


def _kl_loss(
    mu_dict: Dict[str, torch.Tensor],
    logvar_dict: Dict[str, Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    KL divergence: KL(q(z|x,c) || N(0,I)) summed over modalities.
    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    Only computed for modalities that have a logvar_head.
    """
    kls = []
    for mod, mu in mu_dict.items():
        logvar = logvar_dict.get(mod)
        if logvar is None:
            continue
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        kls.append(kl)
    if not kls:
        return next(iter(mu_dict.values())).new_tensor(0.0)
    return torch.stack(kls).mean()


# ─── Training Loops ───────────────────────────────────────────────────────────

def conditional_finetune_epoch(
    model: ConditionalMultiModalWithSharedSpace,
    dataloader: DataLoader,
    optimizer,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_impute: float = 1.0,
    lambda_kl: float = 1.0,
    modality_dropout_prob: float = 0.2,
    feature_mask_p: float = 0.1,
    alpha_mask_recon: float = 0.5,
    grad_clip: float = 1.0,
    gaussian_noise_std: float = 0.0,
) -> Dict[str, float]:
    """
    One training epoch for ConditionalMultiModalWithSharedSpace.
    Dataloader must yield (x_dict, c) batches (ConditionalMultiOmicDataset).
    """
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "contrast": 0.0, "impute": 0.0, "kl": 0.0}
    n = 0

    for batch_x, batch_c in dataloader:
        batch_x = {k: v.to(device) for k, v in batch_x.items()}
        batch_c = batch_c.to(device)

        batch_x = apply_modality_dropout(batch_x, modality_dropout_prob)
        batch_clean, orig_missing = _prepare_clean_batch(batch_x, mask_values)
        noisy, art_masks = _apply_feature_mask_sentinels(batch_clean, mask_values, feature_mask_p)

        if gaussian_noise_std > 0.0:
            noisy = {m: x + torch.randn_like(x) * gaussian_noise_std for m, x in noisy.items()}

        optimizer.zero_grad()
        shared_emb, recons, _, mu_dict, logvar_dict = model(noisy, batch_c, return_kl_params=True)

        r_loss, _ = _recon_loss(batch_clean, recons, orig_missing, art_masks, alpha_mask_recon)
        c_loss = _contrastive_loss(shared_emb)
        i_loss, _ = _imputation_loss(batch_clean, shared_emb, model, orig_missing, batch_c)
        kl = _kl_loss(mu_dict, logvar_dict)

        total = r_loss + lambda_contrastive * c_loss + lambda_impute * i_loss + lambda_kl * kl
        total.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        sums["total"]    += total.item()
        sums["recon"]    += r_loss.item()
        sums["contrast"] += c_loss.item()
        sums["impute"]   += i_loss.item()
        sums["kl"]       += kl.item()
        n += 1

    avg = max(n, 1)
    return {
        "total_loss":    sums["total"] / avg,
        "recon_loss":    sums["recon"] / avg,
        "contrast_loss": sums["contrast"] / avg,
        "impute_loss":   sums["impute"] / avg,
        "kl_loss":       sums["kl"] / avg,
    }


@torch.no_grad()
def conditional_eval_finetune_epoch(
    model: ConditionalMultiModalWithSharedSpace,
    dataloader: DataLoader,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_impute: float = 1.0,
    lambda_kl: float = 1.0,
    feature_mask_p: float = 0.1,
    alpha_mask_recon: float = 0.5,
) -> Dict[str, float]:
    """
    One eval epoch for ConditionalMultiModalWithSharedSpace.
    Dataloader must yield (x_dict, c) batches.
    """
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "contrast": 0.0, "impute": 0.0, "kl": 0.0}
    n = 0

    for batch_x, batch_c in dataloader:
        batch_x = {k: v.to(device) for k, v in batch_x.items()}
        batch_c = batch_c.to(device)

        batch_clean, orig_missing = _prepare_clean_batch(batch_x, mask_values)
        noisy, art_masks = _apply_feature_mask_sentinels(batch_clean, mask_values, feature_mask_p)

        shared_emb, recons, _, mu_dict, logvar_dict = model(noisy, batch_c, return_kl_params=True)

        r_loss, _ = _recon_loss(batch_clean, recons, orig_missing, art_masks, alpha_mask_recon)
        c_loss = _contrastive_loss(shared_emb)
        i_loss, _ = _imputation_loss(batch_clean, shared_emb, model, orig_missing, batch_c)
        kl = _kl_loss(mu_dict, logvar_dict)

        total = r_loss + lambda_contrastive * c_loss + lambda_impute * i_loss + lambda_kl * kl
        sums["total"]    += total.item()
        sums["recon"]    += r_loss.item()
        sums["contrast"] += c_loss.item()
        sums["impute"]   += i_loss.item()
        sums["kl"]       += kl.item()
        n += 1

    avg = max(n, 1)
    return {
        "total_loss":    sums["total"] / avg,
        "recon_loss":    sums["recon"] / avg,
        "contrast_loss": sums["contrast"] / avg,
        "impute_loss":   sums["impute"] / avg,
        "kl_loss":       sums["kl"] / avg,
    }
