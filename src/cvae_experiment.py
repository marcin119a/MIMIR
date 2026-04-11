"""
Standalone CVAE implementation for architectural transition experiments.
Parameterized to allow toggling features between Marcin's CVAE and Mimir's AE.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .mae_masked import ModalityDecoder, build_mlp

class ModalityCVAEExperiment(nn.Module):
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
        # Experimental Toggles
        use_batchnorm: bool = False,
        add_final_activation: bool = True,
        kl_reduction: str = "mean", # "mean" (weak) or "sum" (standard)
        deterministic: bool = False,
        condition_on_site: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes if condition_on_site else 0
        self.latent_dim = hidden_layers[-1]
        self.denoising = denoising
        self.mask_p = mask_p
        self.mask_value = mask_value
        self.loss_on_masked = loss_on_masked
        self.beta = beta
        
        self.use_batchnorm = use_batchnorm
        self.kl_reduction = kl_reduction
        self.deterministic = deterministic
        self.condition_on_site = condition_on_site

        self._last_mask: Optional[torch.Tensor] = None
        self._last_kl: Optional[torch.Tensor] = None

        # Encoder backbone
        enc_in = input_dim + (num_classes if condition_on_site else 0)
        if len(hidden_layers) > 1:
            backbone_dims = [enc_in] + hidden_layers[:-1]
            self.backbone = build_mlp(backbone_dims, 
                                      add_final_activation=add_final_activation,
                                      activation_dropout=activation_dropout,
                                      use_batchnorm=use_batchnorm)
            intermediate_dim = hidden_layers[-2]
        else:
            self.backbone = nn.Identity()
            intermediate_dim = enc_in

        self.mu_head = nn.Linear(intermediate_dim, self.latent_dim)
        self.logvar_head = nn.Linear(intermediate_dim, self.latent_dim)

        # Decoder
        dec_in = self.latent_dim + (num_classes if condition_on_site else 0)
        dec_dims = [dec_in] + list(reversed(hidden_layers[:-1])) + [input_dim]
        self.decoder = ModalityDecoder(dec_dims, 
                                       activation_dropout=activation_dropout,
                                       use_batchnorm=use_batchnorm)

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
        
        if self.condition_on_site:
            xc = torch.cat([x_in, c], dim=-1)
        else:
            xc = x_in
            
        h = self.backbone(xc)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        # KL Calculation
        kl_element = 1.0 + logvar - mu.pow(2) - logvar.exp()
        if self.kl_reduction == "sum":
            self._last_kl = -0.5 * torch.sum(kl_element, dim=1).mean()
        else:
            self._last_kl = -0.5 * kl_element.mean()
            
        if self.deterministic or not self.training:
            z = mu
        else:
            z = self._reparameterise(mu, logvar)
            
        if self.condition_on_site:
            zc = torch.cat([z, c], dim=-1)
        else:
            zc = z
            
        x_recon = self.decoder(zc)
        return mu, x_recon
