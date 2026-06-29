""" Module flex.py

Frequency Adaptive Latent EXtractor (FLEX) model for EMG signal processing.

Usage:


Authors:
    Will Raftery

Date:
    23/06/2026
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


class FlexEMG(nn.Module):
    """
    Frequency Adaptive Latent EXtractor (FLEX) model for EMG signal processing.

    Args:
        cfg: Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = FlexEncoder(cfg)
        self.target_encoder = ...

    def

    

class GatedGlobalInstanceNorm(nn.Module):
    """
    Learnable gated global instance normalisation.

    For each sample in the batch, computes global mean and std over
    C x T, normalises the signal, then interpolates between the
    normalised and raw signal using a learnable scalar gate:

        g = sigmoid(w)
        X_out = g * (X - mu) / (sigma + eps) + (1 - g) * X

    The gate w is initialised to a large negative value so the model
    starts close to the identity (no normalisation) and learns to
    apply normalisation only if it is beneficial.

    This avoids the need for any preprocessing normalisation while
    remaining fully differentiable and shape-agnostic over C and T.

    Args:
        eps:        small constant for numerical stability
        init_gate:  initial value of the raw gate parameter w.
                    sigmoid(-4) ≈ 0.018, so model starts near identity.
                    Set to 0.0 for a 50/50 initialisation.
    """

    def __init__(self, eps: float = 1e-6, init_gate: float = -4.0):
        super().__init__()
        self.eps = eps
        # Learnable scalar gate parameter
        # sigmoid(w) interpolates between raw (w -> -inf) and normalised (w -> +inf)
        self.w = nn.Parameter(torch.tensor(init_gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — raw input signal

        Returns:
            (B, C, T) — adaptively normalised signal
        """
        # Compute global statistics over C x T per sample
        # Flatten to (B, C*T) for mean/std, then restore shape
        B, C, T = x.shape
        x_flat = x.reshape(B, -1)                          # (B, C*T)

        mu = x_flat.mean(dim=1, keepdim=True)           # (B, 1)
        sigma = x_flat.std(dim=1, keepdim=True)            # (B, 1)

        # Reshape stats for broadcasting against (B, C, T)
        mu = mu.unsqueeze(-1)                           # (B, 1, 1)
        sigma = sigma.unsqueeze(-1)                        # (B, 1, 1)

        x_norm = (x - mu) / (sigma + self.eps)            # (B, C, T)

        # Learnable gate: sigmoid(w) in (0, 1)
        g = torch.sigmoid(self.w)

        return g * x_norm + (1 - g) * x

    @property
    def gate_value(self) -> float:
        """Returns the current gate value for logging/inspection."""
        return torch.sigmoid(self.w).item()

    def extra_repr(self) -> str:
        return (
            f"eps={self.eps}, "
            f"init_gate=sigmoid({self.w.item():.2f})="
            f"{torch.sigmoid(self.w).item():.3f}"
        )



@dataclass(eq=False)
class FlexEncoder(nn.Module):
    """
    Frequency Adaptive Latent EXtractor (FLEX) model for EMG signal processing.

    Args:
        cfg: Configuration object containing model parameters.
    """

    def __post_init__(self):
        super().__init__()

        self.input_norm = GatedGlobalInstanceNorm()
        self.filter = ...

        self.embed_patches = ...
        self.transformer = ...

        # Define the model architecture here
        # Example: self.layer1 = nn.Conv1d(...)
        # Add more layers as needed based on the configuration

    def _extract_patches(self, x):
        pass

    def _spatial_embedding(self, electrode_positions):
        pass

    def _positional_embedding(self, patches, sample_rate):
        pass

    def _mask_patches(self, patch_embeds):
        pass


    def forward(self, x, electrode_positions=None, sample_rate=1000):
        """
        Forward pass of the FLEX model.

        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            electrode_positions: Optional tensor of electrode positions for positional embedding. Shape (batch_size, channels, 3) if provided.  

        Returns:
            Output per patch embeddings shape (batch_size, channels, time_steps/patch_len, embed_dim)
        """

        # Normalise input signal with learnt gated normalisation 
        x = self.lin_gate_norm(x)  # (B, C, T)
        
        patches = self._extract_patches(x)      # (B, C, N, patch_len)
        spacial_embeds = self._spatial_embedding(electrode_positions)  # (C, -)
        positional_embeds = self._positional_embedding(patches, sample_rate=sample_rate)  # (N, -)

        # Get channel independant embeddings for each patch
        patch_embeds = self.embed_patches(patches)  # (B, C, N, -)

        # Mask out some patches
        masked_embeds = self._mask_patches(patch_embeds)  # (B, C, N, -)

        # Add spacial and positional embeddings
        masked_embeds = masked_embeds + spacial_embeds.unsqueeze(0).unsqueeze(2) + positional_embeds.unsqueeze(0).unsqueeze(1)  # (B, C, N, -)

        # Flatten to (B, C*N, -) for transformer input
        masked_embeds = masked_embeds.flatten(1, 2)  # (B, C*N, -)

        # Pass through transformer layers
        outputs = self.transformer(masked_embeds)  # (B, C * N, embed_dim)

        # Reshape back to (B, C, N, embed_dim)
        outputs = outputs.view(x.size(0), x.size(1), -1, outputs.size(-1))  # (B, C, N, embed_dim)

        return outputs  # (B, C, N, embed_dim)