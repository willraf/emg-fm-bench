""" Module flex_emg.py

FlexEMG: A flexible sEMG foundation model encoder.

Baseline architecture — standard ViT-style transformer with:
    - Channel-independent patch embedding (shared conv weights across channels)
    - Learnable spatial embeddings (one per channel, to be replaced with
      geometric embeddings in a future version)
    - Sinusoidal temporal positional embeddings
    - Standard transformer encoder blocks (pre-norm)
    - Output: per-patch embeddings (B, C, N, D)

Authors:
    Will Raftery

Date:
    09/07/2026
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Patch Embedding
# --------------------------------------------------------------------------- #

class ChannelIndependentPatchEmbed(nn.Module):
    """
    Embeds sEMG patches into D-dimensional vectors.

    Each channel is patched independently using a shared 1D convolution
    (equivalent to a per-patch fully connected layer). Weights are shared
    across channels, so the model is agnostic to channel count at runtime.

    Args:
        patch_size:  number of time samples per patch (L)
        embed_dim:   output embedding dimension (D)

    Input:  (B, C, T)
    Output: (B, C, N, D)  where N = T // patch_size
    """

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # Conv1d with kernel=stride=patch_size is equivalent to a per-patch
        # linear projection. in_channels=1 because we process each channel
        # independently by reshaping.
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        # Initialise like a linear layer (Xavier uniform)
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            (B, C, N, D)
        """
        B, C, T = x.shape
        assert T % self.patch_size == 0, (
            f"Signal length T={T} must be divisible by patch_size={self.patch_size}. "
            f"Pad or crop your input before passing to the model."
        )

        # Merge batch and channel dims so Conv1d processes each channel independently
        x = x.reshape(B * C, 1, T)          # (B*C, 1, T)
        x = self.proj(x)                     # (B*C, D, N)
        x = x.transpose(1, 2)               # (B*C, N, D)

        N = x.shape[1]
        x = x.reshape(B, C, N, self.embed_dim)  # (B, C, N, D)
        return x


# --------------------------------------------------------------------------- #
# Positional Embeddings
# --------------------------------------------------------------------------- #

class SinusoidalTemporalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings over the patch (time) dimension.

    Follows the original Transformer formulation (Vaswani et al. 2017).
    Embeddings are computed once and cached; the cache is extended
    automatically if a longer sequence is encountered.

    Args:
        embed_dim:   embedding dimension (must be even)
        max_patches: maximum number of patches to pre-cache

    Input:  N (int) — number of patches
    Output: (1, 1, N, D) — broadcastable over (B, C, N, D)
    """

    def __init__(self, embed_dim: int, max_patches: int = 4096):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal embeddings"
        self.embed_dim  = embed_dim
        self.max_patches = max_patches
        self.register_buffer("pe", self._build(max_patches), persistent=False)

    def _build(self, length: int) -> torch.Tensor:
        """Build sinusoidal embedding table of shape (1, 1, length, D)."""
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.embed_dim)
        )                                                                    # (D/2,)
        pe = torch.zeros(length, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).unsqueeze(0)                                  # (1, 1, N, D)

    def forward(self, n_patches: int) -> torch.Tensor:
        """
        Args:
            n_patches: number of patches in the current input

        Returns:
            (1, 1, n_patches, D)
        """
        if n_patches > self.pe.shape[2]:
            # Extend cache if needed
            self.pe = self._build(n_patches).to(self.pe.device)
        return self.pe[:, :, :n_patches, :]


class LearnableSpatialEmbedding(nn.Module):
    """
    Learnable spatial (channel) embeddings — one embedding per channel.

    This is the baseline version. In FlexEMG v2 this will be replaced
    by geometric embeddings conditioned on electrode coordinates.

    Args:
        n_channels: number of input channels (must be fixed at init)
        embed_dim:  embedding dimension

    Input:  C (int) — number of channels (must equal n_channels)
    Output: (1, C, 1, D) — broadcastable over (B, C, N, D)
    """

    def __init__(self, n_channels: int, embed_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim  = embed_dim
        self.embedding  = nn.Embedding(n_channels, embed_dim)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, c: int) -> torch.Tensor:
        """
        Args:
            c: number of channels in current input

        Returns:
            (1, C, 1, D)
        """
        assert c == self.n_channels, (
            f"Expected {self.n_channels} channels, got {c}. "
            f"Learnable spatial embeddings require a fixed channel count. "
            f"Use geometric spatial embeddings for variable channel counts."
        )
        idx = torch.arange(c, device=self.embedding.weight.device)
        return self.embedding(idx).unsqueeze(0).unsqueeze(2)  # (1, C, 1, D)


# --------------------------------------------------------------------------- #
# Transformer Block (Pre-Norm)
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block.

    LayerNorm → Self-Attention → residual
    LayerNorm → MLP → residual

    Args:
        embed_dim:   token dimension
        num_heads:   number of attention heads
        mlp_ratio:   hidden dim of MLP = embed_dim * mlp_ratio
        attn_drop:   dropout on attention weights
        proj_drop:   dropout on attention output projection
        mlp_drop:    dropout in MLP
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop:  float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(proj_drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, N_seq, D) — flattened token sequence
            key_padding_mask: (B, N_seq) bool mask, True = ignore (optional)

        Returns:
            (B, N_seq, D)
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# FlexEMG Encoder
# --------------------------------------------------------------------------- #

class FlexEMGEncoder(nn.Module):
    """
    FlexEMG baseline encoder.

    Architecture:
        1. Channel-independent patch embedding  (B, C, T) → (B, C, N, D)
        2. Add sinusoidal temporal embeddings   (1, 1, N, D)
        3. Add learnable spatial embeddings     (1, C, 1, D)
        4. Flatten to token sequence            (B, C*N, D)
        5. Transformer encoder                  (B, C*N, D)
        6. Reshape to per-patch output          (B, C, N, D)

    The transformer operates over the full (C*N) token sequence, allowing
    attention across both channels and time — full spatiotemporal mixing.

    Args:
        patch_size:  number of time samples per patch
        embed_dim:   token embedding dimension (D)
        depth:       number of transformer blocks
        num_heads:   number of attention heads per block
        mlp_ratio:   MLP hidden dim ratio
        n_channels:  number of input channels (required for spatial embeddings)
        attn_drop:   attention dropout
        proj_drop:   projection dropout
        mlp_drop:    MLP dropout
        drop_path:   stochastic depth rate (applied per block)
    """

    def __init__(
        self,
        patch_size:  int   = 20,
        embed_dim:   int   = 192,
        depth:       int   = 8,
        num_heads:   int   = 8,
        mlp_ratio:   float = 4.0,
        n_channels:  int   = 16,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
        mlp_drop:    float = 0.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.n_channels = n_channels

        # ------------------------------------------------------------------ #
        # Components
        # ------------------------------------------------------------------ #
        self.patch_embed = ChannelIndependentPatchEmbed(patch_size, embed_dim)
        self.temporal_pe = SinusoidalTemporalEmbedding(embed_dim)
        self.spatial_pe  = LearnableSpatialEmbedding(n_channels, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_drop=mlp_drop,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        logging.info(
            f"FlexEMGEncoder initialised — "
            f"patch_size={patch_size}, embed_dim={embed_dim}, "
            f"depth={depth}, num_heads={num_heads}, "
            f"n_channels={n_channels}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self):
        """Initialise transformer weights following ViT conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, C, T) — raw sEMG signal
            key_padding_mask: (B, C*N) bool mask for padding tokens (optional)
                              True = ignore this token

        Returns:
            (B, C, N, D) — per-patch embeddings
        """
        B, C, T = x.shape

        # ------------------------------------------------------------------ #
        # 1. Patch embedding
        # ------------------------------------------------------------------ #
        tokens = self.patch_embed(x)         # (B, C, N, D)
        N = tokens.shape[2]

        # ------------------------------------------------------------------ #
        # 2. Add positional embeddings
        # ------------------------------------------------------------------ #
        tokens = tokens + self.temporal_pe(N)   # (1, 1, N, D) broadcasts
        tokens = tokens + self.spatial_pe(C)    # (1, C, 1, D) broadcasts

        # ------------------------------------------------------------------ #
        # 3. Flatten to token sequence for transformer
        # ------------------------------------------------------------------ #
        # Interleave channels and patches: (B, C, N, D) → (B, C*N, D)
        # We keep channel-major ordering: [c0p0, c0p1, ..., c1p0, c1p1, ...]
        tokens = tokens.reshape(B, C * N, self.embed_dim)

        # ------------------------------------------------------------------ #
        # 4. Transformer blocks
        # ------------------------------------------------------------------ #
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)

        tokens = self.norm(tokens)           # (B, C*N, D)

        # ------------------------------------------------------------------ #
        # 5. Reshape back to (B, C, N, D)
        # ------------------------------------------------------------------ #
        tokens = tokens.reshape(B, C, N, self.embed_dim)

        return tokens

    def load_checkpoint(self, path):
        """Load encoder weights from a checkpoint file."""
        import torch
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # Handle various checkpoint formats
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys: {unexpected}")
        logging.info(f"Checkpoint loaded from {path}")