""" Module fm_wrapper.py

This wrapper standardises the finetuning heads added to foundation models

Usage:


Authors:
    Will Raftery

Date:
    10/03/2026
"""


import logging

from torch import nn
from src.models.neuro_rvq_fm import NeuroRVQFM
import torch 
from dataclasses import dataclass


class BioFMWrapper(nn.Module):
    def __init__(self, cfg, fm):
        super().__init__()
        self.fm = fm

        self.embed_dim = cfg.embed_dim
        self.num_classes = cfg.n_classes
        self.patch_len = cfg.patch_size
        self.pooling_type = cfg.pooling_type
        self.finetune_spacial_embedding = cfg.finetune_spacial_embedding
        self.n_channels = cfg.num_channels

        # Head Definition
        if self.pooling_type == "concat":
            # Concat across channels: dim = embed_dim * n_channels
            self.fc_norm = nn.LayerNorm(self.embed_dim * self.n_channels)
            self.head = nn.Linear(self.embed_dim * self.n_channels, self.num_classes)
        elif self.pooling_type == "attention_pool":
            self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else: # "mean" or "tiny_myo_mean"
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time

        Returns:
            logits: (Batch, num_classes)
        """
        # 1. Get raw tokens from FM (Batch, 256, Embed_Dim)
        # We set return_patch_tokens=True to avoid the hardcoded mean(1) in their forward
        # Using branch_idx=0 as EPN uses the first branch logic
        tokens = self.fm(x)
        
        B, C, N, D = tokens.shape # (B, Channels, num_temporal_patches,Embed_Dim)


        if self.pooling_type == "mean":
            # Global Average across all patches (Simple & robust)
            feat = tokens.mean(dim=(1, 2)) # (B, 200)
            
        elif self.pooling_type == "tiny_myo_mean":
            # Mean across channels, then mean across time
            feat = tokens.mean(dim=1).mean(dim=1) # (B, 200)
            
        elif self.pooling_type == "concat":
            # Concat channels: (B, Time, Chan*Embed) -> (B, 32, 1600)
            feat = tokens.permute(0, 2, 1, 3).reshape(B, N, -1)
            feat = feat.mean(dim=1) # Average over time -> (B, 1600)

        elif self.pooling_type == "attention_pool":
            # Strategy 4: Model learns which patches represent the gesture
            # Flatten to (B, 256, 200) for attention
            flat_tokens = tokens.reshape(B, N, D)
            q = self.query.expand(B, -1, -1)
            feat, _ = self.attn(q, flat_tokens, flat_tokens)
            feat = feat.squeeze(1) # (B, 200)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return self.head(self.fc_norm(feat))
    

    def get_fine_tuning_parameters(self):
        """
        Watch out for defference between module and parameters. Calling nn.module.parameters() returns an iterator over module parameters,
        converting this to list is valid. However, calling list(nn.Parameter) (as in list(self.fm.pos_embed)) creates a view of the parameters
        Basically, because pos_embed is a sigle parameter, list(self.fm.pos_embed) creates a list of the individual elements of the parameter tensor, which are not the actual parameters we want to optimize.
        """
        
        head_params = list(self.head.parameters()) + list(self.fc_norm.parameters())

        if self.pooling_type == "attention_pool":
            head_params += [self.query] + list(self.attn.parameters())

        if self.finetune_spacial_embedding:
            # head_params += list(self.fm.pos_embed) ERROR - I think this creates a view of the parameters
            head_params.append(self.fm.pos_embed)

        return head_params