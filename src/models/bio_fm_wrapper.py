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

        n_patches = 1

        # Head Definition
        if self.pooling_type == "chan_concat_time_mean":
            # Concat across channels: dim = embed_dim * n_channels
            self.fc_norm = nn.LayerNorm(self.embed_dim * self.n_channels)
            self.head = nn.Linear(self.embed_dim * self.n_channels, self.num_classes)

        elif self.pooling_type == "time_concat_chan_mean":
            self.fc_norm = nn.LayerNorm(self.embed_dim * n_patches)
            self.head = nn.Linear(self.embed_dim * n_patches, self.num_classes)

        elif self.pooling_type == "attention_pool":
            self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)

        elif self.pooling_type == "chan_mean" or self.pooling_type == "time_mean" or self.pooling_type == "mean":
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)

        elif self.pooling_type == "chan_concat":
            self.fc_norm = nn.LayerNorm(self.embed_dim * self.n_channels)
            self.head = nn.Linear(self.embed_dim * self.n_channels, self.num_classes)

        elif self.pooling_type == "time_concat":
            self.fc_norm = nn.LayerNorm(self.embed_dim * n_patches)
            self.head = nn.Linear(self.embed_dim * n_patches, self.num_classes)

        else:  
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time

        Returns:
            logits: (Batch, num_classes)
        """
        tokens = self.fm(x)

        feat = self._aggregate(tokens)

        return self.head(self.fc_norm(feat))
    

    def _aggregate(self, tokens):
        """
        Aggregate the tokens across time and/or channels

        Args:
            tokens: (B, C, N, D) per patch embeddings from the foundation model

        Returns:
            feat: (B, D) aggregated feature vector for classification
            feat: (B, N, D) if pooling_type is "channel_mean"
        """

        B, C, N, D = tokens.shape # (B, Channels, num_temporal_patches, Embed_Dim) --> (B, C, N, D)

        if self.pooling_type == "mean":
            # Global Average across all patches (Simple & robust)
            feat = tokens.mean(dim=(1, 2)) # (B, D)
            
        elif self.pooling_type == "chan_concat_time_mean":
            # Concat channels: (B, N, C*D) 
            feat = tokens.permute(0, 2, 1, 3).reshape(B, N, -1)
            feat = feat.mean(dim=1) # Average over time -> (B, C*D)

        elif self.pooling_type == "time_concat_chan_mean":
            feat = tokens.reshape(B, C, -1) # (B, C, N*D)
            feat = feat.mean(dim=1) # Average over channels -> (B, N*D)

        elif self.pooling_type == "attention_pool":
            # TODO: Strategy 4: Model learns which patches represent the gesture
            # Flatten to (B, 256, D) for attention
            flat_tokens = tokens.reshape(B, N, D)
            q = self.query.expand(B, -1, -1)
            feat, _ = self.attn(q, flat_tokens, flat_tokens)
            feat = feat.squeeze(1) # (B, D)

        elif self.pooling_type == "chan_mean":
            # Mean across channels
            feat = tokens.mean(dim=1) # (B, N, D)

        elif self.pooling_type == "time_mean":
            # Mean across time
            feat = tokens.mean(dim=2) # (B, Channels, D)
        
        elif self.pooling_type == "chan_concat":
            feat = tokens.permute(0, 2, 1, 3).reshape(B, N, -1) # (B, N, C*D)

        elif self.pooling_type == "time_concat":
            feat = tokens.reshape(B, C, -1) # (B, C, N*D)   

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return feat


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