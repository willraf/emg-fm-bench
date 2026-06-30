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

        self.num_classes = cfg.n_classes
        self.pooling_type = cfg.pooling_type
        self.finetune_spacial_embedding = cfg.finetune_spacial_embedding

        n_channels = cfg.num_channels
        window_duration = cfg.window_duration
        sample_rate = cfg.sample_rate
        window_length = int(window_duration * sample_rate)

        self._build_head_from_dummy(n_channels, window_length)


    def _build_head_from_dummy(self, n_channels, window_length):
        """
        Create a dummy input to determine the shape of the tokens and build the head accordingly.

        Args:
            n_channels: Number of input channels
            window_length: Length of the input window in samples
        """
        device = next(self.fm.parameters()).device
        # Create a dummy input tensor with shape (B, C, T)
        dummy_input = torch.randn(1, n_channels, window_length, device=device)

        with torch.no_grad():
            self.fm.eval()
            tokens = self.fm(dummy_input)
            self.fm.train()

        if tokens.ndim == 3:
            # If tokens are (B, N, D), we can treat it as a single channel case
            tokens = tokens.unsqueeze(1)  # (B, 1, N, D)

        # Build the head based on the shape of the tokens
        self._build_head(tokens.shape)


    def _build_head(self, input_shape):
        """
        Create the linear head based on the pooling type and input shape.

        Args:
            input_shape: Tuple representing the shape of the input tokens (B, C, N, D)
                - B: Batch size
                - C: Number of channels
                - N: Number of temporal patches
                - D: Embedding dimension
        """
        B, C, N, D = input_shape

        # Head Definition
        if self.pooling_type in ["chan_concat_time_mean", "chan_concat"]: 
            # Concat across channels: dim = embed_dim * n_channels
            self.fc_norm = nn.LayerNorm(D * C)
            self.head = nn.Linear(D * C, self.num_classes)

        elif self.pooling_type in ["time_concat_chan_mean", "time_concat"]:
            self.fc_norm = nn.LayerNorm(D * N)
            self.head = nn.Linear(D * N, self.num_classes)

        elif self.pooling_type == "attention_pool":
            self.query = nn.Parameter(torch.randn(1, 1, D))
            self.attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True)
            self.fc_norm = nn.LayerNorm(D)
            self.head = nn.Linear(D, self.num_classes)

        elif self.pooling_type in ["chan_mean", "time_mean", "mean"]:
            self.fc_norm = nn.LayerNorm(D)
            self.head = nn.Linear(D, self.num_classes)


        else:  
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        logging.info(
            f"BioFMWrapper head built — pooling: {self.pooling_type}, "
            f"C={C}, N={N}, D={D}, head_in={self.head.in_features}, "
            f"head_out={self.num_classes}"
        )


    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time

        Returns:
            logits: (Batch, num_classes)
        """
        tokens = self.fm(x)

        if tokens.ndim == 3:
            # If tokens are (B, N, D), we can treat it as a single channel case
            tokens = tokens.unsqueeze(1)  # (B, 1, N, D)

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
            feat = tokens.permute(0, 2, 1, 3).reshape(B, N, C * D)
            feat = feat.mean(dim=1) # Average over time -> (B, C*D)

        elif self.pooling_type == "time_concat_chan_mean":
            feat = tokens.reshape(B, C, N*D) # (B, C, N*D)
            feat = feat.mean(dim=1) # Average over channels -> (B, N*D)

        elif self.pooling_type == "attention_pool":
            # TODO: Strategy 4: Model learns which patches represent the gesture
            # Flatten to (B, num_patches, D) for attention
            flat_tokens = tokens.reshape(B, C*N, D)
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
            feat = tokens.permute(0, 2, 1, 3).reshape(B, N, C*D) 

        elif self.pooling_type == "time_concat":
            feat = tokens.reshape(B, C, N*D) # (B, C, N*D)   

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