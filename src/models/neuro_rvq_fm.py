""" Module neuro_rvq_fm.py

Contains implementation of the NeuroRVQFM class

NOTE:
    This class expects a tuncated version of the full config (cfg.model.fm)
    This is different from other parts of the repo, where the full config is normally used

Source:
    https://github.com/KonstantinosBarmpas/NeuroRVQ/tree/main

Usage:
    fm = NeuroRVQFM(cfg.model.fm)

Authors:
    Will Raftery

Date:
    06/02/2026
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from src.models.neuro_rvq.NeuroRVQ_modules import Block, trunc_normal_

class PatchEmbed(nn.Module):
    """
    Project each codebook to the patch latent space
    :param in_chans: number of input channels
    :param embed_dim: dimension of embedding space
    """
    def __init__(self, in_chans=1, embed_dim=200):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        
class MultiDimentionalTemporalConv(nn.Module):
    """
    EEG to Patch Embedding - Multidimentional Temporal Filtering
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # --- Group 1: First-level temporal convolutions --- #
        # Branch 1: >20 Hz assuming fs=1000Hz
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 51), padding=(0, 25))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 2: >60 Hz assuming fs=1000Hz
        self.conv1_2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 17), padding=(0, 8))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 3: >125 Hz assuming fs=1000Hz
        self.conv1_3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 8), padding=(0, 4))
        self.norm1_3 = nn.GroupNorm(4, out_chans)
        self.pool1_3 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 4: >250 Hz assuming fs=1000Hz
        self.conv1_4 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_4 = nn.GroupNorm(4, out_chans)
        self.pool1_4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.gelu1 = nn.GELU()

        # --- Group 2: Second-level convolutions --- #
        self.conv2_1 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 25), padding=(0, 12))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 4), padding=(0, 2))
        self.norm2_3 = nn.GroupNorm(4, out_chans)
        self.pool2_3 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_4 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_4 = nn.GroupNorm(4, out_chans)
        self.pool2_4 = nn.AvgPool2d(kernel_size=(1, 4))

        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)

        # --- Group 1 --- #
        x1 = self.pool1_1(self.gelu1(self.norm1_1(self.conv1_1(x))))
        x2 = self.pool1_2(self.gelu1(self.norm1_2(self.conv1_2(x))))
        x3 = self.pool1_3(self.gelu1(self.norm1_3(self.conv1_3(x))))
        x4 = self.pool1_4(self.gelu1(self.norm1_4(self.conv1_4(x))))

        # --- Group 2 --- #
        x1 = self.pool2_1(self.gelu2(self.norm2_1(self.conv2_1(x1))))
        x2 = self.pool2_2(self.gelu2(self.norm2_2(self.conv2_2(x2))))
        x3 = self.pool2_3(self.gelu2(self.norm2_3(self.conv2_3(x3))))
        x4 = self.pool2_4(self.gelu2(self.norm2_4(self.conv2_4(x4))))

        # --- Re-arrange --- #
        x1 = rearrange(x1, 'B C NA T -> B NA (T C)')
        x2 = rearrange(x2, 'B C NA T -> B NA (T C)')
        x3 = rearrange(x3, 'B C NA T -> B NA (T C)')
        x4 = rearrange(x4, 'B C NA T -> B NA (T C)')
        return x1, x2, x3, x4
        


@dataclass(eq=False)
class NeuroRVQFM(nn.Module):
    """
    NeuroRVQ Foundation Model Class
    """

    max_patches: int
    patch_size: int
    in_chans: int
    out_chans: int
    num_classes: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    qk_norm: Optional[Callable]
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float
    init_values: Optional[float]
    init_scale: float
    n_global_electrodes: int
    vocab_size: int
    use_as_encoder: bool
    use_for_pretraining: bool
    sample_channels: tuple
    return_patch_embeddings: bool

    # Channel labels of the global electrodes (from pretraining)
    global_channels: tuple = (
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
        'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16'
    )
    # Channel labels of the sample-specific electrodes
  

    def __post_init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.qk_norm = partial(nn.LayerNorm, eps=1e-6) 

        # print(f'n_patches {n_patches}')
        # print(f'patch_size {self.patch_size}')
        # print(f'in_chans {in_chans}')
        # print(f'out_chans {out_chans}')
        # print(f'num_classes {num_classes}')
        # print(f'embed_dim {embed_dim}')
        # print(f'depth {depth}')
        # print(f'num_heads {num_heads}')
        # print(f'mlp_ratio {mlp_ratio}')
        # print(f'qkv_bias {qkv_bias}')
        # print(f'qk_norm {qk_norm}')
        # print(f'drop_rate {drop_rate}')
        # print(f'attn_drop_rate {attn_drop_rate}')  
        # print(f'drop_path_rate {drop_path_rate}')
        # print(f'init_values {init_values}')
        # print(f'init_scale {init_scale}')
        # print(f'n_global_electrodes {n_global_electrodes}')
        # print(f'vocab_size {vocab_size}')
        # print(f'use_as_encoder {self.use_as_encoder}')
        # print(f'use_for_pretraining {self.use_for_pretraining}')

        # For adding spacial and temporal embeddings
        # Create a mapping from global channel names to their indices for easy lookup
        self.global_channel_map = {name: i for i, name in enumerate(self.global_channels)}
        self._cached_sample_channels = None
        self._cached_spat_ix_tensor = None
        # Sample channels array stored as a lazy property, as may not be determined till runtime
        

        if self.use_as_encoder or self.use_for_pretraining:
            self.patch_embed = MultiDimentionalTemporalConv(out_chans=self.out_chans)
        else:     
            self.patch_embed_1 = PatchEmbed(in_chans=self.in_chans, embed_dim=self.embed_dim)
            self.patch_embed_2 = PatchEmbed(in_chans=self.in_chans, embed_dim=self.embed_dim)
            self.patch_embed_3 = PatchEmbed(in_chans=self.in_chans, embed_dim=self.embed_dim)
            self.patch_embed_4 = PatchEmbed(in_chans=self.in_chans, embed_dim=self.embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(self.n_global_electrodes + 1, self.embed_dim), requires_grad=True)
        self.time_embed = nn.Parameter(torch.zeros(self.max_patches, self.embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_norm=self.qk_norm,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm,
                init_values=self.init_values, window_size=None)
            for i in range(self.depth)])

        # If used for pre-training we need heads to predict 16 * 4 tokens per input sample
        if self.use_for_pretraining:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.norm_pre = nn.LayerNorm(self.embed_dim)
            self.head_pre_1 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_2 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_3 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_4 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_5 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_6 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_7 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_8 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_9 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_10 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_11 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_12 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_13 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_14 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_15 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_16 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_17 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_18 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_19 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_20 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_21 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_22 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_23 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_24 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_25 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_26 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_27 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_28 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_29 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_30 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_31 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_32 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_33 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_34 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_35 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_36 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_37 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_38 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_39 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_40 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_41 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_42 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_43 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_44 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_45 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_46 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_47 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_48 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_49 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_50 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_51 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_52 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_53 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_54 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_55 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_56 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_57 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_58 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_59 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_60 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_61 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_62 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_63 = nn.Linear(self.embed_dim, self.vocab_size)
            self.head_pre_64 = nn.Linear(self.embed_dim, self.vocab_size)
        else:
            self.norm = nn.Identity()
            self.fc_norm_1 = nn.LayerNorm(self.embed_dim)
            self.head_1 = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.fc_norm_2 = nn.LayerNorm(self.embed_dim)
            self.head_2 = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.fc_norm_3 = nn.LayerNorm(self.embed_dim)
            self.head_3 = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.fc_norm_4 = nn.LayerNorm(self.embed_dim)
            self.head_4 = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize the weights of the network
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # Initialization ONLY during pretraining
        if self.use_for_pretraining:
            trunc_normal_(self.mask_token, std=.02)
            trunc_normal_(self.head_pre_1.weight, std=.02)
            trunc_normal_(self.head_pre_2.weight, std=.02)
            trunc_normal_(self.head_pre_3.weight, std=.02)
            trunc_normal_(self.head_pre_4.weight, std=.02)
            trunc_normal_(self.head_pre_5.weight, std=.02)
            trunc_normal_(self.head_pre_6.weight, std=.02)
            trunc_normal_(self.head_pre_7.weight, std=.02)
            trunc_normal_(self.head_pre_8.weight, std=.02)
            trunc_normal_(self.head_pre_9.weight, std=.02)
            trunc_normal_(self.head_pre_10.weight, std=.02)
            trunc_normal_(self.head_pre_11.weight, std=.02)
            trunc_normal_(self.head_pre_12.weight, std=.02)
            trunc_normal_(self.head_pre_13.weight, std=.02)
            trunc_normal_(self.head_pre_14.weight, std=.02)
            trunc_normal_(self.head_pre_15.weight, std=.02)
            trunc_normal_(self.head_pre_16.weight, std=.02)
            trunc_normal_(self.head_pre_17.weight, std=.02)
            trunc_normal_(self.head_pre_18.weight, std=.02)
            trunc_normal_(self.head_pre_19.weight, std=.02)
            trunc_normal_(self.head_pre_20.weight, std=.02)
            trunc_normal_(self.head_pre_21.weight, std=.02)
            trunc_normal_(self.head_pre_22.weight, std=.02)
            trunc_normal_(self.head_pre_23.weight, std=.02)
            trunc_normal_(self.head_pre_24.weight, std=.02)
            trunc_normal_(self.head_pre_25.weight, std=.02)
            trunc_normal_(self.head_pre_26.weight, std=.02)
            trunc_normal_(self.head_pre_27.weight, std=.02)
            trunc_normal_(self.head_pre_28.weight, std=.02)
            trunc_normal_(self.head_pre_29.weight, std=.02)
            trunc_normal_(self.head_pre_30.weight, std=.02)
            trunc_normal_(self.head_pre_31.weight, std=.02)
            trunc_normal_(self.head_pre_32.weight, std=.02)
            trunc_normal_(self.head_pre_33.weight, std=.02)
            trunc_normal_(self.head_pre_34.weight, std=.02)
            trunc_normal_(self.head_pre_35.weight, std=.02)
            trunc_normal_(self.head_pre_36.weight, std=.02)
            trunc_normal_(self.head_pre_37.weight, std=.02)
            trunc_normal_(self.head_pre_38.weight, std=.02)
            trunc_normal_(self.head_pre_39.weight, std=.02)
            trunc_normal_(self.head_pre_40.weight, std=.02)
            trunc_normal_(self.head_pre_41.weight, std=.02)
            trunc_normal_(self.head_pre_42.weight, std=.02)
            trunc_normal_(self.head_pre_43.weight, std=.02)
            trunc_normal_(self.head_pre_44.weight, std=.02)
            trunc_normal_(self.head_pre_45.weight, std=.02)
            trunc_normal_(self.head_pre_46.weight, std=.02)
            trunc_normal_(self.head_pre_47.weight, std=.02)
            trunc_normal_(self.head_pre_48.weight, std=.02)
            trunc_normal_(self.head_pre_49.weight, std=.02)
            trunc_normal_(self.head_pre_50.weight, std=.02)
            trunc_normal_(self.head_pre_51.weight, std=.02)
            trunc_normal_(self.head_pre_52.weight, std=.02)
            trunc_normal_(self.head_pre_53.weight, std=.02)
            trunc_normal_(self.head_pre_54.weight, std=.02)
            trunc_normal_(self.head_pre_55.weight, std=.02)
            trunc_normal_(self.head_pre_56.weight, std=.02)
            trunc_normal_(self.head_pre_57.weight, std=.02)
            trunc_normal_(self.head_pre_58.weight, std=.02)
            trunc_normal_(self.head_pre_59.weight, std=.02)
            trunc_normal_(self.head_pre_60.weight, std=.02)
            trunc_normal_(self.head_pre_61.weight, std=.02)
            trunc_normal_(self.head_pre_62.weight, std=.02)
            trunc_normal_(self.head_pre_63.weight, std=.02)
            trunc_normal_(self.head_pre_64.weight, std=.02)
        else:
            if isinstance(self.head_1, nn.Linear):
                trunc_normal_(self.head_1.weight, std=.02)
            if isinstance(self.head_1, nn.Linear):
                self.head_1.weight.data.mul_(self.init_scale)
                self.head_1.bias.data.mul_(self.init_scale)
            if isinstance(self.head_2, nn.Linear):
                trunc_normal_(self.head_2.weight, std=.02)
            if isinstance(self.head_2, nn.Linear):
                self.head_2.weight.data.mul_(self.init_scale)
                self.head_2.bias.data.mul_(self.init_scale)
            if isinstance(self.head_3, nn.Linear):
                trunc_normal_(self.head_3.weight, std=.02)
            if isinstance(self.head_3, nn.Linear):
                self.head_3.weight.data.mul_(self.init_scale)
                self.head_3.bias.data.mul_(self.init_scale)
            if isinstance(self.head_4, nn.Linear):
                trunc_normal_(self.head_4.weight, std=.02)
            if isinstance(self.head_4, nn.Linear):
                self.head_4.weight.data.mul_(self.init_scale)
                self.head_4.bias.data.mul_(self.init_scale)

        self.apply(self._init_weights)
        self.fix_init_weight()

        
    @property
    def sample_ix_tensor(self):
        """Returns the spatial indices as a GPU-ready tensor."""
        if self.sample_channels is None:
            raise ValueError("sample_channels is not defined")
            
        # Only re-create the tensor if the list has changed
        if self.sample_channels != self._cached_sample_channels:
            indices = [self.global_channel_map[name] for name in self.sample_channels]
            # Create it once and store it
            self._cached_spat_ix_tensor = torch.tensor(
                indices, dtype=torch.long
            )
            self._cached_sample_channels = list(self.sample_channels)
            
        return self._cached_spat_ix_tensor


    def compute_spacial_embedding(self, emg):
        # emg shape: (B, C, T)
        B, C, T = emg.shape
        n_time = T // self.patch_size

        # 1. Get the spatial indices for the channels we are using
        # This maps 'c1', 'c3' etc. to [0, 2, ...] based on your global channel list
        spat_ix_base = self.sample_ix_tensor.to(emg.device)  # Shape: (C,) where C is the number of channels in the sample

        # 2. Repeat for each time patch: [c1, c1, c1 (n times), c2, c2, c2...]
        # Shape: (C * n_time)
        spat_ix_patch = torch.repeat_interleave(spat_ix_base, n_time)

        # 3. Expand to Batch size: (B, C * n_time)
        return spat_ix_patch.unsqueeze(0).expand(B, -1)


    def compute_temporal_embedding(self, emg):
        # emg shape: (B, C, T)
        B, C, T = emg.shape
        n_time = T // self.patch_size

        # 1. Create the temporal sequence for one channel
        # This aligns patches to the 'end' of the max_patches window
        temp_seq = torch.arange(
            self.max_patches - n_time,
            self.max_patches,
            device=emg.device
        ) # Shape: (n_time,)

        # 2. Repeat the sequence for all channels: [0, 1, 2, 0, 1, 2...]
        # Shape: (C * n_time)
        temp_ix_all_channels = temp_seq.repeat(C)

        # 3. Expand to Batch size: (B, C * n_time)
        return temp_ix_all_channels.unsqueeze(0).expand(B, -1)


    # Function to initialize the weights of the network
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Get number of layers from the transformer
    def get_num_layers(self):
        return len(self.blocks)

    # Get classification head
    def get_classifier(self):
        return self.head

    # Reset the classification head
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc_norm = nn.LayerNorm(self.embed_dim * 4) # multiply dim by 4 for concat [x1,x2,x3,x4] 
        self.head = nn.Linear(self.embed_dim * 4, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def forward(self, x, return_patch_tokens=False, bool_masked_pos=None, use_for_pretraining=False, branch_idx=0):
        '''
        Args:
            x: (batch_size, C, T)

        Returns:
            if self.return_patch_embeddings:
                e: (batch_size, num_temporal_patches, C, embed_dim)
        '''
        temporal_embedding_ix = self.compute_temporal_embedding(x)  # (batch_size, n_patches)
        spatial_embedding_ix = self.compute_spacial_embedding(x)  # (batch_size, n_patches)

        B, C, T = x.shape
        x = x.view(B, C, -1, self.patch_size)  # (B, C, temporal_patches, patch_size)

        if (self.use_as_encoder):
            x1, x2, x3, x4 = self.patch_embed(x)
            x = x1
        else:
            if (branch_idx==0):
                x = self.patch_embed_1(x)
            elif (branch_idx==1):
                x = self.patch_embed_2(x)
            elif (branch_idx==2):
                x = self.patch_embed_3(x)
            elif (branch_idx==3):
                x = self.patch_embed_4(x)
            
        batch_size, seq_len, _ = x.size()

        # Concatenate the cls token - Legacy code
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # ONLY in Pre-Training - Masking technique used in LaBraM
        if (use_for_pretraining or bool_masked_pos is not None):
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            symmetric_bool_masked_pos = ~bool_masked_pos
            w_symmetric = symmetric_bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            
            x_symmetric_1 = x1 * (1 - w_symmetric) + mask_token * w_symmetric
            x1 = x1 * (1 - w) + mask_token * w
            x_symmetric_1 = torch.cat((cls_tokens, x_symmetric_1), dim=1)

            x_symmetric_2 = x2 * (1 - w_symmetric) + mask_token * w_symmetric
            x2 = x2 * (1 - w) + mask_token * w
            x_symmetric_2 = torch.cat((cls_tokens, x_symmetric_2), dim=1)

            x_symmetric_3 = x3 * (1 - w_symmetric) + mask_token * w_symmetric
            x3 = x3 * (1 - w) + mask_token * w
            x_symmetric_3 = torch.cat((cls_tokens, x_symmetric_3), dim=1)

            x_symmetric_4 = x4 * (1 - w_symmetric) + mask_token * w_symmetric
            x4 = x4 * (1 - w) + mask_token * w
            x_symmetric_4 = torch.cat((cls_tokens, x_symmetric_4), dim=1)

        if (self.use_as_encoder):
            x1 = torch.cat((cls_tokens, x1), dim=1)
            x2 = torch.cat((cls_tokens, x2), dim=1)
            x3 = torch.cat((cls_tokens, x3), dim=1)
            x4 = torch.cat((cls_tokens, x4), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # Pad the spatial_embedding_ix - spatial_embedding_ix: (batch, n_patches), spatial_embedding: (n_electrodes + 1, embed_dim)
        spatial_embedding_ix = F.pad(input=spatial_embedding_ix, pad=(1, 0), mode='constant', value=0)  # for cls token (batch_size, n_patches + 1)
        # Gets the corresponding pos_embed
        spatial_embedding = self.pos_embed[spatial_embedding_ix.reshape(-1), :]  # (batch_size * (n_patches + 1), embed_dim)
        spatial_embedding = spatial_embedding.reshape(spatial_embedding_ix.shape[0], spatial_embedding_ix.shape[1], spatial_embedding.shape[-1])  # (batch_size, n_patches + 1, embed_dim)
        
        if (self.use_as_encoder):
            x1 = x1 + spatial_embedding
            x2 = x2 + spatial_embedding
            x3 = x3 + spatial_embedding
            x4 = x4 + spatial_embedding
        else:
            x = x + spatial_embedding

        #  temporal_embedding_ix: (batch, n_patches), temporal_embedding: (n_patches, embed_dim)
        temporal_embedding = self.time_embed[temporal_embedding_ix.reshape(-1), :]  # (batch_size * (n_patches), embed_dim)
        temporal_embedding = temporal_embedding.reshape(temporal_embedding_ix.shape[0], temporal_embedding_ix.shape[1], temporal_embedding.shape[-1])  # (batch_size, n_patches, embed_dim)
        
        if (self.use_as_encoder):
            x1[:, 1:, :] += temporal_embedding
            x1 = self.pos_drop(x1)
            x2[:, 1:, :] += temporal_embedding
            x2 = self.pos_drop(x2)
            x3[:, 1:, :] += temporal_embedding
            x3 = self.pos_drop(x3)
            x4[:, 1:, :] += temporal_embedding
            x4 = self.pos_drop(x4)
        else:
            x[:, 1:, :] += temporal_embedding
            x = self.pos_drop(x)
            
        if (self.use_as_encoder):
            # Pass the transformer blocks
            for i, x in enumerate([x1, x2, x3, x4]):
                for blk in self.blocks:
                    x = blk(x)
                    
                if (self.use_for_pretraining or bool_masked_pos is not None):
                    x = self.norm_pre(x)
                else:
                    x = self.norm(x)
                # All except cls token
                if i == 0:
                    x1 = x[:, 1:, :]
                elif i == 1:
                    x2 = x[:, 1:, :]
                elif i == 2:
                    x3 = x[:, 1:, :]
                else:
                    x4 = x[:, 1:, :]
        else:
            # Pass the transformer blocks
            for blk in self.blocks:
                x = blk(x)
            if (self.use_for_pretraining or bool_masked_pos is not None):
                x = self.norm_pre(x)
            else:
                x = self.norm(x)

            # All except cls token
            x = x[:, 1:, :]

        if (use_for_pretraining or bool_masked_pos is not None):
            for i, x_symmetric in enumerate([x_symmetric_1, x_symmetric_2, x_symmetric_3, x_symmetric_4]):
                x_symmetric += spatial_embedding
                x_symmetric[:, 1:, :] += temporal_embedding
                x_symmetric = self.pos_drop(x_symmetric)
                for blk in self.blocks:
                    x_symmetric = blk(x_symmetric)
                x_symmetric = self.norm_pre(x_symmetric)
                # All except cls token
                if i == 0:
                    x_symmetric_1 = x_symmetric[:, 1:, :]
                elif i == 1:
                    x_symmetric_2 = x_symmetric[:, 1:, :]
                elif i == 2:
                    x_symmetric_3 = x_symmetric[:, 1:, :]
                else:
                    x_symmetric_4 = x_symmetric[:, 1:, :]

        # ONLY in Pre-Training
        if (use_for_pretraining or bool_masked_pos is not None):
        
            logits = torch.stack([
                # x1 heads
                self.head_pre_1(x1[bool_masked_pos]),
                self.head_pre_2(x1[bool_masked_pos]),
                self.head_pre_3(x1[bool_masked_pos]),
                self.head_pre_4(x1[bool_masked_pos]),
                self.head_pre_5(x1[bool_masked_pos]),
                self.head_pre_6(x1[bool_masked_pos]),
                self.head_pre_7(x1[bool_masked_pos]),
                self.head_pre_8(x1[bool_masked_pos]),

                # x2 heads
                self.head_pre_9(x2[bool_masked_pos]),
                self.head_pre_10(x2[bool_masked_pos]),
                self.head_pre_11(x2[bool_masked_pos]),
                self.head_pre_12(x2[bool_masked_pos]),
                self.head_pre_13(x2[bool_masked_pos]),
                self.head_pre_14(x2[bool_masked_pos]),
                self.head_pre_15(x2[bool_masked_pos]),
                self.head_pre_16(x2[bool_masked_pos]),

                # x3 heads
                self.head_pre_17(x3[bool_masked_pos]),
                self.head_pre_18(x3[bool_masked_pos]),
                self.head_pre_19(x3[bool_masked_pos]),
                self.head_pre_20(x3[bool_masked_pos]),
                self.head_pre_21(x3[bool_masked_pos]),
                self.head_pre_22(x3[bool_masked_pos]),
                self.head_pre_23(x3[bool_masked_pos]),
                self.head_pre_24(x3[bool_masked_pos]),

                # x4 heads
                self.head_pre_25(x4[bool_masked_pos]),
                self.head_pre_26(x4[bool_masked_pos]),
                self.head_pre_27(x4[bool_masked_pos]),
                self.head_pre_28(x4[bool_masked_pos]),
                self.head_pre_29(x4[bool_masked_pos]),
                self.head_pre_30(x4[bool_masked_pos]),
                self.head_pre_31(x4[bool_masked_pos]),
                self.head_pre_32(x4[bool_masked_pos]),
            ], dim=0)
            
            logits_symmetric = torch.stack([
                # x_symmetric_1 heads
                self.head_pre_1(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_2(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_3(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_4(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_5(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_6(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_7(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_8(x_symmetric_1[~bool_masked_pos]),

                # x_symmetric_2 heads
                self.head_pre_9(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_10(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_11(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_12(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_13(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_14(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_15(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_16(x_symmetric_2[~bool_masked_pos]),

                # x_symmetric_3 heads
                self.head_pre_17(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_18(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_19(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_20(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_21(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_22(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_23(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_24(x_symmetric_3[~bool_masked_pos]),

                # x_symmetric_4 heads
                self.head_pre_25(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_26(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_27(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_28(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_29(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_30(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_31(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_32(x_symmetric_4[~bool_masked_pos]),
            ], dim=0)
        
            return  logits, logits_symmetric
        else:
            # ONLY in RVQ
            if return_patch_tokens:
                if (self.use_as_encoder):
                    return self.head_1(self.fc_norm_1(x1)), self.head_2(self.fc_norm_2(x2)), self.head_3(self.fc_norm_3(x3)), self.head_4(self.fc_norm_4(x4)), _
                else:
                    if (branch_idx==0):
                        return self.head_1(self.fc_norm_1(x)), _
                    elif (branch_idx==1):
                        return self.head_2(self.fc_norm_2(x)), _
                    elif (branch_idx==2):
                        return self.head_3(self.fc_norm_3(x)), _
                    elif (branch_idx==3):
                        return self.head_4(self.fc_norm_4(x)), _
                    
            elif self.return_patch_embeddings:
                x = torch.concat([x1,x2,x3,x4], dim=-1)
                # Reshape to (batch_size, num_temporal_patches, C, embed_dim * 4)
                num_temporal_patches = T // self.patch_size
                x = x.view(B, C, num_temporal_patches, self.embed_dim * 4)
                return x
            else:
                # ONLY in Fine-Tune
                x = torch.concat([x1,x2,x3,x4], dim=-1)
                return self.head(self.fc_norm(x.mean(1))), _