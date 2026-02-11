""" Module soundstream.py

References:
    Soundstream main model:
    https://github.com/haydenshively/SoundStream/tree/master

Usage:


Authors:
    Will Raftery

Date:
    08/10/2025
"""

from functools import reduce
from typing import Literal

import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from src.models.soundstream.encoder import Encoder
from src.models.soundstream.decoder import Decoder


class SoundStream(nn.Module):
    def __init__(self, cfg):
        super(SoundStream, self).__init__()
        in_channels = cfg.data.num_channels
        n_q = cfg.model.n_quantizers
        codebook_size = cfg.model.codebook_size
        D = cfg.model.embedding_dim
        C = cfg.model.channel_multiplier
        strides = tuple(cfg.model.strides)

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(in_channels=in_channels, C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=D,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )
        self.decoder = Decoder(out_channels=in_channels, C=C, D=D, strides=strides)

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
        ):
        # x: batch_size x in_channels x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x in_channels x (T / 1)
        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            return quantized
        
        if mode == 'decode':
            o = self.decoder(x.permute((0,2,1)))
            return o