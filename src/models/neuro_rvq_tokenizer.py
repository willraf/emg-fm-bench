""" Module neuro_rvq_tokeniser.py

Contains implementation of the NeuroRVQTokenizer class

    This tokeniser expects a tuncated version of the full config (cfg.model.tokeniser)
    This is different from other parts of the repo, where the full config is normally used

Source:
    https://github.com/KonstantinosBarmpas/NeuroRVQ/tree/main

Usage:
    tokeniser = NeuroRVQTokenizer(cfg.model.tokeniser)

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
import os
import sys

from src.models.neuro_rvq.NeuroRVQ_modules import trunc_normal_
from src.models.neuro_rvq.RVQ import ResidualVectorQuantization
from src.models.neuro_rvq_fm import NeuroRVQFM

def inverse_fft_cos_sin(fft_amp, fft_sin_pha, fft_cos_pha):
    """
    Inverse FFT function using sin and cos
    :param fft_amp: amplitude
    :param fft_sin_pha: sine
    :param fft_cos_pha: cosine
    :return: inverse fft in time
    """
    imag = fft_amp * fft_sin_pha
    real = fft_amp * fft_cos_pha
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft(fft_y)
    return y


class NeuroRVQTokenizer(nn.Module):
    """
    NeuroRVQ Tokenizer

    Input shape: 
        [B, C, T]: emg signal with B batch size, C channels and T time points
        [B, N]: temporal embedding indices
        [B, N]: spatial embedding indices
    Output:
        [B, N, patch_len]: patched representation of the input signal, where N is the number of patches (N = num_time_points * C)
        [B, N, patch_len]: reconstructed signal from the quantized representation
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.patch_size = cfg.encoder.patch_size
        self.code_dim = cfg.code_dim
        decoder_out_dim = cfg.decoder_out_dim
        n_code = cfg.n_code

        code_dim = self.code_dim

        encoder_config = cfg.encoder
        decoder_config = cfg.decoder

        encoder_config.vocab_size = n_code
        decoder_config.vocab_size = n_code

        self.encoder = NeuroRVQFM(encoder_config)
        self.decoder = NeuroRVQFM(decoder_config)

        self.quantize_1 = ResidualVectorQuantization(num_quantizers = 16,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_2 = ResidualVectorQuantization(num_quantizers = 16,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_3 = ResidualVectorQuantization(num_quantizers = 16,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_4 = ResidualVectorQuantization(num_quantizers = 16,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )

        # Output dimension of the decoder layer
        self.decoder_out_dim = decoder_out_dim

        # Encoding head after the encoder transformer
        self.encode_task_layer_1 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_2 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_3 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_4 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_1.apply(self._init_weights)
        self.encode_task_layer_2.apply(self._init_weights)
        self.encode_task_layer_3.apply(self._init_weights)
        self.encode_task_layer_4.apply(self._init_weights)

        # Decoding heads after the decoder transformer
        self.decode_task_layer_amplitude = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.GELU(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle_sin = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
            nn.Tanh()
        )
        self.decode_task_layer_angle_cos = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
            nn.Tanh()
        )

        # Initialize model weights
        self.decode_task_layer_amplitude.apply(self._init_weights)
        self.decode_task_layer_angle_sin.apply(self._init_weights)
        self.decode_task_layer_angle_cos.apply(self._init_weights)

        # MSE loss function
        self.loss_fn = F.mse_loss

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, temporal_embedding_ix, spatial_embedding_ix):
        quantize, code_ind, loss, usage_ratios = self.encode(data, temporal_embedding_ix, spatial_embedding_ix)
        # Convert [8, B*P] to [8, B, P]
        code_inds = [code_ind_ix.view(8, data.shape[0], -1) for code_ind_ix in code_ind]
        # Stack all codebooks [4, 8, B, P]
        stacked_code_inds = torch.stack(code_inds, dim=0)
        quantize_vecs = [rearrange(quantize_ix, 'b d a c -> b (a c) d').contiguous() for quantize_ix in quantize]
        output = {}
        output['token'] = stacked_code_inds
        output['input_img'] = data
        output['quantize'] = quantize_vecs
        return output

    def encode(self, x, temporal_embedding_ix, spatial_embedding_ix):
        batch_size, n, a, t = x.shape

        encoder_features_1, encoder_features_2, encoder_features_3, encoder_features_4, _ = self.encoder(x, temporal_embedding_ix=temporal_embedding_ix, spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True)
            
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features_1 = self.encode_task_layer_1(encoder_features_1.type_as(self.encode_task_layer_1[-1].weight))
            to_quantizer_features_2 = self.encode_task_layer_2(encoder_features_2.type_as(self.encode_task_layer_2[-1].weight))
            to_quantizer_features_3 = self.encode_task_layer_3(encoder_features_3.type_as(self.encode_task_layer_3[-1].weight))
            to_quantizer_features_4 = self.encode_task_layer_4(encoder_features_4.type_as(self.encode_task_layer_4[-1].weight))
            
        N = to_quantizer_features_1.shape[1]
        h, w = n, N // n
            
        # reshape tokens to feature maps for patch embed in decoder
        to_quantizer_features_1 = rearrange(to_quantizer_features_1, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_1, code_ind_1, loss_1, usage_ratios_1  = self.quantize_1(to_quantizer_features_1)
            
        to_quantizer_features_2 = rearrange(to_quantizer_features_2, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_2, code_ind_2, loss_2, usage_ratios_2  = self.quantize_2(to_quantizer_features_2)
            
        to_quantizer_features_3 = rearrange(to_quantizer_features_3, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_3, code_ind_3, loss_3, usage_ratios_3  = self.quantize_3(to_quantizer_features_3)
            
        to_quantizer_features_4 = rearrange(to_quantizer_features_4, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_4, code_ind_4, loss_4, usage_ratios_4  = self.quantize_4(to_quantizer_features_4)
            
        #  Combine loss
        loss = loss_1 + loss_2 + loss_3 + loss_4

        return [quantize_1, quantize_2, quantize_3, quantize_4], [code_ind_1, code_ind_2, code_ind_3, code_ind_4], loss, [usage_ratios_1, usage_ratios_2, usage_ratios_3, usage_ratios_4]

    def decode(self, quantize, temporal_embedding_ix, spatial_embedding_ix):
        
        for i, quantize_i in enumerate(quantize):
            if i == 0:
                decoder_features_1, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 0)
            elif i == 1:
                decoder_features_2, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 1)
            elif i == 2:
                decoder_features_3, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 2)
            else:
                decoder_features_4, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 3)
        decoder_features = torch.cat([decoder_features_1, decoder_features_2, decoder_features_3, decoder_features_4], dim=2)
        
        # Reconstruct Amplitude, Sine and Cosine
        rec_amplitude = self.decode_task_layer_amplitude(decoder_features)
        rec_angle_sin = self.decode_task_layer_angle_sin(decoder_features)
        rec_angle_cos = self.decode_task_layer_angle_cos(decoder_features)

        return rec_amplitude, rec_angle_sin, rec_angle_cos

    def get_codebook_indices(self, x, temporal_embedding_ix, spatial_embedding_ix):
        return self.get_tokens(x, temporal_embedding_ix, spatial_embedding_ix)['token']
        
    def calculate_phase_loss(self, rec_sin, target_sin, rec_cos, target_cos):
        target_sin = rearrange(target_sin, 'b n a c -> b (n a) c').contiguous()
        target_cos = rearrange(target_cos, 'b n a c -> b (n a) c').contiguous()
        rec = torch.stack((rec_cos, rec_sin), dim=-1)
        target = torch.stack((target_cos, target_sin), dim=-1)
        #  Cosine Similarity for direction and Enforcing Magnitude loss
        phase_loss = 1.0 - F.cosine_similarity(rec, target, dim=-1).mean() + 0.1 * ((rec_sin**2 + rec_cos**2 - 1) ** 2).mean()
        return phase_loss

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c').contiguous()
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def calculate_signal_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c').contiguous()
        rec = rearrange(rec, 'b n a c -> b (n a) c').contiguous()
        mse = self.loss_fn(rec, target)
        return mse

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(torch.var(x, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8))
        x = (x - mean) / std
        return x, mean, std

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix):
        """
        x: shape [B, N, T]
        """
        x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size).contiguous()
        x_fft = torch.fft.fft(x, dim=-1)

        # Get the log ampltitude
        amplitude = torch.abs(x_fft)
        amplitude = torch.log1p(amplitude)
        amplitude, amp_mean, amp_std = self.std_norm(amplitude)
        
        # Get the sine / cosine of the phase
        angle = torch.angle(x_fft)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Encoding and Quantize
        quantize, code_ind, code_loss, usage_ratios = self.encode(x, temporal_embedding_ix, spatial_embedding_ix)

        # Decoding
        xrec_amp, xrec_angle_sin, xrec_angle_cos = self.decode(quantize, temporal_embedding_ix, spatial_embedding_ix)
        
        # Reconstruct raw signal from amplitude and sine / cosine
        ustd_xrec = (rearrange(xrec_amp, 'B N (A T) -> B N A T', T=self.patch_size).contiguous() * amp_std) + amp_mean  # unstandardize
        ustd_xrec = torch.expm1(ustd_xrec)
        ustd_xrec = rearrange(ustd_xrec, 'b n a c -> b (n a) c').contiguous()
        xrec_signal = torch.real(inverse_fft_cos_sin(ustd_xrec, xrec_angle_sin, xrec_angle_cos))

        # Standardize sample and Reconstructed signal for MSE
        std_x, _, _ = self.std_norm(x)
        std_xrec_signal, _, _ = self.std_norm(rearrange(xrec_signal, 'B N (A T) -> B N A T', T=self.patch_size).contiguous())
        signal_rec_loss = self.calculate_signal_rec_loss(std_xrec_signal, std_x)

        # Calculate losses from decoder
        rec_amplitude_loss = self.calculate_rec_loss(xrec_amp, amplitude)
        phase_loss = self.calculate_phase_loss(xrec_angle_sin, sin_angle, xrec_angle_cos, cos_angle)

        # Total loss
        loss = code_loss + rec_amplitude_loss + phase_loss + signal_rec_loss

        std_x = std_x.view(std_x.size(0), -1, 1,std_x.size(-1)).squeeze(2)
        std_xrec_signal = std_xrec_signal.squeeze(2)

        return std_x, std_xrec_signal