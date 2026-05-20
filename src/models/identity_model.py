""" Module identity_model.py

Sanity check model that simply returns the input as output. Useful for debugging and benchmarking.

Usage:


Authors:
    Will Raftery

Date:
    05/05/2026
"""

import torch 
import torch.nn as nn


class IdentityModel(nn.Module):

    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        """
        Simple adjust shape to match expected output shape (B, C, N, embed_dim)

        Args:
            x: Input tensor of shape (B, C, T)
        """
        return x.unsqueeze(2)