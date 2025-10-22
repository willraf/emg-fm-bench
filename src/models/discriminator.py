""" Module discriminator.py

Discriminator for SoundStream

Usage:


Authors:
    Will Raftery

Date:
    13/10/2025
"""

import torch.nn as nn
from torch.nn.utils import weight_norm


class STFTDiscriminator(nn.Module):
    """
    Input: (B, in_channels, T, F)
    Output: List of feature maps from each layer 
        - (B, 32, T, F)
        - (B, 32, T, F) ELU
        - (B, 2C, T, F/2)
        - (B, 4C, T/2, F/4)
        - (B, 4C, T/2, F/8)
        - (B, 8C, T/4, F/16)
        - (B, 8C, T/4, F/32)
        - (B, 16C, T/8, F/64)
        - (B, 1, T/8, 1)
    """
    def __init__(self, in_channels, C, frequency_bins):
        super(STFTDiscriminator, self).__init__()

        out_kernel_size = (1, frequency_bins // 64)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(7, 7),
                padding="same"
            ),
            nn.ELU(),
            ResidualUnit(
                in_channels=32,
                N=C,
                out_channels=2*C,
                strides=(1, 2),
            ),
            ResidualUnit(
                in_channels=2*C,
                N=2*C,
                out_channels=4*C,
                strides=(2, 2),
            ),
            ResidualUnit(
                in_channels=4*C,
                N=4*C,
                out_channels=4*C,
                strides=(1, 2),
            ),
            ResidualUnit(
                in_channels=4*C,
                N=4*C,
                out_channels=8*C,
                strides=(2, 2),
            ),
            ResidualUnit(
                in_channels=8*C,
                N=8*C,
                out_channels=8*C,
                strides=(1, 2),
            ),
            ResidualUnit(
                in_channels=8*C,
                N=8*C,
                out_channels=16*C,
                strides=(2, 2),
            ),
            # Esentially just a fc layer
            nn.Conv2d(
                in_channels=16*C,
                out_channels=1,
                kernel_size=out_kernel_size,
            )
        )

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class ResidualUnit(nn.Module):
    """
    
    Input: (B, in_channels, T, F)
    Output: (B, out_channels, T//s_t, F//s_f)
    """
    def __init__(
            self,
            in_channels,
            N,              # Intermediate channels
            out_channels,
            strides,
        ):
        super(ResidualUnit, self).__init__()
        s_t, s_f = strides
        kernel_size = (s_t+2, s_f+2)
        padding = ((s_t+1)//2, (s_f+1)//2)  

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            ),
            nn.ELU(),   
        )

        # 1x1 conv for skip connection to match dimensions
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1), 
            stride=strides,
        )

    def forward(self, x):
        out = self.layers(x)
        sc = self.skip_connection(x)
        sc = sc[:, :, :out.shape[2], :out.shape[3]]  # Crop skip connection to match output shape
        return sc + out

##### Wave Discriminator #####
# http://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L9

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WaveDiscriminator(nn.Module):
    def __init__(self, in_channels, num_D, ndf, n_layers, downsampling_factor):
        """
        Args:
            num_D: Number of discriminators at different scales (3)
            ndf: Number of channels after first convolution (16)
            n_layers: Number of convolutional layers (4)
            downsampling_factor: Downsampling factor for each layer (4)
        """
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = DiscriminatorBlock(
                in_channels, ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, ndf, n_layers, downsampling_factor):
        """
        Args:
            in_channels: Number of input channels
            ndf: Number of channels after first convolution (16)
            n_layers: Number of convolutional layers (4)
            downsampling_factor: Downsampling factor for each layer (4)
        """
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(in_channels, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


