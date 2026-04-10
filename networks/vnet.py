"""VNet — 3D volumetric segmentation network.
Copied from BCP (DeepMed-Lab-ECNU/BCP) with minor cleanup.
Original: Milletari et al., "V-Net: Fully Convolutional Neural Networks for
Volumetric Medical Image Segmentation", 3DV 2016.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(16, n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super().__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, stride, stride=stride, padding=0)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super().__init__()
        ops = [nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, stride=stride, padding=0)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16,
                 normalization='instancenorm', has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout

        self.block_one    = ConvBlock(1, n_channels,    n_filters,      normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters,    n_filters * 2,  normalization=normalization)

        self.block_two    = ConvBlock(2, n_filters * 2, n_filters * 2,  normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4,  normalization=normalization)

        self.block_three    = ConvBlock(3, n_filters * 4,  n_filters * 4,  normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4,  n_filters * 8,  normalization=normalization)

        self.block_four    = ConvBlock(3, n_filters * 8,  n_filters * 8,  normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8,  n_filters * 16, normalization=normalization)

        self.block_five    = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8,  normalization=normalization)

        self.block_six    = ConvBlock(3, n_filters * 8,  n_filters * 8,  normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8,  n_filters * 4,  normalization=normalization)

        self.block_seven    = ConvBlock(3, n_filters * 4,  n_filters * 4,  normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4,  n_filters * 2,  normalization=normalization)

        self.block_eight    = ConvBlock(2, n_filters * 2,  n_filters * 2,  normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2,  n_filters,      normalization=normalization)

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

        head_layers = [ConvBlock(1, n_filters, n_filters, normalization=normalization)]
        if has_dropout:
            head_layers.append(nn.Dropout3d(p=0.5))
        head_layers.append(nn.Conv3d(n_filters, n_classes, 1))
        self.head = nn.Sequential(*head_layers)

    def encoder(self, x):
        x1    = self.block_one(x)
        x1_dw = self.block_one_dw(x1)
        x2    = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3    = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4    = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5    = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)
        return x1, x2, x3, x4, x5

    def decoder(self, x1, x2, x3, x4, x5):
        x = self.block_five_up(x5) + x4
        x = self.block_six_up(self.block_six(x)) + x3
        x = self.block_seven_up(self.block_seven(x)) + x2
        x = self.block_eight_up(self.block_eight(x)) + x1
        return self.head(x)

    def forward(self, x):
        feats = self.encoder(x)
        out   = self.decoder(*feats)
        return out,   # tuple for BCP-compatible unpacking: net(x)[0]
