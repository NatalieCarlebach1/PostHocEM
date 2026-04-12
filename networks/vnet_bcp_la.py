"""VNetBCP_LA — VNet variant matching BCP's original LA pretrained checkpoint.

This mirrors the architecture of the BCP repo VNet exactly
(DeepMed-Lab-ECNU/BCP, code/networks/VNet.py), so that checkpoints such as
``LA_10.pth`` load cleanly with ``strict=True``.

Key differences vs our ``networks/vnet.VNet``:
  * Uses nested ``encoder`` / ``decoder`` submodules (not flat block_* attrs).
  * Decoder ends with ``block_nine`` + ``out_conv`` (flat), not a ``head``
    ``nn.Sequential``.
  * Extra SSL heads at the top level: ``projection_head``, ``prediction_head``,
    ``contrastive_class_selector_{0,1}``, ``contrastive_class_selector_memory{0,1}``,
    plus a parameter-free ``pool`` (nn.MaxPool3d).
  * Defaults to ``normalization='batchnorm'`` and ``has_dropout=True``, matching
    the LA training recipe in the BCP paper.

Do NOT modify ``networks/vnet.py`` — that one is tuned for Pancreas and stays
as-is.
"""

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks (verbatim from BCP VNet.py, trimmed of unused code paths)
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3,
                 padding=1, normalization='none'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out,
                                 kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f'unknown normalization: {normalization}')
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,
                 normalization='none'):
        super().__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, stride,
                         padding=padding, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f'unknown normalization: {normalization}')
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,
                 normalization='none'):
        super().__init__()
        ops = [nn.ConvTranspose3d(n_filters_in, n_filters_out, stride,
                                  padding=padding, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f'unknown normalization: {normalization}')
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder (nested, matching BCP checkpoint key prefixes)
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='batchnorm',
                 has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout

        self.block_one    = ConvBlock(1, n_channels,     n_filters,      normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters,     n_filters * 2,  normalization=normalization)

        self.block_two    = ConvBlock(2, n_filters * 2,  n_filters * 2,  normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4,  normalization=normalization)

        self.block_three    = ConvBlock(3, n_filters * 4,  n_filters * 4,  normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4,  n_filters * 8,  normalization=normalization)

        self.block_four    = ConvBlock(3, n_filters * 8,  n_filters * 8,  normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8,  n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class _Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='batchnorm',
                 has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six    = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven    = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight    = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up


# ---------------------------------------------------------------------------
# Top-level VNetBCP_LA
# ---------------------------------------------------------------------------


class VNetBCP_LA(nn.Module):
    """VNet matching BCP's LA pretrained checkpoint layout.

    ``forward(x)`` returns ``(logits,)`` — a 1-tuple — so that call sites using
    the BCP convention ``net(x)[0]`` (e.g. ``train_posthoc_em.py``) work
    unchanged.
    """

    def __init__(self, n_channels=1, n_classes=2, n_filters=16,
                 normalization='batchnorm', has_dropout=True):
        super().__init__()

        self.encoder = _Encoder(
            n_channels=n_channels, n_filters=n_filters,
            normalization=normalization, has_dropout=has_dropout,
        )
        self.decoder = _Decoder(
            n_classes=n_classes, n_filters=n_filters,
            normalization=normalization, has_dropout=has_dropout,
        )

        # --- SSL heads present in the BCP LA checkpoint ---
        # These are part of the original BCP training graph (InfoNCE-style
        # contrastive branch). We keep them with identical structure so the
        # checkpoint loads with strict=True. They are unused at PEM time.
        dim_in = 16
        feat_dim = 32

        # Parameter-free — not actually present in the state_dict, but matches
        # the original module tree so we don't accidentally introduce extras.
        self.pool = nn.MaxPool3d(3, stride=2)

        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        for class_c in range(2):
            self.add_module(
                f'contrastive_class_selector_{class_c}',
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim),
                    nn.BatchNorm1d(feat_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(feat_dim, 1),
                ),
            )
        for class_c in range(2):
            self.add_module(
                f'contrastive_class_selector_memory{class_c}',
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim),
                    nn.BatchNorm1d(feat_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(feat_dim, 1),
                ),
            )

    # ---- public API ----

    def forward(self, x):
        features = self.encoder(x)
        out_seg, _x8_up = self.decoder(features)
        # Return a 1-tuple so callers using `net(x)[0]` (BCP convention,
        # train_posthoc_em.py) keep working.
        return (out_seg,)

    def freeze_bn(self):
        """Put every BatchNorm* layer in eval mode.

        PEM fine-tunes with batch_size=2, which makes BN running stats extremely
        noisy. Calling this after ``.train()`` keeps BN frozen to the LA
        checkpoint's statistics while still letting conv weights update.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.SyncBatchNorm)):
                m.eval()
