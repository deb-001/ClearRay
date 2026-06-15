"""
ESRGAN + XAAHA Model Architecture
Extracted from ESRGAN_XAAHA_Training.ipynb for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── XAAHA Attention Mechanism ───────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CARP(nn.Module):
    """Stage 1: Coarse Anatomical Region Prioritization."""
    def __init__(self, channels):
        super().__init__()
        self.saliency_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_in):
        m_coarse = self.saliency_net(f_in)
        a_coarse = self.sigmoid(m_coarse)
        return f_in * (1 + a_coarse)  # Eq. 1


class FDAF(nn.Module):
    """Stage 2: Fine-Grained Detail & Anomaly Focus."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.conv_dilated = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False)
        self.texture_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.local_norm = nn.InstanceNorm2d(channels, affine=False)
        self.se = SqueezeExcitation(channels, reduction)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_coarse_attended):
        t3 = self.conv3(f_coarse_attended)
        t5 = self.conv5(f_coarse_attended)
        td = self.conv_dilated(f_coarse_attended)
        f_texture = self.texture_fuse(torch.cat([t3, t5, td], dim=1))
        f_contrast = self.local_norm(f_texture)
        f_channel_attended = self.se(f_contrast)
        a_fine = self.spatial_att(f_channel_attended)
        return f_coarse_attended * (1 + a_fine)  # Eq. 2


class XAAHA(nn.Module):
    """X-Ray Anatomy-Aware Hierarchical Attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.carp = CARP(channels)
        self.fdaf = FDAF(channels, reduction)

    def forward(self, x):
        return self.fdaf(self.carp(x))


# ─── RRDB Blocks ─────────────────────────────────────────────────────────────

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        c, g = in_channels, growth_rate
        self.conv1 = nn.Conv2d(c, g, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c + g, g, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(c + 2*g, g, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(c + 3*g, g, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(c + 4*g, c, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5


class RRDB(nn.Module):
    def __init__(self, channels, growth_rate=32, residual_scaling=0.2):
        super().__init__()
        self.db1 = DenseBlock(channels, growth_rate)
        self.db2 = DenseBlock(channels, growth_rate)
        self.db3 = DenseBlock(channels, growth_rate)
        self.beta = residual_scaling

    def forward(self, x):
        out1 = self.db1(x) * self.beta + x
        out2 = self.db2(out1) * self.beta + out1
        out3 = self.db3(out2) * self.beta + out2
        return out3


# ─── Generator ────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_rrdb=23,
                 growth_rate=32, scale_factor=4, xaaha_reduction=16):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.xaaha = XAAHA(num_features, xaaha_reduction)
        self.body = nn.Sequential(*[RRDB(num_features, growth_rate) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_last = nn.Conv2d(num_features, in_channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = self.xaaha(feat)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk
        feat = self.lrelu(self.upsample1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.upsample2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
