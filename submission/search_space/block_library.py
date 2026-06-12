import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Normalization helper ───────────────────────────────────────────────────────

def make_norm(channels: int, norm_type: str = 'batch') -> nn.Module:
    if norm_type == 'group':
        # find largest divisor of channels that is ≤ 32 and ≥ 1
        for g in [32, 16, 8, 4, 2, 1]:
            if channels % g == 0:
                return nn.GroupNorm(g, channels)
    return nn.BatchNorm2d(channels)


def make_act(act_type: str = 'relu') -> nn.Module:
    if act_type == 'silu':
        return nn.SiLU()
    if act_type == 'gelu':
        return nn.GELU()
    return nn.ReLU(inplace=True)


# ── SE squeeze-excitation ──────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ── DropPath (stochastic depth) ───────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, rate: float = 0.0):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        keep = 1 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep
        return x * mask.float() / keep


# ── Block implementations ──────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_out, norm_type),
            make_act(act_type),
        )
        self.se = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.dp(self.se(self.conv(x))) + self.skip(x)


class SepConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.dw = nn.Conv2d(c_in, c_in, kernel, stride=stride, padding=pad,
                            dilation=dilation, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.norm = make_norm(c_out, norm_type)
        self.act  = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        out = self.act(self.norm(self.pw(self.dw(x))))
        return self.dp(self.se(out)) + self.skip(x)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_out, norm_type),
            make_act(act_type),
            nn.Conv2d(c_out, c_out, kernel, padding=pad, dilation=dilation, bias=False),
            make_norm(c_out, norm_type),
        )
        self.act  = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.act(self.dp(self.se(self.body(x))) + self.skip(x))


class MBConvBlock(nn.Module):
    """Inverted residual / MobileNetV2-style block."""
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 expansion=4, norm_type='batch', act_type='relu',
                 se=False, se_ratio=0.25, drop_path=0.0, **kw):
        super().__init__()
        c_mid = max(c_in, int(c_in * expansion))
        pad = (kernel // 2) * dilation
        layers = []
        if c_mid != c_in:
            layers += [nn.Conv2d(c_in, c_mid, 1, bias=False),
                       make_norm(c_mid, norm_type), make_act(act_type)]
        layers += [
            nn.Conv2d(c_mid, c_mid, kernel, stride=stride, padding=pad,
                      dilation=dilation, groups=c_mid, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_out, 1, bias=False),
            make_norm(c_out, norm_type),
        ]
        self.body = nn.Sequential(*layers)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.dp(self.se(self.body(x))) + self.skip(x)


class BottleneckBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 expansion=4, norm_type='batch', act_type='relu',
                 se=False, se_ratio=0.25, drop_path=0.0, **kw):
        super().__init__()
        c_mid = max(1, c_out // expansion)
        pad = (kernel // 2) * dilation
        self.body = nn.Sequential(
            nn.Conv2d(c_in,  c_mid, 1, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_mid, kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_out, 1, bias=False),
            make_norm(c_out, norm_type),
        )
        self.act  = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.act(self.dp(self.se(self.body(x))) + self.skip(x))


class AnisotropicBlock(nn.Module):
    """Factored convolutions: (1×k) + (k×1) to handle elongated spatial dims."""
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.h_conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, kernel), stride=(1, stride),
                      padding=(0, pad), dilation=(1, dilation), bias=False),
            make_norm(c_out, norm_type), make_act(act_type),
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(c_out, c_out, (kernel, 1), padding=(pad, 0),
                      dilation=(dilation, 1), bias=False),
            make_norm(c_out, norm_type), make_act(act_type),
        )
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=(1, stride), bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.dp(self.se(self.v_conv(self.h_conv(x)))) + self.skip(x)


class DilatedConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=2,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        # use dilation on the conv but output at stride=1 (dilated convs typically don't stride)
        pad = (kernel // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride=1, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_out, norm_type), make_act(act_type),
        )
        # separate stride step if needed
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.dp(self.se(self.pool(self.conv(x)))) + self.skip(x)


class GridLogicBlock(nn.Module):
    """
    Convolves both with a 1×1 and a full-kernel convolution independently
    then mixes — suited for rule-based grid patterns (Sudoku, GameOfLife).
    """
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        c_half = max(1, c_out // 2)
        self.local = nn.Sequential(
            nn.Conv2d(c_in, c_half, kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_half, norm_type), make_act(act_type),
        )
        self.point = nn.Sequential(
            nn.Conv2d(c_in, c_out - c_half, 1, stride=stride, bias=False),
            make_norm(c_out - c_half, norm_type), make_act(act_type),
        )
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        out = torch.cat([self.local(x), self.point(x)], dim=1)
        return self.dp(self.se(out)) + self.skip(x)


class ChannelMixingBlock(nn.Module):
    """1×1 bottleneck heavy mixing — for channel-heavy inputs."""
    def __init__(self, c_in, c_out, kernel=1, stride=1, dilation=1,
                 expansion=4, norm_type='batch', act_type='relu',
                 se=False, se_ratio=0.25, drop_path=0.0, **kw):
        super().__init__()
        c_mid = max(c_in, c_out * expansion)
        self.body = nn.Sequential(
            nn.Conv2d(c_in,  c_mid, 1, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_mid, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_out, 1, bias=False),
            make_norm(c_out, norm_type),
        )
        self.act  = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.act(self.dp(self.se(self.body(x))) + self.skip(x))


class GlobalContextBlock(nn.Module):
    """GC block: global context pooled as a bias on local features."""
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.local_conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            make_norm(c_out, norm_type), make_act(act_type),
        )
        # global context: pool → transform → broadcast as channel bias
        self.ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, c_out, 1, bias=True),
            make_act(act_type),
            nn.Conv2d(c_out, c_out, 1, bias=True),
        )
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        out = self.local_conv(x) + self.ctx(x)
        return self.dp(self.se(out)) + self.skip(x)


class LightAttentionBlock(nn.Module):
    """
    Lightweight self-attention on spatial tokens.
    Only safe for small spatial dims (H×W ≤ 256 recommended).
    """
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 norm_type='batch', act_type='relu', se=False, se_ratio=0.25,
                 drop_path=0.0, **kw):
        super().__init__()
        n_heads = max(1, min(8, c_in // 8))
        head_dim = max(1, c_in // n_heads)
        embed_dim = n_heads * head_dim
        self.proj_in  = nn.Conv2d(c_in, embed_dim, 1, bias=False)
        self.norm_in  = make_norm(embed_dim, norm_type)
        self.attn     = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, bias=False)
        self.proj_out = nn.Conv2d(embed_dim, c_out, 1, bias=False)
        self.norm_out = make_norm(c_out, norm_type)
        self.act      = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        # Use AvgPool for BOTH main and skip so spatial size is always floor(H/stride).
        # Conv2d(stride=2) gives ceil on odd dims; AvgPool2d gives floor — mismatch!
        _pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.pool = _pool
        if c_in != c_out or stride != 1:
            _skip_pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
            self.skip = nn.Sequential(_skip_pool, nn.Conv2d(c_in, c_out, 1, bias=False))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.norm_in(self.proj_in(x))                 # B, E, H, W
        z = z.flatten(2).permute(0, 2, 1)                 # B, H*W, E
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.permute(0, 2, 1).view(B, -1, H, W)         # B, E, H, W
        z = self.act(self.norm_out(self.proj_out(z)))
        z = self.pool(z)
        return self.dp(self.se(z)) + self.skip(x)


class GroupedBottleneckBlock(nn.Module):
    """RegNet X-block: 1×1 expand → 3×3 grouped conv → 1×1 project."""
    def __init__(self, c_in, c_out, kernel=3, stride=1, dilation=1,
                 group_w=16, expansion=1, norm_type='batch', act_type='relu',
                 se=False, se_ratio=0.25, drop_path=0.0, **kw):
        super().__init__()
        c_mid = max(c_out, int(c_out * expansion))
        pad = (kernel // 2) * dilation
        # Walk down from c_mid//group_w to find a valid group count
        g = max(1, c_mid // group_w)
        while c_mid % g != 0 and g > 1:
            g -= 1
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_mid, kernel, stride=stride, padding=pad,
                      dilation=dilation, groups=g, bias=False),
            make_norm(c_mid, norm_type), make_act(act_type),
            nn.Conv2d(c_mid, c_out, 1, bias=False),
            make_norm(c_out, norm_type),
        )
        self.act  = make_act(act_type)
        self.se   = SEBlock(c_out, se_ratio) if se else nn.Identity()
        self.dp   = DropPath(drop_path)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False) \
                    if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.act(self.dp(self.se(self.body(x))) + self.skip(x))


# ── Registry ──────────────────────────────────────────────────────────────────

BLOCK_REGISTRY = {
    'ConvBlock':           ConvBlock,
    'SepConvBlock':        SepConvBlock,
    'ResidualBlock':       ResidualBlock,
    'MBConvBlock':         MBConvBlock,
    'BottleneckBlock':     BottleneckBlock,
    'AnisotropicBlock':    AnisotropicBlock,
    'DilatedConvBlock':    DilatedConvBlock,
    'GridLogicBlock':      GridLogicBlock,
    'ChannelMixingBlock':  ChannelMixingBlock,
    'GlobalContextBlock':  GlobalContextBlock,
    'LightAttentionBlock':     LightAttentionBlock,
    'GroupedBottleneckBlock':  GroupedBottleneckBlock,
}
