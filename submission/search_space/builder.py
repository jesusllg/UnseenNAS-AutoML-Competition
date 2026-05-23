import torch
import torch.nn as nn
from typing import List, Tuple

from .genotype import (
    Genotype, StageGene,
    CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST,
    DILATION_LIST, SE_RATIO_LIST, DROPOUT_LIST, DROP_PATH_LIST,
)
from .block_library import BLOCK_REGISTRY, make_norm, make_act


# ── Stem builders ─────────────────────────────────────────────────────────────

def build_stem(stem_type: str, c_in: int, c_out: int,
               norm_type: str = 'batch') -> nn.Module:
    if stem_type == 'conv7x7':
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 7, stride=1, padding=3, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
        )
    elif stem_type == 'conv1x1':
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
        )
    elif stem_type == 'double_conv':
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
        )
    else:  # 'conv3x3'
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
        )


# ── Downsample builders ───────────────────────────────────────────────────────

def build_downsample(downsample_op: str, channels: int,
                     aniso_axis=None) -> nn.Module:
    if downsample_op == 'maxpool':
        if aniso_axis == 'W':
            return nn.MaxPool2d((1, 2), stride=(1, 2))
        elif aniso_axis == 'H':
            return nn.MaxPool2d((2, 1), stride=(2, 1))
        return nn.MaxPool2d(2, stride=2)
    elif downsample_op == 'avgpool':
        if aniso_axis == 'W':
            return nn.AvgPool2d((1, 2), stride=(1, 2))
        elif aniso_axis == 'H':
            return nn.AvgPool2d((2, 1), stride=(2, 1))
        return nn.AvgPool2d(2, stride=2)
    elif downsample_op == 'identity':
        return nn.Identity()
    else:  # 'stride2' — done inside the block
        return None  # signal to caller: no separate pooling needed


# ── Head modules ──────────────────────────────────────────────────────────────

class GapLinear(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc   = nn.Linear(c_in, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(x).flatten(1)))


class GmpLinear(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc   = nn.Linear(c_in, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(x).flatten(1)))


class GapGmpLinear(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.max  = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc   = nn.Linear(c_in * 2, num_classes)

    def forward(self, x):
        z = torch.cat([self.avg(x).flatten(1), self.max(x).flatten(1)], dim=1)
        return self.fc(self.drop(z))


class FlattenMlp(nn.Module):
    def __init__(self, c_in, final_h, final_w, num_classes, dropout=0.0, **kw):
        super().__init__()
        flat_dim = c_in * final_h * final_w
        hidden   = max(num_classes * 4, min(1024, flat_dim // 4))
        self.pool = nn.AdaptiveAvgPool2d((min(final_h, 4), min(final_w, 4)))
        out_h = min(final_h, 4)
        out_w = min(final_w, 4)
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * out_h * out_w, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.body(self.pool(x))


class AttentionPool(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.attn_w = nn.Conv2d(c_in, 1, 1)
        self.drop   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc     = nn.Linear(c_in, num_classes)

    def forward(self, x):
        w = torch.softmax(self.attn_w(x).flatten(2), dim=-1)  # B,1,HW
        pooled = (x.flatten(2) * w).sum(dim=2)                 # B,C
        return self.fc(self.drop(pooled))


class SpatialPyramidPool(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(sz) for sz in [1, 2, 4]
        ])
        total = c_in * (1 + 4 + 16)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc   = nn.Linear(total, num_classes)

    def forward(self, x):
        parts = [p(x).flatten(1) for p in self.pools]
        z = torch.cat(parts, dim=1)
        return self.fc(self.drop(z))


class GatedPool(nn.Module):
    def __init__(self, c_in, num_classes, dropout=0.0, **kw):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   nn.Linear(c_in, c_in),
                                   nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc   = nn.Linear(c_in, num_classes)

    def forward(self, x):
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        z = (self.pool(x) * g).flatten(1)
        return self.fc(self.drop(z))


HEAD_REGISTRY = {
    'GapLinear':         GapLinear,
    'GmpLinear':         GmpLinear,
    'GapGmpLinear':      GapGmpLinear,
    'FlattenMlp':        FlattenMlp,
    'AttentionPool':     AttentionPool,
    'SpatialPyramidPool': SpatialPyramidPool,
    'GatedPool':         GatedPool,
}


# ── Spatial size tracker ──────────────────────────────────────────────────────

def compute_output_spatial(genotype: Genotype, H: int, W: int) -> Tuple[int, int]:
    h, w = H, W
    for i, stage in enumerate(genotype.active_stages):
        ds = stage.downsample
        if ds == 'identity':
            pass
        else:  # stride2, maxpool, avgpool all halve spatial
            h = max(1, h // 2)
            w = max(1, w // 2)
    return h, w


# ── Core model ────────────────────────────────────────────────────────────────

class SearchSpaceModel(nn.Module):
    """
    Decodable architecture model.
    Exposes extract_layer_features(x) for AZ-NAS proxy scoring.
    """
    def __init__(self, stem: nn.Module, stage_list: nn.ModuleList,
                 downsamples: nn.ModuleList, neck: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.stem       = stem
        self.stage_list = stage_list
        self.downsamples = downsamples
        self.neck       = neck
        self.head       = head

    def extract_layer_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns feature maps at stem output and after each stage.
        All returned tensors retain gradient for AZ-NAS proxy.
        Required by proxies.az_nas_score().
        """
        features = []
        x = self.stem(x)
        x = x.clone().requires_grad_(True)
        x.retain_grad()
        features.append(x)

        for stage, ds in zip(self.stage_list, self.downsamples):
            if ds is not None:
                x = ds(x)
            x = stage(x)
            x = x.clone().requires_grad_(True)
            x.retain_grad()
            features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage, ds in zip(self.stage_list, self.downsamples):
            if ds is not None:
                x = ds(x)
            x = stage(x)
        x = self.neck(x)
        return self.head(x)


# ── Stage builder ─────────────────────────────────────────────────────────────

def _build_stage(gene: StageGene, c_in: int, norm_type: str,
                 aniso_axis=None) -> Tuple[nn.Module, nn.Module, int]:
    """Returns (downsample_or_None, stage_module, c_out)."""
    block_cls  = BLOCK_REGISTRY[gene.block_type]
    kernel     = KERNEL_LIST[gene.kernel_idx]
    n_blk      = N_BLOCKS_LIST[gene.n_blocks]
    c_out      = CHANNEL_LIST[gene.channels_idx]
    expansion  = EXPANSION_LIST[gene.expansion_idx]
    dilation   = DILATION_LIST[gene.dilation_idx]
    se         = bool(gene.se_enabled)
    se_ratio   = SE_RATIO_LIST[gene.se_ratio_idx]
    drop_path  = DROP_PATH_LIST[gene.drop_path_idx]
    downsample_op = gene.downsample

    # Separate pooling or strided-in-block
    ds_module  = build_downsample(downsample_op, c_in, aniso_axis)
    use_stride = (downsample_op == 'stride2')
    stride     = 2 if use_stride else 1

    layers = []
    for i in range(n_blk):
        cin = c_in if i == 0 else c_out
        layers.append(block_cls(
            c_in      = cin,
            c_out     = c_out,
            kernel    = kernel,
            stride    = stride if i == 0 else 1,
            dilation  = dilation,
            expansion = expansion,
            norm_type = norm_type,
            se        = se,
            se_ratio  = se_ratio,
            drop_path = drop_path,
        ))

    if len(layers) == 0:
        # guard: at least identity + projection
        layers.append(nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
            make_norm(c_out, norm_type),
            nn.ReLU(inplace=True),
        ))

    return ds_module, nn.Sequential(*layers), c_out


# ── Neck ──────────────────────────────────────────────────────────────────────

def build_neck(neck_type: str, c_in: int, norm_type: str) -> nn.Module:
    if neck_type == 'conv1x1':
        return nn.Sequential(
            nn.Conv2d(c_in, c_in, 1, bias=False),
            make_norm(c_in, norm_type),
            nn.ReLU(inplace=True),
        )
    elif neck_type == 'global_avg':
        return nn.AdaptiveAvgPool2d(1)
    return nn.Identity()


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_model(genotype: Genotype, C: int, H: int, W: int,
                num_classes: int, aniso_axis=None) -> SearchSpaceModel:
    norm_type  = genotype.norm_type
    stem_c     = CHANNEL_LIST[genotype.stem_channels]
    dropout    = DROPOUT_LIST[genotype.head_dropout]

    stem = build_stem(genotype.stem_type, C, stem_c, norm_type)

    stage_modules = []
    down_modules  = []
    c_current     = stem_c

    for stage_gene in genotype.active_stages:
        ds, stage_mod, c_out = _build_stage(
            stage_gene, c_current, norm_type, aniso_axis)
        down_modules.append(ds)        # None means stride inside block
        stage_modules.append(stage_mod)
        c_current = c_out

    neck = build_neck(genotype.neck_type, c_current, norm_type)
    if genotype.neck_type == 'global_avg':
        final_h, final_w = 1, 1
    else:
        final_h, final_w = compute_output_spatial(genotype, H, W)

    head_cls = HEAD_REGISTRY[genotype.head_type]
    head = head_cls(
        c_in       = c_current,
        num_classes= num_classes,
        dropout    = dropout,
        final_h    = final_h,
        final_w    = final_w,
    )

    return SearchSpaceModel(
        stem       = stem,
        stage_list = nn.ModuleList(stage_modules),
        downsamples= nn.ModuleList([m if m is not None else nn.Identity()
                                    for m in down_modules]),
        neck       = neck,
        head       = head,
    )
