import math
from dataclasses import dataclass, field
from typing import List, Optional

from .genotype import MAX_STAGES


@dataclass
class FamilyProfile:
    name: str
    # Hard constraints
    max_pool_steps: int          = 4     # max downsampling stages
    is_anisotropic: bool         = False # spatial dims very unequal
    aniso_axis: Optional[str]    = None  # 'H' or 'W' — larger spatial axis
    enable_attention: bool       = True  # LightAttentionBlock allowed globally
                                         # per-stage spatial check in repair overrides
    force_groupnorm: bool        = False # must use GroupNorm instead of BN
    min_channels: int            = 16
    max_params_m: float          = 50.0  # soft cap in millions
    # Augmentation policy: conservative by default for unseen data. Flips and
    # crops silently destroy semantics on symbolic grids, sequences, volumes
    # and channel-stacked data — only the natural-image families opt in.
    augment_hflip: bool          = False
    # Hints (for initial sampling bias only — NOT hard constraints)
    preferred_blocks: List[str]  = field(default_factory=list)
    forbidden_blocks: List[str]  = field(default_factory=list)
    preferred_heads:  List[str]  = field(default_factory=list)


def infer_family(C: int, H: int, W: int, num_classes: int) -> FamilyProfile:
    """
    Derive hard architectural constraints from input geometry.
    No priors on head type — head is a fully free gene.
    """
    spatial_min  = min(H, W)
    spatial_max  = max(H, W)
    spatial_area = H * W
    aniso_ratio  = spatial_max / max(spatial_min, 1)

    # ── anisotropic: one spatial dim >> other (e.g. 1D sequences stored as 2D)
    # Example: Cryptic 1×6×768 → W is the "sequence" axis (BERT embeddings)
    if aniso_ratio >= 6.0 and spatial_min <= 8:
        axis = 'W' if W > H else 'H'
        # Downsampling in this family only ever halves the LONG axis (builder
        # emits (1,2)/(2,1) pools and repair converts stride2 → maxpool), so
        # the pool budget is sized from the long axis — enough halvings to
        # bring the sequence axis down to ~8-24, capped by the stage count.
        # Sizing it from the short axis (as before) starved sequence data of
        # hierarchy: 1×768 inputs got ZERO pools and ran full-width throughout.
        pool_budget = min(MAX_STAGES, max(1, int(math.log2(max(spatial_max / 8, 2)))))
        return FamilyProfile(
            name            = 'anisotropic',
            max_pool_steps  = pool_budget,
            is_anisotropic  = True,
            aniso_axis      = axis,
            enable_attention= True,      # per-stage spatial guard in repair governs
            force_groupnorm = True,
            augment_hflip   = False,     # flipping W reverses embedding dim indices (meaningless)
            preferred_blocks= ['AnisotropicBlock', 'ChannelMixingBlock',
                               'GlobalContextBlock', 'ConvBlock'],
            forbidden_blocks= [],
        )

    # ── small_grid: tiny spatial (≤8×8), typically symbolic/board-like
    # Example: Chesseract 12×8×8, Sudoku 1×9×9, GameOfLife 1×9×9
    if spatial_max <= 10:
        return FamilyProfile(
            name            = 'small_grid',
            max_pool_steps  = 1,
            enable_attention= True,
            force_groupnorm = True,   # small spatial → small effective batch per position
            preferred_blocks= ['GridLogicBlock', 'ConvBlock', 'ResidualBlock', 'LightAttentionBlock'],
            forbidden_blocks= ['AnisotropicBlock'],
        )

    # ── possible_voxel: cube-like: H≈W≈C or "thick" volume data
    # Example: Voxel 20×20×20 (treated as C=20, H=20, W=20)
    if abs(H - W) <= 4 and abs(C - H) <= max(C, H) * 0.5 and C >= 8 and spatial_area <= 625:
        return FamilyProfile(
            name            = 'possible_voxel',
            max_pool_steps  = 2,
            enable_attention= True,   # per-stage spatial guard in repair governs
            force_groupnorm = True,
            # Volume-as-channels: the "depth" axis lives in C, so channel
            # mixing is the natural primitive alongside spatial residuals.
            preferred_blocks= ['ChannelMixingBlock', 'ResidualBlock',
                               'BottleneckBlock', 'ConvBlock'],
            forbidden_blocks= ['AnisotropicBlock'],
        )

    # ── channel_heavy: many input channels, small-to-medium spatial
    # Example: Chesseract (12 channels), Windspeed multi-channel
    if C >= 8 and spatial_area <= 1024:
        return FamilyProfile(
            name            = 'channel_heavy',
            max_pool_steps  = 2,
            enable_attention= True,   # per-stage spatial guard in repair governs
            force_groupnorm = C % 8 == 0,
            preferred_blocks= ['ChannelMixingBlock', 'MBConvBlock', 'BottleneckBlock', 'ResidualBlock'],
            forbidden_blocks= ['AnisotropicBlock'],
        )

    # ── spatiotemporal_like: medium spatial with multiple channels suggesting time/multimodal
    # Example: Windspeed, Language (1×28×768)
    if C >= 3 and (H == 1 or W == 1) and spatial_max >= 32:
        axis = 'W' if W > H else 'H'
        return FamilyProfile(
            name            = 'spatiotemporal_like',
            max_pool_steps  = 3,
            is_anisotropic  = True,
            aniso_axis      = axis,
            enable_attention= True,
            force_groupnorm = False,
            preferred_blocks= ['AnisotropicBlock', 'ChannelMixingBlock', 'DilatedConvBlock'],
            forbidden_blocks= [],
        )

    # ── visual_large: standard large-resolution images
    # Example: Myofibre 3×128×128, GeoClassing 3×64×64, CIFARTile 3×96×96+
    if spatial_min >= 64:
        # 64×64 → 4 pools → 4×4 final (standard for this resolution)
        # 128×128+ → 5 pools → 4×4 final
        pool_steps = 5 if spatial_min >= 128 else 4
        return FamilyProfile(
            name            = 'visual_large',
            max_pool_steps  = pool_steps,
            enable_attention= True,  # per-stage repair enforces the 256-token limit
            force_groupnorm = False,
            augment_hflip   = True,  # natural images: flip is safe
            preferred_blocks= ['MBConvBlock', 'ResidualBlock', 'SepConvBlock',
                               'BottleneckBlock', 'DilatedConvBlock'],
            forbidden_blocks= [],
        )

    # ── visual_medium: typical 28-63 images
    # Example: AddNIST 3×28×28, MultNIST, GeoClassing 3×32×32
    if spatial_min >= 24:
        return FamilyProfile(
            name            = 'visual_medium',
            max_pool_steps  = 3,
            enable_attention= True,
            force_groupnorm = False,
            augment_hflip   = True,  # natural images: flip is safe
            preferred_blocks= ['ResidualBlock', 'MBConvBlock', 'ConvBlock', 'SepConvBlock'],
            forbidden_blocks= [],
        )

    # ── compact_general: everything else (small but not tiny)
    # A family-level attention ban keyed on INPUT area was too blunt here: e.g.
    # Gutenberg 27×18 (area 486) blocked attention everywhere, yet after one
    # pool the map is 13×9 = 117 tokens — exactly what the per-stage guard in
    # repair exists to allow. Defer to it, like visual_large always did.
    return FamilyProfile(
        name            = 'compact_general',
        max_pool_steps  = 2,
        enable_attention= True,   # per-stage spatial guard in repair governs
        force_groupnorm = False,
        preferred_blocks= ['ConvBlock', 'ResidualBlock', 'SepConvBlock',
                           'GlobalContextBlock', 'DilatedConvBlock'],
        forbidden_blocks= [],
    )
