import math
from typing import Optional

from .genotype import (
    Genotype, stem_stride,
    CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST,
    DILATION_LIST, MAX_STAGES,
)
from .family import FamilyProfile
from .builder import compute_output_spatial


# ── Lightweight param estimator ───────────────────────────────────────────────

def estimate_params(genotype: Genotype, C: int, H: int, W: int,
                    num_classes: int) -> int:
    p = 0
    stem_c = CHANNEL_LIST[genotype.stem_channels]
    # stem: 3×3 conv
    p += C * stem_c * 9
    c_in = stem_c
    for gene in genotype.active_stages:
        k       = KERNEL_LIST[gene.kernel_idx]
        n_blk   = N_BLOCKS_LIST[gene.n_blocks]
        c_out   = CHANNEL_LIST[gene.channels_idx]
        expand  = EXPANSION_LIST[gene.expansion_idx]
        c_mid   = max(c_in, int(c_in * expand))
        # approx per-block params (two conv layers each)
        block_p = c_in * c_mid * k * k + c_mid * c_out * k * k
        p += block_p * n_blk
        # skip/projection if needed
        if c_in != c_out:
            p += c_in * c_out
        c_in = c_out
    # head: linear
    p += c_in * num_classes
    return p


def _actual_spatial(h: int, ds: str) -> int:
    """Spatial size after downsample op, matching actual PyTorch behaviour."""
    if ds == 'identity':
        return h
    if ds == 'stride2':
        return (h + 1) // 2   # ceil — Conv2d(stride=2, pad=k//2)
    return max(1, h // 2)     # floor — MaxPool2d / AvgPool2d


def _block_mid_channels(block_type: str, c_in: int, c_out: int, expand: int) -> int:
    """
    The largest intermediate channel count a block actually materialises, taken
    directly from each block's forward in block_library.py. This must stay in
    sync with the blocks — it is what lets the memory guard see a block for its
    real cost instead of assuming every block expands uniformly (the blind spot
    that let GroupedBottleneck's old 6× expansion OOM training undetected).

    Only MBConv and ChannelMixing expand; Bottleneck compresses; every other
    block (incl. the constant-width GroupedBottleneck) runs at c_out.
    """
    if block_type == 'MBConvBlock':
        return max(c_in, c_in * expand)
    if block_type == 'ChannelMixingBlock':
        return max(c_in, c_out * expand)
    if block_type == 'BottleneckBlock':
        return max(1, c_out // expand)
    return c_out


def estimate_activations_mb(genotype: Genotype, C: int, H: int, W: int,
                             batch_size: int = 32, aniso_axis=None) -> float:
    """
    Estimate peak training memory (MB, float32 basis).

    Accounts for:
      - Strided stems and (for anisotropic families) axis-aware pooling, so the
        tracked spatial matches what the builder actually constructs
      - Output activation of each stage (stored for backward pass)
      - Intermediate activation within blocks, sized per block type (only MBConv
        and ChannelMixing expand channels; see _block_mid_channels)
      - Training factor ×3 for backward gradients + Adam m/v states
    """
    h, w = H, W
    if stem_stride(genotype.stem_type) == 2:
        h, w = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
    total_elements = 0
    c_in = CHANNEL_LIST[genotype.stem_channels]
    total_elements += batch_size * c_in * h * w  # stem output

    for gene in genotype.active_stages:
        c_out  = CHANNEL_LIST[gene.channels_idx]
        expand = EXPANSION_LIST[gene.expansion_idx]
        n_blk  = N_BLOCKS_LIST[gene.n_blocks]
        if gene.downsample == 'identity' or aniso_axis is None:
            h = _actual_spatial(h, gene.downsample)
            w = _actual_spatial(w, gene.downsample)
        elif aniso_axis == 'W':   # aniso pools halve only the long axis
            w = max(1, w // 2)
        else:                     # aniso_axis == 'H'
            h = max(1, h // 2)

        # Output activation (must be kept for backward pass)
        total_elements += batch_size * c_out * h * w

        # Intermediate activation within each block, sized by its real forward.
        c_mid = _block_mid_channels(gene.block_type, c_in, c_out, expand)
        total_elements += batch_size * c_mid * h * w * n_blk

        c_in = c_out

    # ×3: backward-pass gradient buffers (~1×) + Adam m/v states (~2× params,
    # but activations dominate at large spatial dims so this factor is conservative).
    _TRAIN_FACTOR = 3
    return total_elements * 4 * _TRAIN_FACTOR / (1024 ** 2)  # float32 MB


def _default_memory_budget_mb() -> float:
    """
    Memory budget = a fraction of the ACTUAL GPU so the cap reflects real
    hardware, not an arbitrary number.

    The estimator above is deliberately conservative (fp32 basis × 3, while
    training actually runs in fp16/AMP), so we can safely target ~80% of the
    physical GPU memory. This guarantees:
      • small/medium-spatial models never hit the cap (search is unchanged —
        the strong, SOTA-ish architectures we already find still pass), and
      • only models that genuinely exceed the GPU (e.g. huge-spatial Windspeed)
        get shrunk, and only by the minimum needed to fit.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            return total_mb * 0.80
    except Exception:
        pass
    return 16384.0  # 16 GB fallback when no GPU info is available


# ── Repair rules ──────────────────────────────────────────────────────────────

def repair(genotype: Genotype, C: int, H: int, W: int,
           num_classes: int, family: FamilyProfile,
           memory_budget_mb: Optional[float] = None) -> Genotype:
    """
    Apply hard-constraint repair rules in order.
    Returns a valid (possibly modified) Genotype.

    memory_budget_mb: if None, defaults to ~80% of the physical GPU memory so
    the cap tracks real hardware instead of an arbitrary fixed limit.
    """
    import copy
    if memory_budget_mb is None:
        memory_budget_mb = _default_memory_budget_mb()
    g = copy.deepcopy(genotype)

    # R1: n_stages must be in [1, MAX_STAGES]
    g.n_stages = max(1, min(MAX_STAGES, g.n_stages))

    # R2: apply family norm override
    if family.force_groupnorm:
        g.norm_type = 'group'

    # R3: forbidden blocks → replace with ConvBlock
    for stage in g.active_stages:
        if stage.block_type in family.forbidden_blocks:
            stage.block_type = 'ConvBlock'

    # R4: family-level attention ban (rare — per-stage rule (e) is the normal
    # gate; this only fires if a family explicitly disables attention outright)
    if not family.enable_attention:
        for stage in g.active_stages:
            if stage.block_type == 'LightAttentionBlock':
                stage.block_type = 'ConvBlock'

    # R4b: strided stems need spatial room. Demote to the stride-1 variant on
    # small inputs (halving 27×18 in the stem wastes most of the signal) and on
    # anisotropic data (a symmetric stem stride would crush the short axis).
    if stem_stride(g.stem_type) == 2 and (min(H, W) < 32 or family.is_anisotropic):
        g.stem_type = g.stem_type[:-len('_s2')]

    # R5-R7: single spatial-aware pass — kernel, downsample, pool budget, block compat.
    #
    # Tracks ACTUAL h/w at each stage after the chosen downsample op:
    #   • maxpool / avgpool  →  h = h // 2   (floor, PyTorch default)
    #   • stride2 (conv)     →  h = (h-1)//2 + 1 = (h+1)//2  (ceil on odd dims)
    #   • anisotropic family →  pools halve ONLY the aniso axis (builder emits
    #     (1,2)/(2,1) pools), so the tracking must too or every kernel clamp
    #     downstream is computed against a phantom shrunken short axis.
    #
    # Key repair actions per stage:
    #   a) anisotropic: stride2 → maxpool. Inside most blocks stride2 halves
    #      BOTH axes while aniso pools halve only the long one — two different
    #      semantics under one gene. Converting makes every downsample
    #      axis-aware and the tracking below exact.
    #   a2) stride2 on odd dim → replace with maxpool to keep floor semantics
    #      everywhere; avoids ceil/floor mismatch between main path and skip.
    #   b) Enforce family.max_pool_steps budget (a strided stem consumes one).
    #   c) Clamp kernel to fit current spatial (k < min(h,w)).
    #   d) Clamp dilation so effective receptive field fits spatial.
    #   e) Block-specific constraints (attention needs spatial > 1 in each dim).
    aniso = family.aniso_axis if family.is_anisotropic else None
    h, w = H, W
    pool_steps = 0
    if stem_stride(g.stem_type) == 2:   # survived R4b → input is large enough
        h, w = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
        pool_steps = 1
    for stage in g.active_stages:
        ds = stage.downsample

        # (a) anisotropic: all downsampling must be axis-aware pooling
        if aniso and ds == 'stride2':
            ds = stage.downsample = 'maxpool'

        # (a2) stride2 with odd spatial → switch to maxpool (both give floor)
        if ds == 'stride2' and (h % 2 != 0 or w % 2 != 0):
            ds = stage.downsample = 'maxpool'

        # (a3) pooling needs ≥2 px on every axis it halves — MaxPool2d(2) on a
        # size-1 axis is a hard PyTorch error, not a no-op. Unseen data can be
        # 1×1 spatial (pure channel vectors) or collapse to 1 mid-net; without
        # this, repair's floor-at-1 tracking says "fine" while the built model
        # crashes on its first forward.
        if ds in ('maxpool', 'avgpool'):
            poolable = w if aniso == 'W' else h if aniso == 'H' else min(h, w)
            if poolable < 2:
                ds = stage.downsample = 'identity'

        # (b) pool budget
        if ds != 'identity':
            pool_steps += 1
            if pool_steps > family.max_pool_steps:
                ds = stage.downsample = 'identity'

        # Compute post-downsample spatial for this stage's ops
        if ds == 'identity':
            sh, sw = h, w
        elif aniso == 'W':
            sh, sw = h, max(1, w // 2)
        elif aniso == 'H':
            sh, sw = max(1, h // 2), w
        elif ds == 'stride2':
            sh, sw = (h + 1) // 2, (w + 1) // 2
        else:
            sh, sw = max(1, h // 2), max(1, w // 2)

        # (c) kernel must be < min spatial dim (needs at least 1 pixel of output)
        k = KERNEL_LIST[stage.kernel_idx]
        max_k = max(1, min(sh, sw))
        if k >= max_k:
            for ki in range(len(KERNEL_LIST) - 1, -1, -1):
                if KERNEL_LIST[ki] < max_k:
                    stage.kernel_idx = ki
                    break
            else:
                stage.kernel_idx = 0  # fallback: kernel=1

        # (d) clamp dilation so dilated kernel fits in spatial
        k = KERNEL_LIST[stage.kernel_idx]
        d = DILATION_LIST[stage.dilation_idx]
        # effective span = d*(k-1)+1; must be <= min(sh,sw)
        max_span = min(sh, sw)
        while d > 1 and d * (k - 1) + 1 > max_span:
            stage.dilation_idx = max(0, stage.dilation_idx - 1)
            d = DILATION_LIST[stage.dilation_idx]

        # (e) LightAttentionBlock: per-stage spatial guard.
        #   Too-large (sh*sw > 256): O(N²) attention is prohibitive — this fires
        #   for early stages of visual_large (e.g. 64×64=4096) while letting deep
        #   stages through once spatial shrinks (e.g. 8×8=64 passes freely).
        #   Too-small (min dim ≤ 1): collapses to identity; replace.
        _ATTN_SPATIAL_LIMIT = 256
        if stage.block_type == 'LightAttentionBlock':
            if sh * sw > _ATTN_SPATIAL_LIMIT or min(sh, sw) <= 1:
                stage.block_type = 'ConvBlock'

        # (f) anisotropic axis: AnisotropicBlock already uses (1,stride) — fine
        # no action needed for aniso

        # advance tracked spatial
        h, w = sh, sw

    # R8: channels must be monotonically non-decreasing (by index)
    prev_idx = g.stem_channels
    for stage in g.active_stages:
        if stage.channels_idx < prev_idx:
            stage.channels_idx = prev_idx
        prev_idx = stage.channels_idx

    # R9 + R10 + R12: head compatibility — use actual final spatial dims,
    # accounting for neck which may collapse spatial independently of stages.
    # compute_output_spatial is the SAME function the builder sizes heads with
    # (stem-stride and aniso-axis aware), so repair and build can never differ.
    # Exact here: after (a)/(a2) above, any surviving stride2 acts on even dims
    # where its ceil semantics equal the pools' floor.
    final_h, final_w = compute_output_spatial(g, H, W, aniso)

    # global_avg neck collapses spatial to 1×1 before the head
    eff_h = 1 if g.neck_type == 'global_avg' else final_h
    eff_w = 1 if g.neck_type == 'global_avg' else final_w

    if g.head_type == 'FlattenMlp':
        c_out = CHANNEL_LIST[g.stages[g.n_stages - 1].channels_idx]
        if c_out * eff_h * eff_w > 65536:
            g.head_type = 'GapLinear'

    if g.head_type == 'SpatialPyramidPool' and (eff_h < 4 or eff_w < 4):
        g.head_type = 'GapLinear'

    # R12: heads that degenerate when spatial collapses to 1×1 via global_avg neck:
    #   AttentionPool  → single token, attention weight always 1.0 → identity
    #   GapGmpLinear   → avg and max of same 1×1 pixel → duplicate input features
    _DEGENERATE_WITH_GLOBAL_AVG = {'AttentionPool', 'GapGmpLinear'}
    if g.neck_type == 'global_avg' and g.head_type in _DEGENERATE_WITH_GLOBAL_AVG:
        g.head_type = 'GapLinear'

    # R11: memory budget guard (includes intermediates + training overhead).
    # Apply up to 3 rounds of channel reduction so a single repair pass can
    # catch severely oversized genotypes without needing repeated calls.
    #
    # Estimate at the SAME batch size DataProcessor will train with (shared rule
    # in helpers.select_batch_size). A mismatch here is a silent under/over
    # estimate — e.g. wide Cryptic (6×768) trains at batch 64, not 32.
    from helpers import select_batch_size
    train_bs = select_batch_size(C, H, W)
    for _shrink in range(3):
        mem_mb = estimate_activations_mb(g, C, H, W, batch_size=train_bs,
                                         aniso_axis=aniso)
        if mem_mb <= memory_budget_mb:
            break
        # Reduce expansion first (cheapest quality loss), then channels, then stages
        for stage in g.active_stages:
            if stage.expansion_idx > 0:
                stage.expansion_idx -= 1
        for stage in g.stages:
            if stage.channels_idx > 0:
                stage.channels_idx -= 1
        if g.n_stages > 2:
            g.n_stages -= 1

    return g
