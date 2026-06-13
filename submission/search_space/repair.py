import math
from typing import Optional

from .genotype import (
    Genotype, StageGene, sample_random_genotype,
    CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST,
    DILATION_LIST, DROP_PATH_LIST, HEAD_TYPES, DOWNSAMPLE_OPS,
    MAX_STAGES, GROUP_W_LIST,
)
from .family import FamilyProfile


# ── Lightweight param estimator ───────────────────────────────────────────────

def estimate_params(genotype: Genotype, C: int, H: int, W: int,
                    num_classes: int) -> int:
    from .genotype import CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST
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


def estimate_activations_mb(genotype: Genotype, C: int, H: int, W: int,
                             batch_size: int = 32) -> float:
    """
    Estimate peak training memory (MB, float32 basis).

    Accounts for:
      - Output activation of each stage (stored for backward pass)
      - Intermediate expanded activations within blocks (MBConv/ChannelMixing
        can expand channels by 4–6×, dominating memory at large spatial dims)
      - Training factor ×3 for backward gradients + Adam m/v states
    """
    from .genotype import CHANNEL_LIST, EXPANSION_LIST, N_BLOCKS_LIST
    h, w = H, W
    total_elements = 0
    c_in = CHANNEL_LIST[genotype.stem_channels]
    total_elements += batch_size * c_in * h * w  # stem output

    for gene in genotype.active_stages:
        c_out  = CHANNEL_LIST[gene.channels_idx]
        expand = EXPANSION_LIST[gene.expansion_idx]
        n_blk  = N_BLOCKS_LIST[gene.n_blocks]
        h = _actual_spatial(h, gene.downsample)
        w = _actual_spatial(w, gene.downsample)

        # Output activation (must be kept for backward pass)
        total_elements += batch_size * c_out * h * w

        # Intermediate expanded activation within each block.
        # MBConvBlock: c_in → c_in*expand → c_out
        # ChannelMixingBlock: c_in → max(c_in, c_out*expand) → c_out
        # Use the most conservative (largest) intermediate estimate.
        c_mid = max(c_in * expand, c_out * expand)
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

    # R4: LightAttentionBlock requires small spatial (≤256 total)
    if not family.enable_attention:
        for stage in g.active_stages:
            if stage.block_type == 'LightAttentionBlock':
                stage.block_type = 'ConvBlock'

    # R5-R7: single spatial-aware pass — kernel, downsample, pool budget, block compat.
    #
    # Tracks ACTUAL h/w at each stage after the chosen downsample op:
    #   • maxpool / avgpool  →  h = h // 2   (floor, PyTorch default)
    #   • stride2 (conv)     →  h = (h-1)//2 + 1 = (h+1)//2  (ceil on odd dims)
    #
    # Key repair actions per stage:
    #   a) stride2 on odd dim → replace with maxpool to keep floor semantics
    #      everywhere; avoids ceil/floor mismatch between main path and skip.
    #   b) Enforce family.max_pool_steps budget.
    #   c) Clamp kernel to fit current spatial (k < min(h,w)).
    #   d) Clamp dilation so effective receptive field fits spatial.
    #   e) Block-specific constraints (attention needs spatial > 1 in each dim).
    h, w = H, W
    pool_steps = 0
    for stage in g.active_stages:
        ds = stage.downsample

        # (a) stride2 with odd spatial → switch to maxpool (both give floor)
        if ds == 'stride2' and (h % 2 != 0 or w % 2 != 0):
            ds = stage.downsample = 'maxpool'

        # (b) pool budget
        if ds != 'identity':
            pool_steps += 1
            if pool_steps > family.max_pool_steps:
                ds = stage.downsample = 'identity'

        # Compute post-downsample spatial for this stage's ops
        sh = (h + 1) // 2 if ds == 'stride2' else (max(1, h // 2) if ds != 'identity' else h)
        sw = (w + 1) // 2 if ds == 'stride2' else (max(1, w // 2) if ds != 'identity' else w)

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

        # (e) LightAttentionBlock needs spatial > 1 in each dim to run attn
        if stage.block_type == 'LightAttentionBlock' and min(sh, sw) <= 1:
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
    final_h, final_w = H, W
    for stage in g.active_stages:
        final_h = _actual_spatial(final_h, stage.downsample)
        final_w = _actual_spatial(final_w, stage.downsample)

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
    # Estimate at the SAME batch size DataProcessor will actually use at train
    # time (its rule keyed on pixel count). Previously hardcoded to 32 while wide
    # inputs like Cryptic (6×768 → 4608 px) train at batch 64 → a silent 2×
    # underestimate that let oversized models slip through and OOM the trainer.
    pixels = C * H * W
    train_bs = 16 if pixels > 100_000 else (32 if pixels > 10_000 else 64)
    for _shrink in range(3):
        mem_mb = estimate_activations_mb(g, C, H, W, batch_size=train_bs)
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
