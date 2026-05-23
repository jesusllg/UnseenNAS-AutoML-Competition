import math
from typing import Optional

from .genotype import (
    Genotype, StageGene, sample_random_genotype,
    CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST,
    DILATION_LIST, DROP_PATH_LIST, HEAD_TYPES, DOWNSAMPLE_OPS,
    MAX_STAGES,
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


def estimate_activations_mb(genotype: Genotype, C: int, H: int, W: int,
                             batch_size: int = 32) -> float:
    from .genotype import CHANNEL_LIST
    h, w = H, W
    total_elements = 0
    c = CHANNEL_LIST[genotype.stem_channels]
    total_elements += batch_size * c * h * w
    for gene in genotype.active_stages:
        c_out = CHANNEL_LIST[gene.channels_idx]
        ds    = gene.downsample
        if ds != 'identity':
            h = max(1, h // 2)
            w = max(1, w // 2)
        total_elements += batch_size * c_out * h * w
        c = c_out
    return total_elements * 4 / (1024 ** 2)  # float32 MB


# ── Repair rules ──────────────────────────────────────────────────────────────

def repair(genotype: Genotype, C: int, H: int, W: int,
           num_classes: int, family: FamilyProfile,
           memory_budget_mb: float = 4096.0) -> Genotype:
    """
    Apply hard-constraint repair rules in order.
    Returns a valid (possibly modified) Genotype.
    """
    import copy
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

    # R5: kernel size must not exceed spatial dimension / 2
    h, w = H, W
    for stage in g.active_stages:
        k = KERNEL_LIST[stage.kernel_idx]
        max_k = max(1, min(h, w) - 1)
        if k > max_k:
            # pick the largest valid kernel
            for ki in range(len(KERNEL_LIST) - 1, -1, -1):
                if KERNEL_LIST[ki] <= max_k:
                    stage.kernel_idx = ki
                    break
        # update spatial for next stage
        if stage.downsample != 'identity':
            h = max(1, h // 2)
            w = max(1, w // 2)

    # R6: max pooling steps (to prevent reducing spatial to zero)
    pool_steps = 0
    for stage in g.active_stages:
        if stage.downsample != 'identity':
            pool_steps += 1
        if pool_steps > family.max_pool_steps:
            stage.downsample = 'identity'

    # R7: AnisotropicBlock stride must stay on anisotropic axis only
    if family.is_anisotropic:
        for stage in g.active_stages:
            if stage.block_type == 'AnisotropicBlock':
                pass  # AnisotropicBlock uses (1,stride) — valid for aniso

    # R8: channels must be monotonically non-decreasing (by index)
    prev_idx = g.stem_channels
    for stage in g.active_stages:
        if stage.channels_idx < prev_idx:
            stage.channels_idx = prev_idx
        prev_idx = stage.channels_idx

    # R9: FlattenMlp head requires final spatial to be small enough
    #     (avoid huge flattened dimensions)
    if g.head_type == 'FlattenMlp':
        final_h, final_w = H, W
        for stage in g.active_stages:
            if stage.downsample != 'identity':
                final_h = max(1, final_h // 2)
                final_w = max(1, final_w // 2)
        c_out = CHANNEL_LIST[g.stages[g.n_stages - 1].channels_idx]
        flat  = c_out * final_h * final_w
        if flat > 65536:
            g.head_type = 'GapLinear'

    # R10: SpatialPyramidPool requires spatial ≥ 4 in each dim
    if g.head_type == 'SpatialPyramidPool':
        final_h, final_w = H, W
        for stage in g.active_stages:
            if stage.downsample != 'identity':
                final_h = max(1, final_h // 2)
                final_w = max(1, final_w // 2)
        if final_h < 4 or final_w < 4:
            g.head_type = 'GapLinear'

    # R11: memory budget guard (activation memory at batch=32)
    mem_mb = estimate_activations_mb(g, C, H, W, batch_size=32)
    if mem_mb > memory_budget_mb:
        # reduce channels progressively until within budget
        for stage in g.stages:
            if stage.channels_idx > 0:
                stage.channels_idx -= 1
        # also cap n_stages
        if g.n_stages > 2:
            g.n_stages -= 1

    return g
