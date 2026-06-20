"""Architecture visualization: PNG diagram of a NAS-found genotype."""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BLOCK_SHORT = {
    'ConvBlock':               'ConvBn',
    'SepConvBlock':            'SepConv',
    'ResidualBlock':           'ResBlock',
    'MBConvBlock':             'MBConv',
    'BottleneckBlock':         'Bottleneck',
    'GroupedBottleneckBlock':  'GroupedBN',
    'AnisotropicBlock':        'AnisoConv',
    'DilatedConvBlock':        'DilConv',
    'GridLogicBlock':          'GridLogic',
    'ChannelMixingBlock':      'ChanMix',
    'GlobalContextBlock':      'GlobCtx',
    'LightAttentionBlock':     'LightAttn',
}

_BLOCK_COLORS = {
    'ConvBlock':               '#4E79A7',
    'SepConvBlock':            '#F28E2B',
    'ResidualBlock':           '#E15759',
    'MBConvBlock':             '#76B7B2',
    'BottleneckBlock':         '#59A14F',
    'GroupedBottleneckBlock':  '#EDC948',
    'AnisotropicBlock':        '#B07AA1',
    'DilatedConvBlock':        '#FF9DA7',
    'GridLogicBlock':          '#9C755F',
    'ChannelMixingBlock':      '#C4A882',
    'GlobalContextBlock':      '#D37295',
    'LightAttentionBlock':     '#A0CBE8',
}

_DS_LABEL = {
    'stride2':  '÷2 stride',
    'maxpool':  '÷2 maxpool',
    'avgpool':  '÷2 avgpool',
    'identity': '',
}


def save_arch(genotype, metadata: dict, save_path,
              proxy_score: Optional[float] = None,
              n_params: Optional[int] = None) -> bool:
    """
    Draw and save architecture diagram as PNG.
    Returns True on success, False on any failure (e.g. matplotlib missing).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — arch diagram skipped.")
        return False

    try:
        from search_space.genotype import (
            CHANNEL_LIST, KERNEL_LIST, N_BLOCKS_LIST, EXPANSION_LIST,
            DILATION_LIST, SE_RATIO_LIST, DROP_PATH_LIST, DROPOUT_LIST,
        )
        try:
            from search_space.genotype import GROUP_W_LIST
        except ImportError:
            GROUP_W_LIST = [4, 8, 16, 32]
    except Exception as e:
        logger.warning("genotype import failed (%s) — arch diagram skipped.", e)
        return False

    codename = metadata.get('codename', 'unknown')
    shape    = metadata.get('input_shape', [0, 1, 1, 1])
    n_cls    = metadata.get('num_classes', '?')
    C, H, W  = shape[1], shape[2], shape[3]

    # ── Build the list of visual nodes ───────────────────────────────────────
    # Each node: {'label': str, 'color': str, 'ds_label': str}
    # ds_label is shown on the arrow ENTERING this node (the downsampling that
    # happens before the node's computations begin).
    nodes = []

    nodes.append({
        'label':    f'Input\n{C} × {H} × {W}',
        'color':    '#E8E8E8',
        'ds_label': '',
    })

    stem_ch = CHANNEL_LIST[genotype.stem_channels]
    act = getattr(genotype, 'act_type', 'relu')
    nodes.append({
        'label':    f'Stem  [{genotype.stem_type}]\n{stem_ch} ch  │  {genotype.norm_type}norm  │  {act}',
        'color':    '#B3CDE3',
        'ds_label': '',
    })

    for i, stage in enumerate(genotype.active_stages):
        ch   = CHANNEL_LIST[stage.channels_idx]
        k    = KERNEL_LIST[stage.kernel_idx]
        n    = N_BLOCKS_LIST[stage.n_blocks]
        exp  = EXPANSION_LIST[stage.expansion_idx]
        dil  = DILATION_LIST[stage.dilation_idx]
        se   = bool(stage.se_enabled)
        se_r = SE_RATIO_LIST[stage.se_ratio_idx]
        dp   = DROP_PATH_LIST[stage.drop_path_idx]
        gw_str = ''
        if stage.block_type == 'GroupedBottleneckBlock':
            gw = GROUP_W_LIST[getattr(stage, 'group_w_idx', 2)]
            gw_str = f'  │  gw={gw}'
        parts = [f'{ch} ch  │  {n}× k{k}']
        if exp > 1:   parts.append(f'exp×{exp}')
        if dil > 1:   parts.append(f'dil={dil}')
        if se:        parts.append(f'SE r={se_r:.3f}')
        if dp > 0:    parts.append(f'dp={dp}')
        detail = '  │  '.join(parts) + gw_str
        bshort = _BLOCK_SHORT.get(stage.block_type, stage.block_type[:12])
        color  = _BLOCK_COLORS.get(stage.block_type, '#BBBBBB')
        nodes.append({
            'label':    f'Stage {i+1}  [{bshort}]\n{detail}',
            'color':    color,
            'ds_label': _DS_LABEL.get(stage.downsample, stage.downsample),
        })

    if genotype.neck_type != 'none':
        nodes.append({
            'label':    f'Neck  [{genotype.neck_type}]',
            'color':    '#CCEBC5',
            'ds_label': '',
        })

    drop = DROPOUT_LIST[genotype.head_dropout]
    drop_str = f'  │  drop={drop}' if drop > 0 else ''
    nodes.append({
        'label':    f'Head  [{genotype.head_type}]\n→ {n_cls} classes{drop_str}',
        'color':    '#FED9A6',
        'ds_label': '',
    })

    # ── Layout ────────────────────────────────────────────────────────────────
    N     = len(nodes)
    BOX_H = 0.70
    GAP   = 0.48
    BOX_W = 0.80
    FIG_W = 5.8
    FIG_H = N * (BOX_H + GAP) + 1.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')

    title_parts = [codename]
    if proxy_score is not None:
        title_parts.append(f'AZ-NAS = {proxy_score:.4f}')
    if n_params is not None:
        p_str = f'{n_params/1e6:.2f}M' if n_params >= 1_000_000 else f'{n_params/1e3:.1f}K'
        title_parts.append(f'{p_str} params')
    ax.set_title('  │  '.join(title_parts), fontsize=10.5, fontweight='bold', pad=6)

    x0       = (1 - BOX_W) / 2
    y_cursor = FIG_H - 0.6  # top of first box

    for i, node in enumerate(nodes):
        y_bottom = y_cursor - BOX_H

        rect = mpatches.FancyBboxPatch(
            (x0, y_bottom), BOX_W, BOX_H,
            boxstyle='round,pad=0.04',
            facecolor=node['color'],
            edgecolor='#444444',
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(rect)

        fs = 7.8 if 'Stage' in node['label'] else 8.8
        ax.text(0.5, y_bottom + BOX_H / 2, node['label'],
                ha='center', va='center', fontsize=fs,
                fontfamily='monospace', zorder=3, linespacing=1.5)

        if i < N - 1:
            ay_top = y_bottom
            ay_bot = y_bottom - GAP
            ax.annotate('',
                        xy=(0.5, ay_bot + 0.05),
                        xytext=(0.5, ay_top - 0.04),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5),
                        zorder=1)
            # ds_label goes on arrow ENTERING the NEXT node
            next_ds = nodes[i + 1].get('ds_label', '')
            if next_ds:
                ax.text(0.5 + BOX_W / 2 + 0.02,
                        (ay_top + ay_bot) / 2,
                        next_ds,
                        ha='left', va='center',
                        fontsize=6.5, color='#555', style='italic')

        y_cursor = y_bottom - GAP

    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True
