from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import random
import copy

# ── Block type catalog ─────────────────────────────────────────────────────────
BLOCK_TYPES = [
    'ConvBlock',
    'SepConvBlock',
    'ResidualBlock',
    'MBConvBlock',
    'BottleneckBlock',
    'AnisotropicBlock',
    'DilatedConvBlock',
    'GridLogicBlock',
    'ChannelMixingBlock',
    'GlobalContextBlock',
    'LightAttentionBlock',
    'GroupedBottleneckBlock',
]

# ── Discrete option lists ──────────────────────────────────────────────────────
CHANNEL_LIST      = [16, 24, 32, 48, 64, 96, 128, 192, 256]
KERNEL_LIST       = [1, 3, 5, 7]
N_BLOCKS_LIST     = [1, 2, 3, 4]
EXPANSION_LIST    = [1, 2, 4, 6]
DILATION_LIST     = [1, 2, 3, 4]
SE_RATIO_LIST     = [0.0625, 0.125, 0.25]
DROPOUT_LIST      = [0.0, 0.1, 0.2, 0.3]
DROP_PATH_LIST    = [0.0, 0.05, 0.1, 0.2]
GROUP_W_LIST      = [4, 8, 16, 32]
HEAD_TYPES        = ['GapLinear', 'GmpLinear', 'GapGmpLinear', 'FlattenMlp',
                     'AttentionPool', 'SpatialPyramidPool', 'GatedPool']
STEM_TYPES        = ['conv3x3', 'conv7x7', 'conv1x1', 'double_conv']
NECK_TYPES        = ['none', 'conv1x1', 'global_avg']
DOWNSAMPLE_OPS    = ['stride2', 'maxpool', 'avgpool', 'identity']
SKIP_MODES        = ['none', 'residual', 'dense']
NORM_TYPES        = ['batch', 'group']
ACT_TYPES         = ['relu', 'silu', 'gelu']

MAX_STAGES = 5
DEFAULT_N_STAGES = 4

# ── Cardinality map (for sampling / mutation) ──────────────────────────────────
STAGE_FIELD_CARDINALITY: Dict[str, int] = {
    'block_type':    len(BLOCK_TYPES),
    'kernel_idx':    len(KERNEL_LIST),
    'n_blocks':      len(N_BLOCKS_LIST),
    'channels_idx':  len(CHANNEL_LIST),
    'expansion_idx': len(EXPANSION_LIST),
    'dilation_idx':  len(DILATION_LIST),
    'se_enabled':    2,
    'se_ratio_idx':  len(SE_RATIO_LIST),
    'skip_mode':     len(SKIP_MODES),
    'downsample':    len(DOWNSAMPLE_OPS),
    'drop_path_idx': len(DROP_PATH_LIST),
    'group_w_idx':   len(GROUP_W_LIST),
}

GLOBAL_FIELD_CARDINALITY: Dict[str, int] = {
    'n_stages':      MAX_STAGES,          # 1..MAX_STAGES
    'stem_type':     len(STEM_TYPES),
    'stem_channels': len(CHANNEL_LIST),
    'neck_type':     len(NECK_TYPES),
    'head_type':     len(HEAD_TYPES),
    'head_dropout':  len(DROPOUT_LIST),
    'norm_type':     len(NORM_TYPES),
}

FIELD_CARDINALITY = {**GLOBAL_FIELD_CARDINALITY, **STAGE_FIELD_CARDINALITY}


@dataclass
class StageGene:
    block_type:    str  = 'ResidualBlock'
    kernel_idx:    int  = 1          # index into KERNEL_LIST
    n_blocks:      int  = 0          # index into N_BLOCKS_LIST
    channels_idx:  int  = 4          # index into CHANNEL_LIST
    expansion_idx: int  = 0          # index into EXPANSION_LIST
    dilation_idx:  int  = 0          # index into DILATION_LIST
    se_enabled:    int  = 0          # 0/1 bool
    se_ratio_idx:  int  = 1          # index into SE_RATIO_LIST
    skip_mode:     str  = 'residual'
    downsample:    str  = 'stride2'
    drop_path_idx: int  = 0          # index into DROP_PATH_LIST
    group_w_idx:   int  = 2          # index into GROUP_W_LIST (default 16)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'StageGene':
        return StageGene(**{k: v for k, v in d.items() if k in StageGene.__dataclass_fields__})

    def clone(self) -> 'StageGene':
        return copy.deepcopy(self)


@dataclass
class Genotype:
    n_stages:      int              = DEFAULT_N_STAGES
    stem_type:     str              = 'conv3x3'
    stem_channels: int              = 2          # index into CHANNEL_LIST
    neck_type:     str              = 'none'
    head_type:     str              = 'GapLinear'
    head_dropout:  int              = 0          # index into DROPOUT_LIST
    norm_type:     str              = 'batch'
    stages:        List[StageGene] = field(default_factory=lambda: [StageGene() for _ in range(MAX_STAGES)])

    def __post_init__(self):
        # Always keep exactly MAX_STAGES stage genes (vestigial stages are ignored during build)
        while len(self.stages) < MAX_STAGES:
            self.stages.append(StageGene())
        self.stages = self.stages[:MAX_STAGES]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'n_stages':      self.n_stages,
            'stem_type':     self.stem_type,
            'stem_channels': self.stem_channels,
            'neck_type':     self.neck_type,
            'head_type':     self.head_type,
            'head_dropout':  self.head_dropout,
            'norm_type':     self.norm_type,
            'stages':        [s.to_dict() for s in self.stages],
        }
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Genotype':
        stages = [StageGene.from_dict(s) for s in d.get('stages', [])]
        return Genotype(
            n_stages      = d.get('n_stages',      DEFAULT_N_STAGES),
            stem_type     = d.get('stem_type',     'conv3x3'),
            stem_channels = d.get('stem_channels', 2),
            neck_type     = d.get('neck_type',     'none'),
            head_type     = d.get('head_type',     'GapLinear'),
            head_dropout  = d.get('head_dropout',  0),
            norm_type     = d.get('norm_type',     'batch'),
            stages        = stages,
        )

    def clone(self) -> 'Genotype':
        return copy.deepcopy(self)

    @property
    def active_stages(self) -> List[StageGene]:
        return self.stages[:self.n_stages]


# ── Random sampling ────────────────────────────────────────────────────────────

def _rand_stage(forbidden_blocks=None) -> StageGene:
    allowed = [b for b in BLOCK_TYPES if b not in (forbidden_blocks or [])]
    return StageGene(
        block_type    = random.choice(allowed),
        kernel_idx    = random.randrange(len(KERNEL_LIST)),
        n_blocks      = random.randrange(len(N_BLOCKS_LIST)),
        channels_idx  = random.randrange(len(CHANNEL_LIST)),
        expansion_idx = random.randrange(len(EXPANSION_LIST)),
        dilation_idx  = random.randrange(len(DILATION_LIST)),
        se_enabled    = random.randint(0, 1),
        se_ratio_idx  = random.randrange(len(SE_RATIO_LIST)),
        skip_mode     = random.choice(SKIP_MODES),
        downsample    = random.choice(DOWNSAMPLE_OPS),
        drop_path_idx = random.randrange(len(DROP_PATH_LIST)),
        group_w_idx   = random.randrange(len(GROUP_W_LIST)),
    )


def sample_random_genotype(
    n_stages: int = None,
    preferred_blocks=None,
    forbidden_blocks=None,
) -> Genotype:
    if n_stages is None:
        n_stages = random.randint(2, MAX_STAGES)
    stages = []
    for _ in range(MAX_STAGES):
        if preferred_blocks:
            block_pool = preferred_blocks + [b for b in BLOCK_TYPES if b not in (forbidden_blocks or [])]
        else:
            block_pool = [b for b in BLOCK_TYPES if b not in (forbidden_blocks or [])]
        stage = _rand_stage(forbidden_blocks)
        stage.block_type = random.choice(block_pool)
        stages.append(stage)
    return Genotype(
        n_stages      = n_stages,
        stem_type     = random.choice(STEM_TYPES),
        stem_channels = random.randrange(len(CHANNEL_LIST)),
        neck_type     = random.choice(NECK_TYPES),
        head_type     = random.choice(HEAD_TYPES),
        head_dropout  = random.randrange(len(DROPOUT_LIST)),
        norm_type     = random.choice(NORM_TYPES),
        stages        = stages,
    )


# ── Mutation operators ─────────────────────────────────────────────────────────

def _mutate_stage_field(stage: StageGene, field_name: str) -> None:
    if field_name == 'block_type':
        stage.block_type = random.choice(BLOCK_TYPES)
    elif field_name == 'kernel_idx':
        stage.kernel_idx = random.randrange(len(KERNEL_LIST))
    elif field_name == 'n_blocks':
        stage.n_blocks = random.randrange(len(N_BLOCKS_LIST))
    elif field_name == 'channels_idx':
        stage.channels_idx = random.randrange(len(CHANNEL_LIST))
    elif field_name == 'expansion_idx':
        stage.expansion_idx = random.randrange(len(EXPANSION_LIST))
    elif field_name == 'dilation_idx':
        stage.dilation_idx = random.randrange(len(DILATION_LIST))
    elif field_name == 'se_enabled':
        stage.se_enabled = 1 - stage.se_enabled
    elif field_name == 'se_ratio_idx':
        stage.se_ratio_idx = random.randrange(len(SE_RATIO_LIST))
    elif field_name == 'skip_mode':
        stage.skip_mode = random.choice(SKIP_MODES)
    elif field_name == 'downsample':
        stage.downsample = random.choice(DOWNSAMPLE_OPS)
    elif field_name == 'drop_path_idx':
        stage.drop_path_idx = random.randrange(len(DROP_PATH_LIST))
    elif field_name == 'group_w_idx':
        stage.group_w_idx = random.randrange(len(GROUP_W_LIST))


def mutate_small(g: Genotype, n_mutations: int = 1) -> Genotype:
    """Change 1-2 fields within a single stage."""
    g = g.clone()
    stage_idx = random.randrange(g.n_stages)
    stage_fields = list(STAGE_FIELD_CARDINALITY.keys())
    for _ in range(n_mutations):
        f = random.choice(stage_fields)
        _mutate_stage_field(g.stages[stage_idx], f)
    return g


def mutate_medium(g: Genotype, n_mutations: int = 2) -> Genotype:
    """Change fields across multiple stages."""
    g = g.clone()
    for _ in range(n_mutations):
        stage_idx = random.randrange(g.n_stages)
        f = random.choice(list(STAGE_FIELD_CARDINALITY.keys()))
        _mutate_stage_field(g.stages[stage_idx], f)
    return g


def mutate_large(g: Genotype) -> Genotype:
    """Replace one stage entirely or change global gene."""
    g = g.clone()
    if random.random() < 0.5:
        # replace a full stage
        stage_idx = random.randrange(g.n_stages)
        g.stages[stage_idx] = _rand_stage()
    else:
        # mutate a global gene
        global_fields = list(GLOBAL_FIELD_CARDINALITY.keys())
        f = random.choice(global_fields)
        if f == 'n_stages':
            g.n_stages = random.randint(2, MAX_STAGES)
        elif f == 'stem_type':
            g.stem_type = random.choice(STEM_TYPES)
        elif f == 'stem_channels':
            g.stem_channels = random.randrange(len(CHANNEL_LIST))
        elif f == 'neck_type':
            g.neck_type = random.choice(NECK_TYPES)
        elif f == 'head_type':
            g.head_type = random.choice(HEAD_TYPES)
        elif f == 'head_dropout':
            g.head_dropout = random.randrange(len(DROPOUT_LIST))
        elif f == 'norm_type':
            g.norm_type = random.choice(NORM_TYPES)
    return g


def mutate(g: Genotype, scale: str = 'medium') -> Genotype:
    if scale == 'small':
        return mutate_small(g)
    elif scale == 'large':
        return mutate_large(g)
    return mutate_medium(g)
