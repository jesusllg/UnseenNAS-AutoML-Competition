"""
Curated seed genotypes per geometry family.

Aging Evolution previously initialised its population from random genotypes
only — on a short clock (phase 2) or an unlucky seed, the whole population
could start from junk. These are small/medium, known-trainable architectures
that anchor the population; the rest is still filled with repaired random
genotypes, so exploration is preserved. Every seed passes through the normal
repair → build → dry-run → proxy path, so an invalid seed for some exotic
geometry is silently skipped, never fatal.

Index cheatsheet (see genotype.py):
  CHANNEL_LIST  [16, 24, 32, 48, 64, 96, 128, 192, 256]   idx 2=32 4=64 6=128
  KERNEL_LIST   [1, 3, 5, 7]                              idx 1=k3  2=k5
  N_BLOCKS_LIST [1, 2, 3, 4]                              idx 1=2 blocks
  EXPANSION_LIST[1, 2, 4, 6]                              idx 0=1×  2=4×
"""
from typing import List

from .genotype import Genotype, StageGene
from .family import FamilyProfile


def _stage(block: str, ch_i: int, k_i: int = 1, n_i: int = 1,
           ds: str = 'maxpool', exp_i: int = 0, se: int = 0) -> StageGene:
    return StageGene(block_type=block, channels_idx=ch_i, kernel_idx=k_i,
                     n_blocks=n_i, downsample=ds, expansion_idx=exp_i,
                     se_enabled=se)


def _geno(stages: List[StageGene], stem_ch_i: int = 2, stem: str = 'conv3x3',
          head: str = 'GapLinear', norm: str = 'batch') -> Genotype:
    return Genotype(n_stages=len(stages), stem_type=stem, stem_channels=stem_ch_i,
                    head_type=head, norm_type=norm, stages=list(stages))


def seed_genotypes_for_family(family: FamilyProfile, C: int, H: int, W: int,
                              num_classes: int) -> List[Genotype]:
    """
    3-6 safe starting architectures for the family. Deliberately small/medium:
    the point is a trainable, competitive baseline in the population from
    round zero, not a giant that games the proxy. repair() adapts each one to
    the exact geometry (norm override, pool budget, kernel clamps).
    """
    name = family.name

    if name == 'anisotropic':
        # Factorised convs along the long axis + channel mixing + global
        # context. Downsampling via pools only (repair enforces this anyway).
        return [
            _geno([_stage('AnisotropicBlock', 2), _stage('AnisotropicBlock', 4),
                   _stage('GlobalContextBlock', 5, ds='identity')], norm='group'),
            _geno([_stage('AnisotropicBlock', 2, k_i=2), _stage('ChannelMixingBlock', 4),
                   _stage('AnisotropicBlock', 5), _stage('GlobalContextBlock', 6, ds='identity')],
                  norm='group'),
            _geno([_stage('ConvBlock', 2), _stage('AnisotropicBlock', 4),
                   _stage('ChannelMixingBlock', 5, ds='identity')], norm='group'),
            _geno([_stage('AnisotropicBlock', 3, n_i=1), _stage('AnisotropicBlock', 5, n_i=1),
                   _stage('AnisotropicBlock', 6, ds='identity')], norm='group',
                  head='AttentionPool'),
        ]

    if name == 'compact_general':
        # Small non-natural images (e.g. text-as-image): compact, stable,
        # moderate receptive-field growth; nothing deep or wide.
        return [
            _geno([_stage('ConvBlock', 2), _stage('ResidualBlock', 4),
                   _stage('ConvBlock', 5, ds='identity')]),
            _geno([_stage('ResidualBlock', 2, n_i=1), _stage('ResidualBlock', 4, n_i=1),
                   _stage('GlobalContextBlock', 5, ds='identity')]),
            _geno([_stage('ConvBlock', 2), _stage('DilatedConvBlock', 4, ds='identity'),
                   _stage('ResidualBlock', 5)]),
            _geno([_stage('SepConvBlock', 3), _stage('ResidualBlock', 4, se=1),
                   _stage('GlobalContextBlock', 6, ds='identity')], norm='group'),
        ]

    if name == 'possible_voxel':
        # Volume-as-channels: mix channels early and often, keep spatial
        # structure, GroupNorm (repair forces it anyway), modest width.
        return [
            _geno([_stage('ChannelMixingBlock', 3), _stage('ResidualBlock', 4),
                   _stage('BottleneckBlock', 5, exp_i=2, ds='identity')], norm='group'),
            _geno([_stage('ConvBlock', 3, k_i=0), _stage('ChannelMixingBlock', 4),
                   _stage('ResidualBlock', 5, ds='identity')], norm='group'),
            _geno([_stage('BottleneckBlock', 4, exp_i=2), _stage('ChannelMixingBlock', 5, ds='identity'),
                   _stage('ResidualBlock', 6, ds='identity')], norm='group'),
        ]

    if name == 'small_grid':
        return [
            _geno([_stage('GridLogicBlock', 2, ds='identity'), _stage('ConvBlock', 4),
                   _stage('ResidualBlock', 5, ds='identity')], norm='group'),
            _geno([_stage('ConvBlock', 2, ds='identity'), _stage('GridLogicBlock', 4, ds='identity'),
                   _stage('LightAttentionBlock', 5, ds='identity')], norm='group'),
            _geno([_stage('ResidualBlock', 3, n_i=1, ds='identity'),
                   _stage('ConvBlock', 5, ds='identity')], norm='group', head='FlattenMlp'),
        ]

    if name == 'channel_heavy':
        return [
            _geno([_stage('ChannelMixingBlock', 3), _stage('MBConvBlock', 4, exp_i=1),
                   _stage('ResidualBlock', 5, ds='identity')]),
            _geno([_stage('ChannelMixingBlock', 4, ds='identity'), _stage('BottleneckBlock', 5, exp_i=2),
                   _stage('ResidualBlock', 6, ds='identity')], norm='group'),
            _geno([_stage('ConvBlock', 3, k_i=0), _stage('ChannelMixingBlock', 5),
                   _stage('GlobalContextBlock', 6, ds='identity')]),
        ]

    if name in ('visual_large', 'visual_medium'):
        big = name == 'visual_large'
        stem = 'conv3x3_s2' if big else 'conv3x3'
        return [
            _geno([_stage('ResidualBlock', 2, n_i=1, ds='stride2'),
                   _stage('ResidualBlock', 4, n_i=1, ds='stride2'),
                   _stage('ResidualBlock', 5, n_i=1, ds='stride2'),
                   _stage('ResidualBlock', 6, ds='identity')], stem=stem),
            _geno([_stage('MBConvBlock', 2, exp_i=2, ds='stride2'),
                   _stage('MBConvBlock', 4, exp_i=2, se=1, ds='stride2'),
                   _stage('MBConvBlock', 5, exp_i=2, se=1, ds='stride2'),
                   _stage('SepConvBlock', 6, ds='identity')], stem=stem),
            _geno([_stage('ConvBlock', 2, ds='stride2'), _stage('BottleneckBlock', 4, exp_i=2, ds='stride2'),
                   _stage('BottleneckBlock', 6, exp_i=2, ds='stride2')], stem=stem),
            _geno([_stage('SepConvBlock', 2, ds='stride2'), _stage('ResidualBlock', 4, ds='stride2'),
                   _stage('DilatedConvBlock', 5, ds='stride2'),
                   _stage('GlobalContextBlock', 6, ds='identity')], stem=stem),
        ]

    # spatiotemporal_like and anything unforeseen: safe generic ladder
    return [
        _geno([_stage('ConvBlock', 2), _stage('ResidualBlock', 4),
               _stage('ConvBlock', 5, ds='identity')]),
        _geno([_stage('SepConvBlock', 2), _stage('ResidualBlock', 4),
               _stage('GlobalContextBlock', 5, ds='identity')]),
    ]
