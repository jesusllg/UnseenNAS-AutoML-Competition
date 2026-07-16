"""
P0 diagnostic / regression test: Sudoku-style 9x9 grids and the trainability
proxy.

Hypothesis (confirmed on V3 results — Sudoku 87.6% -> 54.5%): on odd spatial
dims, pooled stage transitions (9->4) are non-divisible, trainability skips
those layer pairs, can return -inf, and az_nas_score silently DROPS the
component — so architectures whose trainability was never measured avoid a
(normally negative) contribution and outrank architectures that were fully
measured.

Run BEFORE the proxy fix to record the baseline, and AFTER to verify:
  1. T is finite for >=95% of pooled genotypes (alignment fix), and
  2. the pooled-without-T scoring advantage disappears.

Usage:  python tests/test_proxy_sudoku_bias.py
Exit code 0 = healthy (post-fix expectations met), 1 = bias present.
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

import numpy as np
import torch

from search_space import infer_family, repair, build_model, sample_random_genotype
from search_space.proxies import az_nas_score_full, trainability

C, H, W, N_CLS = 1, 9, 9, 9          # Sudoku-like geometry
N_GENOTYPES = 100
SEED = 0


def sample_rows(C, H, W, n_cls, n_genotypes, seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fam = infer_family(C, H, W, n_cls)
    batch = torch.randn(16, C, H, W)
    rows, tried = [], 0
    while len(rows) < n_genotypes and tried < n_genotypes * 20:
        tried += 1
        g = sample_random_genotype(preferred_blocks=fam.preferred_blocks,
                                   forbidden_blocks=fam.forbidden_blocks)
        try:
            g = repair(g, C, H, W, n_cls, fam)
            m = build_model(g, C, H, W, n_cls, aniso_axis=fam.aniso_axis)
            with torch.no_grad():
                m.cpu()(torch.zeros(2, C, H, W))
            comp = az_nas_score_full(m, batch, 'cpu')
        except Exception:
            continue
        stats  = getattr(trainability, 'last_stats', {'pairs': 0, 'skipped': 0})
        pooled = any(s.downsample != 'identity' for s in g.active_stages)
        rows.append({
            'pooled':   pooled,
            't':        comp['trainability'],
            't_finite': bool(np.isfinite(comp['trainability'])),
            'pairs':    stats['pairs'],
            'skipped':  stats['skipped'],
        })
    return fam, rows


def main():
    fam, rows = sample_rows(C, H, W, N_CLS, N_GENOTYPES)

    pooled    = [r for r in rows if r['pooled']]
    t_lost    = [r for r in pooled if not r['t_finite']]
    frac_t_ok = 1 - len(t_lost) / max(1, len(pooled))
    n_pairs   = sum(r['pairs'] for r in rows)
    n_skip    = sum(r['skipped'] for r in rows)
    any_skip  = sum(1 for r in rows if r['skipped'] > 0)

    print(f"family={fam.name}  genotypes={len(rows)}  pooled={len(pooled)}")
    print(f"T finite among pooled: {frac_t_ok*100:.0f}%")
    print(f"layer pairs measured: {n_pairs - n_skip}/{n_pairs}"
          f"  (architectures with >=1 unmeasured pair: {any_skip}/{len(rows)})")
    # Baseline (pre-fix, seed 0): 100/100 pooled genotypes had a SKIPPED 9->4
    # transition pair — T was averaged over a different pair-subset per
    # architecture, leaving the most informative (downsampling) boundary
    # unmeasured on odd grids. Post-fix both counters must be ~zero.
    coverage_ok = frac_t_ok >= 0.95 and n_skip / max(n_pairs, 1) <= 0.02

    # Distribution comparability: adaptive_avg_pool2d alignment (odd 9x9) is
    # an APPROXIMATION of the PixelUnshuffle path (even 8x8) — T measured
    # through it must live on the same scale, not just be finite.
    _, rows8 = sample_rows(1, 8, 8, N_CLS, 40, seed=1)
    t_odd  = np.array([r['t'] for r in rows  if r['t_finite']])
    t_even = np.array([r['t'] for r in rows8 if r['t_finite']])
    m_o, s_o = t_odd.mean(), t_odd.std()
    m_e, s_e = t_even.mean(), t_even.std()
    print(f"T dist  9x9 (adaptive-aligned): mean={m_o:.3f} std={s_o:.3f}"
          f"   8x8 (PixelUnshuffle): mean={m_e:.3f} std={s_e:.3f}")
    dist_ok = abs(m_o - m_e) <= 3 * max(s_o, s_e, 1e-3)
    if not dist_ok:
        print(">> T distributions diverge across alignment paths — recalibrate.")

    healthy = coverage_ok and dist_ok
    print("RESULT:", "HEALTHY (full coverage, comparable T scale)" if healthy
          else "DEFECT present (expected pre-fix)")
    return 0 if healthy else 1


if __name__ == '__main__':
    sys.exit(main())
