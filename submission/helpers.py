import json
import math
import random as _random
import time
from pathlib import Path

import numpy as np
import torch


GLOBAL_SEED = 42

# ── Competition-wide time constants ──────────────────────────────────────────

N_COMPETITION_DATASETS     = 3     # known: 3 datasets
TOTAL_COMPETITION_HOURS    = 24.0  # known: 24h total budget
COMPETITION_OVERHEAD_HOURS = 0.5   # safety margin for I/O, imports, scoring

_FAMILY_COST_WEIGHT = {
    'small_grid':          0.40,
    'compact_general':     0.70,
    'visual_medium':       1.00,
    'channel_heavy':       1.20,
    'possible_voxel':      1.30,
    'anisotropic':         1.80,
    'spatiotemporal_like': 1.80,
    'visual_large':        2.00,
}


def set_seeds(seed: int = GLOBAL_SEED) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


def estimate_dataset_cost(meta: dict, dataset_dir=None) -> float:
    """
    Estimate relative compute cost from metadata.

    Uses actual train_x.npy file size when dataset_dir is provided —
    much more reliable than shape alone since log() compresses 10 GB vs 600 MB
    into almost the same value. Falls back to input_shape when file is absent.
    """
    shape = meta.get('input_shape', [0, 1, 1, 1])
    N, C, H, W = shape[0], shape[1], shape[2], shape[3]
    n_cls = meta.get('num_classes', 10)

    # Volume in elements; prefer actual file size (avoids log-compression of N)
    volume = max(N * C * H * W, 1)
    if dataset_dir is not None:
        npy = Path(dataset_dir) / 'train_x.npy'
        if npy.exists():
            volume = max(npy.stat().st_size / 4, 1)  # float32 → element count

    try:
        from search_space import infer_family
        fw = _FAMILY_COST_WEIGHT.get(infer_family(C, H, W, n_cls).name, 1.0)
    except Exception:
        fw = 1.0
    class_factor = max(1.0, n_cls / 10.0)
    # x^0.4 preserves large differences better than log while avoiding pure linearity
    return (volume ** 0.4) * fw * class_factor


def compute_allocations(costs: dict, pool_seconds: float,
                        min_h: float = 2.0, max_h: float = 12.0) -> dict:
    """Proportional allocation with floor/ceiling; re-scales if ceiling clips."""
    names = list(costs)
    raw = np.array([costs[n] for n in names], dtype=float)
    if raw.sum() == 0:
        raw = np.ones(len(names))
    props = raw / raw.sum()
    allocs = np.clip(props * pool_seconds, min_h * 3600, max_h * 3600)
    if allocs.sum() > pool_seconds:
        allocs *= pool_seconds / allocs.sum()
    return dict(zip(names, allocs.tolist()))


class GlobalBudgetGovernor:
    """
    Tracks cumulative time usage across datasets via predictions/.global_budget.json.

    Call get_allocation() at NAS init to learn how many seconds this dataset
    should consume. Call record_start() to mark when work begins. Call
    record_done() when the full pipeline for this dataset is complete so the
    next dataset's allocation is computed correctly.

    Never writes to any datasets/*/metadata file.
    """

    _STATE_FILE = Path("predictions") / ".global_budget.json"

    def __init__(self, n_total: int = N_COMPETITION_DATASETS,
                 total_hours: float = TOTAL_COMPETITION_HOURS,
                 overhead_hours: float = COMPETITION_OVERHEAD_HOURS):
        self.n_total = n_total
        self.pool_s  = (total_hours - overhead_hours) * 3600
        self._state  = self._load()

    def _load(self) -> dict:
        try:
            if self._STATE_FILE.exists():
                state = json.loads(self._STATE_FILE.read_text())
                # Auto-reset when a previous full run completed so the next run starts clean
                if state.get('n_done', 0) >= self.n_total:
                    return {'n_done': 0, 'seconds_used': 0.0}
                return state
        except Exception:
            pass
        return {'n_done': 0, 'seconds_used': 0.0}

    def _save(self):
        self._STATE_FILE.parent.mkdir(exist_ok=True)
        self._STATE_FILE.write_text(json.dumps(self._state, indent=2))

    def get_allocation(self, meta: dict, datasets_dir: Path = None) -> float:
        """Return effective budget in seconds for the current dataset."""
        n_done       = self._state.get('n_done', 0)
        seconds_used = self._state.get('seconds_used', 0.0)
        remaining    = max(0.0, self.pool_s - seconds_used)
        n_remaining  = max(1, self.n_total - n_done)

        if datasets_dir is not None:
            costs = self._scan_costs(meta, Path(datasets_dir))
            if len(costs) > 1:
                allocs = compute_allocations(costs, remaining)
                name   = meta.get('codename', '')
                for k, v in allocs.items():
                    if k == name or name in k or k in name:
                        alloc_h = v / 3600
                        print(f"  [GBG] proportional alloc={alloc_h:.2f}h"
                              f" ({remaining/3600:.2f}h pool, {len(costs)} datasets visible)")
                        return float(v)

        alloc = remaining / n_remaining
        print(f"  [GBG] equal-split alloc={alloc/3600:.2f}h"
              f" ({remaining/3600:.2f}h pool ÷ {n_remaining} remaining)")
        return alloc

    def _scan_costs(self, current_meta: dict, datasets_dir: Path) -> dict:
        import re as _re, json as _json
        costs = {}
        if not datasets_dir.exists():
            return costs
        for p in sorted(datasets_dir.iterdir()):
            if not (p.is_dir() and (p / "metadata").exists()):
                continue
            try:
                raw = (p / "metadata").read_bytes().decode("utf-8-sig").strip()
                raw = _re.sub(
                    r':\s*(?:\?|N/A|NA|NaN|nan|None|undefined)(?=\s*[,}\]\r\n])',
                    ': null', raw)
                m   = _json.loads(raw)
                costs[m.get('codename', p.name)] = estimate_dataset_cost(m, dataset_dir=p)
            except Exception:
                costs[p.name] = estimate_dataset_cost(current_meta, dataset_dir=p)
        return costs

    def record_start(self, dataset_name: str = ''):
        self._state['current'] = dataset_name
        self._save()

    def record_done(self, seconds_actual: float):
        self._state['n_done']       = self._state.get('n_done', 0) + 1
        self._state['seconds_used'] = self._state.get('seconds_used', 0.0) + seconds_actual
        self._save()
        n_remaining = max(0, self.n_total - self._state['n_done'])
        pool_left   = max(0.0, self.pool_s - self._state['seconds_used'])
        carry       = pool_left / max(1, n_remaining)
        print(f"  [GBG] done={self._state['n_done']}/{self.n_total}"
              f"  used={show_time(seconds_actual)}"
              f"  pool_left={show_time(pool_left)}"
              f"  ~{show_time(carry)}/dataset")
