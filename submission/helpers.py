import json
import math
import random as _random
import re
import time
from pathlib import Path

import numpy as np
import torch


# Reproducibility + competition facts live in config.py (the single source of
# truth). Re-exported here so existing `from helpers import GLOBAL_SEED` etc.
# keep working unchanged.
from config import (GLOBAL_SEED, N_COMPETITION_DATASETS,
                    TOTAL_COMPETITION_HOURS, COMPETITION_OVERHEAD_HOURS)

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


# ── Hardware/runtime helpers (single source of truth) ────────────────────────

def select_batch_size(C: int, H: int, W: int) -> int:
    """
    THE batch-size rule, keyed on input pixel count (channels × H × W).

    One definition used by both sides so they can never disagree:
      • DataProcessor builds the loaders with it.
      • repair.py estimates training memory with it, so the estimate matches
        the batch the trainer will actually run (a mismatch here silently
        under/over-estimates memory and lets oversized models OOM the trainer).
    """
    pixels = C * H * W
    if pixels > 100_000:
        return 16
    if pixels > 10_000:
        return 32
    return 64


def free_gpu() -> None:
    """
    THE GPU-memory reclaim idiom: drop unreferenced Python tensors, then return
    the CUDA caching allocator's unused blocks to the driver.

    Call at phase boundaries (entering NAS, handing a model from search to
    training, after an OOM) so the next phase starts without the previous
    phase's reserved-but-idle memory inflating the allocator's footprint.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


# ── Metadata loading (single shared implementation) ──────────────────────────

def sanitize_metadata_json(raw: str) -> str:
    """
    Coerce non-standard bare values in metadata JSON to valid JSON equivalents.
    Handles: ? NA N/A NaN nan Inf -Inf Infinity None undefined True False.
    Quoted string values like "benchmark": "NA" are left untouched.
    """
    _NULL_TOKENS = (
        r'\?'
        r'|N/A|NA'
        r'|NaN|nan'
        r'|[+-]?[Ii]nfinity|[+-]?[Ii]nf'
        r'|None'
        r'|undefined'
    )
    raw = re.sub(r':\s*(?:' + _NULL_TOKENS + r')(?=\s*[,}\]\r\n])', ': null', raw)
    raw = re.sub(r':\s*True(?=\s*[,}\]\r\n])',  ': true',  raw)
    raw = re.sub(r':\s*False(?=\s*[,}\]\r\n])', ': false', raw)
    return raw


def load_metadata(dataset_path) -> dict:
    """Read and sanitize a dataset's metadata file (handles BOM + bare tokens)."""
    raw = (Path(dataset_path) / "metadata").read_bytes().decode("utf-8-sig").strip()
    return json.loads(sanitize_metadata_json(raw))


# ── Dataset cost & time-allocation policy (single source of truth) ───────────

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


def halving_allocation(pool_s: float, n_remaining: int) -> float:
    """
    THE time-allocation policy. Everything that allocates time calls this.

      - last dataset  → entire remaining pool (inherits all unused time)
      - otherwise     → half the remaining pool
                        (DS1 = pool/2, DS2 = leftover/2, DS3 = everything left)
    """
    if n_remaining <= 1:
        return pool_s
    return pool_s / 2


class GlobalBudgetGovernor:
    """
    Owns the entire per-dataset time-budget lifecycle. Persists cumulative
    usage across datasets in predictions/.global_budget.json so the allocation
    survives the organizer creating a fresh Clock per dataset.

    Lifecycle (the ONLY contract between NAS and Trainer is this object,
    handed over in-memory as metadata['_gbg'] — never written to disk):

        gbg = GlobalBudgetGovernor(...)
        gbg.begin_dataset(codename)      # NAS.__init__: start wall clock, fix allocation
        gbg.allocation_s                 # NAS search: full allocation for this dataset
        gbg.effective_remaining()        # Trainer: allocation minus elapsed wall time
        gbg.finish_dataset()             # Trainer finally: record usage (idempotent)

    Never writes to any datasets/*/metadata file.
    """

    _STATE_FILE = Path("predictions") / ".global_budget.json"

    def __init__(self, n_total: int = N_COMPETITION_DATASETS,
                 total_hours: float = TOTAL_COMPETITION_HOURS,
                 overhead_hours: float = COMPETITION_OVERHEAD_HOURS):
        self.n_total      = n_total
        self.pool_s       = (total_hours - overhead_hours) * 3600
        self._state       = self._load()
        self._wall_start  = None   # set by begin_dataset()
        self.allocation_s = None   # set by begin_dataset()
        self._done        = False  # finish_dataset() idempotency guard

    # ── persistence ───────────────────────────────────────────────────────────

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

    @classmethod
    def reset_state(cls):
        """Delete the persisted state file (fresh local test run)."""
        if cls._STATE_FILE.exists():
            cls._STATE_FILE.unlink()
            print("  [GBG] Cleared stale state file — fresh run.")

    # ── allocation ────────────────────────────────────────────────────────────

    def get_allocation(self) -> float:
        """Seconds this dataset should consume, per the halving policy."""
        n_done       = self._state.get('n_done', 0)
        seconds_used = self._state.get('seconds_used', 0.0)
        remaining    = max(0.0, self.pool_s - seconds_used)
        n_remaining  = max(1, self.n_total - n_done)

        alloc = halving_allocation(remaining, n_remaining)
        tag   = "last dataset → full pool" if n_remaining == 1 else "halving"
        print(f"  [GBG] {tag} alloc={alloc/3600:.2f}h"
              f"  (pool={remaining/3600:.2f}h, done={n_done}/{self.n_total})")
        return alloc

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def begin_dataset(self, dataset_name: str = '') -> float:
        """Fix this dataset's allocation and start its wall clock."""
        self._wall_start  = time.perf_counter()
        self.allocation_s = self.get_allocation()
        self._state['current'] = dataset_name
        self._save()
        return self.allocation_s

    def current_allocation(self) -> float:
        """
        This dataset's allocation in seconds. Computes it on demand if
        begin_dataset() hasn't run yet, so callers can never read a None
        allocation (defensive — normal flow sets it in begin_dataset()).
        """
        if self.allocation_s is None:
            self.allocation_s = self.get_allocation()
        return self.allocation_s

    def elapsed(self) -> float:
        """Wall seconds since begin_dataset() (0 if not begun)."""
        if self._wall_start is None:
            return 0.0
        return time.perf_counter() - self._wall_start

    def effective_remaining(self) -> float:
        """Allocation minus elapsed wall time — what's truly left for this dataset."""
        if self.allocation_s is None:
            return self.pool_s
        return max(0.0, self.allocation_s - self.elapsed())

    def finish_dataset(self):
        """Record this dataset's actual usage. Safe to call more than once."""
        if self._done:
            return
        self._done = True
        seconds_actual = self.elapsed()
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
