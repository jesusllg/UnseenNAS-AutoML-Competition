import json
import math
import os
import random as _random
import re
import time
from pathlib import Path

import numpy as np
import torch


# Reproducibility lives in config.py (the single source of truth).
# Re-exported here so existing `from helpers import GLOBAL_SEED` keeps working.
from config import (GLOBAL_SEED, SEARCH_FRAC,
                    PREDICT_RESERVE_FRAC, PREDICT_RESERVE_MIN_S,
                    PREDICT_RESERVE_MAX_FRAC, PREDICT_RESERVE_MAX_S)


# ── Time authority (single source of truth for "how long do we have?") ────────

def get_safe_time_remaining(metadata, clock) -> float:
    """
    Seconds remaining for THIS dataset, reconciled conservatively.

    Per the organisers: time applies per dataset; metadata['time_remaining']
    is the remaining time at the last checkpoint. The live clock, when
    present, is at least as fresh, so we take the MINIMUM of the available
    sources — never the most optimistic one. Fallback chain when both are
    missing: metadata['time_limit'] (hours) with a 10% haircut, then the
    official 0.5 h default with the same haircut.
    """
    vals = []
    if clock is not None:
        try:
            vals.append(float(clock.check()))
        except Exception:
            pass
    if isinstance(metadata, dict):
        tr = metadata.get('time_remaining')
        if isinstance(tr, (int, float)):
            vals.append(float(tr))
    if vals:
        return max(0.0, min(vals))
    tl = metadata.get('time_limit') if isinstance(metadata, dict) else None
    if isinstance(tl, (int, float)) and tl > 0:
        return float(tl) * 3600.0 * 0.9
    return 0.5 * 3600.0 * 0.9   # official default time_limit, conservatively


def predict_reserve_s(remaining_s: float) -> float:
    """Seconds held back from training for prediction + output writing."""
    return min(
        max(PREDICT_RESERVE_MIN_S, remaining_s * PREDICT_RESERVE_FRAC),
        remaining_s * PREDICT_RESERVE_MAX_FRAC,
        PREDICT_RESERVE_MAX_S,
    )


def split_budget(remaining_s: float, search_frac: float = SEARCH_FRAC) -> dict:
    """
    Map "seconds left for this dataset" onto phase budgets:
      search_s  — NAS budget when called at search start
      train_s   — training budget when called at training start
      reserve_s — held back for predict + writing outputs (never trained away)
    """
    reserve = predict_reserve_s(remaining_s)
    return {
        'search_s':  remaining_s * search_frac,
        'train_s':   max(0.0, remaining_s - reserve),
        'reserve_s': reserve,
    }

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

def gpu_total_mb() -> float:
    """Total VRAM of GPU 0 in MB, or 0.0 when CUDA is unavailable/broken."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def select_batch_size(C: int, H: int, W: int) -> int:
    """
    THE batch-size rule, keyed on input pixel count (channels × H × W) and
    scaled down on small GPUs (evaluation hardware is unknown — could be a T4
    or worse).

    One definition used by both sides so they can never disagree:
      • DataProcessor builds the loaders with it.
      • repair.py estimates training memory with it, so the estimate matches
        the batch the trainer will actually run (a mismatch here silently
        under/over-estimates memory and lets oversized models OOM the trainer).
    """
    from config import LOW_VRAM_MB
    pixels = C * H * W
    if pixels > 100_000:
        bs = 16
    elif pixels > 10_000:
        bs = 32
    else:
        bs = 64
    vram = gpu_total_mb()
    if 0.0 < vram < LOW_VRAM_MB:
        bs = max(8, bs // 2)
    return bs


def select_num_workers(n_samples: int) -> int:
    """
    DataLoader workers for unknown hardware: 0 on weak CPUs or small datasets
    (worker startup costs more than it saves), at most 2 otherwise.
    """
    try:
        cpus = os.cpu_count() or 1
    except Exception:
        cpus = 1
    if cpus <= 2 or n_samples < 2000:
        return 0
    return 2


def make_amp_tools(use_amp: bool):
    """
    (autocast_factory, grad_scaler) working on both PyTorch 1.10 and 2.x.
    2.x: torch.amp.GradScaler('cuda') / torch.amp.autocast('cuda').
    1.10: torch.cuda.amp.GradScaler() / torch.cuda.amp.autocast().
    """
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        def autocast():
            return torch.amp.autocast('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        from torch.cuda import amp as _cuda_amp
        scaler = _cuda_amp.GradScaler(enabled=use_amp)

        def autocast():
            return _cuda_amp.autocast(enabled=use_amp)
    return autocast, scaler


def is_oom(err: BaseException) -> bool:
    """CUDA OOM check across torch versions (OutOfMemoryError is 1.13+)."""
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    return isinstance(err, RuntimeError) and 'out of memory' in str(err).lower()


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
    Halving time-allocation policy for LOCAL --total-time test runs only.
    (The official 2026 evaluator gives each dataset its own fixed clock, so
    nothing in the submission itself allocates across datasets.)

      - last dataset  → entire remaining pool (inherits all unused time)
      - otherwise     → half the remaining pool
                        (DS1 = pool/2, DS2 = leftover/2, DS3 = everything left)
    """
    if n_remaining <= 1:
        return pool_s
    return pool_s / 2
