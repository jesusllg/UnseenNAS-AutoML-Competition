"""
AZ-NAS zero-cost proxies adapted from cvlab-yonsei/AZ-NAS (CVPR 2024).

Original source:
  https://github.com/cvlab-yonsei/AZ-NAS
  NB201/ZeroShotProxy/compute_az_nas_score.py

Adaptations:
- Removed dependency on NB201-specific model interface.
- Assumes model exposes extract_layer_features(x) → List[Tensor].
- Works with arbitrary NCHW inputs (no fixed resolution or search-space).
- Kaiming normal fan_in re-init applied before scoring.
- Complexity uses log(params) instead of FLOPs (search-space agnostic).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


# ── Model init (from AZ-NAS) ──────────────────────────────────────────────────

def _kaiming_normal_fanin_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _init_model(model: nn.Module) -> None:
    model.apply(_kaiming_normal_fanin_init)


# ── Individual proxies ────────────────────────────────────────────────────────

def _expressivity_score(feat: torch.Tensor) -> float:
    """
    Entropy of eigenvalue distribution of per-layer spatial covariance.
    Adapted from AZ-NAS compute_az_nas_score.py.
    """
    feat = feat.detach().clone()
    b, c, h, w = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
    m    = feat.mean(dim=0, keepdim=True)
    feat = feat - m
    sigma = torch.mm(feat.transpose(1, 0), feat) / feat.size(0)
    s = torch.linalg.eigvalsh(sigma)
    s = s.clamp(min=0)
    total = s.sum()
    if total <= 0:
        return 0.0
    prob_s = s / total
    score  = (-prob_s * torch.log(prob_s + 1e-8)).sum().item()
    return score


def expressivity(model: nn.Module, batch_x: torch.Tensor,
                 device: torch.device) -> float:
    model.train()
    model.to(device)
    x = batch_x.to(device)
    with torch.no_grad():
        feats = model.extract_layer_features(x)
    scores = [_expressivity_score(f) for f in feats]
    return float(np.sum(scores)) if scores else -np.inf


def progressivity(model: nn.Module, batch_x: torch.Tensor,
                  device: torch.device) -> float:
    model.train()
    model.to(device)
    x = batch_x.to(device)
    with torch.no_grad():
        feats = model.extract_layer_features(x)
    if len(feats) < 2:
        return -np.inf
    scores = np.array([_expressivity_score(f) for f in feats])
    diffs  = scores[1:] - scores[:-1]
    return float(np.min(diffs))


def trainability(model: nn.Module, batch_x: torch.Tensor,
                 device: torch.device) -> float:
    """
    Inter-layer Jacobian singular-value-based trainability.
    Adapted from AZ-NAS compute_az_nas_score.py.
    """
    model.train()
    model.to(device)
    x = batch_x.to(device)
    layer_features = model.extract_layer_features(x)

    scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in  = layer_features[i - 1]
        if not f_out.requires_grad or not f_in.requires_grad:
            continue

        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2

        try:
            g_in = torch.autograd.grad(
                outputs=f_out, inputs=f_in,
                grad_outputs=g_out, retain_graph=False,
                allow_unused=True,
            )[0]
        except Exception:
            continue

        if g_in is None:
            continue

        if g_out.size() == g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
            continue

        # align spatial dims — PixelUnshuffle requires integer, divisible stride
        if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
            bo, co, ho, wo = g_out.size()
            bi, ci, hi, wi = g_in.size()
            if ho == 0 or wo == 0 or hi % ho != 0 or wi % wo != 0:
                continue  # non-divisible (e.g. 9→4): skip this pair
            sh, sw = hi // ho, wi // wo
            if sh != sw or sh < 1:
                continue  # non-uniform stride: skip
            if sh > 1:
                try:
                    g_in = nn.PixelUnshuffle(sh)(g_in)
                except Exception:
                    continue
            bi, ci, hi, wi = g_in.size()

        bo, co, ho, wo = g_out.size()
        bi, ci, hi, wi = g_in.size()

        # flattened spatial must match for the Jacobian mm
        if bo * ho * wo != bi * hi * wi:
            continue

        g_out_flat = g_out.permute(0, 2, 3, 1).contiguous().view(bo * ho * wo, co)
        g_in_flat  = g_in.permute(0, 2, 3, 1).contiguous().view(bi * hi * wi, ci)
        mat = torch.mm(g_in_flat.transpose(1, 0), g_out_flat) / (bo * ho * wo)

        if mat.size(0) < mat.size(1):
            mat = mat.transpose(0, 1)

        try:
            s = torch.linalg.svdvals(mat)
            sv_max = s.max().item()
            scores.append(-sv_max - 1.0 / (sv_max + 1e-6) + 2.0)
        except Exception:
            continue

    if not scores:
        return -np.inf
    finite = [v for v in scores if np.isfinite(v)]
    return float(np.mean(finite)) if finite else -np.inf


def complexity_penalty(model: nn.Module) -> float:
    """log(total_params) — higher params ↔ higher penalty."""
    n = sum(p.numel() for p in model.parameters())
    return float(np.log(n + 1))


# ── Combined AZ-NAS score ─────────────────────────────────────────────────────

# Weight of the complexity penalty -lambda_c*log(params) in the combined score.
# Single source of truth in config.py (see LAMBDA_COMPLEXITY there for the full
# rationale: a light-model tie-breaker, not a dominant size pressure).
try:
    from config import LAMBDA_COMPLEXITY as _LAMBDA_COMPLEXITY
except ImportError:
    # search_space can be imported without submission/ on sys.path (e.g. an
    # isolated unit test). Fall back to the documented default so scoring works.
    _LAMBDA_COMPLEXITY = 0.05


def az_nas_score(
    model:    nn.Module,
    batch_x:  torch.Tensor,
    device,
    lambda_c: float = _LAMBDA_COMPLEXITY,
    reinit:   bool  = True,
) -> float:
    """
    Assembly: E + P + T - lambda_c * C

    Scores are raw (no rank-normalization), which is appropriate when
    evaluating a single architecture. When evaluating a population,
    use az_nas_score_population() for rank-normalised comparison.

    Returns the combined scalar score (higher = better architecture).
    """
    if reinit:
        _init_model(model)

    if isinstance(device, str):
        device = torch.device(device)

    def _safe(fn):
        try:
            return fn(model, batch_x, device)
        except Exception:
            return -np.inf

    e = _safe(expressivity)
    p = _safe(progressivity)
    t = _safe(trainability)
    c = complexity_penalty(model)

    # Use whichever proxies are finite — if all fail, return -inf.
    # Gracefully handles datasets with odd spatial dims (e.g. Sudoku 9×9)
    # where some proxies consistently fail.
    parts = [v for v in (e, p, t) if np.isfinite(v)]
    if not parts:
        return -np.inf
    return sum(parts) - lambda_c * c


def az_nas_score_full(
    model:    nn.Module,
    batch_x:  torch.Tensor,
    device,
    lambda_c: float = _LAMBDA_COMPLEXITY,
    reinit:   bool  = True,
) -> dict:
    """Same as az_nas_score but returns each component."""
    if reinit:
        _init_model(model)

    if isinstance(device, str):
        device = torch.device(device)

    e = expressivity(model, batch_x, device)
    p = progressivity(model, batch_x, device)
    t = trainability(model, batch_x, device)
    c = complexity_penalty(model)
    combined = (e + p + t - lambda_c * c) if all(np.isfinite(v) for v in [e, p, t]) else -np.inf

    return {
        'expressivity':  e,
        'progressivity': p,
        'trainability':  t,
        'complexity':    c,
        'az_nas':        combined,
    }
