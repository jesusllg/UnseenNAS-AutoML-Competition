import copy
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

from helpers import show_time


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, padding=p, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel, padding=p, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


class SearchableCNN(nn.Module):
    """Configurable CNN built from a list of stage configs."""

    def __init__(self, in_channels, num_classes, stage_configs, input_hw):
        """
        stage_configs: list of (out_channels, num_blocks, kernel_size, use_residual)
        input_hw: (H, W) — controls how many safe downsampling steps exist
        """
        super().__init__()
        h = min(input_hw)
        max_pool_steps = 0
        while h > 4:
            h //= 2
            max_pool_steps += 1

        layers = []
        c = in_channels
        for i, (out_c, n_blocks, ks, use_res) in enumerate(stage_configs):
            layers.append(ConvBnAct(c, out_c, ks))
            for _ in range(n_blocks - 1):
                layers.append(ResBlock(out_c, ks) if use_res else ConvBnAct(out_c, out_c, ks))
            if i < max_pool_steps:
                layers.append(nn.MaxPool2d(2, 2))
            c = out_c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


# ── Search space ───────────────────────────────────────────────────────────────

_CHANNELS = [32, 64, 128, 256]
_KERNELS  = [3, 5]
_BLOCKS   = [1, 2, 3]


def _sample_stage_configs(rng, n_stages, min_c_idx=0):
    """Sample a random stage config list. min_c_idx forces larger starting channels."""
    configs = []
    c_idx = min_c_idx
    for _ in range(n_stages):
        c_idx = int(rng.randint(c_idx, min(c_idx + 2, len(_CHANNELS) - 1) + 1))
        configs.append((
            _CHANNELS[c_idx],
            int(rng.choice(_BLOCKS)),
            int(rng.choice(_KERNELS)),
            bool(rng.randint(0, 2)),
        ))
    return configs


# ── Benchmark-aware search calibration ────────────────────────────────────────

def _search_params(benchmark):
    """
    Return (budget_frac, proxy_epochs, proxy_batches, min_c_idx) tuned to benchmark.

    Scoring formula: adj = (acc% - benchmark%) * 10/(100 - benchmark%)
    High-benchmark datasets yield more points per % gain → invest more effort.
    Low-benchmark datasets are easy to beat → a quick search suffices.
    """
    if benchmark is None:
        return 0.30, 3, 40, 0
    b = float(benchmark)
    if b >= 85:
        # Hard to beat but each % is worth a lot — maximise search depth
        return 0.35, 5, 50, 1   # start channels at 64+
    elif b >= 65:
        return 0.30, 3, 40, 0
    else:
        # Easy win — short search, any reasonable model will beat baseline
        return 0.20, 3, 30, 0


# ── Proxy (fast) evaluation ────────────────────────────────────────────────────

def _proxy_train(model, loader, device, n_epochs, max_batches):
    model.to(device).train()
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return model


def _proxy_acc(model, loader, device, max_batches=20):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


# ── Default fallback architectures, scaled by benchmark tier ──────────────────

_FALLBACK = {
    'hard':   [(64, 2, 3, True),  (128, 2, 3, True),  (256, 2, 3, True)],
    'medium': [(64, 2, 3, True),  (128, 2, 3, True),  (256, 1, 3, False)],
    'easy':   [(32, 2, 3, False), (64,  1, 3, True)],
}

def _fallback_stages(benchmark):
    if benchmark is None or float(benchmark) >= 65:
        return _FALLBACK['medium']
    return _FALLBACK['easy']


# ── NAS class ──────────────────────────────────────────────────────────────────

class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def search(self):
        try:
            return self._search()
        except Exception:
            print("  [NAS] Unexpected error — using fallback architecture.")
            print(traceback.format_exc())
            in_c  = self.metadata['input_shape'][1]
            n_cls = self.metadata['num_classes']
            hw    = tuple(self.metadata['input_shape'][2:])
            benchmark = self.metadata.get('benchmark', None)
            return SearchableCNN(in_c, n_cls, _fallback_stages(benchmark), hw)

    def _search(self):
        in_c      = self.metadata['input_shape'][1]
        n_cls     = self.metadata['num_classes']
        hw        = tuple(self.metadata['input_shape'][2:])
        benchmark = self.metadata.get('benchmark', None)

        budget_frac, proxy_epochs, proxy_batches, min_c_idx = _search_params(benchmark)

        search_budget = self.clock.check() * budget_frac
        deadline      = time.perf_counter() + search_budget

        print(f"  NAS | benchmark={benchmark} → budget={show_time(search_budget)}"
              f" proxy={proxy_epochs}ep×{proxy_batches}b | device={self.device}")

        rng = np.random.RandomState(42)
        best_model, best_acc = None, -1.0
        n_tried = 0

        while time.perf_counter() < deadline:
            n_stages  = int(rng.randint(2, 5))
            stage_cfg = _sample_stage_configs(rng, n_stages, min_c_idx)

            try:
                model = SearchableCNN(in_c, n_cls, stage_cfg, hw)
                model = _proxy_train(model, self.train_loader, self.device,
                                     proxy_epochs, proxy_batches)
                acc   = _proxy_acc(model, self.valid_loader, self.device)
                n_tried += 1
                marker = " ★" if acc > best_acc else ""
                print(f"  [{n_tried:>2}] stages={n_stages} cfg={stage_cfg}"
                      f" → val={acc:.3f}{marker}")

                if acc > best_acc:
                    best_acc  = acc
                    best_model = copy.deepcopy(model).cpu()

            except RuntimeError as e:
                print(f"  Architecture skipped: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"  Search done: {n_tried} tried, best val={best_acc:.3f}")

        if best_model is None:
            print("  No valid architecture found — using fallback.")
            best_model = SearchableCNN(in_c, n_cls, _fallback_stages(benchmark), hw)

        return best_model
