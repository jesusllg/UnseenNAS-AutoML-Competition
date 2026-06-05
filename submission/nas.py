import copy
import time
import traceback
import logging

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from helpers import (show_time, set_seeds, GLOBAL_SEED,
                     GlobalBudgetGovernor,
                     N_COMPETITION_DATASETS, TOTAL_COMPETITION_HOURS)

logger = logging.getLogger(__name__)

# ── Legacy fallback model (SearchableCNN) ─────────────────────────────────────
# Kept intact so that if the new search space fails for any reason,
# we still produce a working model.

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
    """Configurable CNN built from a list of stage configs (legacy fallback)."""

    def __init__(self, in_channels, num_classes, stage_configs, input_hw):
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
        self.fc   = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


_FALLBACK = {
    'medium': [(64, 2, 3, True), (128, 2, 3, True), (256, 1, 3, False)],
}


# ── Search helpers ────────────────────────────────────────────────────────────

_N_POPULATION    = 100
_N_ROUNDS        = 2000   # effectively "run until time budget expires"
_TOURNAMENT_SIZE = 25
_SEARCH_FRAC     = 0.30


def _get_proxy_batch(train_loader, device, batch_size=16):
    """Grab a single small batch for zero-cost proxy evaluation."""
    try:
        x, _ = next(iter(train_loader))
        if x.size(0) > batch_size:
            x = x[:batch_size]
        return x.to(device)
    except Exception:
        return None


# ── NAS class ─────────────────────────────────────────────────────────────────

class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Release any GPU memory the previous dataset may have left behind
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Global Budget Governor ────────────────────────────────────────────
        self._wall_start = time.perf_counter()
        gbg = GlobalBudgetGovernor(
            n_total     = metadata.get('n_competition_datasets', N_COMPETITION_DATASETS),
            total_hours = metadata.get('total_competition_hours', TOTAL_COMPETITION_HOURS),
        )
        self._effective_budget_s = gbg.get_allocation(
            metadata, datasets_dir=Path('datasets')
        )
        # Inject into in-memory metadata dict so Trainer can read it (no disk writes)
        metadata['effective_budget_s']  = self._effective_budget_s
        metadata['_gbg']                = gbg
        metadata['_pipeline_wall_start'] = self._wall_start
        gbg.record_start(metadata.get('codename', 'unknown'))

    def search(self):
        try:
            return self._search()
        except Exception:
            print("  [NAS] Unexpected error — using fallback architecture.")
            print(traceback.format_exc())
            in_c = self.metadata['input_shape'][1]
            n_cls = self.metadata['num_classes']
            hw   = tuple(self.metadata['input_shape'][2:])
            return SearchableCNN(in_c, n_cls, _FALLBACK['medium'], hw)

    def _search(self):
        seed = self.metadata.get('seed', GLOBAL_SEED)
        set_seeds(seed)

        shape = self.metadata['input_shape']
        in_c  = shape[1]
        H, W  = shape[2], shape[3]
        n_cls = self.metadata['num_classes']

        n_pop     = self.metadata.get('n_population',    _N_POPULATION)
        n_rounds  = self.metadata.get('n_rounds',        _N_ROUNDS)
        tourney_k = self.metadata.get('tournament_size', _TOURNAMENT_SIZE)
        search_frac   = self.metadata.get('search_frac', _SEARCH_FRAC)
        # Cap search to our effective per-dataset budget, not just clock remaining
        search_budget = min(self.clock.check(), self._effective_budget_s) * search_frac
        t_search_start = time.perf_counter()
        print(f"  NAS | sf={search_frac:.2f} → budget={show_time(search_budget)}"
              f"  pop={n_pop} rounds={n_rounds} | device={self.device}")

        # ── Import search space ───────────────────────────────────────────────
        try:
            from search_space import (
                infer_family, sample_random_genotype, repair,
                build_model, az_nas_score, aging_evolution, best_individual,
            )
        except ImportError as e:
            print(f"  [NAS] search_space import failed ({e}), using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        # ── Proxy batch ───────────────────────────────────────────────────────
        proxy_batch = _get_proxy_batch(self.train_loader, self.device)
        if proxy_batch is None:
            print("  [NAS] Could not get proxy batch — using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        family = infer_family(in_c, H, W, n_cls)
        print(f"  NAS | family={family.name}  aniso={family.is_anisotropic}"
              f"  max_pools={family.max_pool_steps}  groupnorm={family.force_groupnorm}")

        def proxy_fn(model, batch_x, device):
            return az_nas_score(model, batch_x, device, reinit=True)

        # ── Aging evolution ───────────────────────────────────────────────────
        try:
            population = aging_evolution(
                family          = family,
                C               = in_c,
                H               = H,
                W               = W,
                num_classes     = n_cls,
                proxy_fn        = proxy_fn,
                batch_x         = proxy_batch,
                device          = self.device,
                n_population    = n_pop,
                n_rounds        = n_rounds,
                tournament_size = tourney_k,
                time_budget_s   = search_budget,
                seed            = seed,
                verbose         = True,
            )
        except Exception as e:
            print(f"  [NAS] Evolution failed ({e}) — using legacy fallback.")
            traceback.print_exc()
            return self._legacy_search(in_c, n_cls, (H, W))

        best = best_individual(population)
        if best is None:
            print("  [NAS] Empty population — using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        n_evaluated = len(population)
        search_elapsed = time.perf_counter() - t_search_start
        print(f"  Search done: {n_evaluated} evaluated, best AZ-NAS={best.fitness:.4f}")
        self.metadata['nas_report'] = {
            'n_tried':        n_evaluated,
            'best_proxy_val': round(float(best.fitness), 4),
            'best_stages':    str(best.genotype),
            'search_s':       round(search_elapsed, 1),
        }

        try:
            model = build_model(
                best.genotype, in_c, H, W, n_cls,
                aniso_axis=family.aniso_axis,
            )
            # quick sanity check
            with torch.no_grad():
                dummy = torch.randn(2, in_c, H, W).to(self.device)
                model.to(self.device)(dummy)
            return model.cpu()
        except Exception as e:
            print(f"  [NAS] Model build failed ({e}) — using legacy fallback.")
            traceback.print_exc()
            return self._legacy_search(in_c, n_cls, (H, W))

    # ── Legacy search (random + proxy train) ─────────────────────────────────

    _CHANNELS_L = [32, 64, 128, 256]
    _KERNELS_L  = [3, 5]
    _BLOCKS_L   = [1, 2, 3]

    def _sample_stage_configs_legacy(self, rng, n_stages, min_c_idx=0):
        configs = []
        c_idx = min_c_idx
        for _ in range(n_stages):
            c_idx = int(rng.randint(c_idx, min(c_idx + 2, len(self._CHANNELS_L) - 1) + 1))
            configs.append((
                self._CHANNELS_L[c_idx],
                int(rng.choice(self._BLOCKS_L)),
                int(rng.choice(self._KERNELS_L)),
                bool(rng.randint(0, 2)),
            ))
        return configs

    def _legacy_search(self, in_c, n_cls, hw):
        budget_frac, proxy_epochs, proxy_batches = 0.25, 3, 40
        search_budget = self.clock.check() * budget_frac
        deadline = time.perf_counter() + search_budget

        rng = np.random.RandomState(42)
        best_model, best_acc = None, -1.0
        n_tried = 0

        while time.perf_counter() < deadline:
            n_stages  = int(rng.randint(2, 5))
            stage_cfg = self._sample_stage_configs_legacy(rng, n_stages)
            try:
                model = SearchableCNN(in_c, n_cls, stage_cfg, hw)
                model = self._proxy_train(model, proxy_epochs, proxy_batches)
                acc   = self._proxy_acc(model)
                n_tried += 1
                marker = " ★" if acc > best_acc else ""
                print(f"  [legacy {n_tried:>2}] → val={acc:.3f}{marker}")
                if acc > best_acc:
                    best_acc   = acc
                    best_model = copy.deepcopy(model).cpu()
            except RuntimeError as e:
                print(f"  Architecture skipped: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if best_model is None:
            best_model = SearchableCNN(in_c, n_cls, _FALLBACK['medium'], hw)
            best_acc = 0.0
        self.metadata.setdefault('nas_report', {
            'n_tried':        n_tried,
            'best_proxy_val': round(best_acc, 4),
            'best_stages':    'legacy_random',
            'search_s':       round(self.clock.check(), 1),
        })
        return best_model

    def _proxy_train(self, model, n_epochs, max_batches):
        model.to(self.device).train()
        opt  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                if i >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
        return model

    def _proxy_acc(self, model, max_batches=20):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_loader):
                if i >= max_batches:
                    break
                pred = model(x.to(self.device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total   += y.size(0)
        return correct / total if total else 0.0
