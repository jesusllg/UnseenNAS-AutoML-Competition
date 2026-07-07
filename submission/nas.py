import copy
import time
import traceback
import logging

import numpy as np
import torch
import torch.nn as nn

from helpers import (show_time, set_seeds, GLOBAL_SEED, free_gpu,
                     get_safe_time_remaining, split_budget, gpu_total_mb)

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

# Pipeline hyperparameters: single source of truth in config.py.
from config import (NAS_POPULATION, NAS_ROUNDS, NAS_TOURNAMENT, SEARCH_FRAC,
                    NAS_PROXY_BATCH, LEARNING_RATE, WEIGHT_DECAY,
                    SMOKE_TIME_S, LOW_VRAM_MB,
                    MIN_AFFORDABLE_EPOCHS, TRAINABILITY_TOP_K)


def _adaptive_search_params(remaining_s: float):
    """
    Scale search effort to the actual time budget and hardware — the
    evaluation environment (clock AND GPU) is unknown.

      smoke        (little time left)   → tiny population, few rounds,
                                          smaller search fraction
      low-resource (small/absent GPU)   → halve population, smaller proxy batch

    Returns (population, rounds, tournament_k, proxy_batch, search_frac).
    """
    pop, rounds  = NAS_POPULATION, NAS_ROUNDS
    k, pbatch    = NAS_TOURNAMENT, NAS_PROXY_BATCH
    frac         = SEARCH_FRAC
    mode         = 'normal'
    if remaining_s < SMOKE_TIME_S:
        pop    = max(12, pop // 8)
        rounds = min(rounds, 200)
        frac   = 0.25
        mode   = 'smoke'
    vram = gpu_total_mb()
    if vram < LOW_VRAM_MB:   # includes 0.0 = CPU-only
        pop    = max(10, pop // 2)
        pbatch = min(pbatch, 8)
        mode  += '+low-resource'
    k = max(3, min(k, pop // 4))   # keep selection pressure ∝ population
    if mode != 'normal':
        print(f"  NAS | adaptive mode: {mode} (remaining={show_time(remaining_s)},"
              f" vram={vram:.0f}MB) → pop={pop} rounds={rounds} k={k}")
    return pop, rounds, k, pbatch, frac


def _get_proxy_batch(train_loader, device, batch_size=NAS_PROXY_BATCH):
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
        free_gpu()

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
        seed = GLOBAL_SEED
        set_seeds(seed)

        shape = self.metadata['input_shape']
        in_c  = shape[1]
        H, W  = shape[2], shape[3]
        n_cls = self.metadata['num_classes']

        # Time authority: metadata['time_remaining'] and the live clock,
        # reconciled conservatively. Never a global pool (per-dataset model).
        remaining = get_safe_time_remaining(self.metadata, self.clock)
        n_pop, n_rounds, tourney_k, pbatch, search_frac = \
            _adaptive_search_params(remaining)
        search_budget  = split_budget(remaining, search_frac)['search_s']
        t_search_start = time.perf_counter()
        print(f"  NAS | sf={search_frac:.2f} → budget={show_time(search_budget)}"
              f"  pop={n_pop} rounds={n_rounds} | device={self.device}")

        # ── Import search space ───────────────────────────────────────────────
        try:
            from search_space import (
                infer_family, sample_random_genotype, repair,
                build_model, az_nas_score, aging_evolution, best_individual,
                seed_genotypes_for_family,
            )
        except ImportError as e:
            print(f"  [NAS] search_space import failed ({e}), using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        # ── Proxy batch ───────────────────────────────────────────────────────
        proxy_batch = _get_proxy_batch(self.train_loader, self.device,
                                       batch_size=pbatch)
        if proxy_batch is None:
            print("  [NAS] Could not get proxy batch — using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        family = infer_family(in_c, H, W, n_cls)
        print(f"  NAS | family={family.name}  aniso={family.is_anisotropic}"
              f"  max_pools={family.max_pool_steps}  groupnorm={family.force_groupnorm}")

        def proxy_fn(model, batch_x, device):
            return az_nas_score(model, batch_x, device, reinit=True)

        # Curated seeds anchor the population with known-trainable baselines;
        # invalid ones for this geometry are skipped inside evolution.
        try:
            seeds = seed_genotypes_for_family(family, in_c, H, W, n_cls)
        except Exception:
            seeds = []

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
                seed_genotypes  = seeds,
            )
        except Exception as e:
            print(f"  [NAS] Evolution failed ({e}) — using legacy fallback.")
            traceback.print_exc()
            return self._legacy_search(in_c, n_cls, (H, W))

        if not population:
            print("  [NAS] Empty population — using legacy fallback.")
            return self._legacy_search(in_c, n_cls, (H, W))

        # ── Budget-aware final selection (anti proxy-gaming) ─────────────────
        # The proxy knows nothing about the clock: picking the raw best let
        # 10×-param models through that never converged inside the budget.
        best = self._pick_trainable(population, family, in_c, H, W, n_cls,
                                    build_model)

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

            # Save arch diagram + genotype JSON
            import json
            from pathlib import Path
            pred_dir = Path('predictions')
            pred_dir.mkdir(exist_ok=True)
            codename = self.metadata.get('codename', 'unknown')
            try:
                geno_path = pred_dir / f'{codename}_genotype.json'
                geno_path.write_text(json.dumps(best.genotype.to_dict(), indent=2))
                print(f"  Genotype JSON  → {geno_path}")
            except Exception as eg:
                print(f"  [NAS] Genotype save failed: {eg}")
            try:
                from arch_viz import save_arch
                n_p = sum(p.numel() for p in model.parameters())
                viz_path = pred_dir / f'{codename}_arch.png'
                ok = save_arch(best.genotype, self.metadata, viz_path,
                               proxy_score=best.fitness, n_params=n_p)
                if ok:
                    print(f"  Arch diagram   → {viz_path}")
            except Exception as ev:
                print(f"  [NAS] Arch viz failed: {ev}")

            # Release the NAS search memory before handing the model to training.
            free_gpu()
            return model.cpu()
        except Exception as e:
            print(f"  [NAS] Model build failed ({e}) — using legacy fallback.")
            traceback.print_exc()
            return self._legacy_search(in_c, n_cls, (H, W))

    # ── Budget-aware selection ────────────────────────────────────────────────

    def _pick_trainable(self, population, family, in_c, H, W, n_cls, build_model):
        """
        Walk the population best-first and return the first individual whose
        MEASURED fwd+bwd step time affords >= MIN_AFFORDABLE_EPOCHS of training
        in the remaining budget. If none qualifies, return the candidate with
        the most affordable epochs among the top K (most-trainable compromise).

        Rationale (V1→V2 data): the AZ proxy rated a 13M-param model ~1 point
        above a 1.3M one; the big model trained 6× longer and gained nothing,
        and on GeoClassing never converged at all. Capacity only helps if it
        can be trained to convergence inside THIS dataset's clock.
        """
        remaining = get_safe_time_remaining(self.metadata, self.clock)
        train_s   = split_budget(remaining)['train_s']
        steps     = max(1, len(self.train_loader))
        bs        = getattr(self.train_loader, 'batch_size', None) or 32

        scored = []
        for rank, ind in enumerate(population[:TRAINABILITY_TOP_K]):
            try:
                model = build_model(ind.genotype, in_c, H, W, n_cls,
                                    aniso_axis=family.aniso_axis).to(self.device)
                model.train()
                crit = nn.CrossEntropyLoss()
                x = torch.randn(bs, in_c, H, W, device=self.device)
                y = torch.randint(0, n_cls, (bs,), device=self.device)
                step_s = 0.0
                for timed in (False, True):   # warmup pass, then timed pass
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model.zero_grad(set_to_none=True)
                    crit(model(x), y).backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    step_s = time.perf_counter() - t0
                del model
                free_gpu()
                epoch_s    = step_s * steps * 1.2   # + eval/loader overhead margin
                affordable = train_s / max(epoch_s, 1e-6)
                scored.append((ind, affordable))
                if affordable >= MIN_AFFORDABLE_EPOCHS:
                    if rank > 0:
                        print(f"  NAS | trainability gate: skipped {rank} bigger"
                              f" candidate(s); picked AZ={ind.fitness:.3f}"
                              f" (~{affordable:.0f} epochs affordable)")
                    return ind
            except Exception as e:
                logger.debug("trainability probe failed: %s", e)
                free_gpu()
                continue

        if scored:
            ind, aff = max(scored, key=lambda t: t[1])
            print(f"  NAS | trainability gate: no candidate affords"
                  f" {MIN_AFFORDABLE_EPOCHS} epochs — picking most trainable"
                  f" (~{aff:.0f} epochs, AZ={ind.fitness:.3f})")
            return ind
        return population[0]   # probes all failed — fall back to raw best

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
        search_budget = get_safe_time_remaining(self.metadata, self.clock) * budget_frac
        deadline = time.perf_counter() + search_budget

        rng = np.random.RandomState(GLOBAL_SEED)
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
                free_gpu()

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
        opt  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
