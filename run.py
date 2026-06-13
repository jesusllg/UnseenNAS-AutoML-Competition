#!/usr/bin/env python3
"""
Local test runner for the NAS competition pipeline.

Replaces calling main.py directly — adds dataset selection, time override,
auto-creates predictions/, sorted iteration, and prints a summary table.

Usage:
  python run.py                          # run all datasets in datasets/
  python run.py --dataset AddNIST        # run one specific dataset
  python run.py --dataset AddNIST MultNIST  # run several
  python run.py --time 0.25             # override time limit per dataset (hours)
  python run.py --total-time 24         # global pool shared across datasets
  python run.py --truncate              # quick smoke-test (64 samples)

Time budget modes (mutually exclusive):
  --time HOURS        Fixed per-dataset time limit (overrides metadata).
  --total-time HOURS  Global pool with halving allocation: each dataset gets
                      pool/2 (the last one gets everything left), floored at
                      --min-time. Saved time flows to the remaining datasets.
  (default)           Same halving policy via the GlobalBudgetGovernor with
                      the competition constants (24h total, 3 datasets).

Datasets always run cheapest→heaviest so surplus time reaches the heavy ones.
"""

import argparse
import copy
import json
import numpy as np
import pickle as pkl
import sys
import time
import traceback
from pathlib import Path

import torch

# ── submission files live in ./submission/ ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "submission"))
from nas import NAS
from data_processor import DataProcessor
from trainer import Trainer
from helpers import (show_time, load_metadata, estimate_dataset_cost,
                     halving_allocation, GlobalBudgetGovernor,
                     N_COMPETITION_DATASETS, TOTAL_COMPETITION_HOURS)


# ── helpers (mirrors evaluation/main.py) ─────────────────────────────────────

class Clock:
    def __init__(self, hours):
        self.limit = time.perf_counter() + hours * 3600

    def check(self):
        return self.limit - time.perf_counter()


class GlobalTimeBudget:
    """
    Dynamically redistributes a total time pool across datasets (--total-time).

    Allocation policy is helpers.halving_allocation — the same single source
    of truth the GlobalBudgetGovernor uses, so the local clock can never be
    smaller than the GBG's internal allocation.
    """

    def __init__(self, total_hours, n_datasets, min_hours=0.1):
        self.pool        = total_hours * 3600   # seconds remaining
        self.n_remaining = n_datasets
        self.min_s       = min_hours * 3600

    def next_allocation_h(self):
        """Hours for the next dataset (halving policy, floored at min)."""
        alloc_s = halving_allocation(self.pool, self.n_remaining)
        return max(alloc_s, self.min_s) / 3600

    def record_usage(self, used_seconds):
        self.pool        = max(0.0, self.pool - used_seconds)
        self.n_remaining = max(0,   self.n_remaining - 1)
        if self.n_remaining > 0:
            carry = self.pool / self.n_remaining
            print(f"  [Budget] used={show_time(used_seconds)}"
                  f"  pool={show_time(self.pool)}"
                  f"  ~{show_time(carry)}/dataset remaining")


def load_dataset(dataset_path: Path, truncate: bool):
    def npy(name):
        arr = np.load(dataset_path / name)
        return arr[:64] if truncate else arr

    return (
        npy("train_x.npy"), npy("train_y.npy"),
        npy("valid_x.npy"), npy("valid_y.npy"),
        npy("test_x.npy"),
    )


def num_params(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)


def fail_dataset(codename, pred_dir):
    pkl.dump({'Failed': True, 'Runtime': -1, 'Params': None},
             open(pred_dir / f"{codename}_stats.pkl", "wb"))


# ── per-dataset run ───────────────────────────────────────────────────────────

def run_one(dataset_path: Path, args, pred_dir: Path, hours: float):
    """Run the full pipeline for one dataset. Returns (ok, runtime_seconds)."""
    meta     = load_metadata(dataset_path)
    codename = meta["codename"]
    clock    = Clock(hours)
    imut     = copy.deepcopy(clock)

    # Inject all tunable pipeline params so NAS/Trainer read them from metadata
    meta['seed']             = args.seed
    meta['search_frac']      = args.search_frac
    meta['n_population']     = args.nas_population
    meta['n_rounds']         = args.nas_rounds
    meta['tournament_size']  = args.nas_tournament
    meta['proxy_epochs']     = args.proxy_epochs
    meta['proxy_batches']    = args.proxy_batches
    meta['train_frac']       = args.train_frac
    meta['weight_decay']     = args.weight_decay
    meta['es_enabled']       = args.es_enabled
    meta['es_patience']      = args.es_patience
    meta['es_min_epochs']    = args.es_min_epochs
    meta['es_delta_start']      = args.es_delta_start
    meta['es_delta_min']        = args.es_delta_min
    meta['es_delta_decay']      = args.es_delta_decay
    meta['es_plateau_patience'] = args.es_plateau_patience
    meta['es_regression_delta'] = args.es_regression_delta

    es_str = (f"↑δ{args.es_delta_start:.4f}↘{args.es_delta_min:.4f} "
              f"↓{args.es_regression_delta:.4f} p={args.es_patience}") if args.es_enabled else "off"
    print(f"\n{'='*10} {codename:^20} {'='*10}")
    print(f"  time={hours}h | truncate={args.truncate}"
          f" | search={args.search_frac} train={args.train_frac} wd={args.weight_decay:.0e}"
          f" | NAS pop={args.nas_population} rounds={args.nas_rounds} k={args.nas_tournament}"
          f" | ES={es_str}")
    _injected = {'seed','search_frac','n_population','n_rounds','tournament_size',
                 'proxy_epochs','proxy_batches','train_frac','weight_decay',
                 'es_enabled','es_patience','es_plateau_patience','es_min_epochs',
                 'es_delta_start','es_delta_min','es_delta_decay','es_regression_delta'}
    [print(f"  {k}: {v}") for k, v in meta.items() if k not in _injected]

    t0 = time.perf_counter()
    try:
        train_x, train_y, valid_x, valid_y, test_x = load_dataset(dataset_path, args.truncate)
        meta["time_remaining"] = imut.check()

        print("\n=== DataProcessor ===")
        dp = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, meta, clock)
        train_loader, valid_loader, test_loader = dp.process()

        # framework asserts
        from torch.utils.data import RandomSampler
        assert not isinstance(test_loader.sampler, RandomSampler), "test_loader must not shuffle"
        assert not test_loader.drop_last, "test_loader must not drop_last"

        print("\n=== NAS ===")
        meta["time_remaining"] = imut.check()
        model = NAS(train_loader, valid_loader, meta, clock).search()
        params = int(num_params(model))

        print("\n=== Trainer ===")
        meta["time_remaining"] = imut.check()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model, device, train_loader, valid_loader, meta, clock)
        trained = trainer.train()

        print("\n=== Predict ===")
        preds = trainer.predict(test_loader)
        np.save(pred_dir / f"{codename}.npy", preds)

        # Unload model from GPU so the next dataset starts with a clean slate
        trainer.model.cpu()
        del trainer, model, trained
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Auto test accuracy if test_y.npy available (local evaluation only)
        test_acc = None
        test_y_path = dataset_path / "test_y.npy"
        if test_y_path.exists():
            test_y = np.load(test_y_path)
            if args.truncate:
                test_y = test_y[:64]
            test_acc = float((np.array(preds) == test_y.reshape(-1)).mean())
            print(f"  Test accuracy: {test_acc*100:.4f}%  (test_y found — local eval only)")
            meta['test_acc'] = round(test_acc, 4)

        runtime = time.perf_counter() - t0
        pkl.dump({'Failed': False, 'Runtime': round(runtime, 2), 'Params': params,
                  'TestAcc': test_acc},
                 open(pred_dir / f"{codename}_stats.pkl", "wb"))
        print(f"  Done in {show_time(runtime)} | params={params:,}")
        return True, runtime

    except Exception:
        runtime = time.perf_counter() - t0
        print(f"\n  [ERROR] {codename} failed:")
        print(traceback.format_exc())
        fail_dataset(codename, pred_dir)
        return False, runtime


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NAS competition local runner.")
    ap.add_argument("--dataset", nargs="*", metavar="NAME",
                    help="Dataset folder name(s) to run. Default: all in datasets/.")
    ap.add_argument("--time", type=float, metavar="HOURS",
                    help="Fixed time limit per dataset (hours). "
                         "Default: use metadata field, fallback 7h.")
    ap.add_argument("--total-time", type=float, metavar="HOURS",
                    help="Global time pool (hours) shared across all datasets. "
                         "Unused time flows to remaining datasets. "
                         "Mutually exclusive with --time.")
    ap.add_argument("--min-time", type=float, default=0.1, metavar="HOURS",
                    help="Floor time per dataset when using --total-time. (default: 0.1h)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed for NAS search and training. (default: 42)")
    ap.add_argument("--truncate", action="store_true",
                    help="Use only 64 samples per split (quick smoke-test).")
    ap.add_argument("--datasets-dir", default="datasets", metavar="DIR",
                    help="Root folder containing dataset subdirectories.")
    # ── Pipeline hyperparameters ──────────────────────────────────────────────
    # search_frac + train_frac should sum to ≤ 0.95 (5% buffer for predict/overhead)
    ap.add_argument("--search-frac",      type=float, default=0.30,
                    help="Fraction of total budget for NAS search. (default: 0.30)")
    ap.add_argument("--train-frac",       type=float, default=0.65,
                    help="Fraction of total budget for final training. (default: 0.65)")
    ap.add_argument("--nas-population",   type=int,   default=100,
                    help="NAS population size. (default: 100)")
    ap.add_argument("--nas-rounds",       type=int,   default=2000,
                    help="Max NAS evolution rounds (stops at time budget). (default: 2000)")
    ap.add_argument("--nas-tournament",   type=int,   default=25,
                    help="NAS tournament size. (default: 25)")
    ap.add_argument("--proxy-epochs",     type=int,   default=3,
                    help="Training epochs per candidate during proxy eval. (default: 3)")
    ap.add_argument("--proxy-batches",    type=int,   default=40,
                    help="Max batches per epoch during proxy eval. (default: 40)")
    ap.add_argument("--weight-decay",     type=float, default=1e-4,
                    help="AdamW weight decay for final training. (default: 1e-4)")
    # ── Early stopping (dynamic delta) ───────────────────────────────────────
    ap.add_argument("--no-es", dest="es_enabled", action="store_false", default=True,
                    help="Disable early stopping entirely.")
    ap.add_argument("--es-patience",     type=int,   default=20,
                    help="Consecutive non-improving epochs before stopping. (default: 20)")
    ap.add_argument("--es-min-epochs",   type=int,   default=10,
                    help="Minimum epochs before ES can trigger. (default: 10)")
    ap.add_argument("--es-delta-start",  type=float, default=0.002,
                    help="Initial min-improvement threshold. (default: 0.002 = 0.2pp)")
    ap.add_argument("--es-delta-min",    type=float, default=0.001,
                    help="Floor for delta after decay. (default: 0.001 = 0.1pp)")
    ap.add_argument("--es-delta-decay",       type=int,   default=3,
                    help="Improvements before halving delta. (default: 3)")
    ap.add_argument("--es-plateau-patience", type=int,   default=20,
                    help="Consecutive plateau epochs before stopping. (default: 20)")
    ap.add_argument("--es-regression-delta", type=float, default=0.010,
                    help="Fall >X below best to count as bad epoch. (default: 0.010 = 1pp)")
    args = ap.parse_args()

    # Validate fractions
    if args.search_frac + args.train_frac > 0.95:
        print(f"WARNING: search_frac ({args.search_frac}) + train_frac ({args.train_frac})"
              f" = {args.search_frac + args.train_frac:.2f} > 0.95 — little time left for predict/overhead.")
    if args.time is not None and args.total_time is not None:
        print("ERROR: --time and --total-time are mutually exclusive.")
        sys.exit(1)

    datasets_dir = Path(args.datasets_dir)
    pred_dir     = Path("predictions")
    pred_dir.mkdir(exist_ok=True)

    # Reset GBG state at the start of every run.py invocation so a previously
    # killed or partial test run doesn't leave stale n_done counts behind.
    GlobalBudgetGovernor.reset_state()

    if not datasets_dir.exists():
        print(f"ERROR: datasets directory not found: {datasets_dir}")
        print("Run:  python download_datasets.py")
        sys.exit(1)

    # Collect available dataset folders (must contain a metadata file)
    available = sorted(
        p for p in datasets_dir.iterdir()
        if p.is_dir() and (p / "metadata").exists()
    )

    if not available:
        print(f"No datasets found in {datasets_dir}/")
        sys.exit(1)

    # Filter to requested datasets (exact folder name match)
    if args.dataset:
        name_map = {p.name: p for p in available}
        missing  = [n for n in args.dataset if n not in name_map]
        if missing:
            print(f"Dataset(s) not found: {missing}")
            print(f"Available: {[p.name for p in available]}")
            sys.exit(1)
        targets = [name_map[n] for n in args.dataset]
    else:
        targets = available

    n_ds = len(targets)
    n_avail = len(available)
    print(f"Detected {n_avail} dataset(s) in {datasets_dir}/")
    if args.dataset:
        print(f"Running {n_ds}/{n_avail} selected: {[p.name for p in targets]}")
    else:
        print(f"Running all {n_ds}: {[p.name for p in targets]}")

    # ── Order datasets cheapest→heaviest (ALL modes) ──────────────────────────
    # Easy/fast datasets must run first so their unused time flows to the heavy
    # ones at the end (GBG halving gives the last dataset the entire remaining
    # pool). Previously this sort only ran with --total-time, so in default mode
    # the order was alphabetical and a heavy dataset like CIFARTile could run
    # first and starve everything after it.
    costs = {}
    for p in targets:
        try:
            m = load_metadata(p)
            costs[p.name] = estimate_dataset_cost(m, dataset_dir=p)
        except Exception:
            costs[p.name] = float('inf')   # unreadable metadata → assume heavy, run last
    if args.dataset:
        # User gave an explicit list — keep their order but warn if it's suboptimal
        cost_order = sorted(targets, key=lambda p: costs[p.name])
        if [p.name for p in cost_order] != [p.name for p in targets]:
            print(f"NOTE: running in your given order. Cheapest→heaviest would be:"
                  f" {[p.name for p in cost_order]}")
    else:
        targets = sorted(targets, key=lambda p: costs[p.name])
    if len(targets) > 1:
        print("Execution order (cheapest→heaviest):")
        for p in targets:
            print(f"  {p.name}: cost={costs[p.name]:,.0f}")

    # ── Time budget setup ─────────────────────────────────────────────────────
    budget = None
    if args.total_time is not None:
        budget = GlobalTimeBudget(args.total_time, n_ds, args.min_time)
        per = args.total_time / n_ds
        print(f"Global budget: {args.total_time}h / {n_ds} datasets"
              f" = ~{per:.2f}h initial each (floor {args.min_time}h)")

    results = {}
    for ds_path in targets:
        # Determine the local clock for this dataset. In every mode the clock
        # must be >= the GBG's internal allocation, otherwise min(clock, GBG)
        # collapses to the clock and the GBG never controls anything. Both
        # GlobalTimeBudget and GlobalBudgetGovernor use helpers.halving_allocation,
        # so they always agree by construction.
        if budget is not None:
            hours = budget.next_allocation_h()
        elif args.time is not None:
            hours = args.time
        else:
            # Default mode: read the same persisted GBG state NAS will read,
            # so the clock equals the GBG allocation exactly.
            hours = GlobalBudgetGovernor(
                n_total=N_COMPETITION_DATASETS,
                total_hours=TOTAL_COMPETITION_HOURS,
            ).get_allocation() / 3600

        ok, runtime = run_one(ds_path, args, pred_dir, hours)
        results[ds_path.name] = ok

        if budget is not None:
            budget.record_usage(runtime)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"{'DATASET':<20} {'STATUS':<10} {'PARAMS':>10} {'RUNTIME':>10} {'TRAIN':>8} {'VAL':>8} {'TEST':>8}")
    print(f"{'-'*82}")
    for ds_name, ok in results.items():
        codename = load_metadata(datasets_dir / ds_name).get("codename", ds_name)
        stats_path = pred_dir / f"{codename}_stats.pkl"
        params = runtime = "N/A"
        test_str = "—"
        if stats_path.exists():
            s = pkl.load(open(stats_path, "rb"))
            params   = f"{s['Params']:,}" if s['Params'] else "N/A"
            runtime  = show_time(s['Runtime']) if s['Runtime'] >= 0 else "N/A"
            if s.get('TestAcc') is not None:
                test_str = f"{s['TestAcc']*100:.2f}%"

        report_path = pred_dir / f"{codename}_report.json"
        train_str = val_str = "N/A"
        if report_path.exists():
            r = json.loads(report_path.read_text())
            if 'final_train_acc' in r:
                train_str = f"{float(r['final_train_acc'])*100:.2f}%"
            if 'final_val_acc' in r:
                val_str = f"{float(r['final_val_acc'])*100:.2f}%"
            elif 'best_val_acc' in r:
                val_str = f"{float(r['best_val_acc'])*100:.2f}%"

        status = "✓ OK" if ok else "✗ FAIL"
        print(f"{ds_name:<20} {status:<10} {params:>10} {runtime:>10} {train_str:>8} {val_str:>8} {test_str:>8}")

    n_ok = sum(results.values())
    print(f"{'='*82}")
    print(f"Completed: {n_ok}/{len(results)} datasets")


if __name__ == "__main__":
    main()
