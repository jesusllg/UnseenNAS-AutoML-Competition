#!/usr/bin/env python3
"""
Local test runner for the NAS competition pipeline.

Replaces calling main.py directly — adds dataset selection, time override,
auto-creates predictions/, sorted iteration, and prints a summary table.

Usage:
  python run.py                          # run all datasets in datasets/
  python run.py --dataset AddNIST        # run one specific dataset
  python run.py --dataset AddNIST MultNIST  # run several
  python run.py --time 0.25             # override time limit (hours)
  python run.py --truncate              # quick smoke-test (64 samples)
"""

import argparse
import copy
import json
import math
import numpy as np
import os
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


# ── helpers (mirrors evaluation/main.py) ─────────────────────────────────────

def show_time(seconds):
    def div_rem(n, d):
        return math.floor(n / d), int(n - math.floor(n / d) * d)
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        m, s = div_rem(seconds, 60)
        return f"{m}m,{s}s"
    else:
        h, s  = div_rem(seconds, 3600)
        m, s  = div_rem(s, 60)
        return f"{h}h,{m}m,{s}s"


class Clock:
    def __init__(self, hours):
        self.limit = time.perf_counter() + hours * 3600

    def check(self):
        return self.limit - time.perf_counter()


def load_metadata(dataset_path: Path) -> dict:
    return json.loads((dataset_path / "metadata").read_text())


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

def run_one(dataset_path: Path, time_override: float | None, truncate: bool,
            pred_dir: Path):
    meta = load_metadata(dataset_path)
    codename = meta["codename"]

    # Time limit: CLI override > metadata field > default 0.5 h
    hours = time_override if time_override is not None else meta.get("time_limit", 0.5)
    clock = Clock(hours)
    imut  = copy.deepcopy(clock)

    print(f"\n{'='*10} {codename:^20} {'='*10}")
    print(f"  time_limit={hours}h | truncate={truncate}")
    [print(f"  {k}: {v}") for k, v in meta.items()]

    t0 = time.perf_counter()
    try:
        train_x, train_y, valid_x, valid_y, test_x = load_dataset(dataset_path, truncate)
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

        runtime = time.perf_counter() - t0
        np.save(pred_dir / f"{codename}.npy", preds)
        pkl.dump({'Failed': False, 'Runtime': round(runtime, 2), 'Params': params},
                 open(pred_dir / f"{codename}_stats.pkl", "wb"))
        print(f"  Done in {show_time(runtime)} | params={params:,}")
        return True

    except Exception:
        print(f"\n  [ERROR] {codename} failed:")
        print(traceback.format_exc())
        fail_dataset(codename, pred_dir)
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NAS competition local runner.")
    ap.add_argument("--dataset", nargs="*", metavar="NAME",
                    help="Dataset folder name(s) to run. Default: all in datasets/.")
    ap.add_argument("--time", type=float, metavar="HOURS",
                    help="Override time limit for every dataset (hours).")
    ap.add_argument("--truncate", action="store_true",
                    help="Use only 64 samples per split (quick smoke-test).")
    ap.add_argument("--datasets-dir", default="datasets", metavar="DIR",
                    help="Root folder containing dataset subdirectories.")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir)
    pred_dir     = Path("predictions")
    pred_dir.mkdir(exist_ok=True)         # auto-create — no more missing folder errors

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

    print(f"Running {len(targets)} dataset(s): {[p.name for p in targets]}")

    results = {}
    for ds_path in targets:
        ok = run_one(ds_path, args.time, args.truncate, pred_dir)
        results[ds_path.name] = ok

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'DATASET':<20} {'STATUS':<10} {'PARAMS':>10} {'RUNTIME':>10} {'VAL':>8}")
    print(f"{'-'*60}")
    for ds_name, ok in results.items():
        codename = load_metadata(datasets_dir / ds_name).get("codename", ds_name)
        stats_path = pred_dir / f"{codename}_stats.pkl"
        if stats_path.exists():
            s = pkl.load(open(stats_path, "rb"))
            params   = f"{s['Params']:,}" if s['Params'] else "N/A"
            runtime  = show_time(s['Runtime']) if s['Runtime'] >= 0 else "N/A"
        else:
            params = runtime = "N/A"

        report_path = pred_dir / f"{codename}_report.json"
        val = "N/A"
        if report_path.exists():
            r = json.loads(report_path.read_text())
            val = f"{float(r.get('best_val_acc', 0))*100:.2f}%"

        status = "✓ OK" if ok else "✗ FAIL"
        print(f"{ds_name:<20} {status:<10} {params:>10} {runtime:>10} {val:>8}")

    n_ok = sum(results.values())
    print(f"{'='*60}")
    print(f"Completed: {n_ok}/{len(results)} datasets")


if __name__ == "__main__":
    main()
