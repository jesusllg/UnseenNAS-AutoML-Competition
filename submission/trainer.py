import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score

from helpers import show_time, set_seeds, GLOBAL_SEED

# train_frac is a fraction of the TOTAL clock budget (not of remaining time).
# search_frac + train_frac should sum to ≤ 0.95, leaving ~5% for predict/overhead.
_TRAIN_FRAC   = 0.65
_SEARCH_FRAC  = 0.30   # kept here only as the default denominator reference
_WEIGHT_DECAY = 1e-4

# Early stopping defaults
_ES_ENABLED      = True
_ES_PATIENCE     = 20     # consecutive non-improving epochs before stopping
_ES_MIN_EPOCHS   = 10     # warmup: ES cannot trigger before this epoch
_ES_DELTA_START  = 0.005  # initial min-improvement threshold (0.5 pp)
_ES_DELTA_MIN    = 0.001  # floor for delta (0.1 pp) — keeps threshold achievable
_ES_DELTA_DECAY  = 5      # improvements before halving delta


class _EarlyStopper:
    """
    Consecutive early stopping with dynamic delta.

    `wait` counts consecutive epochs where val_acc <= best + delta.
    On a good epoch (val > best + delta), best is updated and wait resets to 0.
    On a bad epoch, wait increments by 1.
    Stops when consecutive bad epochs reach `patience`.

    Delta decays every `delta_decay` total improvements, making the bar
    progressively stricter. Floor at delta_min keeps the threshold achievable
    (avoids impossible targets like >100% accuracy).
    Disabled entirely when enabled=False.
    """

    def __init__(self, patience, min_epochs, delta_start, delta_min,
                 delta_decay, enabled):
        self.patience      = patience
        self.min_epochs    = min_epochs
        self.delta         = delta_start
        self.delta_min     = delta_min
        self.delta_decay   = delta_decay
        self.enabled       = enabled
        self.best          = -float('inf')
        self.wait          = 0   # consecutive bad epochs
        self.improve_count = 0   # total improvements (for delta decay)

    def step(self, val_acc, epoch):
        """Return True if training should stop."""
        if not self.enabled or epoch < self.min_epochs:
            return False

        if val_acc > self.best + self.delta:
            self.best = val_acc
            self.wait = 0
            self.improve_count += 1
            # Tighten the bar every delta_decay total improvements
            if self.improve_count % self.delta_decay == 0:
                old = self.delta
                self.delta = max(self.delta * 0.5, self.delta_min)
                if self.delta < old:
                    print(f"  [ES] Δ {old:.5f}→{self.delta:.5f} "
                          f"({self.improve_count} improvements)")
        else:
            self.wait += 1

        return self.wait >= self.patience


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model    = model
        self.device   = device
        self.train_dl = train_dataloader
        self.valid_dl = valid_dataloader
        self.metadata = metadata
        self.clock    = clock
        # XLA (TPU) support: import mark_step if device is xla
        self._xm = None
        if str(device).startswith('xla'):
            try:
                import torch_xla.core.xla_model as xm
                self._xm = xm
            except ImportError:
                pass

    # ------------------------------------------------------------------
    def train(self):
        try:
            return self._train()
        except Exception:
            print("  [Trainer] Unexpected error — returning model as-is.")
            print(traceback.format_exc())
            return self.model

    def _train(self):
        set_seeds(self.metadata.get('seed', GLOBAL_SEED))
        self.model.to(self.device)

        # Complementary fractions of the TOTAL clock budget:
        #   search_frac takes X% of total at NAS time
        #   train_frac  is also a fraction of total — we recover it from remaining time
        search_frac  = self.metadata.get('search_frac',  _SEARCH_FRAC)
        train_frac   = self.metadata.get('train_frac',   _TRAIN_FRAC)
        weight_decay = self.metadata.get('weight_decay', _WEIGHT_DECAY)

        # remaining ≈ (1 - search_frac) * total  →  train_budget = train_frac * total
        remaining_frac = max(1.0 - search_frac, 1e-6)
        train_budget   = self.clock.check() * (train_frac / remaining_frac)
        deadline       = time.perf_counter() + train_budget
        t_train_start  = time.perf_counter()

        # Early stopping (dynamic delta)
        es_enabled     = self.metadata.get('es_enabled',     _ES_ENABLED)
        es_patience    = self.metadata.get('es_patience',    _ES_PATIENCE)
        es_min_epochs  = self.metadata.get('es_min_epochs',  _ES_MIN_EPOCHS)
        es_delta_start = self.metadata.get('es_delta_start', _ES_DELTA_START)
        es_delta_min   = self.metadata.get('es_delta_min',   _ES_DELTA_MIN)
        es_delta_decay = self.metadata.get('es_delta_decay', _ES_DELTA_DECAY)
        stopper = _EarlyStopper(es_patience, es_min_epochs,
                                es_delta_start, es_delta_min,
                                es_delta_decay, es_enabled)

        n_cls = self.metadata['num_classes']
        label_smoothing = 0.1 if n_cls >= 10 else 0.0

        es_desc = (f"δ {es_delta_start:.5f}↘{es_delta_min:.5f} every {es_delta_decay}↑"
                   f" p={es_patience}") if es_enabled else "off"
        print(f"  Trainer | budget={show_time(train_budget)} wd={weight_decay:.0e}"
              f" | ES={es_desc} | device={self.device}")

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

        use_amp = torch.cuda.is_available()
        scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_acc   = 0.0
        best_state = None
        epoch_times: list[float] = []
        epoch = 0

        while True:
            t_left = deadline - time.perf_counter()
            if t_left <= 0:
                print(f"  Time limit reached after {epoch} epochs.")
                break
            if epoch_times:
                avg = sum(epoch_times[-3:]) / len(epoch_times[-3:])
                if avg > t_left * 0.9:
                    print(f"  Stopping — ~{show_time(avg)}/epoch, {show_time(t_left)} left.")
                    break

            t0 = time.perf_counter()
            self.model.train()
            labels, preds = [], []

            for x, y in self.train_dl:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    out  = self.model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if self._xm is not None:
                    self._xm.mark_step()

                labels += y.cpu().tolist()
                preds  += out.detach().argmax(1).cpu().tolist()

            scheduler.step()
            epoch += 1
            epoch_times.append(time.perf_counter() - t0)

            train_acc = accuracy_score(labels, preds)
            val_acc   = self._evaluate(self.valid_dl)
            lr_now    = scheduler.get_last_lr()[0]

            print("  Epoch {:>3} | Train {:>6.2f}% | Val {:>6.2f}% | {} | lr {:.2e}".format(
                epoch, train_acc * 100, val_acc * 100,
                show_time(epoch_times[-1]), lr_now))

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if stopper.step(val_acc, epoch):
                saved = show_time(max(0.0, deadline - time.perf_counter()))
                print(f"  Early stop at epoch {epoch} "
                      f"(plateau {es_patience} ep, Δ={stopper.delta:.5f})."
                      f" ~{saved} returned to pool.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation with best weights
        final_val_acc   = self._evaluate(self.valid_dl)
        final_train_acc = self._evaluate(self.train_dl)
        print(f"  {'─'*55}")
        print(f"  Final | Train {final_train_acc*100:.2f}% | Val {final_val_acc*100:.2f}%"
              f"  (best val seen: {best_acc*100:.2f}%)")
        print(f"  {'─'*55}")

        train_elapsed = time.perf_counter() - t_train_start
        self._save_report(epoch, final_train_acc, final_val_acc, train_elapsed)
        self._save_model()

        return self.model

    # ------------------------------------------------------------------
    def _evaluate(self, loader):
        self.model.eval()
        labels, preds = [], []
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x.to(self.device))
                labels += y.tolist()
                preds  += out.argmax(1).cpu().tolist()
        return accuracy_score(labels, preds)

    # ------------------------------------------------------------------
    def _save_report(self, n_epochs, final_train_acc, final_val_acc, train_s):
        """Write per-dataset JSON + append row to cumulative CSV."""
        codename   = self.metadata.get('codename', 'unknown')
        nas_report = self.metadata.get('nas_report', {})

        record = {
            'timestamp':        datetime.now().isoformat(timespec='seconds'),
            'codename':         codename,
            'n_architectures':  nas_report.get('n_tried', '?'),
            'best_proxy_val':   nas_report.get('best_proxy_val', '?'),
            'best_stages':      nas_report.get('best_stages', '?'),
            'search_s':         nas_report.get('search_s', '?'),
            'n_epochs':         n_epochs,
            'final_train_acc':  round(final_train_acc, 4),
            'final_val_acc':    round(final_val_acc, 4),
            'train_s':          round(train_s, 1),
            'num_classes':      self.metadata.get('num_classes', '?'),
            'input_shape':      str(self.metadata.get('input_shape', '?')),
        }

        pred_dir = Path('predictions')
        pred_dir.mkdir(exist_ok=True)

        # Per-run JSON
        (pred_dir / f'{codename}_report.json').write_text(
            json.dumps(record, indent=2))

        # Cumulative CSV — append one row per run
        csv_path = pred_dir / 'report.csv'
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        print(f"  Report saved → {pred_dir / f'{codename}_report.json'}")

    # ------------------------------------------------------------------
    def _save_model(self):
        """Save trained model weights to predictions/<codename>_model.pt."""
        codename = self.metadata.get('codename', 'unknown')
        pred_dir = Path('predictions')
        pred_dir.mkdir(exist_ok=True)
        path = pred_dir / f'{codename}_model.pt'
        torch.save(self.model.cpu().state_dict(), path)
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  Model saved  → {path}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    def predict(self, test_loader):
        self.model.to(self.device).eval()
        preds = []
        with torch.no_grad():
            for x in test_loader:
                preds += self.model(x.to(self.device)).argmax(1).cpu().tolist()
        return preds
