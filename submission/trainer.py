import csv
import json
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score

from helpers import show_time

SEED = 42

# train_frac is a fraction of the TOTAL clock budget (not of remaining time).
# search_frac + train_frac should sum to ≤ 0.95, leaving ~5% for predict/overhead.
_TRAIN_FRAC   = 0.65
_SEARCH_FRAC  = 0.30   # kept here only as the default denominator reference
_WEIGHT_DECAY = 1e-4

# Early stopping defaults
_ES_PATIENCE   = 15
_ES_MIN_DELTA  = 0.001
_ES_MIN_EPOCHS = 10


def _set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _EarlyStopper:
    """Prechelt-style patience-based early stopping on validation accuracy."""

    def __init__(self, patience, min_delta, min_epochs):
        self.patience   = patience
        self.min_delta  = min_delta
        self.min_epochs = min_epochs
        self.best       = -float('inf')
        self.wait       = 0

    def step(self, val_acc, epoch):
        """Return True if training should stop."""
        if epoch < self.min_epochs:
            return False
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.wait = 0
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
        _set_seeds()
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

        # Early stopping
        es_patience   = self.metadata.get('es_patience',   _ES_PATIENCE)
        es_min_delta  = self.metadata.get('es_min_delta',  _ES_MIN_DELTA)
        es_min_epochs = self.metadata.get('es_min_epochs', _ES_MIN_EPOCHS)
        stopper = _EarlyStopper(es_patience, es_min_delta, es_min_epochs)

        n_cls = self.metadata['num_classes']
        label_smoothing = 0.1 if n_cls >= 10 else 0.0

        print(f"  Trainer | budget={show_time(train_budget)} wd={weight_decay:.0e}"
              f" es={es_patience}p/{es_min_epochs}min | device={self.device}")

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
            val_acc   = self._evaluate()
            lr_now    = scheduler.get_last_lr()[0]

            print("  Epoch {:>3} | Train {:>6.2f}% | Val {:>6.2f}% | {} | lr {:.2e}".format(
                epoch, train_acc * 100, val_acc * 100,
                show_time(epoch_times[-1]), lr_now))

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if stopper.step(val_acc, epoch):
                saved = show_time(deadline - time.perf_counter())
                print(f"  Early stop at epoch {epoch} (patience={es_patience})."
                      f" ~{saved} returned to pool.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  Restored best weights (val={best_acc:.3f})")

        train_elapsed = time.perf_counter() - t_train_start
        self._save_report(epoch, best_acc, train_elapsed)

        return self.model

    # ------------------------------------------------------------------
    def _evaluate(self):
        self.model.eval()
        labels, preds = [], []
        with torch.no_grad():
            for x, y in self.valid_dl:
                out = self.model(x.to(self.device))
                labels += y.tolist()
                preds  += out.argmax(1).cpu().tolist()
        return accuracy_score(labels, preds)

    # ------------------------------------------------------------------
    def _save_report(self, n_epochs, best_val_acc, train_s):
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
            'best_val_acc':     round(best_val_acc, 4),
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
    def predict(self, test_loader):
        self.model.to(self.device).eval()
        preds = []
        with torch.no_grad():
            for x in test_loader:
                preds += self.model(x.to(self.device)).argmax(1).cpu().tolist()
        return preds
