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

from helpers import show_time, set_seeds, GLOBAL_SEED, free_gpu

# Pipeline hyperparameters: single source of truth in config.py.
from config import (
    SEARCH_FRAC, TRAIN_FRAC, WEIGHT_DECAY,
    LEARNING_RATE, GRAD_CLIP_NORM, LR_T_MAX, LR_ETA_MIN,
    LABEL_SMOOTHING, LABEL_SMOOTHING_MIN_CLASSES,
    ES_ENABLED, ES_PATIENCE, ES_PLATEAU_PATIENCE, ES_MIN_EPOCHS,
    ES_DELTA_START, ES_DELTA_MIN, ES_DELTA_DECAY, ES_REGRESSION_DELTA,
)


class _EarlyStopper:
    """
    Regression-based early stopping with dynamic improvement delta.

    Three zones per epoch (after min_epochs warmup):
      ★ Improvement : val > best + delta        → reset both counters, update best
      ↓ Regression  : val < best - reg_delta    → regression_wait += 1
      ~ Plateau     : in between                → plateau_wait += 1

    Stops when regression_wait >= patience  OR  plateau_wait >= plateau_patience.
    Plateau patience catches models stuck at theoretical maximum (e.g. 100%).
    """

    def __init__(self, patience, plateau_patience, min_epochs, delta_start,
                 delta_min, delta_decay, regression_delta, enabled):
        self.patience         = patience
        self.plateau_patience = plateau_patience
        self.min_epochs       = min_epochs
        self.delta            = delta_start
        self.delta_min        = delta_min
        self.delta_decay      = delta_decay
        self.regression_delta = regression_delta
        self.enabled          = enabled
        self.best             = -float('inf')
        self.wait             = 0   # regression counter
        self.plateau_wait     = 0   # plateau counter
        self.improve_count    = 0

    def step(self, val_acc, epoch):
        """Return (stop, zone_char) for logging."""
        if not self.enabled or epoch < self.min_epochs:
            return False, '~'

        if val_acc > self.best + self.delta:
            self.best = val_acc
            self.wait = 0
            self.plateau_wait = 0
            self.improve_count += 1
            if self.improve_count % self.delta_decay == 0:
                old = self.delta
                self.delta = max(self.delta * 0.5, self.delta_min)
                if self.delta < old:
                    print(f"  [ES] Δ {old:.5f}→{self.delta:.5f} "
                          f"({self.improve_count} improvements)")
            return False, '★'
        elif val_acc < self.best - self.regression_delta:
            self.wait += 1
            self.plateau_wait = 0
            return self.wait >= self.patience, '↓'
        else:
            self.plateau_wait += 1
            return self.plateau_wait >= self.plateau_patience, '~'


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
        finally:
            # Always notify GBG of completion, even on failure, so state stays
            # consistent for the next dataset. finish_dataset() is idempotent.
            gbg = self.metadata.get('_gbg')
            if gbg is not None:
                gbg.finish_dataset()

    def _train(self):
        set_seeds(GLOBAL_SEED)
        # Start training with the NAS search phase's idle GPU memory reclaimed.
        free_gpu()
        self.model.to(self.device)

        weight_decay = WEIGHT_DECAY

        # Training budget. SEARCH_FRAC / TRAIN_FRAC are complementary fractions of
        # the TOTAL per-dataset clock; the GBG already knows how much NAS consumed.
        gbg = self.metadata.get('_gbg')
        if gbg is not None:
            # GBG is active: training gets everything left of the allocated budget.
            # effective_remaining() already accounts for NAS time, so no fraction
            # needed. clock.check() is the hard ceiling (organizer's clock).
            train_budget = min(self.clock.check(), gbg.effective_remaining())
        else:
            # Legacy fallback when GBG is not available: estimate from fractions.
            remaining_frac = max(1.0 - SEARCH_FRAC, 1e-6)
            train_budget   = self.clock.check() * (TRAIN_FRAC / remaining_frac)
        deadline = time.perf_counter() + train_budget
        t_train_start  = time.perf_counter()

        # Early stopping (regression-based with dynamic improvement delta)
        stopper = _EarlyStopper(ES_PATIENCE, ES_PLATEAU_PATIENCE, ES_MIN_EPOCHS,
                                ES_DELTA_START, ES_DELTA_MIN,
                                ES_DELTA_DECAY, ES_REGRESSION_DELTA, ES_ENABLED)

        n_cls = self.metadata['num_classes']
        label_smoothing = LABEL_SMOOTHING if n_cls >= LABEL_SMOOTHING_MIN_CLASSES else 0.0

        es_desc = (f"δ↑{ES_DELTA_START:.4f}↘{ES_DELTA_MIN:.4f}/{ES_DELTA_DECAY} "
                   f"↓{ES_REGRESSION_DELTA:.4f} p={ES_PATIENCE}/~{ES_PLATEAU_PATIENCE}") if ES_ENABLED else "off"
        print(f"  Trainer | budget={show_time(train_budget)} wd={weight_decay:.0e}"
              f" | ES={es_desc} | device={self.device}")

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN)

        use_amp = torch.cuda.is_available()
        scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_acc   = 0.0
        best_epoch = 0
        best_state = None
        epoch_times: list[float] = []
        epoch = 0
        # Adaptive micro-batch: None = use the full batch. On OOM we halve it and
        # accumulate gradients so the EFFECTIVE batch size never changes. Persists
        # across epochs so we don't re-discover the limit every epoch.
        self._micro_bs = None

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

                out = self._forward_backward(x, y, optimizer, criterion,
                                             scaler, use_amp)

                if self._xm is not None:
                    self._xm.mark_step()

                labels += y[:out.size(0)].cpu().tolist()
                preds  += out.argmax(1).cpu().tolist()

            scheduler.step()
            epoch += 1
            epoch_times.append(time.perf_counter() - t0)

            train_acc = accuracy_score(labels, preds)
            val_acc   = self._evaluate(self.valid_dl)
            lr_now    = scheduler.get_last_lr()[0]

            if val_acc > best_acc:
                best_acc   = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            stop, zone = stopper.step(val_acc, epoch)
            wait_str = f" ↓{stopper.wait}/{ES_PATIENCE}" if zone == '↓' else ""
            print("  Epoch {:>3} | Train {:>6.2f}% | Val {:>6.2f}% | {} | lr {:.2e} {}{}".format(
                epoch, train_acc * 100, val_acc * 100,
                show_time(epoch_times[-1]), lr_now, zone, wait_str))

            if stop:
                saved = show_time(max(0.0, deadline - time.perf_counter()))
                reason = (f"plateau ~×{ES_PLATEAU_PATIENCE}" if zone == '~'
                          else f"regression ↓>{ES_REGRESSION_DELTA*100:.1f}pp ×{ES_PATIENCE}")
                print(f"  Early stop at epoch {epoch} ({reason}). ~{saved} returned to pool.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  ← Restored best weights from epoch {best_epoch} (val={best_acc*100:.2f}%)")

        # Final evaluation with best weights
        final_val_acc   = self._evaluate(self.valid_dl)
        final_train_acc = self._evaluate(self.train_dl)
        print(f"  {'─'*55}")
        print(f"  Final | Train {final_train_acc*100:.2f}% | Val {final_val_acc*100:.2f}%"
              f"  (best epoch: {best_epoch})")
        print(f"  {'─'*55}")

        train_elapsed = time.perf_counter() - t_train_start
        self._save_report(epoch, final_train_acc, final_val_acc, train_elapsed)
        self._save_model()

        return self.model

    # ------------------------------------------------------------------
    def _forward_backward(self, x, y, optimizer, criterion, scaler, use_amp):
        """
        One optimizer step on (x, y), resilient to CUDA OOM.

        If a prior batch OOM'd, self._micro_bs holds a chunk size < full batch.
        The batch is split into chunks of that size and gradients are accumulated
        across them, so the EFFECTIVE batch (and thus the gradient) is identical
        to a single full-batch step — only peak activation memory shrinks. Each
        chunk's mean loss is weighted by its sample fraction so the sum equals the
        full-batch mean. (Anisotropic family uses GroupNorm, which is batch-size
        independent → exactly equivalent; BN families drift slightly, acceptable
        since this only triggers on OOM.)

        On OOM we halve the chunk size and retry the same batch. If it still OOMs
        at chunk size 1, the model genuinely cannot fit and we raise.

        Returns the concatenated detached logits (ordered) for metric tracking.
        """
        bs = x.size(0)
        optimizer.zero_grad(set_to_none=True)
        while True:
            chunk = bs if self._micro_bs is None else self._micro_bs
            try:
                outs = []
                for i in range(0, bs, chunk):
                    xs, ys = x[i:i + chunk], y[i:i + chunk]
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        out  = self.model(xs)
                        loss = criterion(out, ys) * (xs.size(0) / bs)
                    scaler.scale(loss).backward()
                    outs.append(out.detach())
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                return torch.cat(outs, dim=0)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                free_gpu()
                new_chunk = max(1, chunk // 2)
                if new_chunk == chunk:   # already at 1 and still OOM
                    raise RuntimeError(
                        "CUDA OOM even at micro-batch size 1 — model genuinely "
                        "exceeds GPU memory. Tighten repair.py memory estimator."
                    )
                self._micro_bs = new_chunk
                print(f"  [OOM] micro-batch → {self._micro_bs}"
                      f"  (effective batch {bs} preserved via grad accumulation)")

    # ------------------------------------------------------------------
    def _evaluate(self, loader):
        self.model.eval()
        labels, preds = [], []
        # Mirror the training micro-batch limit so eval can't OOM where train fit.
        eval_chunk = self._micro_bs if getattr(self, '_micro_bs', None) else None
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                bs = x.size(0)
                chunk = eval_chunk or bs
                for i in range(0, bs, chunk):
                    out = self.model(x[i:i + chunk])
                    preds += out.argmax(1).cpu().tolist()
                labels += y.tolist()
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
        # Honour the micro-batch limit discovered during training so inference
        # over the (often larger) test set cannot OOM where training fit.
        eval_chunk = self._micro_bs if getattr(self, '_micro_bs', None) else None
        with torch.no_grad():
            for x in test_loader:
                x = x.to(self.device)
                bs = x.size(0)
                chunk = eval_chunk or bs
                for i in range(0, bs, chunk):
                    preds += self.model(x[i:i + chunk]).argmax(1).cpu().tolist()
        return preds
