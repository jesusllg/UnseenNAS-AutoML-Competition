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

from helpers import (show_time, set_seeds, GLOBAL_SEED, free_gpu,
                     get_safe_time_remaining, split_budget,
                     make_amp_tools, is_oom)

# Pipeline hyperparameters: single source of truth in config.py.
from config import (
    WEIGHT_DECAY,
    LEARNING_RATE, GRAD_CLIP_NORM, LR_T_MAX, LR_ETA_MIN,
    LABEL_SMOOTHING, LABEL_SMOOTHING_MIN_CLASSES,
    ES_ENABLED, ES_PATIENCE, ES_PLATEAU_PATIENCE, ES_MIN_EPOCHS,
    ES_DELTA_START, ES_DELTA_MIN, ES_DELTA_DECAY, ES_REGRESSION_DELTA,
    ES_PATIENCE_MAX_MULT, SECOND_SHOT_MIN_S, SECOND_SHOT_SKIP_VAL,
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

    def step(self, val_acc, epoch, patience_mult: float = 1.0):
        """
        Return (stop, zone_char) for logging.

        patience_mult stretches both patience thresholds — the Trainer passes
        a clock-aware multiplier (×MAX while ≥50% of the dataset clock is
        left, tapering to ×1) so early stopping is generous when hours remain
        and strict when time is scarce.
        """
        if not self.enabled or epoch < self.min_epochs:
            return False, '~'

        eff_pat     = self.patience * max(1.0, patience_mult)
        eff_plateau = self.plateau_patience * max(1.0, patience_mult)

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
            return self.wait >= eff_pat, '↓'
        else:
            self.plateau_wait += 1
            return self.plateau_wait >= eff_plateau, '~'


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

    def _patience_mult(self) -> float:
        """
        Clock-aware early-stopping patience multiplier: ×ES_PATIENCE_MAX_MULT
        while ≥50% of the dataset clock remains, tapering linearly to ×1 as
        the clock is consumed. Short clocks are effectively unchanged.
        """
        tl = self.metadata.get('time_limit') if isinstance(self.metadata, dict) else None
        total_s = float(tl) * 3600.0 if isinstance(tl, (int, float)) and tl > 0 \
            else 0.5 * 3600.0
        try:
            frac_left = max(0.0, float(self.clock.check())) / total_s
        except Exception:
            return 1.0
        return 1.0 + (ES_PATIENCE_MAX_MULT - 1.0) * min(1.0, frac_left / 0.5)

    def _fit(self, model, deadline, tag=''):
        """
        Train ONE model until deadline / early stop. Restores its best weights
        before returning. Reused by the main run and the second shot.
        """
        model.to(self.device)
        self._micro_bs = None   # per-model OOM micro-batch state

        n_cls = self.metadata['num_classes']
        label_smoothing = LABEL_SMOOTHING if n_cls >= LABEL_SMOOTHING_MIN_CLASSES else 0.0

        stopper = _EarlyStopper(ES_PATIENCE, ES_PLATEAU_PATIENCE, ES_MIN_EPOCHS,
                                ES_DELTA_START, ES_DELTA_MIN,
                                ES_DELTA_DECAY, ES_REGRESSION_DELTA, ES_ENABLED)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX,
                                                         eta_min=LR_ETA_MIN)
        use_amp = torch.cuda.is_available()
        # AMP tools resolved per torch version (2.x torch.amp / 1.10 cuda.amp)
        self._autocast, scaler = make_amp_tools(use_amp)

        best_acc, best_epoch, best_state = 0.0, 0, None
        epoch_times: list = []
        epoch, train_acc = 0, 0.0
        stopped_early = False
        self.model, _model_bak = model, self.model   # _forward_backward uses self.model

        try:
            while True:
                t_left = deadline - time.perf_counter()
                if t_left <= 0:
                    print(f"  {tag}Time limit reached after {epoch} epochs.")
                    break
                if epoch_times:
                    avg = sum(epoch_times[-3:]) / len(epoch_times[-3:])
                    if avg > t_left * 0.9:
                        print(f"  {tag}Stopping — ~{show_time(avg)}/epoch, {show_time(t_left)} left.")
                        break

                t0 = time.perf_counter()
                model.train()
                labels, preds = [], []
                hit_deadline = False

                for x, y in self.train_dl:
                    # Mid-epoch deadline check: bounds clock overrun to one
                    # batch instead of one epoch (the average-time guard can't
                    # protect the FIRST epoch — no history yet).
                    if time.perf_counter() >= deadline:
                        hit_deadline = True
                        break
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

                if hit_deadline:
                    print(f"  {tag}Time limit hit mid-epoch {epoch} — stopping with "
                          f"{'best' if best_state is not None else 'current'} weights.")
                    break

                train_acc = accuracy_score(labels, preds)
                val_acc   = self._evaluate(self.valid_dl)
                lr_now    = scheduler.get_last_lr()[0]

                if val_acc > best_acc:
                    best_acc   = val_acc
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                mult = self._patience_mult()
                stop, zone = stopper.step(val_acc, epoch, patience_mult=mult)
                wait_str = f" ↓{stopper.wait}/{int(ES_PATIENCE*mult)}" if zone == '↓' else ""
                print("  {}Epoch {:>3} | Train {:>6.2f}% | Val {:>6.2f}% | {} | lr {:.2e} {}{}".format(
                    tag, epoch, train_acc * 100, val_acc * 100,
                    show_time(epoch_times[-1]), lr_now, zone, wait_str))

                if stop:
                    saved = show_time(max(0.0, deadline - time.perf_counter()))
                    reason = (f"plateau ~×{int(ES_PLATEAU_PATIENCE*mult)}" if zone == '~'
                              else f"regression ↓>{ES_REGRESSION_DELTA*100:.1f}pp"
                                   f" ×{int(ES_PATIENCE*mult)}")
                    print(f"  {tag}Early stop at epoch {epoch} ({reason})."
                          f" ~{saved} returned to pool.")
                    stopped_early = True
                    break
        finally:
            self.model = _model_bak

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  {tag}← Restored best weights from epoch {best_epoch}"
                  f" (val={best_acc*100:.2f}%)")

        return {'best_acc': best_acc, 'best_epoch': best_epoch,
                'has_best': best_state is not None, 'epochs': epoch,
                'epoch_times': epoch_times, 'train_acc': train_acc,
                'stopped_early': stopped_early}

    def _train(self):
        set_seeds(GLOBAL_SEED)
        # Start training with the NAS search phase's idle GPU memory reclaimed.
        free_gpu()

        # Training budget: everything left on this dataset's clock (metadata
        # time_remaining and live clock reconciled conservatively) minus a
        # reserve held back for prediction + output writing. A missed predict
        # scores -10; a slightly shorter training run costs far less.
        remaining = get_safe_time_remaining(self.metadata, self.clock)
        budget = split_budget(remaining)
        self._reserve_s = budget['reserve_s']
        train_budget    = budget['train_s']
        t_train_start   = time.perf_counter()

        es_desc = (f"δ↑{ES_DELTA_START:.4f}↘{ES_DELTA_MIN:.4f}/{ES_DELTA_DECAY} "
                   f"↓{ES_REGRESSION_DELTA:.4f} p={ES_PATIENCE}/~{ES_PLATEAU_PATIENCE}"
                   f"·adapt≤×{ES_PATIENCE_MAX_MULT:.0f}") if ES_ENABLED else "off"
        print(f"  Trainer | budget={show_time(train_budget)}"
              f" (predict reserve {show_time(self._reserve_s)})"
              f" wd={WEIGHT_DECAY:.0e} | ES={es_desc} | device={self.device}")

        fit1   = self._fit(self.model, time.perf_counter() + train_budget)
        chosen, fit2 = fit1, None

        # ── Second shot: reinvest the idle clock into the rerank runner-up ────
        # V4 evidence: early stopping routinely ended runs with hours unused;
        # every regression was "small model picked + early exit + wasted time".
        # If enough trainable time remains, fully train the runner-up and keep
        # whichever model validates better. Uses ONLY otherwise-wasted time.
        info = self.metadata.get('_second_shot')
        idle = self.clock.check() - self._reserve_s
        if info is not None and idle > SECOND_SHOT_MIN_S \
                and fit1['best_acc'] >= SECOND_SHOT_SKIP_VAL:
            print(f"  Second shot skipped — first model already at"
                  f" {fit1['best_acc']*100:.2f}% val, nothing to gain.")
        elif info is not None and idle > SECOND_SHOT_MIN_S:
            try:
                from search_space import build_model
                shape = self.metadata['input_shape']
                print(f"  {'─'*55}")
                print(f"  Second shot | {show_time(idle)} idle — training rerank"
                      f" runner-up ({info.get('params', 0)/1e6:.2f}M,"
                      f" probe val={info.get('val_probe', 0)*100:.1f}%)")
                self.model.cpu()
                free_gpu()
                set_seeds(GLOBAL_SEED)
                m2 = build_model(info['genotype'], shape[1], shape[2], shape[3],
                                 self.metadata['num_classes'],
                                 aniso_axis=info.get('aniso_axis'))
                fit2 = self._fit(m2, time.perf_counter() +
                                 (self.clock.check() - self._reserve_s), tag='[2nd] ')
                if fit2['best_acc'] > fit1['best_acc']:
                    print(f"  Second shot WINS: val {fit2['best_acc']*100:.2f}%"
                          f" > {fit1['best_acc']*100:.2f}% — switching model.")
                    self.model = m2
                    chosen = fit2
                else:
                    print(f"  Second shot stays second: val {fit2['best_acc']*100:.2f}%"
                          f" ≤ {fit1['best_acc']*100:.2f}% — first model stands.")
                    del m2
                    free_gpu()
            except Exception as e:
                print(f"  [Trainer] second shot failed (non-fatal): {e}")
                free_gpu()

        self.model.to(self.device)
        best_epoch  = chosen['best_epoch']
        train_acc   = chosen['train_acc']
        epoch       = fit1['epochs'] + (fit2['epochs'] if fit2 else 0)
        epoch_times = chosen['epoch_times']
        self._ss_summary = None if fit2 is None else {
            'winner':     'second' if chosen is fit2 else 'first',
            'first_val':  round(fit1['best_acc'], 4),
            'second_val': round(fit2['best_acc'], 4),
        }

        if chosen['has_best']:
            # Re-evaluating the restored weights would reproduce best_acc by
            # construction (eval is deterministic) — don't spend a valid pass.
            final_val_acc = chosen['best_acc']
        elif self.clock.check() - self._reserve_s > 0:
            final_val_acc = self._evaluate(self.valid_dl)
        else:
            final_val_acc = 0.0   # no time left to measure — predict still runs

        # The full train-set pass is diagnostic only; skip it when it could eat
        # into the predict reserve (cost ≈ one epoch's forward).
        train_eval_cost = epoch_times[-1] if epoch_times else 0.0
        if self.clock.check() - self._reserve_s > train_eval_cost:
            final_train_acc = self._evaluate(self.train_dl)
        else:
            final_train_acc = train_acc
            print("  (skipped final train-set eval — protecting predict reserve)")
        print(f"  {'─'*55}")
        print(f"  Final | Train {final_train_acc*100:.2f}% | Val {final_val_acc*100:.2f}%"
              f"  (best epoch: {best_epoch})")
        print(f"  {'─'*55}")

        train_elapsed = time.perf_counter() - t_train_start
        # Artifacts are diagnostics only — an I/O failure must never fail the
        # dataset, and saving must NEVER erode the predict reserve (main.py
        # calls predict() next and relies on that reserve). Gate on slack
        # ABOVE the full reserve: the report is a tiny JSON (a few seconds'
        # slack is plenty); the model file is tens of MB, so require a whole
        # extra reserve to remain intact after it.
        slack = self.clock.check() - self._reserve_s
        try:
            if slack > 5.0:
                self._save_report(epoch, final_train_acc, final_val_acc, train_elapsed)
            else:
                print("  (skipped report save — protecting predict reserve)")
        except Exception as e:
            print(f"  [Trainer] report save failed (non-fatal): {e}")
        try:
            if slack > self._reserve_s:
                self._save_model()
            else:
                print("  (skipped model save — protecting predict reserve)")
        except Exception as e:
            print(f"  [Trainer] model save failed (non-fatal): {e}")

        # Model is safely on CPU after _save_model (or still referenced here
        # regardless) — reclaim search/training GPU memory before prediction.
        free_gpu()

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
                    with self._autocast():
                        out  = self.model(xs)
                        loss = criterion(out, ys) * (xs.size(0) / bs)
                    scaler.scale(loss).backward()
                    outs.append(out.detach())
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                return torch.cat(outs, dim=0)
            except RuntimeError as e:
                # is_oom covers torch.cuda.OutOfMemoryError (1.13+ subclass of
                # RuntimeError) AND the plain-RuntimeError OOM of older torch.
                if not is_oom(e):
                    raise
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
            'second_shot':      json.dumps(getattr(self, '_ss_summary', None)),
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
    def _majority_class(self) -> int:
        """Most frequent training label — last-resort prediction filler."""
        try:
            y = getattr(self.train_dl.dataset, 'y', None)
            if y is not None and len(y) > 0:
                return int(torch.bincount(y).argmax().item())
        except Exception:
            pass
        return 0

    def _predict_model(self, test_loader):
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

    def predict(self, test_loader):
        """
        Predictions MUST come back with exactly one label per test sample —
        anything else breaks scoring. If model inference dies (OOM, corrupt
        weights, anything), degrade to CPU inference, then, as an absolute
        last resort, pad with the majority training class rather than fail
        the dataset outright.
        """
        try:
            preds = self._predict_model(test_loader)
        except Exception:
            print("  [Trainer] predict failed — retrying on CPU.")
            print(traceback.format_exc())
            free_gpu()
            try:
                self.device = torch.device('cpu')
                preds = self._predict_model(test_loader)
            except Exception:
                print("  [Trainer] CPU predict also failed — majority-class fallback.")
                preds = []

        try:
            n_test = len(test_loader.dataset)
        except Exception:
            return preds   # cannot validate length — return what we have

        if len(preds) != n_test:
            maj = self._majority_class()
            print(f"  [Trainer] prediction count {len(preds)} != {n_test} test"
                  f" samples — padding/truncating with class {maj}.")
            preds = list(preds)[:n_test] + [maj] * max(0, n_test - len(preds))
        return preds
