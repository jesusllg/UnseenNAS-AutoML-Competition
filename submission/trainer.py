import time
import traceback

import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score

from helpers import show_time


# ── Benchmark-aware training calibration ──────────────────────────────────────

def _train_params(benchmark):
    """
    Return (budget_frac, weight_decay) tuned to the dataset benchmark.

    High-benchmark datasets need stronger regularisation to avoid over-fitting
    the small margin between baseline and perfection.
    """
    if benchmark is None:
        return 0.85, 1e-4
    b = float(benchmark)
    if b >= 85:
        return 0.87, 3e-4   # more budget + stronger L2 for hard datasets
    elif b >= 65:
        return 0.85, 1e-4
    else:
        return 0.83, 5e-5   # easy datasets don't need heavy regularisation


# ── Trainer class ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model    = model
        self.device   = device
        self.train_dl = train_dataloader
        self.valid_dl = valid_dataloader
        self.metadata = metadata
        self.clock    = clock

    # ------------------------------------------------------------------
    def train(self):
        try:
            return self._train()
        except Exception:
            print("  [Trainer] Unexpected error — returning model as-is.")
            print(traceback.format_exc())
            return self.model

    def _train(self):
        self.model.to(self.device)

        benchmark               = self.metadata.get('benchmark', None)
        budget_frac, weight_decay = _train_params(benchmark)

        train_budget = self.clock.check() * budget_frac
        deadline     = time.perf_counter() + train_budget

        n_cls = self.metadata['num_classes']
        label_smoothing = 0.1 if n_cls >= 10 else 0.0

        print(f"  Trainer | benchmark={benchmark}"
              f" → budget={show_time(train_budget)} wd={weight_decay:.0e} | device={self.device}")

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
            # Skip epoch if rolling average won't fit in remaining time
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

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  Restored best weights (val={best_acc:.3f})")

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
    def predict(self, test_loader):
        self.model.to(self.device).eval()
        preds = []
        with torch.no_grad():
            for x in test_loader:
                preds += self.model(x.to(self.device)).argmax(1).cpu().tolist()
        return preds
