import copy
import time

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
        input_hw: (H, W) of input images — controls how many pooling steps are safe
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


def _sample_stage_configs(rng, n_stages):
    configs = []
    c_idx = 0
    for _ in range(n_stages):
        c_idx = int(rng.randint(c_idx, min(c_idx + 2, len(_CHANNELS) - 1) + 1))
        configs.append((
            _CHANNELS[c_idx],
            int(rng.choice(_BLOCKS)),
            int(rng.choice(_KERNELS)),
            bool(rng.randint(0, 2)),
        ))
    return configs


# ── Proxy (fast) evaluation ────────────────────────────────────────────────────

def _proxy_train(model, loader, device, n_epochs=3, max_batches=40):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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


# ── NAS class ──────────────────────────────────────────────────────────────────

# Sensible default if search time is too short to try anything
_DEFAULT_STAGES = [(64, 2, 3, True), (128, 2, 3, True), (256, 1, 3, False)]


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def search(self):
        in_c  = self.metadata['input_shape'][1]
        n_cls = self.metadata['num_classes']
        hw    = tuple(self.metadata['input_shape'][2:])  # (H, W)

        # Allocate 30 % of remaining time to architecture search
        search_budget = self.clock.check() * 0.30
        deadline = time.perf_counter() + search_budget
        print(f"  NAS budget: {show_time(search_budget)} | device: {self.device}")

        rng = np.random.RandomState(42)
        best_model, best_acc = None, -1.0
        n_tried = 0

        while time.perf_counter() < deadline:
            n_stages = int(rng.randint(2, 5))
            stage_cfg = _sample_stage_configs(rng, n_stages)

            try:
                model = SearchableCNN(in_c, n_cls, stage_cfg, hw)
                model = _proxy_train(model, self.train_loader, self.device)
                acc   = _proxy_acc(model, self.valid_loader, self.device)
                n_tried += 1
                print(f"  [{n_tried}] stages={n_stages} cfg={stage_cfg} → val={acc:.3f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model).cpu()

            except RuntimeError as e:
                print(f"  Architecture skipped ({e})")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"  Search done: {n_tried} architectures tried, best val_acc={best_acc:.3f}")

        if best_model is None:
            print("  Falling back to default architecture.")
            best_model = SearchableCNN(in_c, n_cls, _DEFAULT_STAGES, hw)

        return best_model
