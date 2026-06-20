import numpy as np
import torch
import torchvision.transforms as transforms

from helpers import select_batch_size
from search_space import infer_family


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        x = x.astype(np.float32)
        self.x = torch.from_numpy(x)
        if self.x.dim() == 3:
            self.x = self.x.unsqueeze(1)
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        im = self.x[idx]
        if self.transform is not None:
            im = self.transform(im)
        if self.y is None:
            return im
        return im, self.y[idx]


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def process(self):
        x = self.train_x.astype(np.float32)
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]

        # Per-channel mean/std from training set
        mean = x.mean(axis=(0, 2, 3))
        std = x.std(axis=(0, 2, 3))
        std = np.where(std < 1e-7, 1.0, std)

        self.metadata['norm_mean'] = mean.tolist()
        self.metadata['norm_std'] = std.tolist()

        h, w = x.shape[2], x.shape[3]
        n_cls = self.metadata.get('num_classes', 10)

        normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

        # Whether to apply RandomHorizontalFlip depends on the geometry family.
        # Anisotropic data (extreme aspect ratio ≥6×) likely has sequential structure
        # along the long axis; flipping that axis may invalidate positional semantics.
        # All other families default to flipping on.
        family = infer_family(x.shape[1], h, w, n_cls)
        use_hflip = getattr(family, 'augment_hflip', True)
        flip = [transforms.RandomHorizontalFlip()] if use_hflip else []

        if h >= 32:
            pad = max(4, h // 8)
            train_transform = transforms.Compose(flip + [
                transforms.RandomCrop(h, padding=pad),
                normalize,
            ])
        else:
            train_transform = transforms.Compose(flip + [normalize])

        eval_transform = transforms.Compose([normalize])

        # Smaller batches for large images to avoid OOM (shared rule so repair's
        # memory estimate is computed at the exact batch we train with).
        batch_size = select_batch_size(x.shape[1], h, w)
        self.metadata['batch_size'] = batch_size

        train_ds = _Dataset(self.train_x, self.train_y, transform=train_transform)
        valid_ds = _Dataset(self.valid_x, self.valid_y, transform=eval_transform)
        test_ds  = _Dataset(self.test_x,  None,         transform=eval_transform)

        kw = dict(num_workers=2, pin_memory=torch.cuda.is_available())

        # Seeded generator so shuffle order is reproducible across runs
        g = torch.Generator()
        g.manual_seed(42)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            generator=g, **kw)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, **kw)
        test_loader  = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw)

        return train_loader, valid_loader, test_loader
