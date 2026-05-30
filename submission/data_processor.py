import numpy as np
import torch
import torchvision.transforms as transforms


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

        normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

        if h >= 32:
            pad = max(4, h // 8)
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(h, padding=pad),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                normalize,
            ])

        eval_transform = transforms.Compose([normalize])

        # Smaller batches for large images to avoid OOM
        pixels = h * w * x.shape[1]
        if pixels > 100_000:
            batch_size = 16
        elif pixels > 10_000:
            batch_size = 32
        else:
            batch_size = 64

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
