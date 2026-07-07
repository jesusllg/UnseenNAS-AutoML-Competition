import numpy as np
import torch
import torchvision.transforms as transforms

from config import GLOBAL_SEED
from helpers import select_batch_size, select_num_workers
from search_space import infer_family


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        # copy=False: zero-copy when the source is already float32
        x = np.asarray(x).astype(np.float32, copy=False)
        self.x = torch.from_numpy(x)
        if self.x.dim() == 3:
            self.x = self.x.unsqueeze(1)
        if y is None:
            self.y = None
        else:
            # labels can arrive as (N,), (N,1), lists… normalise to flat (N,)
            y = np.asarray(y).reshape(-1)
            self.y = torch.from_numpy(y).long()
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


def _channel_stats(x, max_samples: int = 4096):
    """
    Per-channel mean/std WITHOUT materialising a float32 copy of the full
    train set (RAM on the evaluation machine is unknown). For big arrays the
    stats come from an evenly-strided sample — statistically plenty for
    normalisation constants.
    """
    step = max(1, len(x) // max_samples)
    xs = np.asarray(x[::step]).astype(np.float32, copy=False)
    if xs.ndim == 3:
        xs = xs[:, np.newaxis]
    mean = xs.mean(axis=(0, 2, 3))
    std = xs.std(axis=(0, 2, 3))
    std = np.where(std < 1e-7, 1.0, std)
    return mean, std


def build_transforms(family, C: int, H: int, W: int, mean, std):
    """
    (train_transform, eval_transform) with a conservative augmentation policy
    for unseen data: flips/crops ONLY for the natural-image families
    (family.augment_hflip), nothing anywhere else — a flip silently corrupts
    symbolic grids, sequences and channel-stacked volumes.

    RandomCrop always uses the full (H, W) target so rectangular inputs keep
    their shape (RandomCrop(h) would crop to h×h and break/deform W≠H data).
    """
    normalize = transforms.Normalize(mean=list(mean), std=list(std))
    aug = []
    if getattr(family, 'augment_hflip', False):
        aug.append(transforms.RandomHorizontalFlip())
        if min(H, W) >= 32:
            pad = max(2, min(H, W) // 8)
            aug.append(transforms.RandomCrop((H, W), padding=pad))
    return transforms.Compose(aug + [normalize]), transforms.Compose([normalize])


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
        shape = self.train_x.shape
        if len(shape) == 3:            # (N, H, W) → implicit single channel
            n, C, H, W = shape[0], 1, shape[1], shape[2]
        else:                           # (N, C, H, W)
            n, C, H, W = shape[0], shape[1], shape[2], shape[3]
        n_cls = self.metadata.get('num_classes', 10)

        # Stats from a sample — never a full float32 copy of train_x.
        mean, std = _channel_stats(self.train_x)

        # Family decides the augmentation policy (geometry-derived, never
        # dataset identity); we never write anything back into metadata.
        family = infer_family(C, H, W, n_cls)
        train_transform, eval_transform = build_transforms(
            family, C, H, W, mean.tolist(), std.tolist())

        # Batch size: shared rule with repair's memory estimator (also
        # VRAM-aware). Workers: adapted to CPU count and dataset size.
        batch_size = select_batch_size(C, H, W)
        num_workers = select_num_workers(n)

        train_ds = _Dataset(self.train_x, self.train_y, transform=train_transform)
        valid_ds = _Dataset(self.valid_x, self.valid_y, transform=eval_transform)
        test_ds  = _Dataset(self.test_x,  None,         transform=eval_transform)

        kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())

        # drop_last only when the dataset is comfortably larger than a batch —
        # on tiny datasets dropping the ragged batch can discard real signal
        # (and an epoch could even be empty).
        drop_last = len(train_ds) >= 2 * batch_size

        # Seeded generator so shuffle order is reproducible across runs
        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last,
            generator=g, **kw)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, **kw)
        # Test loader: strictly no shuffle, no drop_last (evaluator asserts it,
        # and predictions must align 1:1 with test samples).
        test_loader  = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw)

        return train_loader, valid_loader, test_loader
