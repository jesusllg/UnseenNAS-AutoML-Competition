# UnseenNAS — AutoML Competition Submission

Our submission for the [5th NAS Unseen-Data Competition](https://www.nascompetition.com). The pipeline is fully time-budget aware: every component inspects the clock and adapts its depth of search or training to the time actually available.

---

## Repository structure

```
.
├── submission/               # Our implementation (what gets submitted)
│   ├── data_processor.py     # Normalisation + augmentation → DataLoaders
│   ├── nas.py                # Random architecture search over SearchableCNN
│   ├── trainer.py            # Budget-aware training loop with AMP + best-weight restore
│   └── helpers.py            # show_time() utility
├── evaluation/               # Server-side scripts (do not modify for submission)
│   ├── main.py               # Full evaluation pipeline
│   └── score.py              # Scoring against ground-truth labels
├── Makefile                  # build / run / score / zip targets
├── download_datasets.py      # Downloads the 13 practice datasets
└── requirements.txt
```

---

## Design overview

### DataProcessor (`submission/data_processor.py`)

- Computes per-channel mean and std from the training set and stores them in `metadata` for downstream use.
- Applies **RandomHorizontalFlip** on all datasets; adds **RandomCrop** with adaptive padding on images ≥ 32 px.
- Selects batch size automatically based on pixel count: `64` (small) → `32` (medium) → `16` (large), to avoid OOM on high-resolution datasets.

### NAS (`submission/nas.py`)

Search space is a configurable stack of stages, each stage being a sequence of `ConvBnAct` or `ResBlock` layers followed by optional max-pooling.

| Axis | Choices |
|---|---|
| Channels per stage | 32, 64, 128, 256 |
| Blocks per stage | 1, 2, 3 |
| Kernel size | 3, 5 |
| Use residual | True / False |

**Benchmark-aware calibration** — the search budget, proxy epochs, proxy batches, and minimum starting channel width are all scaled to the dataset benchmark score:

| Benchmark | Budget fraction | Proxy epochs × batches | Min channel index |
|---|---|---|---|
| ≥ 85 (hard) | 35 % | 5 × 50 | 64+ |
| 65 – 84 (medium) | 30 % | 3 × 40 | 32+ |
| < 65 (easy) | 20 % | 3 × 30 | 32+ |

If the entire search budget expires without producing a valid model, a tier-matched fallback architecture is returned.

### Trainer (`submission/trainer.py`)

- Trains for as many epochs as the remaining time allows, using a 3-epoch rolling average to decide whether another epoch fits.
- **AdamW** optimizer with **CosineAnnealingLR** (T_max=200, eta_min=1e-5).
- **Mixed-precision** (torch AMP) + gradient clipping at norm 1.0 when a GPU is available.
- **Label smoothing** (ε=0.1) for datasets with ≥ 10 classes.
- Saves and restores the best validation checkpoint at the end.
- `budget_frac` and `weight_decay` are also tuned to the benchmark tier (harder datasets get stronger L2).

---

## Setup

```bash
pip install -r requirements.txt
```

### Download practice datasets

```bash
python download_datasets.py          # all 13 datasets
python download_datasets.py AddNIST  # single dataset
python download_datasets.py --list   # show available datasets
```

Datasets are saved under `datasets/<Name>/` with the expected NumPy + metadata structure.

---

## Testing locally

The Makefile mirrors the server evaluation pipeline exactly.

```bash
# Full end-to-end test (clean → build → run → score)
make submission=submission all

# Individual steps
make submission=submission build   # assembles package/
make submission=submission run     # runs main.py inside package/
make submission=submission score   # scores predictions against labels

# Bundle for submission
make submission=submission zip     # creates submission.zip
```

> Each dataset's `metadata` file must include a `time_limit` field (hours). Default is 0.5 h if missing.
> Example: `{"num_classes": 20, "input_shape": [50000, 3, 28, 28], "codename": "Adaline", "benchmark": 89.85, "time_limit": 1.0}`

---

## Practice datasets

| Dataset | DOI |
|---|---|
| AddNIST | https://doi.org/10.25405/data.ncl.24574354.v1 |
| Language | https://doi.org/10.25405/data.ncl.24574729.v1 |
| MultNIST | https://doi.org/10.25405/data.ncl.24574678.v1 |
| CIFARTile | https://doi.org/10.25405/data.ncl.24551539.v1 |
| Gutenberg | https://doi.org/10.25405/data.ncl.24574753.v1 |
| GeoClassing | https://doi.org/10.25405/data.ncl.24050256.v3 |
| Chesseract | https://doi.org/10.25405/data.ncl.24118743.v2 |
| Sudoku | https://doi.org/10.25405/data.ncl.26976121.v1 |
| Voxel | https://doi.org/10.25405/data.ncl.26970223.v1 |
| Myofibre | https://doi.org/10.25405/data.ncl.26969998.v1 |
| GameOfLife | https://doi.org/10.25405/data.ncl.30000835 |
| Cryptic | https://doi.org/10.7488/ds/8054 |
| Windspeed | https://doi.org/10.7488/ds/8053 |

---

## Competition reference

Starter kit: https://github.com/Towers-D/NAS-Comp-Starter-Kit
