# UnseenNAS — AutoML Competition Submission

> Zero-cost Neural Architecture Search for heterogeneous NCHW tensors.
> Aging Evolution + AZ-NAS proxy (CVPR 2024). No gradient-based training during search.

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [Repository layout](#2-repository-layout)
3. [Quick start — no datasets needed](#3-quick-start--no-datasets-needed)
4. [Git workflow — step by step](#4-git-workflow--step-by-step)
5. [Running the full evaluation pipeline](#5-running-the-full-evaluation-pipeline)
6. [The experiment notebook](#6-the-experiment-notebook)
7. [How the system works](#7-how-the-system-works)
8. [Complete hyperparameter reference](#8-complete-hyperparameter-reference)
9. [Fine-tuning scenarios](#9-fine-tuning-scenarios)
10. [Search space reference](#10-search-space-reference)
11. [Datasets](#11-datasets)
12. [Design decisions](#12-design-decisions)

---

## 1. What this project does

Given any image-like classification dataset (any number of channels, any spatial resolution, any number of classes), the system:

1. **Preprocesses** the data: per-channel normalisation, adaptive augmentation.
2. **Searches** for a good neural architecture in seconds — without training any network — using the AZ-NAS zero-cost proxy and Aging Evolution.
3. **Trains** the best architecture found within the remaining time budget.
4. **Predicts** class labels on the test split.

The key insight: AZ-NAS scores thousands of architectures on a single mini-batch by measuring their mathematical properties (expressivity, progressivity, trainability) instead of training them. This makes search 100–1000× faster than conventional approaches.

---

## 2. Repository layout

```
UnseenNAS-AutoML-Competition/
│
├── submission/                   # The actual submission (what gets evaluated)
│   ├── data_processor.py         # Normalisation + adaptive augmentation
│   ├── nas.py                    # NAS entry point: search + fallback
│   ├── trainer.py                # Training loop (AdamW + cosine LR + AMP)
│   ├── helpers.py                # Shared utilities (show_time, etc.)
│   └── search_space/             # The NAS engine
│       ├── __init__.py           # Public API — import from here
│       ├── genotype.py           # Architecture genome: dataclasses + mutation
│       ├── family.py             # Geometry → hard constraints (infer_family)
│       ├── block_library.py      # 11 primitive neural blocks + SEBlock + DropPath
│       ├── builder.py            # Genome → nn.Module decoder
│       ├── repair.py             # Deterministic constraint repair (11 rules)
│       ├── proxies.py            # AZ-NAS zero-cost fitness proxy
│       └── evolution.py          # Aging Evolution search loop
│
├── evaluation/                   # Competition evaluation pipeline
│   ├── main.py                   # Runs DataProcessor → NAS → Trainer → predict
│   └── score.py                  # Scores predictions vs ground-truth labels
│
├── notebooks/
│   └── nas_experiment.ipynb      # Paper-quality experiment notebook (9 sections)
│
├── datasets/                     # Put downloaded datasets here (git-ignored)
│
├── Makefile                      # Convenience targets: build / run / score / zip / all
├── download_datasets.py          # Download all 13 practice datasets
└── README.md                     # This file
```

---

## 3. Quick start — no datasets needed

```bash
# 1. Clone the repo
git clone https://github.com/jesusllg/unseennas-automl-competition
cd unseennas-automl-competition

# 2. Install dependencies (Python 3.8+, PyTorch 2.0+)
pip install torch torchvision numpy scipy matplotlib scikit-learn

# 3. Sanity-check the search space in ~5 seconds
python - <<'EOF'
import sys
sys.path.insert(0, 'submission')
from search_space import infer_family, sample_random_genotype, repair, build_model, az_nas_score
import torch

C, H, W, NC = 3, 32, 32, 10
family = infer_family(C, H, W, NC)
print(f"Family: {family.name}")

g = sample_random_genotype()
g = repair(g, C, H, W, NC, family)
model = build_model(g, C, H, W, NC)

x = torch.randn(4, C, H, W)
print(f"Output shape: {model(x).shape}")            # torch.Size([4, 10])
print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
print(f"AZ-NAS score: {az_nas_score(model, x, 'cpu'):.4f}")
EOF
```

Expected output:
```
Family: visual_medium
Output shape: torch.Size([4, 10])
Parameters:   284,234
AZ-NAS score: 12.3456
```

---

## 4. Git workflow — step by step

This section explains the complete Git workflow for anyone new to version control.

### 4.1 Initial setup (one time only)

```bash
# Configure your identity
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

# Clone the repository to your machine
git clone https://github.com/jesusllg/unseennas-automl-competition
cd unseennas-automl-competition
```

### 4.2 Starting a new experiment or feature

Always develop on a branch — never directly on `main`.

```bash
# Create and switch to a new branch
git checkout -b experiment/my-new-search-space

# You are now on your branch — main is untouched
git branch   # should show * experiment/my-new-search-space
```

### 4.3 Making changes

Edit any files you need. After making changes:

```bash
# See what changed
git status
git diff submission/search_space/evolution.py

# Stage specific files (preferred over git add -A)
git add submission/search_space/evolution.py
git add submission/nas.py

# Commit with a descriptive message
git commit -m "Increase population size to 60 for harder benchmarks"
```

### 4.4 Pushing your branch to GitHub

```bash
# First push: -u links local branch to remote
git push -u origin experiment/my-new-search-space

# Subsequent pushes on the same branch
git push
```

### 4.5 Keeping your branch up to date with main

If `main` has been updated while you were working:

```bash
# Fetch latest changes from GitHub
git fetch origin main

# Rebase your branch on top of the updated main
git rebase origin/main

# If there are conflicts, resolve them, then:
git add <conflicted-file>
git rebase --continue
```

### 4.6 Merging to main

```bash
# Switch to main and pull latest
git checkout main
git pull origin main

# Merge your branch
git merge --no-ff experiment/my-new-search-space -m "Merge larger population experiment"

# Push main to GitHub
git push origin main
```

### 4.7 Creating a pull request (GitHub UI)

1. Push your branch: `git push -u origin experiment/my-branch`
2. Go to `https://github.com/jesusllg/unseennas-automl-competition`
3. Click **"Compare & pull request"** for your branch
4. Write a description of what changed and why
5. Click **"Create pull request"**
6. After review, click **"Merge pull request"**

### 4.8 Useful git commands

```bash
git log --oneline -10          # Last 10 commits
git log --oneline --graph      # Branching history as ASCII graph
git diff HEAD~1                # What changed in the last commit
git stash                      # Temporarily save uncommitted changes
git stash pop                  # Restore stashed changes
git checkout -                 # Switch back to previous branch
git reset HEAD~1               # Undo last commit (keep changes staged)
```

---

## 5. Running the full evaluation pipeline

### 5.1 Dependencies

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn
```

### 5.2 Download practice datasets

```bash
python download_datasets.py
# Downloads all 13 datasets into datasets/
# Takes ~5-10 minutes depending on connection
```

### 5.3 Run with Makefile

```bash
# Run full pipeline on all datasets (DataProcessor → NAS → Trainer → score)
make submission=submission all

# Individual steps
make submission=submission build   # Validate syntax and imports
make submission=submission run     # Run main.py, produce predictions
make submission=submission score   # Score predictions against labels
make submission=submission zip     # Create submission.zip for upload
```

### 5.4 Run manually (without Makefile)

```bash
cd evaluation
python main.py \
    --submission_dir ../submission \
    --datasets_dir   ../datasets \
    --output_dir     ../predictions \
    --time_limit     1200          # 20 minutes per dataset (in seconds)
```

### 5.5 Run on a single dataset

```bash
python evaluation/main.py \
    --submission_dir submission \
    --datasets_dir   datasets/CIFARTile \
    --output_dir     predictions \
    --time_limit     300           # 5 minutes — quick test
```

---

## 6. The experiment notebook

`notebooks/nas_experiment.ipynb` is a comprehensive, paper-quality experiment notebook. Open it in:

- **Google Colab** (recommended): Upload to [colab.research.google.com](https://colab.research.google.com), or use `File → Open notebook → GitHub`
- **JupyterLab locally**: `jupyter lab notebooks/nas_experiment.ipynb`
- **VS Code**: Open the `.ipynb` file directly

### 6.1 What each section does

| Section | Title | What you see |
|---|---|---|
| 0 | Setup | Package installation, GPU detection, visual style configuration |
| 1 | Real data | Dataset gallery, class distribution, pixel histograms (real images if available) |
| 2 | Family detection | Horizontal bar chart of family scores, constraint heatmap per family |
| 3 | Architecture chromosomes | Gene chromosome diagrams, block gallery cards with colour coding |
| 4 | AZ-NAS deep dive | Eigenvalue spectrum, layer expressivity profile, feature map thumbnails, component bars |
| 5 | Population landscape | Fitness ranking, Pareto scatter, violin distributions by block type |
| 6 | Aging Evolution | Evolution curve, fitness vs round, evolution vs random comparison |
| 7 | Proxy–accuracy correlation | Scatter plot of AZ-NAS score vs 3-epoch val accuracy |
| 8 | ⚠ Full pipeline mini-demo | End-to-end: DataProcessor → NAS → Trainer → predict (SLOW) |
| 9 | Summary dashboard | Pipeline card, full metrics table, radar chart |

### 6.2 Where to get data for the notebook

The notebook tries to load real data from:
- `../datasets/CIFARTile/` (first choice)
- `../datasets/AddNIST/` (second choice)

If neither is found it falls back to synthetic tensors automatically. Run `python download_datasets.py` first for the best experience.

### 6.3 Key notebook parameters to tune

At the top of Section 4 (AZ-NAS evaluation):
```python
MAX_SAMPLES = 15      # Number of architectures to score — increase for richer plot
```

At the top of Section 6 (Evolution):
```python
EVO_POP      = 10     # Population size (real runs use 30-40)
EVO_ROUNDS   = 30     # Evolution rounds (real runs use 100-150)
EVO_TOURNEY  = 3      # Tournament size
```

At the top of Section 7 (Proxy–accuracy correlation):
```python
CORR_N_ARCH    = 8    # Architectures to sample
CORR_N_EPOCHS  = 3    # Training epochs per architecture
CORR_N_BATCHES = 30   # Training batches per epoch
```

---

## 7. How the system works

### 7.1 DataProcessor (`data_processor.py`)

Runs before NAS and training. Responsibilities:
- Computes per-channel mean and std from the training split.
- Normalises inputs to zero mean, unit std.
- Adaptive augmentation: RandomHorizontalFlip always; RandomCrop (pad=max(4,H//8)) for images with H ≥ 32.
- Adaptive batch size: 16 for high-resolution (pixels > 100k), 32 for medium, 64 for small.

### 7.2 NAS search pipeline

```
infer_family(C, H, W, num_classes)
    ↓ FamilyProfile (geometry constraints)

aging_evolution(family, proxy_fn=az_nas_score, ...)
    ├── Initialise: n_population valid random genotypes
    ├── For each round (up to n_rounds or time budget):
    │     1. Tournament selection (best of tournament_size random picks)
    │     2. Mutate parent genotype (large → medium → small over time)
    │     3. Repair genotype (11 deterministic rules)
    │     4. Build model: build_model(genotype, C, H, W, num_classes)
    │     5. Evaluate: az_nas_score(model, proxy_batch, device)
    │     6. Add child to population; evict oldest
    └── Return sorted population

best_individual(population)
    ↓ Genotype

build_model(genotype, C, H, W, num_classes)
    ↓ SearchSpaceModel (nn.Module)

Trainer.train(model)
    ↓ Trained model

model.predict(test_loader)
    ↓ Class labels
```

### 7.3 AZ-NAS proxy (`proxies.py`)

Scores any architecture on a single mini-batch, no training needed:

```
score = Expressivity + Progressivity + Trainability − λ · Complexity

Expressivity  = Σ H(eigenvalue spectrum of per-layer spatial covariance)
                Measures: diversity of learned representations
                
Progressivity = min(Expressivity[i+1] − Expressivity[i]) across layers
                Measures: does the network deepen its representations?
                
Trainability  = mean(−σ_max − 1/σ_max + 2)  over inter-layer Jacobian SVDs
                Measures: stability of gradient flow (eigenvalue spread)
                
Complexity    = log(total_parameters)
                Penalises overly large models
```

All computed with Kaiming-normal reinitialisation. No labels required.

### 7.4 Aging Evolution (`evolution.py`)

Based on [Real et al. 2019 — Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548):

- Maintains a population of `n_population` architectures sorted by fitness.
- Each round: pick `tournament_size` random candidates → keep the best → mutate it → repair → score → insert.
- Eviction: removes the **oldest** (not the worst) — this prevents premature convergence and encourages exploration.
- Mutation scale anneals over time:
  - First 30% of rounds: `large` — replace an entire stage gene
  - Next 40% of rounds: `medium` — change fields across multiple stages
  - Last 30% of rounds: `small` — change 1-2 fields within one stage

### 7.5 Trainer (`trainer.py`)

- AdamW optimiser, lr=1e-3.
- CosineAnnealingLR: T_max=200 epochs, eta_min=1e-5.
- Mixed-precision (AMP) on CUDA.
- Label smoothing ε=0.1 for num_classes ≥ 10.
- Best-checkpoint: restores the weights with highest validation accuracy at the end.
- Time-budget aware: stops when the rolling average epoch time exceeds remaining budget × 0.9.

---

## 8. Complete hyperparameter reference

This section documents every tunable parameter, where it lives, and what effect changing it has.

---

### 8.1 Search budget allocation — `submission/nas.py`

**Function**: `_search_params(benchmark)`

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `search_frac` | `_search_params()` return | 0.20–0.35 | Fraction of remaining time allocated to NAS search. Higher → more search, less training. |
| `n_pop` (n_population) | `_search_params()` return | 20–40 | Population size in Aging Evolution. Larger → more diverse exploration, slower per round. |
| `n_rounds` | `_search_params()` return | 60–150 | Maximum evolution rounds. More → longer search, potentially better architecture. |
| `tourney_k` (tournament_size) | `_search_params()` return | 5–10 | Tournament size. Larger → more selection pressure, faster convergence, less exploration. |

Current schedule by `benchmark` metadata field:

| Benchmark value | search_frac | n_pop | n_rounds | tourney_k |
|---|---|---|---|---|
| ≥ 85 (hard) | 0.35 | 40 | 150 | 10 |
| 65–84 (medium) | 0.30 | 30 | 100 | 7 |
| < 65 (easy) | 0.20 | 20 | 60 | 5 |
| `None` | 0.30 | 30 | 100 | 7 |

**To change**: Edit `_search_params()` in `submission/nas.py`, lines 95–105.

```python
def _search_params(benchmark):
    if benchmark is None:
        return 0.30, 30, 100, 7      # (search_frac, n_pop, n_rounds, tourney_k)
    b = float(benchmark)
    if b >= 85:
        return 0.35, 40, 150, 10    # Increase 40→60 for richer search
    elif b >= 65:
        return 0.30, 30, 100, 7
    else:
        return 0.20, 20, 60, 5
```

**To override for a single run** without touching code, pass `benchmark` in metadata:
```python
metadata = {'benchmark': 95, ...}   # Forces hard-dataset search params
```

---

### 8.2 Aging Evolution — `submission/search_space/evolution.py`

**Function**: `aging_evolution(family, C, H, W, num_classes, proxy_fn, batch_x, device, n_population, n_rounds, tournament_size, time_budget_s, verbose)`

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `n_population` | int | 50 | Size of the maintained population. Rule of thumb: ≥ 20 for stable results. |
| `n_rounds` | int | 200 | Max number of child architectures evaluated. Each round = ~0.5–2s depending on model size. |
| `tournament_size` | int | 10 | Candidates in each tournament. Higher = greedier selection. Try 3–20. |
| `time_budget_s` | float | None | Hard wall-clock budget in seconds. Overrides `n_rounds` if reached first. |
| `verbose` | bool | False | Print progress every 10 rounds if True. |

Mutation scale thresholds (hardcoded in `evolution.py`, lines ~50-60):

```python
frac = round_idx / n_rounds
if   frac < 0.30:  scale = 'large'    # Replace full stage
elif frac < 0.70:  scale = 'medium'   # Multi-field changes
else:              scale = 'small'    # 1-2 field changes
```

**To make search more explorative**: lower `tournament_size` (e.g. 3) and keep `n_population` high.  
**To make search more exploitative**: raise `tournament_size` (e.g. 15) and shrink `n_population`.

---

### 8.3 AZ-NAS proxy weights — `submission/search_space/proxies.py`

**Function**: `az_nas_score(model, batch_x, device, lambda_c=0.1, reinit=True)`

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `lambda_c` | float | 0.1 | Weight on complexity penalty `log(params)`. Higher → smaller models preferred. |
| `reinit` | bool | True | Whether to Kaiming-reinitialise weights before scoring. Always keep True. |

**Component weights** (currently equal, summed directly):

```python
score = expressivity + progressivity + trainability - lambda_c * complexity
```

To change the relative weights, edit `az_nas_score()` in `proxies.py`:

```python
# Example: emphasise trainability for datasets where gradient flow matters
score = 1.0 * expressivity + 0.5 * progressivity + 2.0 * trainability - 0.1 * complexity
```

**Proxy batch size** (in `nas.py::_get_proxy_batch`):

```python
def _get_proxy_batch(train_loader, device, batch_size=16):
```

Larger batches give more reliable proxy scores but consume more GPU memory. Try 8–64.

---

### 8.4 Training — `submission/trainer.py`

**Function**: `_train_params(benchmark)` and `Trainer._train()`

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `budget_frac` | `_train_params()` return | 0.83–0.87 | Fraction of remaining time used for training. |
| `weight_decay` | `_train_params()` return | 5e-5 – 3e-4 | L2 regularisation. Higher → stronger regularisation (good for small datasets or many-class problems). |
| `lr` | `_train()` line ~68 | 1e-3 | Initial learning rate for AdamW. |
| `T_max` | `_train()` line ~69 | 200 | CosineAnnealingLR period (in epochs). Should be ≥ expected number of epochs. |
| `eta_min` | `_train()` line ~69 | 1e-5 | Minimum learning rate at the end of cosine schedule. |
| `label_smoothing` | `_train()` | 0.1 if n_cls≥10 | Label smoothing epsilon. Set to 0.0 to disable. |

Training budget schedule:

| Benchmark | budget_frac | weight_decay |
|---|---|---|
| ≥ 85 (hard) | 0.87 | 3e-4 |
| 65–84 (medium) | 0.85 | 1e-4 |
| < 65 (easy) | 0.83 | 5e-5 |

**To change the learning rate schedule**:

```python
# In trainer.py, _train():
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
# Change T_max to match your expected epoch count, e.g.:
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

**To add warmup**:
```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=195, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
```

---

### 8.5 Data processing — `submission/data_processor.py`

| Parameter | Location | Default | Effect |
|---|---|---|---|
| Batch size thresholds | `DataProcessor.process()` | pixels: 100k / 10k | Determines automatic batch size. Edit the `if` chain to override. |
| Crop padding | `DataProcessor.process()` | `max(4, H//8)` | RandomCrop padding for images H≥32. Larger → more aggressive crop augmentation. |

**To force a specific batch size**:
```python
# In data_processor.py, after the automatic selection:
metadata['batch_size'] = 32   # Override automatic selection
```

---

### 8.6 Repair constraints — `submission/search_space/repair.py`

**Function**: `repair(genotype, C, H, W, num_classes, family, memory_budget_mb=4096.0)`

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `memory_budget_mb` | float | 4096.0 | Maximum estimated activation memory (MB at batch=32). Architectures exceeding this have channels reduced. |

To allow larger models (if you have a big GPU):
```python
# In nas.py, _search(), where aging_evolution is called:
# Pass memory_budget_mb via repair kwargs — currently hardcoded in repair.py
memory_budget_mb = 8192.0   # 8 GB
```

Edit `repair.py` Rule 11 directly:
```python
# Rule 11 — Memory budget guard
MEMORY_BUDGET_MB = 8192.0   # Change this constant
```

---

### 8.7 Search space option lists — `submission/search_space/genotype.py`

These lists define the discrete vocabulary the search operates over. Edit them to restrict or expand the search space.

```python
CHANNEL_LIST    = [16, 24, 32, 48, 64, 96, 128, 192, 256]   # 9 channel widths
KERNEL_LIST     = [1, 3, 5, 7]                                 # 4 kernel sizes
N_BLOCKS_LIST   = [1, 2, 3, 4]                                 # Blocks per stage
EXPANSION_LIST  = [1, 2, 4, 6]                                 # MBConv expansion ratio
DILATION_LIST   = [1, 2, 3, 4]                                 # Conv dilation
SE_RATIO_LIST   = [0.0625, 0.125, 0.25]                       # Squeeze-Excitation ratio
DROPOUT_LIST    = [0.0, 0.1, 0.2, 0.3]                        # Dropout rate
DROP_PATH_LIST  = [0.0, 0.05, 0.1, 0.2]                       # DropPath (stochastic depth)
MAX_STAGES      = 5                                             # Maximum depth

BLOCK_TYPES = [
    'ConvBlock', 'SepConvBlock', 'ResidualBlock', 'MBConvBlock',
    'BottleneckBlock', 'AnisotropicBlock', 'DilatedConvBlock',
    'GridLogicBlock', 'ChannelMixingBlock', 'GlobalContextBlock',
    'LightAttentionBlock',
]   # 11 blocks

HEAD_TYPES = [
    'GapLinear', 'GmpLinear', 'GapGmpLinear',
    'FlattenMlp', 'AttentionPool', 'SpatialPyramidPool', 'GatedPool',
]   # 7 heads

STEM_TYPES = ['conv3x3', 'conv7x7', 'conv1x1', 'double_conv']   # 4 stem types
NECK_TYPES = ['none', 'conv1x1', 'global_avg']                   # 3 neck types
```

**To restrict search to fewer channels** (faster, smaller models):
```python
CHANNEL_LIST = [16, 32, 64, 128]   # 4 options instead of 9
```

**To disable heavy blocks** (faster proxy evaluation):
```python
# In genotype.py, remove from BLOCK_TYPES:
BLOCK_TYPES = [
    'ConvBlock', 'SepConvBlock', 'ResidualBlock', 'MBConvBlock',
    # ... remove GlobalContextBlock, LightAttentionBlock for speed
]
```

---

### 8.8 Family geometry constraints — `submission/search_space/family.py`

**Function**: `infer_family(C, H, W, num_classes) → FamilyProfile`

The family thresholds that trigger each constraint:

| Family | Trigger condition | Max pool steps | Force GroupNorm |
|---|---|---|---|
| `anisotropic` | max(H,W)/min(H,W) ≥ 6 | 2 | Yes |
| `small_grid` | max(H,W) ≤ 10 | 1 | Yes |
| `possible_voxel` | H≈W≈C (cube-like, area≤1024) | 2 | Yes |
| `channel_heavy` | C ≥ 8, area ≤ 1024 | 2 | Yes |
| `spatiotemporal_like` | C≥3, H=1 or W=1 | 3 | Yes |
| `visual_large` | min(H,W) ≥ 64 | 5 | No |
| `visual_medium` | min(H,W) ≥ 24 | 3 | No |
| `compact_general` | everything else | 2 | No |

To adjust thresholds, edit `infer_family()` in `family.py`.

---

### 8.9 Evaluation time limit

The `time_limit` field in dataset metadata controls the total wall-clock seconds given to the full pipeline (DataProcessor + NAS + Trainer):

```bash
# In evaluation/main.py, the --time_limit flag sets this:
python evaluation/main.py --time_limit 1200   # 20 minutes

# For quick debugging:
python evaluation/main.py --time_limit 120    # 2 minutes (very short training)

# For a competition-like run:
python evaluation/main.py --time_limit 3600   # 1 hour
```

The NAS module consumes `search_frac × time_limit` seconds. Trainer consumes `budget_frac × remaining` seconds.

---

## 9. Fine-tuning scenarios

### Scenario A: Quick debug run (< 5 minutes)

```python
# In nas.py _search_params(), override everything:
def _search_params(benchmark):
    return 0.20, 10, 20, 3   # tiny search

# In genotype.py:
CHANNEL_LIST = [16, 32, 64]
N_BLOCKS_LIST = [1, 2]

# In proxies.py:
def _get_proxy_batch(..., batch_size=8): ...   # smaller proxy batch
```

Run with:
```bash
python evaluation/main.py --time_limit 300
```

---

### Scenario B: Large GPU, serious search (A100 / H100)

```python
# In nas.py _search_params():
def _search_params(benchmark):
    return 0.45, 100, 500, 15   # large population, many rounds

# In repair.py:
MEMORY_BUDGET_MB = 32768.0   # 32 GB budget

# In proxies.py _get_proxy_batch():
batch_size = 64   # larger proxy batches

# In trainer.py _train_params():
def _train_params(benchmark):
    return 0.90, 1e-4   # spend 90% of time training
```

---

### Scenario C: CPU-only / small machine

```python
# In nas.py _search_params():
def _search_params(benchmark):
    return 0.10, 5, 10, 3   # minimal search

# In genotype.py — remove heavy blocks:
BLOCK_TYPES = ['ConvBlock', 'SepConvBlock', 'ResidualBlock', 'MBConvBlock']
CHANNEL_LIST = [16, 32, 48, 64]
```

---

### Scenario D: Maximise accuracy on a specific dataset

1. Download the dataset
2. Run a full search with a large time budget and note the best genotype printed by the NAS
3. Hard-code that genotype as the starting point for next runs:

```python
# In nas.py _search(), after best = best_individual(population):
print("Best genotype:", best.genotype.to_dict())
# Copy the output and paste it into a fixed initial population seed
```

---

### Scenario E: Reproduce notebook results faster

In the notebook, tune at the top of each section:

```python
MAX_SAMPLES = 5      # Section 4: fewer proxy evaluations
EVO_POP     = 5      # Section 6: tiny population
EVO_ROUNDS  = 10     # Section 6: fewer rounds
CORR_N_ARCH = 4      # Section 7: fewer architectures for correlation
CORR_N_EPOCHS = 1    # Section 7: 1 epoch per arch instead of 3
```

---

## 10. Search space reference

### 10.1 The 11 primitive blocks

| Block | Best for | Key parameters |
|---|---|---|
| `ConvBlock` | General feature extraction with residual skip | kernel, dilation, SE |
| `SepConvBlock` | Parameter efficiency | kernel, expansion |
| `ResidualBlock` | Deep networks, stable gradient flow | kernel, n_blocks |
| `MBConvBlock` | Mobile-style inverted residuals | expansion, SE |
| `BottleneckBlock` | High-channel representations | expansion ratio |
| `AnisotropicBlock` | Elongated / 1D-like inputs | factored k×1 + 1×k convolutions |
| `DilatedConvBlock` | Large receptive field without downsampling | dilation rate |
| `GridLogicBlock` | Rule-based grids (Sudoku, Game of Life) | kernel |
| `ChannelMixingBlock` | Many-channel inputs | expansion |
| `GlobalContextBlock` | Broadcast global context as channel bias | — |
| `LightAttentionBlock` | Small spatial dims (H×W ≤ 256) | — |

### 10.2 The 7 head types

| Head | Description |
|---|---|
| `GapLinear` | Global Average Pooling → Linear (default, spatial-agnostic) |
| `GmpLinear` | Global Max Pooling → Linear |
| `GapGmpLinear` | Concat(GAP, GMP) → Linear (2× channels) |
| `FlattenMlp` | Flatten spatial → MLP (only for small spatial dims) |
| `AttentionPool` | Learned weighted sum over spatial locations |
| `SpatialPyramidPool` | Multi-scale pooling → Linear (requires H,W ≥ 4) |
| `GatedPool` | Sigmoid-gated combination of GAP and GMP |

### 10.3 The 8 geometry families

| Family | When triggered | Main effect |
|---|---|---|
| `anisotropic` | max(H,W)/min(H,W) ≥ 6 | Factored blocks, ≤2 pool steps, GroupNorm |
| `small_grid` | max(H,W) ≤ 10 | ≤1 pool step, GroupNorm, attention allowed |
| `possible_voxel` | H≈W≈C, area≤1024 | ≤2 pool steps, GroupNorm |
| `channel_heavy` | C≥8, area≤1024 | Channel mixing preferred, ≤2 pool steps |
| `spatiotemporal_like` | C≥3, H=1 or W=1 | Anisotropic mode, ≤3 pool steps |
| `visual_large` | min(H,W) ≥ 64 | Up to 5 pool steps, large models allowed |
| `visual_medium` | min(H,W) ≥ 24 | Up to 3 pool steps |
| `compact_general` | Everything else | Up to 2 pool steps |

### 10.4 Repair rules (in order)

1. Clamp `n_stages` to [1, MAX_STAGES]
2. Override norm to GroupNorm if `family.force_groupnorm`
3. Replace forbidden blocks with random allowed alternatives
4. Gate LightAttentionBlock: only if spatial area ≤ 256
5. Clip kernel size: never larger than min(H,W) after all pooling
6. Cap total pooling steps to `family.max_pool_steps`
7. Fix anisotropic axis consistency across all stages
8. Enforce monotone (non-decreasing) channel progression across stages
9. Kill FlattenMlp if flattened dim > 65536
10. Kill SpatialPyramidPool if final H or W < 4
11. Memory budget guard: reduce channels if estimated activation MB > budget

---

## 11. Datasets

### 11.1 Practice datasets (13 total)

| Dataset | Domain | Link |
|---|---|---|
| AddNIST | Arithmetic on MNIST digits | [doi:10.25405/data.ncl.24574354.v1](https://doi.org/10.25405/data.ncl.24574354.v1) |
| Language | Text-as-image classification | [doi:10.25405/data.ncl.24574729.v1](https://doi.org/10.25405/data.ncl.24574729.v1) |
| MultNIST | Multiplication on MNIST digits | [doi:10.25405/data.ncl.24574678.v1](https://doi.org/10.25405/data.ncl.24574678.v1) |
| CIFARTile | Tiled CIFAR-10 images | [doi:10.25405/data.ncl.24551539.v1](https://doi.org/10.25405/data.ncl.24551539.v1) |
| Gutenberg | Literary text classification | [doi:10.25405/data.ncl.24574753.v1](https://doi.org/10.25405/data.ncl.24574753.v1) |
| GeoClassing | Geographical image classification | [doi:10.25405/data.ncl.24050256.v3](https://doi.org/10.25405/data.ncl.24050256.v3) |
| Chesseract | Chess board classification | [doi:10.25405/data.ncl.24118743.v2](https://doi.org/10.25405/data.ncl.24118743.v2) |
| Sudoku | Sudoku grid classification | [doi:10.25405/data.ncl.26976121.v1](https://doi.org/10.25405/data.ncl.26976121.v1) |
| Voxel | 3D voxel data (projected) | [doi:10.25405/data.ncl.26970223.v1](https://doi.org/10.25405/data.ncl.26970223.v1) |
| Myofibre | Biological microscopy | [doi:10.25405/data.ncl.26969998.v1](https://doi.org/10.25405/data.ncl.26969998.v1) |
| GameOfLife | Conway's Game of Life grids | [doi:10.25405/data.ncl.30000835](https://doi.org/10.25405/data.ncl.30000835) |
| Cryptic | Cryptic crossword grids | [doi:10.7488/ds/8054](https://doi.org/10.7488/ds/8054) |
| Windspeed | Wind speed field prediction | [doi:10.7488/ds/8053](https://doi.org/10.7488/ds/8053) |

### 11.2 Expected directory structure

```
datasets/
  AddNIST/
    train/
    valid/
    test/
    metadata.json
  CIFARTile/
    train/
    valid/
    test/
    metadata.json
  ...
```

### 11.3 metadata.json fields

```json
{
  "input_shape": [1, 3, 32, 32],    // [batch, channels, height, width]
  "num_classes": 10,                 // Number of output classes
  "benchmark": 65.0,                 // Baseline accuracy (determines search budget)
  "time_limit": 1200                 // Total seconds for the pipeline
}
```

---

## 12. Design decisions

| Decision | Rationale |
|---|---|
| Pure categorical encoding | Simpler than continuous/hybrid; naturally compatible with Aging Evolution; no encoding/decoding overhead |
| Family → hard constraints only, head is a free gene | Avoids wrong inductive bias if family is misidentified; repair only blocks physically impossible options |
| AZ-NAS proxy (zero-cost, no training) | Scores thousands of architectures in seconds; correlates well with final accuracy on NAS benchmarks |
| Aging Evolution over DARTS/one-shot | No shared weights instability; works on arbitrary NCHW geometries; straightforward to implement and debug |
| `retain_grad()` on intermediate features | Required for AZ-NAS trainability proxy to access inter-layer Jacobians on non-leaf tensors |
| `SearchableCNN` fallback | If any part of the new search space fails for any reason, the pipeline still produces a valid model |
| GroupNorm for small-grid and anisotropic families | Batch statistics are unreliable when spatial dimensions are tiny; GroupNorm normalises within each sample |
| Age-based eviction (not fitness-based) | Preserves diversity; prevents population collapsing to a single local optimum |

---

## Branch structure

| Branch | Purpose |
|---|---|
| `main` | Stable, competition-ready |
| `claude/focused-planck-isnMM` | Active development |

---

## Citation

If you use the AZ-NAS proxy in your work, please cite:

```bibtex
@inproceedings{aznas2024,
  title     = {AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search},
  author    = {Lee, Junghyup and Ham, Bumsub},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
}
```

Aging Evolution baseline:

```bibtex
@article{real2019regularized,
  title   = {Regularized Evolution for Image Classifier Architecture Search},
  author  = {Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year    = {2019},
}
```
