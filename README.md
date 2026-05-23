# UnseenNAS — AutoML Competition Submission

Our submission to the [NAS Unseen-Data Competition](https://www.nascompetition.com) (2025/2026 edition).
The core idea: **zero-cost architecture search** over a rich, geometry-aware search space, evaluated
with the AZ-NAS proxy (CVPR 2024) and optimised with Aging Evolution — no gradient-based training
during search.

---

## Repository layout

```
submission/
  data_processor.py       # Per-channel normalisation + adaptive augmentation
  nas.py                  # NAS entry point: aging evolution + AZ-NAS proxy
  trainer.py              # Time-budget training loop (AdamW + CosineAnnealingLR + AMP)
  helpers.py              # Shared utilities
  search_space/
    __init__.py           # Public API
    genotype.py           # Genotype + StageGene dataclasses, mutation operators
    family.py             # infer_family() — geometry → hard constraints
    block_library.py      # 11 primitive blocks + SEBlock + DropPath
    builder.py            # build_model() decoder, SearchSpaceModel
    repair.py             # 11-rule deterministic constraint repair
    proxies.py            # AZ-NAS: expressivity, progressivity, trainability
    evolution.py          # Aging Evolution loop

evaluation/
  main.py                 # Competition evaluation pipeline (copy)
  score.py                # Scoring script (copy)

notebooks/
  nas_experiment.ipynb    # Colab-compatible experiment notebook (10 sections)

Makefile                  # build / run / score / zip / all
download_datasets.py      # Download all 13 practice datasets
```

---

## How it works

### 1. DataProcessor (`data_processor.py`)

- Per-channel mean/std normalisation computed from training data.
- Adaptive augmentation (random flip/crop for spatial data, no-op for symbolic).
- Adaptive batch size stored in metadata for downstream use.

### 2. NAS (`nas.py` + `search_space/`)

The search runs **without any gradient-based training**.

#### Pipeline

```
infer_family(C, H, W, num_classes)
    → FamilyProfile (hard constraints)

aging_evolution(family, proxy_fn=az_nas_score, ...)
    ├── initialise population of repaired random genotypes
    ├── for each round:
    │     tournament → parent → mutate → repair → build_model → proxy
    │     add child, evict oldest
    └── return sorted population

best_individual(population).genotype
    → build_model(genotype, C, H, W, num_classes)
    → SearchSpaceModel  (nn.Module, ready for Trainer)
```

Fallback: if the search space fails for any reason, `SearchableCNN` (simple random CNN) takes over.

#### Search space

**Genotype** — pure categorical dict, no continuous encoding:

| Gene | Options |
|---|---|
| `n_stages` | 1–5 |
| `stem_type` | conv3x3, conv7x7, conv1x1, double_conv |
| `stem_channels` | 16–256 (9 levels) |
| `neck_type` | none, conv1x1, global_avg |
| `head_type` | GapLinear, GmpLinear, GapGmpLinear, FlattenMlp, AttentionPool, SpatialPyramidPool, GatedPool |
| `norm_type` | batch, group |
| Per-stage × 5 | block_type (11), kernel (4), n_blocks (4), channels (9), expansion (4), dilation (4), SE, skip_mode (3), downsample (4), drop_path (4) |

**11 primitive blocks:**

| Block | Best for |
|---|---|
| `ConvBlock` | General feature extraction with residual skip |
| `SepConvBlock` | Parameter efficiency |
| `ResidualBlock` | Deep networks with stable gradient flow |
| `MBConvBlock` | Mobile-style inverted residuals |
| `BottleneckBlock` | High-channel representations |
| `AnisotropicBlock` | Elongated/1D-like inputs (factored k×1 + 1×k) |
| `DilatedConvBlock` | Large receptive field without downsampling |
| `GridLogicBlock` | Rule-based grids (Sudoku, GameOfLife) |
| `ChannelMixingBlock` | Channel-heavy inputs with many input channels |
| `GlobalContextBlock` | Global context broadcast as channel bias |
| `LightAttentionBlock` | Small spatial dims (H×W ≤ 256) |

**8 geometry families** inferred from `(C, H, W, num_classes)`:

| Family | Trigger | Key constraints |
|---|---|---|
| `anisotropic` | one spatial dim ≥ 6× the other | factored blocks, max 2 pool steps, GroupNorm |
| `small_grid` | max(H,W) ≤ 10 | max 1 pool step, GroupNorm, attention allowed |
| `possible_voxel` | H≈W≈C cube-like | max 2 pool steps, GroupNorm |
| `channel_heavy` | C ≥ 8, area ≤ 1024 | channel mixing preferred |
| `spatiotemporal_like` | C ≥ 3, H=1 or W=1, long axis | anisotropic, max 3 pool steps |
| `visual_large` | min(H,W) ≥ 64 | up to 5 pool steps |
| `visual_medium` | min(H,W) ≥ 24 | up to 3 pool steps |
| `compact_general` | everything else | up to 2 pool steps |

Family drives **hard constraints only** (max pooling, attention enable/disable, GroupNorm forcing).
The head type is a fully free searchable gene.

**AZ-NAS proxy** (adapted from [cvlab-yonsei/AZ-NAS](https://github.com/cvlab-yonsei/AZ-NAS)):

```
score = Expressivity + Progressivity + Trainability − λ · Complexity

Expressivity  = Σ H(eigenvalues of per-layer spatial covariance)
Progressivity = min(inter-layer expressivity deltas)
Trainability  = mean(−σ_max − 1/σ_max + 2)  over inter-layer Jacobian SVDs
Complexity    = log(total_params)
```

All computed on a single mini-batch with Kaiming-normal reinit. No labels required.

**Aging Evolution** ([Real et al. 2019](https://arxiv.org/abs/1802.01548)):
- Tournament selection → mutate (large→medium→small scheduling) → repair → proxy eval.
- Age-based eviction (oldest removed, not worst) encourages exploration.
- Mutation scale anneals from `large` (full stage replacement) to `small` (1-field tweak).

**Repair pipeline** (11 rules in order):
1. `n_stages` bounds
2. GroupNorm override when family requires it
3. Forbidden block replacement
4. LightAttention gated by spatial area
5. Kernel size bounded by spatial dimension
6. Max pooling steps cap
7. Anisotropic axis consistency
8. Monotone channel progression (non-decreasing by stage)
9. FlattenMlp head killed if flat dim > 65536
10. SpatialPyramidPool requires final H, W ≥ 4
11. Memory budget guard (activation MB at batch=32)

### 3. Trainer (`trainer.py`)

- AdamW optimiser + CosineAnnealingLR scheduler.
- Mixed-precision (AMP) on CUDA.
- Best-weights checkpoint: restores best val-acc model at end.
- Time-budget aware: stops training when `clock.check()` falls below a grace threshold.

---

## Datasets

Practice datasets from previous competition rounds:

| Dataset | Link |
|---|---|
| AddNIST | [doi:10.25405/data.ncl.24574354.v1](https://doi.org/10.25405/data.ncl.24574354.v1) |
| Language | [doi:10.25405/data.ncl.24574729.v1](https://doi.org/10.25405/data.ncl.24574729.v1) |
| MultNIST | [doi:10.25405/data.ncl.24574678.v1](https://doi.org/10.25405/data.ncl.24574678.v1) |
| CIFARTile | [doi:10.25405/data.ncl.24551539.v1](https://doi.org/10.25405/data.ncl.24551539.v1) |
| Gutenberg | [doi:10.25405/data.ncl.24574753.v1](https://doi.org/10.25405/data.ncl.24574753.v1) |
| GeoClassing | [doi:10.25405/data.ncl.24050256.v3](https://doi.org/10.25405/data.ncl.24050256.v3) |
| Chesseract | [doi:10.25405/data.ncl.24118743.v2](https://doi.org/10.25405/data.ncl.24118743.v2) |
| Sudoku | [doi:10.25405/data.ncl.26976121.v1](https://doi.org/10.25405/data.ncl.26976121.v1) |
| Voxel | [doi:10.25405/data.ncl.26970223.v1](https://doi.org/10.25405/data.ncl.26970223.v1) |
| Myofibre | [doi:10.25405/data.ncl.26969998.v1](https://doi.org/10.25405/data.ncl.26969998.v1) |
| GameOfLife | [doi:10.25405/data.ncl.30000835](https://doi.org/10.25405/data.ncl.30000835) |
| Cryptic | [doi:10.7488/ds/8054](https://doi.org/10.7488/ds/8054) |
| Windspeed | [doi:10.7488/ds/8053](https://doi.org/10.7488/ds/8053) |

Download all at once:

```bash
python download_datasets.py
```

Place downloaded folders inside `datasets/` at the repo root.

---

## Quick start

### Install dependencies

```bash
pip install torch torchvision numpy scipy matplotlib
```

### Run the full evaluation pipeline locally

```bash
make submission=submission all
```

This runs `DataProcessor → NAS → Trainer → predict → score` for every dataset in `datasets/`.

### Individual steps

```bash
make submission=submission build   # validate syntax / imports
make submission=submission run     # run main.py, produce predictions
make submission=submission score   # score predictions against labels
make submission=submission zip     # bundle submission.zip for submission
```

### Quick sanity check (no datasets needed)

```python
import sys
sys.path.insert(0, 'submission')
from search_space import *
import torch

C, H, W, NC = 3, 32, 32, 10
family = infer_family(C, H, W, NC)           # → visual_medium
g = sample_random_genotype()
g = repair(g, C, H, W, NC, family)
model = build_model(g, C, H, W, NC)

x = torch.randn(4, C, H, W)
print(model(x).shape)                        # → torch.Size([4, 10])
print(az_nas_score(model, x, 'cpu'))         # → float
```

### Experiment notebook

Open `notebooks/nas_experiment.ipynb` in Colab or locally. Covers:
1. Setup
2. Synthetic datasets per family
3. Family inference
4. Architecture sampling + repair
5. Model build and inspection
6. AZ-NAS proxy evaluation on 15 architectures
7. Aging Evolution loop with fitness plot
8. Proxy–accuracy correlation (3-epoch training)
9. End-to-end integration test
10. Multi-family sweep

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Pure categorical encoding (no hybrid vector) | Simpler, fully compatible with Aging Evolution; hybrid adds complexity without benefit at this stage |
| Family → hard constraints only, head is a free gene | Avoids wrong inductive bias if family is misidentified; repair only blocks physically impossible options |
| AZ-NAS proxy (zero-cost, no training) | Scores thousands of architectures in seconds; correlates well with final accuracy on NAS benchmarks |
| Aging Evolution vs. DARTS/one-shot | No shared weights, works on arbitrary NCHW geometries, no optimisation instability |
| `extract_layer_features()` with `retain_grad()` | Enables inter-layer Jacobian computation required by AZ-NAS trainability proxy |
| `SearchableCNN` fallback | If any part of the new search space fails, the pipeline still produces a valid model |

---

## Branch structure

| Branch | Purpose |
|---|---|
| `main` | Stable, competition-ready |
| `claude/focused-planck-isnMM` | Active development |
