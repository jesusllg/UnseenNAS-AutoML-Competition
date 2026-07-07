"""
Central configuration — the single source of truth for every tunable
pipeline hyperparameter.

Why this file exists
--------------------
The competition evaluator (evaluation/main.py) hands each stage the dataset's
own `metadata` dict, loaded verbatim from the dataset's `metadata` JSON. That
dict belongs to the organisers; it carries facts about the data (input_shape,
num_classes, codename, time_limit) and we must NOT write our own knobs into it
or rely on keys that only exist in our local test harness. Doing so makes the
local run diverge from the real evaluation.

So all of OUR tuning lives here instead. Every module imports its constants
from this file, nothing is injected through metadata, and the behaviour is
identical locally and under the official evaluator.

Dataset-specific behaviour is NOT configured here. It is derived geometrically
at runtime from (C, H, W, n_classes) via search_space.infer_family — we never
hardcode per-dataset choices, because the real datasets are unseen.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────

GLOBAL_SEED = 42


# ── Time budget (per-dataset, 2026 competition model) ─────────────────────────
# Per the organisers' clarification: time applies PER DATASET (final run is
# ~24 h across 3 datasets, roughly 8 h each); nothing carries over, and
# exceeding a dataset's clock fails that dataset. The authoritative signal is
# metadata['time_remaining'] / the live clock — helpers.get_safe_time_remaining
# reconciles them conservatively (min of the available sources).

SEARCH_FRAC = 0.30   # NAS search: fraction of the clock remaining at NAS start

# Absolute ceiling on search time. SEARCH_FRAC alone scales with the clock —
# on an 8 h dataset that is ~2.4 h of proxy search, which is past the point of
# diminishing returns for NAS_ROUNDS evaluations and steals real training
# epochs. Search stops at whichever bound hits first; anything unused flows
# back to training automatically (the trainer budgets from the LIVE clock at
# its own start, budgets are ceilings, never pre-allocated slices).
SEARCH_MAX_S = 3600.0   # never spend more than 1 h searching, on any clock

# Training runs until the clock minus a reserve kept back for prediction and
# artifact saving. The reserve is a fraction of the time remaining when
# training starts, with an absolute floor so normal clocks always leave room
# to predict — capped both as a share (short clocks must still train) and in
# absolute seconds (an 8 h clock does not need a 30-minute reserve).
# Failing to predict scores -10; a slightly shorter training run costs far less.
PREDICT_RESERVE_FRAC     = 0.07    # of clock remaining at training start
PREDICT_RESERVE_MIN_S    = 90.0    # absolute floor (seconds)
PREDICT_RESERVE_MAX_FRAC = 0.25    # ceiling: reserve never exceeds this share
PREDICT_RESERVE_MAX_S    = 600.0   # absolute ceiling (seconds)

# Adaptive modes (nas.py scales its effort with these):
#   smoke  — remaining time below SMOKE_TIME_S → minimal search, short training
#   low-resource — small/absent GPU → smaller population and proxy batch
SMOKE_TIME_S     = 1200.0  # < 20 min left → phase-2-style minimal mode
LOW_VRAM_MB      = 6000.0  # below this (or CPU-only) → low-resource scaling

# Budget-aware final selection. The AZ-NAS proxy knows nothing about the
# clock: V1→V2 evidence showed 10× params bought ~+1 AZ point but made epochs
# ~10× slower — models stopped converging inside the budget (GeoClassing
# regressed and never finished). After evolution we walk the population
# best-first and pick the first candidate whose measured step time affords at
# least MIN_AFFORDABLE_EPOCHS of training in the remaining budget.
MIN_AFFORDABLE_EPOCHS = 25   # a model that can't run this many epochs won't converge
TRAINABILITY_TOP_K    = 8    # how many top-fitness candidates to time-test


# ── NAS search (aging evolution) ──────────────────────────────────────────────

NAS_POPULATION = 100
NAS_ROUNDS     = 2000   # effectively "run until the time budget expires"

# Selection intensity = tournament / population. Real et al. used 25/100 with a
# TRUE-accuracy fitness; our zero-cost proxy is far noisier, so high pressure
# over a long search can exploit proxy noise and over-converge to
# proxy-gaming, over-shrunk architectures. Lower it if that reappears.
NAS_TOURNAMENT = 25

# Number of samples in the single batch fed to the zero-cost proxy. Small by
# design: the proxies score a forward/backward on one batch, so this trades
# proxy-estimate variance against per-candidate cost.
NAS_PROXY_BATCH = 16


# ── Zero-cost proxy (AZ-NAS) ──────────────────────────────────────────────────
# Weight of the complexity penalty -lambda_c*log(params) in the combined score.
# Deliberately small: a tie-breaker toward lighter/faster models (we are time-
# and hardware-limited), NOT a dominant pressure that shrinks architectures past
# the point where they can fit the task. Empirically 0.1 over-shrank on hard
# datasets; 0.05 keeps the light-model bias while leaving capacity on the table.
# The hard memory ceiling (repair R11, ~80% of GPU) is what actually bounds size.
LAMBDA_COMPLEXITY = 0.05


# ── Final training (optimizer + schedule) ─────────────────────────────────────

LEARNING_RATE = 1e-3    # AdamW initial learning rate
WEIGHT_DECAY  = 1e-4    # AdamW weight decay
GRAD_CLIP_NORM = 1.0    # global-norm gradient clipping

# Cosine annealing schedule. T_MAX is the cosine half-period in epochs; it is
# intentionally larger than most runs reach so the LR decays smoothly rather
# than completing a full cosine cycle (early stopping / time budget end first).
LR_T_MAX   = 200
LR_ETA_MIN = 1e-5

# Label smoothing is applied only when the task has enough classes for it to
# help (it hurts very-few-class problems by capping confidence too aggressively).
LABEL_SMOOTHING              = 0.1
LABEL_SMOOTHING_MIN_CLASSES  = 10


# ── Early stopping (regression-based with dynamic improvement delta) ──────────

ES_ENABLED          = True
ES_PATIENCE         = 20     # consecutive regression epochs before stopping
ES_PLATEAU_PATIENCE = 20     # consecutive plateau epochs before stopping
ES_MIN_EPOCHS       = 10     # warmup: ES cannot trigger before this epoch
ES_DELTA_START      = 0.002  # initial improvement threshold (0.2 pp)
ES_DELTA_MIN        = 0.001  # floor for improvement delta (0.1 pp)
ES_DELTA_DECAY      = 3      # improvements before halving delta
ES_REGRESSION_DELTA = 0.010  # fall >1 pp below best → counts as bad epoch
