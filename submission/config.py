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


# ── Competition facts (given by the organisers, not tuning knobs) ─────────────

N_COMPETITION_DATASETS     = 3     # known: 3 datasets
TOTAL_COMPETITION_HOURS    = 24.0  # known: 24h total budget
COMPETITION_OVERHEAD_HOURS = 0.5   # safety margin for I/O, imports, scoring


# ── Time-budget split ─────────────────────────────────────────────────────────
# Fractions of the TOTAL per-dataset clock budget (not of remaining time).
# search + train should sum to ≤ 0.95, leaving ~5% for predict/overhead.

SEARCH_FRAC = 0.30   # NAS search phase
TRAIN_FRAC  = 0.65   # final training phase


# ── NAS search (aging evolution) ──────────────────────────────────────────────

NAS_POPULATION = 100
NAS_ROUNDS     = 2000   # effectively "run until the time budget expires"

# Selection intensity = tournament / population. Real et al. used 25/100 with a
# TRUE-accuracy fitness; our zero-cost proxy is far noisier, so high pressure
# over a long search can exploit proxy noise and over-converge to
# proxy-gaming, over-shrunk architectures. Lower it if that reappears.
NAS_TOURNAMENT = 25


# ── Zero-cost proxy (AZ-NAS) ──────────────────────────────────────────────────
# Weight of the complexity penalty -lambda_c*log(params) in the combined score.
# Deliberately small: a tie-breaker toward lighter/faster models (we are time-
# and hardware-limited), NOT a dominant pressure that shrinks architectures past
# the point where they can fit the task. Empirically 0.1 over-shrank on hard
# datasets; 0.05 keeps the light-model bias while leaving capacity on the table.
# The hard memory ceiling (repair R11, ~80% of GPU) is what actually bounds size.
LAMBDA_COMPLEXITY = 0.05


# ── Final training ────────────────────────────────────────────────────────────

WEIGHT_DECAY = 1e-4


# ── Early stopping (regression-based with dynamic improvement delta) ──────────

ES_ENABLED          = True
ES_PATIENCE         = 20     # consecutive regression epochs before stopping
ES_PLATEAU_PATIENCE = 20     # consecutive plateau epochs before stopping
ES_MIN_EPOCHS       = 10     # warmup: ES cannot trigger before this epoch
ES_DELTA_START      = 0.002  # initial improvement threshold (0.2 pp)
ES_DELTA_MIN        = 0.001  # floor for improvement delta (0.1 pp)
ES_DELTA_DECAY      = 3      # improvements before halving delta
ES_REGRESSION_DELTA = 0.010  # fall >1 pp below best → counts as bad epoch
