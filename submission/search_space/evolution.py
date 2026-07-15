"""
Aging Evolution (Regularized Evolution) for NAS.

Ref: Real et al. "Regularized Evolution for Image Classifier Architecture Search" (AAAI 2019).

Key idea:
  - Maintain a fixed-size population.
  - On each round: sample a tournament, take the best, mutate it, evaluate,
    add to population, remove the OLDEST (not the worst).
  - This encourages exploration because old, "used up" winners get evicted.
"""
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch

from .genotype import Genotype, sample_random_genotype, mutate
from .family import FamilyProfile
from .repair import repair
from .builder import build_model

try:
    from config import GLOBAL_SEED as _DEFAULT_SEED
    from config import LAMBDA_COMPLEXITY_RANK as _LAMBDA_RANK
except ImportError:
    _DEFAULT_SEED = 42   # standalone import without submission/ on sys.path
    _LAMBDA_RANK  = 0.3

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    genotype:   Genotype
    fitness:    float
    age:        int = 0
    # Raw AZ-NAS components {e, p, t, c} when the proxy returns them; fitness
    # is then a population-relative rank combination (see _rank_fitness).
    components: Optional[dict] = None


def _rank_fitness(group: List['Individual'],
                  lambda_rank: float = _LAMBDA_RANK) -> None:
    """
    Set each Individual.fitness from within-group percentile ranks of its raw
    components:  fitness = rank_E + rank_P + rank_T − lambda_rank·rank_C.

    Raw component scales are incomparable (E grows with depth; T ≤ 0), so raw
    sums let one component dominate — this is the rank normalisation the AZ-NAS
    paper uses across a candidate pool. A non-finite component gets the WORST
    rank (0.0) instead of being dropped: failing a measurement can never help.
    """
    n = len(group)
    if n == 0:
        return
    denom = max(n - 1, 1)
    ranks = [dict() for _ in range(n)]
    for key in ('e', 'p', 't', 'c'):
        vals = [ind.components.get(key, -np.inf) for ind in group]
        keyf = lambda i: vals[i] if np.isfinite(vals[i]) else -np.inf
        order = sorted(range(n), key=keyf)
        # Tie-aware: equal raw values share the AVERAGE of their positions —
        # otherwise ties hand out arbitrary rank differences that can outweigh
        # a real penalty (e.g. a failed-T candidate luckily out-ranking a
        # measured twin purely on tie order).
        pos = 0
        while pos < n:
            end = pos
            while end + 1 < n and keyf(order[end + 1]) == keyf(order[pos]):
                end += 1
            avg = (pos + end) / 2 / denom
            for j in range(pos, end + 1):
                i = order[j]
                ranks[i][key] = avg if np.isfinite(vals[i]) else 0.0
            pos = end + 1
    for ind, rk in zip(group, ranks):
        ind.fitness = rk['e'] + rk['p'] + rk['t'] - lambda_rank * rk['c']


def _default_proxy(model, batch_x, device) -> float:
    """Fallback proxy when no proxy_fn provided: random score."""
    return random.random()


def aging_evolution(
    family:          FamilyProfile,
    C:               int,
    H:               int,
    W:               int,
    num_classes:     int,
    proxy_fn:        Callable,       # (model, batch_x, device) → float
    batch_x,                         # representative input tensor
    device,
    n_population:    int  = 50,
    n_rounds:        int  = 200,
    tournament_size: int  = 10,
    time_budget_s:   Optional[float] = None,
    seed:            int  = _DEFAULT_SEED,
    verbose:         bool = False,
    seed_genotypes:  Optional[List[Genotype]] = None,
) -> List[Individual]:
    """
    Runs Aging Evolution and returns the full population sorted by fitness.
    Stops early if time_budget_s is reached.

    seed_genotypes: curated starting architectures (see seeds.py), consumed
    first during initialisation through the SAME repair/dry-run/proxy path as
    random samples — a seed invalid for this geometry is skipped, never fatal.
    The rest of the population is filled randomly, preserving exploration.
    """
    # Seed all randomness so the search is fully reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    population: List[Individual] = []
    start_time = time.time()
    seed_queue = [g.clone() for g in (seed_genotypes or [])]

    # ── Initialise population (curated seeds first, then random) ─────────────
    n_init_attempts = 0
    while len(population) < n_population:
        if time_budget_s and (time.time() - start_time) > time_budget_s * 0.5:
            print(f"  [NAS] Init time budget reached ({len(population)}/{n_population} seeded).")
            break
        n_init_attempts += 1
        if n_init_attempts > n_population * 30:  # generous: 30× pop size
            print(f"  [NAS] Init attempts={n_init_attempts}, seeded={len(population)}/{n_population}."
                  f" Continuing with partial population.")
            break

        if seed_queue:
            g = seed_queue.pop(0)
        else:
            g = sample_random_genotype(
                preferred_blocks=family.preferred_blocks,
                forbidden_blocks=family.forbidden_blocks,
            )
        try:
            g = repair(g, C, H, W, num_classes, family)
            model = build_model(g, C, H, W, num_classes, aniso_axis=family.aniso_axis)
            # dry-run: catch any remaining shape errors before proxy eval
            with torch.no_grad():
                model.cpu()(torch.zeros(2, C, H, W))
            fit   = proxy_fn(model, batch_x, device)
        except Exception as e:
            logger.debug("Init arch failed: %s", e)
            continue

        if isinstance(fit, dict):
            # Rank mode: raw components stored; scalar fitness assigned later
            # relative to the population (see _rank_fitness).
            population.append(Individual(g, 0.0, components=fit))
            if verbose:
                print(f"  [init {len(population)}/{n_population}]"
                      f" E={fit['e']:.2f} P={fit['p']:.2f} T={fit['t']:.2f}")
        else:
            population.append(Individual(g, fit))
            if verbose:
                print(f"  [init {len(population)}/{n_population}] fitness={fit:.4f}")

    if not population:
        logger.error("Could not initialise any valid architecture.")
        return population

    # Rank mode is on when the proxy returns component dicts. The archive
    # keeps EVERY evaluated individual so the final ranking is a single
    # consistent ordering over the whole search history (argmax over history,
    # as in Real et al.) — it supersedes the legacy global-best tracking.
    rank_mode = population[0].components is not None
    archive: List[Individual] = list(population) if rank_mode else []
    if rank_mode:
        _rank_fitness(population)

    # Pre-age initial population so FIFO eviction is correct from round 1.
    # init[0] (first inserted) gets the highest starting age → evicted first.
    # Without this, all initial members start at age=0, the new child also
    # starts at age=0, aging everyone by +1 gives all age=1, and stable-sort
    # puts the child last → pop(-1) removes the child instead of init[0].
    n_init = len(population)
    for i, ind in enumerate(population):
        ind.age = n_init - 1 - i  # init[0] → age n-1 (oldest), init[n-1] → age 0

    if not rank_mode:
        best_fitness = max(ind.fitness for ind in population)
        # Track global best separately — aging eviction removes the OLDEST, not
        # the worst, so the best individual can get expelled and be lost. This
        # is a running max equivalent to argmax(all history) (Real et al.).
        _gb = max(population, key=lambda x: x.fitness)
        global_best: Individual = Individual(_gb.genotype, _gb.fitness)

    # ── Evolution rounds ──────────────────────────────────────────────────────
    for rnd in range(n_rounds):
        if time_budget_s and (time.time() - start_time) > time_budget_s:
            break

        # tournament selection
        tournament = random.sample(population, min(tournament_size, len(population)))
        parent     = max(tournament, key=lambda x: x.fitness)

        # mutation scale scheduling: large early, small late
        frac = rnd / max(n_rounds - 1, 1)
        if frac < 0.3:
            scale = 'large'
        elif frac < 0.7:
            scale = 'medium'
        else:
            scale = 'small'

        # mutate + repair + evaluate
        tries = 0
        while tries < 5:
            tries += 1
            child_g = mutate(parent.genotype, scale=scale)
            try:
                child_g = repair(child_g, C, H, W, num_classes, family)
                model   = build_model(child_g, C, H, W, num_classes,
                                      aniso_axis=family.aniso_axis)
                # dry-run: catch any remaining shape errors before proxy eval
                with torch.no_grad():
                    model.cpu()(torch.zeros(2, C, H, W))
                fit     = proxy_fn(model, batch_x, device)
                break
            except Exception as e:
                logger.debug("Mutation attempt %d failed: %s", tries, e)
                fit = -np.inf

        if rank_mode:
            if not isinstance(fit, dict):
                continue   # all mutation attempts failed — skip the round
            child = Individual(child_g, 0.0, components=fit)
            population.append(child)
            archive.append(child)
        else:
            child = Individual(child_g, fit)
            population.append(child)

        # age all and remove oldest
        for ind in population:
            ind.age += 1
        population.sort(key=lambda x: x.age)
        population.pop(-1)  # remove oldest

        if rank_mode:
            # Refresh population-relative fitness for the next tournament.
            _rank_fitness(population)
            if verbose and max(population, key=lambda x: x.fitness) is child:
                print(f"  [round {rnd+1}/{n_rounds}] child leads population"
                      f"  scale={scale}")
        elif fit > best_fitness:
            best_fitness = fit
            # Snapshot BEFORE this individual ages out of the population.
            # child_g comes from mutate() so it's a fresh object not shared with population.
            global_best = Individual(child_g, fit)
            if verbose:
                print(f"  [round {rnd+1}/{n_rounds}] new best fitness={fit:.4f}  scale={scale}")

    if rank_mode:
        # One consistent ranking over EVERYTHING evaluated during the search
        # (population + everything aged out). Nothing good can be lost to
        # aging eviction, so no separate global-best bookkeeping is needed.
        _rank_fitness(archive)
        archive.sort(key=lambda x: x.fitness, reverse=True)
        return archive[:n_population]

    population.sort(key=lambda x: x.fitness, reverse=True)

    # Reinsert global_best if aging evicted it (replace worst current member)
    if not population or global_best.fitness > population[0].fitness:
        if population and global_best.fitness > population[-1].fitness:
            population[-1] = global_best
            population.sort(key=lambda x: x.fitness, reverse=True)
        elif not population:
            population.append(global_best)

    return population


def best_individual(population: List[Individual]) -> Optional[Individual]:
    if not population:
        return None
    return max(population, key=lambda x: x.fitness)
