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

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    genotype: Genotype
    fitness:  float
    age:      int = 0


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
    verbose:         bool = False,
) -> List[Individual]:
    """
    Runs Aging Evolution and returns the full population sorted by fitness.
    Stops early if time_budget_s is reached.
    """
    population: List[Individual] = []
    start_time = time.time()

    # ── Initialise population ─────────────────────────────────────────────────
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

        population.append(Individual(g, fit))
        if verbose:
            print(f"  [init {len(population)}/{n_population}] fitness={fit:.4f}")

    if not population:
        logger.error("Could not initialise any valid architecture.")
        return population

    best_fitness = max(ind.fitness for ind in population)

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

        child = Individual(child_g, fit)
        population.append(child)

        # age all and remove oldest
        for ind in population:
            ind.age += 1
        population.sort(key=lambda x: x.age)
        population.pop(-1)  # remove oldest

        if fit > best_fitness:
            best_fitness = fit
            if verbose:
                print(f"  [round {rnd+1}/{n_rounds}] new best fitness={fit:.4f}  scale={scale}")

    population.sort(key=lambda x: x.fitness, reverse=True)
    return population


def best_individual(population: List[Individual]) -> Optional[Individual]:
    if not population:
        return None
    return max(population, key=lambda x: x.fitness)
