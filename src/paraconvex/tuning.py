"""Per-exponent step-size tuning.

The optimal ``alpha_0`` is not comparable across exponents.  The subgradient
carries a factor ``p|<a,x>|^{p-1}``, so its typical magnitude changes with ``p``,
and the sharpness constant of the error bound does too.  Running every ``p`` at
one shared step-size would produce a plot showing which exponent happened to
suit that step-size -- not which exponent the method handles better.  Every arm
is therefore tuned independently and reported at its own best.

Tuning uses its own seed and its own (smaller) replication ensemble, kept
disjoint from the evaluation run, so the reported curves are not selected on the
same randomness they are scored on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from paraconvex.algorithms.stepsizes import Diminishing, Geometric, StepSize
from paraconvex.algorithms.stochastic_subgradient import stochastic_subgradient
from paraconvex.problems.phase_retrieval import RobustPhaseRetrieval

__all__ = [
    "TuningOutcome",
    "diminishing_grid",
    "exponent_diminishing_grid",
    "geometric_grid",
    "tune_stepsize",
]


@dataclass
class TuningOutcome:
    """Winning schedule plus the full scored grid, for auditing."""

    best: StepSize
    best_score: float
    table: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        return f"{self.best.label()} -> median final distance {self.best_score:.3e}"


def geometric_grid(
    n_iterations: int,
    alpha0_values: np.ndarray | None = None,
    total_decay_values: tuple[float, ...] = (1e-1, 1e-2, 1e-3),
) -> list[StepSize]:
    """Geometric schedules over a log-spaced ``alpha_0`` and total decay."""
    if alpha0_values is None:
        alpha0_values = np.logspace(-3, 1, 9)
    return [
        Geometric.from_total_decay(float(a0), float(td), n_iterations)
        for a0 in alpha0_values
        for td in total_decay_values
    ]


def diminishing_grid(
    alpha0_values: np.ndarray | None = None,
    powers: tuple[float, ...] = (0.5,),
) -> list[StepSize]:
    if alpha0_values is None:
        alpha0_values = np.logspace(-3, 1, 9)
    return [
        Diminishing(alpha0=float(a0), power=float(q))
        for a0 in alpha0_values
        for q in powers
    ]


def exponent_diminishing_grid(
    p: float,
    alpha0_values: np.ndarray | None = None,
) -> list[StepSize]:
    """``alpha_k = alpha_0 (k+1)^{-1/p}`` over a log-spaced ``alpha_0``.

    The decay exponent is pinned by the loss exponent, so only ``alpha_0`` is
    searched -- a one-dimensional grid, unlike the geometric family where the
    total decay is a second free parameter.
    """
    if alpha0_values is None:
        alpha0_values = np.logspace(-5, -1, 9)
    return [Diminishing.from_loss_exponent(float(a0), p) for a0 in alpha0_values]


def tune_stepsize(
    problem: RobustPhaseRetrieval,
    candidates: list[StepSize],
    n_iterations: int,
    batch_size: int,
    relative_start: float,
    seed: int,
    x0: Tensor | None = None,
    runner: Callable[..., object] = stochastic_subgradient,
) -> TuningOutcome:
    """Score every candidate and return the best.

    The score is the **median** final distance across the tuning replications.
    Median rather than mean throughout: under heavy-tailed noise a single
    replication that lands in a bad basin has unbounded influence on the mean,
    which would let it pick a step-size on the strength of one outlier.

    A candidate is disqualified (score ``inf``) once at least half its
    replications diverge, so a schedule that is spectacular when it survives but
    usually blows up cannot win.

    ``runner`` selects the method being tuned; it must accept
    ``(problem, x0, stepsize, n_iterations, batch_size, seed, record_every)``.
    Bind any method-specific options (``prox_nu``, ``normalize``) with
    :func:`functools.partial` before passing it, so the schedule is tuned for
    exactly the configuration that will later be run.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # A caller comparing several exponents must pass one shared `x0`; deriving
    # it here per call would leave the arms agreeing only by coincidence.
    if x0 is None:
        gen = torch.Generator(device=problem.data.A.device).manual_seed(seed)
        x0 = problem.random_start(gen, relative_start)

    rows = []
    best_score = float("inf")
    best: StepSize | None = None

    for i, schedule in enumerate(candidates):
        result = runner(
            problem,
            x0=x0,
            stepsize=schedule,
            n_iterations=n_iterations,
            batch_size=batch_size,
            seed=seed + 1000 + i,
            record_every=max(n_iterations // 4, 1),
        )

        diverged_fraction = result.diverged.double().mean().item()
        final = result.final_distance()
        score = (
            float("inf")
            if diverged_fraction >= 0.5
            else final.median().item()
        )

        rows.append(
            {
                "schedule": schedule.label(),
                "kind": getattr(schedule, "name", "?"),
                "alpha0": getattr(schedule, "alpha0", np.nan),
                "lam": getattr(schedule, "lam", np.nan),
                "power": getattr(schedule, "power", np.nan),
                "median_final_distance": final.median().item(),
                "diverged_fraction": diverged_fraction,
                "score": score,
            }
        )

        if score < best_score:
            best_score, best = score, schedule

    if best is None:
        raise RuntimeError(
            "every candidate diverged on at least half its replications; "
            "widen the grid towards smaller alpha_0"
        )

    table = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return TuningOutcome(best=best, best_score=best_score, table=table)
