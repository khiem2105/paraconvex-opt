"""Shared iteration loop for the stochastic model-based family.

Every member -- subgradient, prox-linear, proximal point -- runs the same loop
and differs only in how one iterate is produced from the previous one.  Sharing
the loop keeps the recording schedule, the divergence handling and the result
shape identical across methods, so their trajectories are directly comparable
rather than comparable-if-the-two-copies-stayed-in-sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from paraconvex.problems.phase_retrieval import RobustPhaseRetrieval

__all__ = ["StepFn", "TrajectoryResult", "run_trajectory"]


@dataclass
class TrajectoryResult:
    """Recorded trajectory of a stochastic method.

    Attributes
    ----------
    iterations:
        Iteration index of each record, shape ``(T,)``.
    distance:
        Relative distance to ``+-x_true``, shape ``(T, R)``.
    objective:
        Full-sample objective value, shape ``(T, R)``.
    x_final:
        Final iterate, shape ``(R, n)``.
    diverged:
        Per-replication flag for a non-finite iterate, shape ``(R,)``.  Step-size
        grids necessarily probe values that blow up; the tuner reads this rather
        than silently propagating NaNs into the medians.
    """

    iterations: Tensor
    distance: Tensor
    objective: Tensor
    x_final: Tensor
    diverged: Tensor

    def final_distance(self) -> Tensor:
        """Distance at the last record, shape ``(R,)``."""
        return self.distance[-1]


class StepFn(Protocol):
    """Produce ``x_{k+1}`` from ``x_k``.

    Each method draws its own sample from ``generator``, because how much it
    needs differs: the subgradient method takes a minibatch of any size, while
    prox-linear takes exactly one measurement per replication.  Sampling here
    rather than in the loop lets each ask for the shape it actually wants
    instead of receiving a batch axis it has to strip.
    """

    def __call__(self, x: Tensor, generator: torch.Generator, k: int) -> Tensor: ...


def run_trajectory(
    problem: RobustPhaseRetrieval,
    x0: Tensor,
    step: StepFn,
    n_iterations: int,
    seed: int = 0,
    record_every: int = 10,
) -> TrajectoryResult:
    """Drive ``step`` for ``n_iterations``, recording distance and objective."""
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    generator = torch.Generator(device=x0.device).manual_seed(seed)
    record_at = _record_schedule(n_iterations, record_every)

    x = x0.clone()
    diverged = torch.zeros(x0.shape[0], dtype=torch.bool, device=x0.device)
    distances: list[Tensor] = []
    objectives: list[Tensor] = []
    iterations: list[int] = []

    for k in range(n_iterations + 1):
        if k in record_at:
            iterations.append(k)
            distances.append(problem.distance_to_truth(x))
            objectives.append(problem.value(x))

        if k == n_iterations:
            break

        x_new = step(x, generator, k)

        # Freeze a replication at its last finite iterate once it blows up, so a
        # single divergent run cannot poison the shared tensor with NaNs that
        # then propagate through every median and quantile. The flag is sticky:
        # `diverged` is what the tuner reads, since the frozen distance would
        # otherwise look like an ordinary (merely bad) result.
        diverged = diverged | ~torch.isfinite(x_new).all(dim=1)
        x = torch.where(diverged.unsqueeze(-1), x, x_new)

    return TrajectoryResult(
        iterations=torch.tensor(iterations),
        distance=torch.stack(distances),
        objective=torch.stack(objectives),
        x_final=x,
        diverged=diverged,
    )


def _record_schedule(n_iterations: int, record_every: int) -> set[int]:
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")
    schedule = set(range(0, n_iterations + 1, record_every))
    schedule.add(0)
    schedule.add(n_iterations)
    return schedule
