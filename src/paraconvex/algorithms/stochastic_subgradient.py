"""Stochastic subgradient method.

    x_{k+1} = x_k - alpha_k * g_k,        g_k in d f(x_k; xi_k),

with ``xi_k`` a minibatch of measurements drawn i.i.d. at each iteration.  This
is the subgradient member of the stochastic model-based family: the model is the
linearisation ``f_{x_k}(y; xi) = f(x_k; xi) + <g_k, y - x_k>``, whose proximal
subproblem with a quadratic term has exactly the closed form above.

The iteration loop, recording and divergence handling live in
:mod:`paraconvex.algorithms.trajectory` and are shared with the other members of
the family, so trajectories are directly comparable across methods.
"""

from __future__ import annotations

import torch
from torch import Tensor

from paraconvex.algorithms.stepsizes import StepContext, StepSize
from paraconvex.algorithms.trajectory import TrajectoryResult, run_trajectory
from paraconvex.problems.phase_retrieval import RobustPhaseRetrieval

__all__ = ["SubgradientResult", "stochastic_subgradient"]

#: Retained name; the result type is shared across the model-based family.
SubgradientResult = TrajectoryResult


def stochastic_subgradient(
    problem: RobustPhaseRetrieval,
    x0: Tensor,
    stepsize: StepSize,
    n_iterations: int,
    batch_size: int = 1,
    seed: int = 0,
    record_every: int = 10,
    normalize: bool = False,
    f_star: Tensor | None = None,
) -> TrajectoryResult:
    """Run the method and record the trajectory.

    Parameters
    ----------
    normalize:
        If ``True``, step along ``g_k / ||g_k||`` instead of ``g_k``.  Off by
        default, matching the stochastic model-based formulation.  Worth knowing
        for cross-``p`` comparisons: the subgradient scale carries a factor
        ``p|u|^{p-1}``, so unnormalised steps of the same ``alpha`` mean
        different things at different ``p`` -- which is precisely why the
        step-size is tuned per exponent rather than shared.
    f_star:
        Per-replication optimal-value estimate, required by Polyak steps.
    record_every:
        Record every this many iterations; iteration 0 and the final iterate are
        always recorded.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    needs_value = getattr(stepsize, "name", "") == "polyak"

    def step(x: Tensor, generator: torch.Generator, k: int) -> Tensor:
        idx = problem.sample(generator, batch_size)  # (R, B)
        g = problem.minibatch_subgradient(x, idx)
        grad_norm = g.norm(dim=1)

        alpha = stepsize(
            StepContext(
                k=k,
                f_val=problem.minibatch_value(x, idx) if needs_value else None,
                grad_norm=grad_norm,
                f_star=f_star,
            )
        )
        if isinstance(alpha, Tensor):
            alpha = alpha.unsqueeze(-1)  # (R, 1) broadcasts over coordinates

        direction = g / grad_norm.clamp_min(1e-12).unsqueeze(-1) if normalize else g
        return x - alpha * direction

    return run_trajectory(
        problem,
        x0=x0,
        step=step,
        n_iterations=n_iterations,
        seed=seed,
        record_every=record_every,
    )
