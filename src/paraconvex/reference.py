"""High-accuracy reference minimum for the empirical objective.

Why this exists: the ground truth ``x_true`` is *not* the minimizer of the noisy
empirical objective.  A deterministic full-batch run reliably finds points with
``f(x) < f(x_true)``, and the gap grows linearly with the noise level.  So

* ``dist(x_k, +-x_true)`` measures optimization error **plus** estimation error,
  and flattens at a statistical floor that carries no information about the
  convergence rate;
* ``f(x_k) - f_min`` measures optimization error alone.

The second is what an algorithmic comparison needs, and it requires an estimate
of ``f_min`` per replication -- supplied here by a long full-batch subgradient
run with a geometrically decaying step.
"""

from __future__ import annotations

import torch
from torch import Tensor

from paraconvex.problems.phase_retrieval import RobustPhaseRetrieval

__all__ = ["converged_minimizer", "deterministic_subgradient", "estimate_minimum"]


def deterministic_subgradient(
    problem: RobustPhaseRetrieval,
    x0: Tensor,
    alpha0: float,
    n_iterations: int,
    total_decay: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Full-batch subgradient descent.

    Returns the best iterate found and its objective value, both tracked
    per replication.  "Best" is by objective, not by the final iterate: the
    method is nonsmooth, so the last point need not be the lowest one seen.
    """
    lam = total_decay ** (1.0 / n_iterations)
    x = x0.clone()
    best_x = x.clone()
    best_f = problem.value(x)

    for k in range(n_iterations):
        x = x - (alpha0 * lam**k) * problem.subgradient(x)
        f = problem.value(x)
        improved = f < best_f
        best_f = torch.where(improved, f, best_f)
        best_x = torch.where(improved.unsqueeze(-1), x, best_x)

    return best_x, best_f


def estimate_minimum(
    problem: RobustPhaseRetrieval,
    seed: int,
    n_iterations: int = 6000,
    alpha0: float = 0.05,
    n_starts: int = 2,
    relative_starts: tuple[float, ...] = (0.02, 0.15),
    observed: Tensor | None = None,
) -> Tensor:
    """Estimate ``min f`` per replication; shape ``(R,)``.

    Runs several full-batch descents from different distances to ``x_true`` and
    keeps the lowest value, then takes the minimum against ``observed`` -- any
    objective values already seen elsewhere.  Folding those in matters because
    the estimate is used as a subtrahend: if a stochastic run ever dips below
    it, ``f(x_k) - f_min`` would go negative and break the log-scale plot.
    """
    candidates = [problem.value(problem.data.x_true)]

    for i, rel in enumerate(relative_starts[:n_starts]):
        gen = torch.Generator(device=problem.data.A.device).manual_seed(seed + i)
        x0 = problem.random_start(gen, rel)
        _, best_f = deterministic_subgradient(problem, x0, alpha0, n_iterations)
        candidates.append(best_f)

    if observed is not None:
        candidates.append(observed)

    return torch.stack(candidates).min(dim=0).values


def converged_minimizer(
    problem: RobustPhaseRetrieval,
    x0: Tensor,
    coarse_alpha0: float = 0.02,
    coarse_iterations: int = 15000,
    polish_alpha0: float = 2e-4,
    polish_iterations: int = 5000,
) -> tuple[Tensor, Tensor, float]:
    """Drive full-batch descent to a converged minimizer of the *empirical* loss.

    Returns ``(x_hat, f_hat, polish_gain)``, the last being the largest objective
    improvement contributed by the polish stage -- a convergence diagnostic the
    caller should check is negligible before trusting ``x_hat``.

    Why two stages: a single geometrically decaying run either starts too small
    to travel or ends too large to settle on a nonsmooth objective.  A coarse
    pass covers the distance, then a short pass with a much smaller step refines
    the corner the coarse pass is oscillating around.

    Why this exists separately from the stochastic solver: robustness is a
    property of ``argmin f_p``, not of the iterates.  Comparing trajectories at a
    fixed ``k`` confounds the estimator with convergence speed, and can actively
    mislead -- early stopping may leave an arm *closer* to ``x_true`` than its own
    minimizer, flattering whichever arm converges slowest.
    """
    x_coarse, f_coarse = deterministic_subgradient(
        problem, x0, coarse_alpha0, coarse_iterations, total_decay=1e-5
    )
    x_hat, f_hat = deterministic_subgradient(
        problem, x_coarse, polish_alpha0, polish_iterations, total_decay=1e-3
    )
    return x_hat, f_hat, float((f_coarse - f_hat).abs().max())
