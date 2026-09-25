"""Stochastic prox-linear method.

At each iteration one measurement is sampled, the sampled loss
``f(y; xi) = |c(y)|`` is replaced by its *convex-composite model* -- the outer
``|.|`` kept exactly, the inner map linearised -- and the model is minimised
with a quadratic proximal term:

    x_{k+1} = argmin_y  |c + <g, y - x_k>| + (1 / (2 alpha_k)) ||y - x_k||^2

where ``c = c(x_k; xi_k)`` and ``g = Dc(x_k; xi_k)``.

Closed form
-----------
Split ``d = y - x_k`` along ``g``: writing ``d = s g/||g|| + d_perp``, the first
term depends on ``d`` only through ``s`` while the second grows with
``||d_perp||``, so ``d_perp = 0`` and the subproblem collapses to one scalar.
Dualising the outer absolute value with ``|t| = max_{lam in [-1,1]} lam t`` and
swapping min and max gives an inner minimisation ``d = -alpha lam g`` and the
concave scalar problem ``max_{lam in [-1,1]} lam c - (alpha/2) lam^2 ||g||^2``,
whence

    lam* = proj_{[-1,1]}( c / (alpha ||g||^2) )
    d*   = -alpha lam* g

Since ``alpha proj_{[-1,1]}(z) = proj_{[-alpha,alpha]}(alpha z)`` for
``alpha > 0``, that is the same as the primal form actually implemented:

    x_{k+1} = x_k - clip( c / ||g||^2,  -alpha,  alpha ) * g

Reading it two ways is useful.  ``c/||g||^2 * g`` is the Gauss-Newton step for
the single equation ``c(y) = 0``, so prox-linear takes that step **truncated at
alpha**; equivalently the dual multiplier is clipped to ``[-1,1]``, and it
saturates exactly when the kink is active.

Two consequences:

* ``alpha`` small -- the clip is inactive and ``d = -alpha sign(c) g``, which is
  precisely ``-alpha`` times the subgradient.  Prox-linear *is* the subgradient
  method in that regime, exactly rather than approximately.
* ``alpha`` large -- the clip binds and the iterate lands on
  ``c + <g,d> = 0``, the zero set of the linearised residual.  It can never
  overshoot, whatever ``alpha`` is.  That automatic trust region is why the
  method is stable where a plain subgradient step is not.

The regulariser is quadratic and the batch size is one throughout: with a scalar
``c`` the subproblem is the closed form above.  (For ``B > 1`` it would become
``min_d (1/B)||C d + c||_1 + ||d||^2/(2 alpha)``, whose dual is a box-constrained
QP in ``B`` variables -- a genuine inner solve, deliberately out of scope.)
"""

from __future__ import annotations

import torch
from torch import Tensor

from paraconvex.algorithms.stepsizes import StepContext, StepSize
from paraconvex.algorithms.trajectory import TrajectoryResult, run_trajectory
from paraconvex.problems.phase_retrieval import RobustPhaseRetrieval

__all__ = ["prox_linear_step", "stochastic_prox_linear"]


def prox_linear_step(
    c: Tensor, g: Tensor, alpha: Tensor | float, eps: float = 1e-12
) -> Tensor:
    """Solve the prox-linear subproblem; returns the displacement ``d``.

    Parameters
    ----------
    c:
        Residual at the current point, shape ``(R,)``.
    g:
        Jacobian row, shape ``(R, n)``.
    alpha:
        Proximal parameter, scalar or shape ``(R,)``.

    Implements ``d = -clip(c/||g||^2, -alpha, alpha) g``; see the module
    docstring for the derivation and for the equivalent dual form.
    """
    bound = torch.as_tensor(alpha, dtype=g.dtype, device=g.device)
    gauss_newton = c / g.pow(2).sum(dim=1).clamp_min(eps)
    clipped = torch.minimum(torch.maximum(gauss_newton, -bound), bound)
    return -clipped.unsqueeze(-1) * g


def stochastic_prox_linear(
    problem: RobustPhaseRetrieval,
    x0: Tensor,
    stepsize: StepSize,
    n_iterations: int,
    batch_size: int = 1,
    seed: int = 0,
    record_every: int = 10,
    f_star: Tensor | None = None,
) -> TrajectoryResult:
    """Run the stochastic prox-linear method and record the trajectory.

    ``batch_size`` exists only so the signature matches the other solvers (the
    tuner drives them through one protocol) and must be 1.
    """
    if batch_size != 1:
        raise ValueError(
            f"prox-linear runs at batch_size=1, where its subproblem is "
            f"closed-form; got {batch_size}"
        )

    needs_value = getattr(stepsize, "name", "") == "polyak"

    def step(x: Tensor, generator: torch.Generator, k: int) -> Tensor:
        idx = problem.sample(generator, 1).squeeze(1)  # (R,) one measurement each
        c, g = problem.linearization(x, idx)  # (R,), (R, n)

        alpha = stepsize(
            StepContext(
                k=k,
                f_val=c.abs() if needs_value else None,
                grad_norm=g.norm(dim=1),
                f_star=f_star,
            )
        )
        return x + prox_linear_step(c, g, alpha)

    return run_trajectory(
        problem,
        x0=x0,
        step=step,
        n_iterations=n_iterations,
        seed=seed,
        record_every=record_every,
    )
