"""Step-size schedules.

Each rule maps iteration state to ``alpha_k``, returned either as a scalar or as
a per-replication tensor of shape ``(R,)`` -- Polyak's rule needs the latter
because its step depends on the current objective value, which differs across
replications.  The solver broadcasts whichever it receives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

__all__ = [
    "Constant",
    "Diminishing",
    "Geometric",
    "Polyak",
    "StepContext",
    "StepSize",
    "build_stepsize",
]


@dataclass(frozen=True)
class StepContext:
    """State a step-size rule may consult at iteration ``k``."""

    k: int
    #: Sampled objective value per replication, shape ``(R,)``. Polyak only.
    f_val: Tensor | None = None
    #: Norm of the stochastic subgradient, shape ``(R,)``. Polyak only.
    grad_norm: Tensor | None = None
    #: Estimate of the optimal value per replication, shape ``(R,)``. Polyak only.
    f_star: Tensor | None = None


class StepSize(Protocol):
    name: str

    def __call__(self, ctx: StepContext) -> Tensor | float: ...


@dataclass(frozen=True)
class Constant:
    """``alpha_k = alpha``."""

    alpha: float
    name: str = "constant"

    def __call__(self, ctx: StepContext) -> float:
        return self.alpha

    def label(self) -> str:
        return f"constant (a={self.alpha:g})"


@dataclass(frozen=True)
class Geometric:
    """``alpha_k = alpha_0 * lam^k``.

    The theory-matched choice: under a Holderian error bound, geometrically
    decaying steps are what deliver a linear rate, so this is the schedule the
    weakly-convex-versus-paraconvex comparison should be read from.
    """

    alpha0: float
    lam: float
    name: str = "geometric"

    def __call__(self, ctx: StepContext) -> float:
        return self.alpha0 * self.lam**ctx.k

    def label(self) -> str:
        return f"geometric (a0={self.alpha0:g}, lam={self.lam:g})"

    @classmethod
    def from_total_decay(
        cls, alpha0: float, total_decay: float, n_iterations: int
    ) -> Geometric:
        """Build from the *end-to-end* shrinkage ``alpha_K / alpha_0``.

        Tuning over ``lam`` directly is awkward: the useful values sit in a
        narrow band just below 1 whose location moves with the horizon (at
        K=5000, lam=0.999 shrinks by 7e-3 while lam=0.99 shrinks by 1e-22).
        Specifying the total decay instead keeps the grid horizon-independent
        and interpretable.
        """
        if not 0.0 < total_decay <= 1.0:
            raise ValueError(f"total_decay must be in (0, 1], got {total_decay}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        return cls(alpha0=alpha0, lam=total_decay ** (1.0 / n_iterations))


@dataclass(frozen=True)
class Diminishing:
    """``alpha_k = alpha_0 * (k+1)^(-power)``; ``power=0.5`` is the classical rule."""

    alpha0: float
    power: float = 0.5
    name: str = "diminishing"
    #: Loss exponent the power was derived from, when it was; labelling only.
    loss_exponent: float | None = None

    def __call__(self, ctx: StepContext) -> float:
        return self.alpha0 / (ctx.k + 1.0) ** self.power

    def label(self) -> str:
        if self.loss_exponent is None:
            return f"diminishing (a0={self.alpha0:g}, q={self.power:g})"
        return (
            f"diminishing (a0={self.alpha0:g}, q=1/p={self.power:g}, "
            f"p={self.loss_exponent:g})"
        )

    @classmethod
    def from_loss_exponent(cls, alpha0: float, p: float) -> Diminishing:
        """``alpha_k = alpha_0 * (k+1)^(-1/p)``, tied to the loss exponent ``p``.

        At ``p = 2`` this is ``k^{-1/2}``, the classical square-summable-but-not-
        summable rule, so the weakly-convex arm recovers the textbook schedule.
        Smaller ``p`` decays faster (``p=1.25`` gives ``k^{-0.8}``), which makes
        the total shrinkage over a run differ substantially between arms -- over
        ``K=10^4`` steps, 100x at ``p=2`` versus ~1600x at ``p=1.25``.  That is
        by design, but it is also why ``alpha_0`` must be tuned per exponent
        rather than shared.
        """
        if p <= 0:
            raise ValueError(f"loss exponent p must be positive, got {p}")
        return cls(alpha0=alpha0, power=1.0 / p, loss_exponent=p)


@dataclass(frozen=True)
class Polyak:
    """``alpha_k = (f(x_k) - f*) / (scale * ||g_k||^2)``, clipped at ``alpha_max``.

    Uses the *minibatch* value and subgradient, so this is the stochastic Polyak
    step and inherits its bias; ``f_star`` must be supplied by the caller since
    the noisy optimum is not zero.  ``scale=4`` reproduces the "scaled Polyak"
    variant of the deterministic literature.
    """

    scale: float = 1.0
    alpha_max: float = 1.0
    eps: float = 1e-12
    name: str = "polyak"

    def __call__(self, ctx: StepContext) -> Tensor:
        if ctx.f_val is None or ctx.grad_norm is None:
            raise ValueError("Polyak step requires f_val and grad_norm in the context")
        f_star = ctx.f_star if ctx.f_star is not None else torch.zeros_like(ctx.f_val)
        gap = (ctx.f_val - f_star).clamp_min(0.0)
        alpha = gap / (self.scale * ctx.grad_norm.pow(2).clamp_min(self.eps))
        return alpha.clamp_max(self.alpha_max)

    def label(self) -> str:
        return f"Polyak (scale={self.scale:g})"


_REGISTRY = {
    "constant": Constant,
    "geometric": Geometric,
    "diminishing": Diminishing,
    "polyak": Polyak,
}


def build_stepsize(kind: str, **kwargs) -> StepSize:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        raise KeyError(
            f"unknown step-size {kind!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return factory(**kwargs)
