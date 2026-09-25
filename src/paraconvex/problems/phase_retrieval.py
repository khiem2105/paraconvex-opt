"""Robust phase retrieval with a tunable loss exponent.

One fixed dataset ``b_i = <a_i,x_true>^2 + noise`` is fitted by

    min_x  f_p(x) := (1/m) sum_i | |<a_i,x>|^p - T_p(b_i) |,      p in (1, 2],

where ``T_p`` maps a squared measurement onto the ``p``-th power scale:

    T_p(b) = sign(b) |b|^{p/2}       ("signed",  the default)
    T_p(b) = max(b, 0)^{p/2}         ("clip")

Noiselessly ``b = <a,x_true>^2 >= 0``, so both give ``T_p(b) = |<a,x_true>|^p``
and ``x_true`` solves every model exactly.  ``p`` is therefore a property of the
*estimator*, not of the data.

Why the exponent buys robustness
--------------------------------
``p/2 < 1`` makes ``T_p`` concave on the positives, so a gross outlier ``xi``
reaches the objective as ``|xi|^{p/2}`` rather than ``|xi|``.  Concretely the
objective is integrable iff ``dof > p/2``: the classical ``p=2`` loss needs
``dof > 1`` while ``p=1.25`` needs only ``dof > 0.625``.

Why it is paraconvex
--------------------
The problem is composite ``phi(g(x))`` with ``phi = ||.||_1/m`` convex Lipschitz
and ``g_i(x) = |<a_i,x>|^p - T_p(b_i)``.  Since

    d/dx |<a,x>|^p = p |<a,x>|^{p-1} sign(<a,x>) a

and ``t -> |t|^{p-1} sign(t)`` is Holder continuous of order ``p-1``, the
composite rule makes ``f_p`` **nu-paraconvex with nu = p - 1**:

* ``p = 2`` => ``nu = 1``: weak convexity.  ``T_2`` is the identity under
  "signed", so this arm is the unmodified classical problem.
* ``p in (1,2)`` => ``nu < 1``: genuinely paraconvex, outside that theory.

Every method is batched over a leading replication axis ``R``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from paraconvex.data import PhaseRetrievalData, dof_for_moment

__all__ = ["RobustPhaseRetrieval", "TargetTransform", "uniform_in_ball"]

TargetTransform = Literal["signed", "clip"]


def uniform_in_ball(
    center: Tensor, generator: torch.Generator, radius: float
) -> Tensor:
    """Draw uniformly from the ball of radius ``radius`` about each row of ``center``.

    ``center`` is ``(R, n)``; the result has the same shape.  A free function
    rather than a method because the draw does **not** depend on the loss
    exponent: every arm of a comparison must start from the same point, and
    hanging this off a ``p``-specific object invites deriving it once per arm
    and hoping the randomness lines up.

    Uniform *in* the ball, not on its surface, requires a non-uniform radius.
    The volume element grows as ``r^{n-1}``, so drawing ``r`` uniformly would
    concentrate points near the centre; the correct draw is

        r = radius * U^{1/n},        U ~ Uniform(0, 1),

    which makes ``(r/radius)^n`` exactly uniform.  Direction and radius are
    drawn independently, the direction from a normalised Gaussian (isotropic by
    spherical symmetry of ``N(0, I_n)``).

    In high dimension this is nearly indistinguishable from sampling the
    surface: at ``n=100``, ``P(r < 0.9 radius) = 0.9^100 ~ 3e-5``, so
    essentially all of the ball's volume sits in a thin outer shell.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    n = center.shape[1]
    opts = {"generator": generator, "device": center.device, "dtype": center.dtype}

    direction = torch.randn(center.shape, **opts)
    direction = direction / direction.norm(dim=1, keepdim=True)

    u = torch.rand((center.shape[0], 1), **opts)
    return center + (radius * u.pow(1.0 / n)) * direction


class RobustPhaseRetrieval:
    """Objective, oracles and recovery metric for one loss exponent ``p``."""

    def __init__(
        self,
        data: PhaseRetrievalData,
        p: float,
        target_transform: TargetTransform = "signed",
        check_moment: bool = True,
    ):
        if not 1.0 < p <= 2.0:
            raise ValueError(
                f"p must lie in (1, 2] so that nu = p - 1 lies in (0, 1]; got p={p}"
            )
        if check_moment and data.dof <= p / 2.0:
            raise ValueError(
                f"Student-t dof={data.dof:g} makes the p={p:g} objective "
                f"non-integrable (an outlier enters as |xi|^{{p/2}}, so dof > "
                f"p/2 = {p / 2:g} is required). Try dof >= "
                f"{dof_for_moment(p):g}, or pass check_moment=False to probe "
                f"the non-integrable regime deliberately."
            )
        if target_transform not in ("signed", "clip"):
            raise ValueError(
                f"target_transform must be 'signed' or 'clip', got {target_transform!r}"
            )

        self.data = data
        self.p = p
        self.target_transform = target_transform
        self.R, self.m, self.n = data.A.shape

        # T_p(b) depends only on the data, so it is computed once.
        self.target = self._transform(data.b)

    def _transform(self, b: Tensor) -> Tensor:
        half = self.p / 2.0
        if self.target_transform == "clip":
            return b.clamp_min(0.0) ** half
        return torch.sign(b) * b.abs() ** half

    # -- basic quantities --------------------------------------------------

    @property
    def nu(self) -> float:
        """Paraconvexity order, ``nu = p - 1``."""
        return self.p - 1.0

    def _measure(self, x: Tensor) -> Tensor:
        """``<a_i, x>`` for every measurement; shape ``(R, m)``."""
        return torch.einsum("rmn,rn->rm", self.data.A, x)

    # -- full-sample oracles ------------------------------------------------

    def value(self, x: Tensor) -> Tensor:
        """Full objective per replication; shape ``(R,)``."""
        u = self._measure(x)
        return (u.abs() ** self.p - self.target).abs().mean(dim=1)

    def subgradient(self, x: Tensor) -> Tensor:
        """Full-sample subgradient; shape ``(R, n)``.  Deterministic baseline."""
        u = self._measure(x)
        coef = self._residual_coefficient(u, self.target)
        return torch.einsum("rm,rmn->rn", coef, self.data.A) / self.m

    # -- stochastic oracles -------------------------------------------------

    def sample(self, generator: torch.Generator, batch_size: int) -> Tensor:
        """Draw measurement indices independently per replication; ``(R, B)``.

        With replacement, matching the i.i.d. oracle the stochastic analysis
        assumes.
        """
        return torch.randint(
            0,
            self.m,
            (self.R, batch_size),
            generator=generator,
            device=self.data.A.device,
        )

    def minibatch_subgradient(self, x: Tensor, idx: Tensor) -> Tensor:
        """Subgradient of the sampled loss; shape ``(R, n)``.

        Unbiased for :meth:`subgradient`: indices are uniform and the full
        objective is the uniform average.
        """
        A_sel, t_sel = self._gather(idx)
        u = (A_sel * x.unsqueeze(1)).sum(dim=-1)  # (R, B)
        coef = self._residual_coefficient(u, t_sel)  # (R, B)
        return (coef.unsqueeze(-1) * A_sel).mean(dim=1)  # (R, n)

    def minibatch_value(self, x: Tensor, idx: Tensor) -> Tensor:
        """Sampled objective value; shape ``(R,)``.  Used by Polyak steps."""
        A_sel, t_sel = self._gather(idx)
        u = (A_sel * x.unsqueeze(1)).sum(dim=-1)
        return (u.abs() ** self.p - t_sel).abs().mean(dim=1)

    def linearization(self, x: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Composite data for the prox-linear model at one measurement each.

        ``idx`` is ``(R,)`` -- a single measurement index per replication, which
        is all prox-linear ever needs, since its subproblem is closed-form only
        for a scalar residual.  Returns the residual ``c`` of shape ``(R,)`` and
        its gradient ``g = grad_x c`` of shape ``(R, n)``:

            c = |<a,x>|^p - T_p(b),    g = p |<a,x>|^{p-1} sign(<a,x>) a

        Distinct from :meth:`minibatch_subgradient`, which returns
        ``sign(c) * g`` averaged over a batch.  Prox-linear needs the two pieces
        *unmixed*: its subproblem decides the sign of the outer ``|.|`` itself
        and may land exactly on the kink ``c + <g,d> = 0``, which is the
        trust-region behaviour that distinguishes it from a subgradient step.
        """
        a, target = (t.squeeze(1) for t in self._gather(idx.unsqueeze(1)))

        u = (a * x).sum(dim=1)  # (R,)
        residual = u.abs() ** self.p - target
        slope = self.p * u.abs() ** (self.p - 1.0) * torch.sign(u)
        return residual, slope.unsqueeze(-1) * a

    def _gather(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Sensing vectors ``(R, B, n)`` and targets ``(R, B)`` for ``idx`` ``(R, B)``.

        Each replication draws its own indices, so ``torch.gather`` along the
        measurement axis.  ``expand`` matches the index rank to the input as a
        stride-0 view, allocating nothing.
        """
        return (
            torch.gather(self.data.A, 1, idx.unsqueeze(-1).expand(-1, -1, self.n)),
            torch.gather(self.target, 1, idx),
        )

    # -- shared chain-rule factor -------------------------------------------

    def _residual_coefficient(self, u: Tensor, target: Tensor) -> Tensor:
        """Scalar multiplying ``a_i`` in the subgradient.

        ``d/dx | |<a,x>|^p - T | = sign(|u|^p - T) * p|u|^{p-1} sign(u) * a``.

        Two nondifferentiable points, both null events under continuous data and
        both resolved to the *zero* subgradient element, which is the convention
        ``torch.sign`` already implements:

        * ``|u|^p = T``: any value in ``[-1,1]`` is admissible; ``0`` is taken.
        * ``u = 0``: for ``p > 1`` the map ``u -> |u|^p`` is genuinely
          differentiable with derivative ``0``, so this is not a convention.
        """
        residual = u.abs() ** self.p - target
        return torch.sign(residual) * self.p * u.abs() ** (self.p - 1.0) * torch.sign(u)

    # -- recovery metric -----------------------------------------------------

    def distance_to_truth(self, x: Tensor) -> Tensor:
        """``min(||x - x*||, ||x + x*||) / ||x*||`` per replication; ``(R,)``.

        The sign ambiguity is intrinsic: ``|<a,x>|^p`` is even in ``x``, so
        ``-x*`` fits the measurements as well as ``x*`` for every ``p``.
        """
        x_true = self.data.x_true
        d_plus = (x - x_true).norm(dim=1)
        d_minus = (x + x_true).norm(dim=1)
        return torch.minimum(d_plus, d_minus) / x_true.norm(dim=1)

    def random_start(
        self, generator: torch.Generator, relative_distance: float
    ) -> Tensor:
        """Uniform in the ball of radius ``relative_distance`` about ``x*``.

        Shape ``(R, n)``.  Robust phase retrieval is nonconvex with spurious
        critical points far from the truth and the convergence theory is local,
        so bounding the initial distance keeps every arm in the same regime: the
        comparison then measures local behaviour rather than which exponent drew
        a luckier basin.

        The radius is ``relative_distance`` directly, which relies on ``x*``
        being unit-norm (:func:`~paraconvex.data.make_design` normalises it).
        That is what keeps this consistent with :meth:`distance_to_truth`, which
        divides by ``||x*||``: together they guarantee
        ``distance_to_truth(x0) <= relative_distance``.  Drop the unit-norm
        convention and both would need a ``||x*||`` factor restored.

        Convenience wrapper over :func:`uniform_in_ball`.  A comparison across
        exponents should call that directly, **once**, and share the result --
        see its docstring.
        """
        return uniform_in_ball(self.data.x_true, generator, relative_distance)
