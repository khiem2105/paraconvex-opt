"""Synthetic data for robust phase retrieval with heavy-tailed noise.

One physical measurement model, shared by every loss exponent:

    b_i = <a_i, x_true>^2 + sigma * xi_i,        xi_i ~ Student-t(dof).

The exponent ``p`` does **not** appear here.  It is a property of the *loss*,
not of the data: :class:`~paraconvex.problems.phase_retrieval.RobustPhaseRetrieval`
fits ``| |<a_i,x>|^p - T_p(b_i) |`` where ``T_p(b) ~ b^{p/2}`` maps the squared
measurement onto the ``p``-th power scale.  Noiselessly ``b = <a,x_true>^2`` and
so ``T_p(b) = |<a,x_true>|^p`` exactly, for every ``p`` -- the ground truth
solves all of them.  Varying ``p`` therefore varies the estimator applied to one
fixed dataset, which is what makes the comparison a robustness comparison rather
than a comparison of different problems.

*Scale-matched noise.*  A Student-t draw is dimensionless, so it needs a scale
before it can be added to ``b``.  ``sigma`` ties that scale to the size of a
typical clean measurement, making ``noise_level`` relative rather than absolute.
It is computed from the squared measurements, so it is exponent-independent: one
``sigma`` per replication serves every ``p``.

The scale used is the **population mean** of a clean measurement, which has an
exact closed form -- no sampling involved:

    E[<a_i, x_true>^2] = x_true^T E[a a^T] x_true = ||x_true||^2

for ``a_i ~ N(0, I_n)``.  So ``sigma = noise_level * ||x_true||^2``, which is
``noise_level`` itself under the unit-norm convention here.  Two properties
follow, and both were the point of using it:

* **No sampling jitter.**  An *empirical* summary of ``clean`` -- the sample
  median, say -- carries a standard error of ``1/(2 f(M) sqrt(m))``, roughly
  13.5% at ``m=300``.  That would make every replication run at a slightly
  different effective noise level, inflating the ensemble spread with variation
  that has nothing to do with the algorithm.  The closed form has none.
* **Interpretability.**  Since ``E[clean] = 1`` here, ``noise_level = 0.1``
  means the noise scale is exactly 10% of the mean clean measurement.

For reference, ``clean_i`` is chi-squared with one degree of freedom (because
``<a_i, x_true> ~ N(0,1)`` for *any* ``n``, by isotropy of the Gaussian), whose
mean is 1 and whose median is ``Phi^{-1}(0.75)^2 = 0.4549``.  The distribution is
strongly right-skewed, so those differ by a factor of 2.2; an earlier version of
this module calibrated against the sample median, and ``noise_level`` there
meant 2.2x less noise than it does now.

*Moment condition.*  Student-t(dof) has ``E|xi|^q < inf`` iff ``q < dof``.  Since
a large outlier enters the loss as ``|xi|^{p/2}``, the objective has a finite
mean iff ``dof > p/2`` -- a *weaker* requirement for smaller ``p``, which is one
precise sense in which the paraconvex loss is more robust.  The condition is
checked in the problem class, where ``p`` is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "PhaseRetrievalData",
    "PhaseRetrievalDesign",
    "dof_for_moment",
    "make_design",
]


def dof_for_moment(p: float, margin: float = 0.5) -> float:
    """Smallest sensible Student-t ``dof`` making the ``p``-loss integrable.

    An outlier enters the objective as ``|xi|^{p/2}``, so integrability needs
    ``dof > p/2``; this returns ``p/2 + margin``.
    """
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return p / 2.0 + margin


@dataclass(frozen=True)
class PhaseRetrievalData:
    """One dataset: ``R`` independent replications, no exponent attached.

    All tensors carry a leading replication axis so a whole experiment advances
    in lockstep under one batched solver.

    Attributes
    ----------
    A:
        Sensing vectors, shape ``(R, m, n)``.
    b:
        Noisy squared measurements, shape ``(R, m)``.  May be negative where a
        heavy-tailed draw overwhelms the signal; the loss decides how to treat
        that.
    x_true:
        Ground truth signals, shape ``(R, n)``.
    sigma:
        Per-replication noise scale actually applied, shape ``(R,)``.
    """

    A: Tensor
    b: Tensor
    x_true: Tensor
    sigma: Tensor
    dof: float
    noise_level: float

    @property
    def n_replications(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[1]

    @property
    def n(self) -> int:
        return self.A.shape[2]

    @property
    def negative_fraction(self) -> float:
        """Share of measurements driven below zero by the noise."""
        return float((self.b < 0).double().mean())

    def describe(self) -> str:
        return (
            f"n={self.n}, m={self.m}, R={self.n_replications}, "
            f"dof={self.dof:g}, noise_level={self.noise_level:g}, "
            f"negative b: {100 * self.negative_fraction:.1f}%"
        )


@dataclass(frozen=True)
class PhaseRetrievalDesign:
    """Raw draw: sensing vectors, ground truth, unscaled noise."""

    A: Tensor
    x_true: Tensor
    xi_raw: Tensor
    dof: float

    def instantiate(self, noise_level: float) -> PhaseRetrievalData:
        """Apply a noise level to produce the squared measurements ``b``."""
        if noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {noise_level}")

        clean = torch.einsum("rmn,rn->rm", self.A, self.x_true) ** 2
        # Population mean of a clean measurement, in closed form:
        #   E[<a,x*>^2] = x*^T E[a a^T] x* = ||x*||^2   for a ~ N(0, I).
        # Using this rather than a sample statistic keeps sigma identical across
        # replications; see the module docstring.
        sigma = noise_level * self.x_true.pow(2).sum(dim=1)  # (R,)
        b = clean + sigma.unsqueeze(1) * self.xi_raw

        return PhaseRetrievalData(
            A=self.A,
            b=b,
            x_true=self.x_true,
            sigma=sigma,
            dof=self.dof,
            noise_level=noise_level,
        )


def make_design(
    n: int,
    m: int,
    n_replications: int,
    dof: float,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> PhaseRetrievalDesign:
    """Draw Gaussian ``a_i``, unit-norm ``x_true`` and raw Student-t noise.

    ``x_true`` is normalised so the reported relative distance is comparable
    across replications.

    ``dtype`` defaults to float64: for ``p`` near 1 the factor ``|u|^{p-1}`` in
    the subgradient varies sharply near ``u = 0``, and float32 rounding there is
    visible in the convergence curves.

    Sampling goes through NumPy because ``torch.distributions`` samplers do not
    accept an explicit ``Generator``, so a torch-side Student-t could not be
    reproduced from ``seed`` alone.
    """
    if dof <= 0:
        raise ValueError(f"dof must be positive, got {dof}")

    rng = np.random.default_rng(seed)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32

    A = rng.standard_normal((n_replications, m, n)).astype(np_dtype)

    x_true = rng.standard_normal((n_replications, n)).astype(np_dtype)
    x_true /= np.linalg.norm(x_true, axis=1, keepdims=True)

    xi_raw = rng.standard_t(dof, size=(n_replications, m)).astype(np_dtype)

    to = {"device": device, "dtype": dtype}
    return PhaseRetrievalDesign(
        A=torch.as_tensor(A).to(**to),
        x_true=torch.as_tensor(x_true).to(**to),
        xi_raw=torch.as_tensor(xi_raw).to(**to),
        dof=dof,
    )
