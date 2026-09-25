"""Optimization algorithms and step-size schedules."""

from paraconvex.algorithms.stepsizes import (
    Constant,
    Diminishing,
    Geometric,
    Polyak,
    StepContext,
    StepSize,
    build_stepsize,
)
from paraconvex.algorithms.stochastic_prox_linear import (
    prox_linear_step,
    stochastic_prox_linear,
)
from paraconvex.algorithms.stochastic_subgradient import (
    SubgradientResult,
    stochastic_subgradient,
)
from paraconvex.algorithms.trajectory import TrajectoryResult, run_trajectory

__all__ = [
    "Constant",
    "Diminishing",
    "Geometric",
    "Polyak",
    "StepContext",
    "StepSize",
    "SubgradientResult",
    "TrajectoryResult",
    "build_stepsize",
    "prox_linear_step",
    "run_trajectory",
    "stochastic_prox_linear",
    "stochastic_subgradient",
]
