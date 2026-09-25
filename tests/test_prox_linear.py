"""Checks for the stochastic prox-linear method."""

from __future__ import annotations

import pytest
import torch

from paraconvex.algorithms import (
    Constant,
    prox_linear_step,
    stochastic_prox_linear,
    stochastic_subgradient,
)
from paraconvex.data import make_design
from paraconvex.problems import RobustPhaseRetrieval


def build(p: float = 1.5, n: int = 20, m: int = 60, R: int = 4, seed: int = 0):
    data = make_design(n=n, m=m, n_replications=R, dof=2.5, seed=seed).instantiate(0.1)
    return RobustPhaseRetrieval(data, p=p, check_moment=False)


def brute_force_step(c: float, g: torch.Tensor, alpha: float) -> torch.Tensor:
    """Minimise |c + <g,d>| + ||d||^2/(2 alpha) over the full vector d.

    Deliberately assumes nothing about the solution being parallel to ``g`` --
    that is the structural claim the closed form rests on, so the test must not
    bake it in.
    """
    d = torch.zeros_like(g, requires_grad=True)
    opt = torch.optim.LBFGS(
        [d], lr=0.3, max_iter=6000, tolerance_grad=1e-16, tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        r = c + g @ d
        # tiny smoothing purely for LBFGS stability, far below the tolerance checked
        obj = torch.sqrt(r * r + 1e-24) + d.pow(2).sum() / (2 * alpha)
        obj.backward()
        return obj

    opt.step(closure)
    return d.detach()


class TestClosedForm:
    @pytest.mark.parametrize("alpha,regime", [(1e-3, "kink"), (10.0, "smooth")])
    def test_matches_brute_force(self, alpha: float, regime: str):
        """The closed form must solve the subproblem, not merely resemble it."""
        gen = torch.Generator().manual_seed(0)
        for _ in range(4):
            g = torch.randn(25, generator=gen, dtype=torch.float64)
            c = torch.randn((), generator=gen, dtype=torch.float64) * 2

            got = prox_linear_step(c.reshape(1), g.reshape(1, -1), alpha)[0]
            expected = brute_force_step(float(c), g, alpha)
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-8)

    def test_equals_the_clipped_gauss_newton_step(self):
        """The step must equal ``-clip(c/||g||^2, -alpha, alpha) g``."""
        gen = torch.Generator().manual_seed(1)
        g = torch.randn(6, 12, generator=gen, dtype=torch.float64)
        c = torch.randn(6, generator=gen, dtype=torch.float64)
        alpha = 0.3

        got = prox_linear_step(c, g, alpha)
        expected = -(c / g.pow(2).sum(1)).clamp(-alpha, alpha).unsqueeze(-1) * g
        torch.testing.assert_close(got, expected)

    def test_large_alpha_lands_on_the_linearised_zero_set(self):
        """The trust-region property: the step never overshoots ``c + <g,d> = 0``."""
        gen = torch.Generator().manual_seed(2)
        g = torch.randn(8, 15, generator=gen, dtype=torch.float64)
        c = torch.randn(8, generator=gen, dtype=torch.float64)

        d = prox_linear_step(c, g, alpha=1e6)
        residual = c + (g * d).sum(dim=1)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10,
                                   rtol=0)

    def test_step_is_parallel_to_the_jacobian(self):
        gen = torch.Generator().manual_seed(3)
        g = torch.randn(5, 10, generator=gen, dtype=torch.float64)
        c = torch.randn(5, generator=gen, dtype=torch.float64)

        d = prox_linear_step(c, g, 0.2)
        cos = torch.nn.functional.cosine_similarity(d, g, dim=1).abs()
        torch.testing.assert_close(cos, torch.ones_like(cos))

    def test_zero_residual_gives_zero_step(self):
        g = torch.randn(4, 9, generator=torch.Generator().manual_seed(4),
                        dtype=torch.float64)
        d = prox_linear_step(torch.zeros(4, dtype=torch.float64), g, 0.5)
        torch.testing.assert_close(d, torch.zeros_like(d))


class TestSolver:
    def test_small_alpha_reproduces_the_subgradient_method(self):
        """For tiny alpha the smooth branch applies and the two coincide exactly.

        There ``s* = -alpha||g||sign(c)``, so ``d = -alpha * sign(c) g``, which
        is precisely ``-alpha`` times the subgradient. Any drift would mean the
        branch logic or the sign convention disagrees between the two solvers.
        """
        problem = build()
        x0 = problem.random_start(torch.Generator().manual_seed(0), 0.1)
        kw = {"n_iterations": 60, "batch_size": 1, "seed": 5, "record_every": 60}

        pl = stochastic_prox_linear(problem, x0, Constant(1e-9), **kw)
        sg = stochastic_subgradient(problem, x0, Constant(1e-9), **kw)
        torch.testing.assert_close(pl.x_final, sg.x_final, rtol=1e-7, atol=1e-12)

    def test_survives_a_step_size_that_destroys_the_subgradient_method(self):
        """The trust region is the point: a huge alpha must not blow up."""
        problem = build()
        x0 = problem.random_start(torch.Generator().manual_seed(0), 0.1)
        kw = {"n_iterations": 300, "batch_size": 1, "seed": 6, "record_every": 300}

        pl = stochastic_prox_linear(problem, x0, Constant(1e4), **kw)
        sg = stochastic_subgradient(problem, x0, Constant(1e4), **kw)

        assert torch.isfinite(pl.x_final).all()
        assert pl.final_distance().median() < sg.final_distance().median()

    @pytest.mark.parametrize("p", [1.25, 1.5, 2.0])
    def test_converges_from_a_nearby_start(self, p: float):
        """Makes substantial progress at a step size in the usable range.

        Two calibration notes, both measured rather than guessed:

        * The trust region bounds a step by ``alpha||g||`` with ``||g|| ~ 4-5``
          here, so ``alpha`` must stay well below the starting distance of 0.1.
          At ``alpha=1e-2`` a single step can exceed the whole distance to the
          solution and the iterate wanders (0.099 -> 0.132).
        * ``alpha=1e-4`` takes 0.0987 to 0.0499 / 0.0430 / 0.0455 at
          ``p = 1.25 / 1.5 / 2.0``. The 0.6 factor clears the worst of those
          (0.506) with margin; a 0.5 threshold would sit on top of it and make
          the test flaky for reasons unrelated to correctness.
        """
        problem = build(p=p, n=30, m=120, R=6)
        x0 = problem.random_start(torch.Generator().manual_seed(0), 0.1)
        result = stochastic_prox_linear(
            problem, x0, Constant(1e-4), n_iterations=4000, seed=7, record_every=4000
        )
        assert result.diverged.sum() == 0
        assert result.final_distance().median() < 0.6 * result.distance[0].median()

    def test_batch_size_above_one_is_refused(self):
        problem = build()
        x0 = problem.random_start(torch.Generator().manual_seed(0), 0.1)
        with pytest.raises(ValueError, match="batch_size=1"):
            stochastic_prox_linear(problem, x0, Constant(0.1), n_iterations=5,
                                   batch_size=4)


class TestLinearizationOracle:
    @pytest.mark.parametrize("p", [1.25, 1.5, 2.0])
    def test_jacobian_matches_autograd(self, p: float):
        problem = build(p=p)
        idx = problem.sample(torch.Generator().manual_seed(0), 1).squeeze(1)
        x = torch.randn(problem.R, problem.n,
                        generator=torch.Generator().manual_seed(1),
                        dtype=problem.data.A.dtype).requires_grad_(True)

        residual, jac = problem.linearization(x, idx)
        assert residual.shape == (problem.R,)
        assert jac.shape == (problem.R, problem.n)

        residual.sum().backward()
        torch.testing.assert_close(jac, x.grad, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("p", [1.25, 1.5, 2.0])
    def test_recovers_the_subgradient_when_recombined(self, p: float):
        """``sign(c) * g`` is the single-sample subgradient."""
        problem = build(p=p)
        idx = problem.sample(torch.Generator().manual_seed(2), 1).squeeze(1)
        x = torch.randn(problem.R, problem.n,
                        generator=torch.Generator().manual_seed(3),
                        dtype=problem.data.A.dtype)

        c, g = problem.linearization(x, idx)
        recombined = torch.sign(c).unsqueeze(-1) * g
        torch.testing.assert_close(
            recombined,
            problem.minibatch_subgradient(x, idx.unsqueeze(1)),
            rtol=1e-12, atol=1e-14,
        )
