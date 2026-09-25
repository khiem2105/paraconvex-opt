"""Correctness checks for the robust phase retrieval oracles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paraconvex.data import dof_for_moment, make_design
from paraconvex.problems import RobustPhaseRetrieval
from paraconvex.problems.phase_retrieval import uniform_in_ball

P_VALUES = [1.25, 1.5, 1.75, 2.0]


def build(p: float, n: int = 12, m: int = 30, R: int = 3, seed: int = 0,
          noise_level: float = 0.1, dof: float = 2.5, **kwargs):
    design = make_design(n=n, m=m, n_replications=R, dof=dof, seed=seed)
    return RobustPhaseRetrieval(design.instantiate(noise_level), p=p, **kwargs)


@pytest.mark.parametrize("p", P_VALUES)
def test_subgradient_matches_autograd(p: float):
    """The hand-derived subgradient must equal autograd at differentiable points.

    A random ``x`` makes both kink sets -- ``u = 0`` and ``|u|^p = T`` -- null
    events, so the objective is differentiable there and the two must agree.
    """
    problem = build(p)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(
        problem.R, problem.n, generator=gen, dtype=problem.data.A.dtype
    ).requires_grad_(True)

    problem.value(x).sum().backward()

    torch.testing.assert_close(
        problem.subgradient(x.detach()), x.grad, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("p", P_VALUES)
def test_minibatch_oracles_are_unbiased(p: float):
    """Averaging over *all* indices reproduces the full-sample oracles exactly."""
    problem = build(p)
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(problem.R, problem.n, generator=gen, dtype=problem.data.A.dtype)
    all_idx = torch.arange(problem.m).unsqueeze(0).expand(problem.R, -1)

    torch.testing.assert_close(
        problem.minibatch_subgradient(x, all_idx), problem.subgradient(x),
        rtol=1e-12, atol=1e-12,
    )
    torch.testing.assert_close(
        problem.minibatch_value(x, all_idx), problem.value(x), rtol=1e-12, atol=1e-12
    )


class TestTargetTransform:
    """Properties of ``T_p(b)`` that the whole comparison rests on."""

    def test_signed_transform_is_identity_at_p2(self):
        """``T_2(b) = b``, so the p=2 arm is the unmodified classical problem.

        This is why "signed" is the default: clipping would silently alter the
        weakly-convex baseline the paraconvex arms are compared against.
        """
        problem = build(2.0)
        torch.testing.assert_close(problem.target, problem.data.b, rtol=0, atol=0)

    def test_clip_transform_differs_at_p2_when_b_negative(self):
        problem = build(2.0, target_transform="clip")
        assert (problem.data.b < 0).any(), "test needs some negative measurements"
        assert not torch.equal(problem.target, problem.data.b)
        assert (problem.target >= 0).all()

    @pytest.mark.parametrize("p", P_VALUES)
    def test_noiseless_target_is_the_pth_power(self, p: float):
        """Noiselessly ``T_p(b) = |<a,x*>|^p`` for every ``p``."""
        design = make_design(n=10, m=25, n_replications=2, dof=2.5, seed=7)
        problem = RobustPhaseRetrieval(design.instantiate(0.0), p=p)
        u_true = torch.einsum("rmn,rn->rm", problem.data.A, problem.data.x_true)
        torch.testing.assert_close(problem.target, u_true.abs() ** p)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_noiseless_truth_is_a_global_minimum(self, p: float):
        """With no noise the objective vanishes exactly at ``x*``, for all ``p``.

        This is the property that makes ``p`` a choice of *estimator* on one
        fixed dataset rather than a change of problem.
        """
        design = make_design(n=10, m=25, n_replications=2, dof=2.5, seed=7)
        problem = RobustPhaseRetrieval(design.instantiate(0.0), p=p)
        value = problem.value(problem.data.x_true)
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)

    def test_smaller_p_compresses_outliers_harder(self):
        """The robustness mechanism, stated as a test.

        A gross outlier reaches the objective as ``|xi|^{p/2}``, so its
        contribution must shrink monotonically as ``p`` decreases.
        """
        design = make_design(n=6, m=12, n_replications=1, dof=2.5, seed=0)
        data = design.instantiate(0.1)
        outlier = torch.tensor(1.0e6, dtype=data.b.dtype)

        magnitudes = [
            RobustPhaseRetrieval(data, p=p)._transform(outlier).item()
            for p in P_VALUES
        ]
        assert magnitudes == sorted(magnitudes), (
            f"expected increasing compression as p falls, got {magnitudes}"
        )


@pytest.mark.parametrize("p", P_VALUES)
def test_sign_ambiguity(p: float):
    """``x*`` and ``-x*`` are indistinguishable to the measurements."""
    problem = build(p)
    x_true = problem.data.x_true
    zeros = torch.zeros(problem.R, dtype=x_true.dtype)

    torch.testing.assert_close(problem.distance_to_truth(x_true), zeros,
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(problem.distance_to_truth(-x_true), zeros,
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(problem.value(x_true), problem.value(-x_true))


class TestRandomStart:
    """`random_start` draws uniformly *in* a ball around x*, not on its surface."""

    @staticmethod
    def relative_radius(problem, x0):
        """||x0 - x*|| / ||x*||, avoiding the +-x* ambiguity of the metric."""
        x_true = problem.data.x_true
        return ((x0 - x_true).norm(dim=1) / x_true.norm(dim=1)).numpy()

    @pytest.mark.parametrize("rho", [0.05, 0.5])
    def test_stays_within_the_ball(self, rho: float):
        problem = build(1.5, n=8, R=400)
        gen = torch.Generator().manual_seed(4)
        r = self.relative_radius(problem, problem.random_start(gen, rho))
        assert r.max() <= rho + 1e-12
        assert r.min() > 0.0

    def test_radius_distribution_is_uniform_in_the_ball(self):
        """``(r/rho)^n`` must be Uniform(0,1) -- the defining property.

        ``rho`` is the ball radius; ``R`` is reserved for the replication count.

        A low dimension is used deliberately: at n=100 essentially all of the
        volume lies near the surface, so an incorrect radial law would be almost
        invisible. At n=3 the difference is stark.
        """
        from scipy import stats

        n, rho = 3, 0.4
        problem = build(1.5, n=n, R=4000)
        gen = torch.Generator().manual_seed(11)
        r = self.relative_radius(problem, problem.random_start(gen, rho))

        assert stats.kstest((r / rho) ** n, "uniform").pvalue > 0.01

        # and the wrong-but-tempting choice (uniform radius) must be rejected,
        # confirming the test can actually tell them apart
        wrong = np.random.default_rng(0).uniform(0, 1, r.size)
        assert stats.kstest(wrong**n, "uniform").pvalue < 1e-6

    def test_direction_is_isotropic(self):
        problem = build(1.5, n=6, R=4000)
        gen = torch.Generator().manual_seed(12)
        x0 = problem.random_start(gen, 0.3)
        offsets = (x0 - problem.data.x_true).numpy()
        unit = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
        # mean of a uniform direction is 0; SE per coordinate ~ 1/sqrt(n_dim*R)
        assert np.abs(unit.mean(axis=0)).max() < 5 / np.sqrt(6 * 4000)

    def test_zero_radius_returns_the_truth(self):
        problem = build(1.5)
        gen = torch.Generator().manual_seed(4)
        torch.testing.assert_close(
            problem.random_start(gen, 0.0), problem.data.x_true, rtol=0, atol=0
        )

    def test_reproducible_from_the_generator(self):
        problem = build(1.5)
        a = problem.random_start(torch.Generator().manual_seed(3), 0.1)
        b = problem.random_start(torch.Generator().manual_seed(3), 0.1)
        assert torch.equal(a, b)

    def test_negative_radius_rejected(self):
        problem = build(1.5)
        with pytest.raises(ValueError, match="non-negative"):
            problem.random_start(torch.Generator().manual_seed(0), -0.1)


class TestValidation:
    def test_dof_must_exceed_half_p(self):
        """An outlier enters as ``|xi|^{p/2}``, so integrability needs dof > p/2."""
        design = make_design(n=5, m=10, n_replications=1, dof=0.8, seed=0)
        with pytest.raises(ValueError, match="non-integrable"):
            RobustPhaseRetrieval(design.instantiate(0.1), p=2.0)

    def test_smaller_p_tolerates_heavier_tails(self):
        """dof=0.8 is fatal for p=2 but fine for p=1.25 -- robustness, precisely."""
        data = make_design(n=5, m=10, n_replications=1, dof=0.8, seed=0).instantiate(0.1)
        with pytest.raises(ValueError):
            RobustPhaseRetrieval(data, p=2.0)
        RobustPhaseRetrieval(data, p=1.25)  # 1.25/2 = 0.625 < 0.8, so admissible

    def test_moment_check_can_be_disabled(self):
        data = make_design(n=5, m=10, n_replications=1, dof=0.8, seed=0).instantiate(0.1)
        RobustPhaseRetrieval(data, p=2.0, check_moment=False)

    @pytest.mark.parametrize("p", [0.9, 1.0, 2.5])
    def test_p_outside_range_rejected(self, p: float):
        data = make_design(n=5, m=10, n_replications=1, dof=3.0, seed=0).instantiate(0.1)
        with pytest.raises(ValueError, match=r"p must lie in \(1, 2\]"):
            RobustPhaseRetrieval(data, p=p)

    def test_bad_transform_rejected(self):
        data = make_design(n=5, m=10, n_replications=1, dof=3.0, seed=0).instantiate(0.1)
        with pytest.raises(ValueError, match="target_transform"):
            RobustPhaseRetrieval(data, p=1.5, target_transform="sqrt")

    def test_dof_for_moment(self):
        assert dof_for_moment(2.0) == 1.5
        assert dof_for_moment(1.25, margin=1.0) == 1.625
        with pytest.raises(ValueError):
            dof_for_moment(0.0)


def test_dataset_is_shared_across_exponents():
    """The paired-comparison guarantee: identical A, x*, and b for every p."""
    data = make_design(n=8, m=20, n_replications=2, dof=2.5, seed=11).instantiate(0.1)
    problems = [RobustPhaseRetrieval(data, p=p) for p in P_VALUES]

    for prob in problems[1:]:
        assert torch.equal(prob.data.A, problems[0].data.A)
        assert torch.equal(prob.data.b, problems[0].data.b)
        assert torch.equal(prob.data.x_true, problems[0].data.x_true)

    # only the target scale differs
    assert not torch.equal(problems[0].target, problems[-1].target)


def test_nu_is_p_minus_one():
    for p in P_VALUES:
        assert build(p).nu == pytest.approx(p - 1.0)


def test_negative_fraction_is_reported():
    data = make_design(n=20, m=200, n_replications=4, dof=2.5, seed=3).instantiate(0.1)
    assert 0.0 < data.negative_fraction < 0.5


class TestNoiseScale:
    """`sigma` uses the closed-form population mean, not a sample statistic."""

    def test_sigma_equals_noise_level_times_squared_norm(self):
        """E[<a,x*>^2] = ||x*||^2 exactly, so sigma is that times noise_level."""
        data = make_design(n=40, m=100, n_replications=5, dof=2.5, seed=0).instantiate(0.3)
        expected = 0.3 * data.x_true.pow(2).sum(dim=1)
        torch.testing.assert_close(data.sigma, expected, rtol=0, atol=0)

    def test_sigma_is_identical_across_replications(self):
        """The whole point: no per-replication jitter in the effective noise.

        A sample-median calibration would vary ~13.5% at m=300; x_true is
        unit-norm, so the closed form is exactly constant.
        """
        data = make_design(n=40, m=300, n_replications=50, dof=2.5, seed=0).instantiate(0.1)
        torch.testing.assert_close(
            data.sigma, torch.full_like(data.sigma, 0.1), rtol=1e-12, atol=1e-12
        )

    def test_sigma_does_not_depend_on_the_measurement_draw(self):
        """Different m (hence different samples of `clean`) must give the same sigma."""
        sigmas = [
            make_design(n=40, m=m, n_replications=3, dof=2.5, seed=0)
            .instantiate(0.2)
            .sigma
            for m in (50, 500, 5000)
        ]
        for s in sigmas[1:]:
            torch.testing.assert_close(s, sigmas[0], rtol=1e-12, atol=1e-12)

    def test_sigma_scales_linearly_with_noise_level(self):
        design = make_design(n=20, m=100, n_replications=3, dof=2.5, seed=0)
        torch.testing.assert_close(
            design.instantiate(0.4).sigma, 4.0 * design.instantiate(0.1).sigma
        )

    def test_zero_noise_gives_zero_sigma(self):
        data = make_design(n=20, m=100, n_replications=3, dof=2.5, seed=0).instantiate(0.0)
        assert data.sigma.abs().max() == 0.0
        assert data.negative_fraction == 0.0


class TestSharedStartAcrossExponents:
    """Every arm of a comparison must begin from the same point.

    Guarded by a test because the invariant used to hold only by coincidence:
    the driver re-seeded a generator inside its per-exponent loop, which agreed
    across arms purely because `random_start` happened to consume identical
    randomness for every `p`. `uniform_in_ball` takes `x_true` rather than a
    problem, so the start cannot depend on the exponent at all.
    """

    def test_start_does_not_depend_on_the_exponent(self):
        data = make_design(n=20, m=60, n_replications=5, dof=2.5, seed=0).instantiate(0.1)
        starts = [
            uniform_in_ball(data.x_true, torch.Generator().manual_seed(0), 0.1)
            for _ in P_VALUES
        ]
        for s in starts[1:]:
            assert torch.equal(starts[0], s)

    def test_method_delegates_to_the_free_function(self):
        problem = build(1.5)
        via_method = problem.random_start(torch.Generator().manual_seed(3), 0.2)
        via_function = uniform_in_ball(
            problem.data.x_true, torch.Generator().manual_seed(3), 0.2
        )
        assert torch.equal(via_method, via_function)

    def test_free_function_needs_no_problem_instance(self):
        """It depends only on the centre, so no exponent can leak in."""
        centre = torch.randn(4, 7, dtype=torch.float64)
        out = uniform_in_ball(centre, torch.Generator().manual_seed(1), 0.3)
        assert out.shape == centre.shape
        assert ((out - centre).norm(dim=1) <= 0.3 + 1e-12).all()

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            uniform_in_ball(torch.zeros(2, 3), torch.Generator().manual_seed(0), -1.0)
