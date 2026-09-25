"""Checks for the step-size schedules."""

from __future__ import annotations

from itertools import pairwise

import pytest

from paraconvex.algorithms import Constant, Diminishing, Geometric
from paraconvex.algorithms.stepsizes import StepContext
from paraconvex.tuning import exponent_diminishing_grid

P_VALUES = [1.25, 1.5, 1.75, 2.0]


def alpha(schedule, k: int) -> float:
    return schedule(StepContext(k=k))


class TestExponentDiminishing:
    """``alpha_k = alpha_0 (k+1)^(-1/p)``."""

    @pytest.mark.parametrize("p", P_VALUES)
    def test_matches_the_formula(self, p: float):
        s = Diminishing.from_loss_exponent(0.01, p)
        for k in (0, 1, 9, 999):
            assert alpha(s, k) == pytest.approx(0.01 * (k + 1) ** (-1.0 / p))

    @pytest.mark.parametrize("p", P_VALUES)
    def test_power_is_reciprocal_of_p(self, p: float):
        assert Diminishing.from_loss_exponent(1.0, p).power == pytest.approx(1.0 / p)

    def test_p2_is_the_classical_square_root_rule(self):
        """At p=2 the schedule must reduce to the textbook ``k^{-1/2}``."""
        tied = Diminishing.from_loss_exponent(0.5, 2.0)
        classical = Diminishing(alpha0=0.5, power=0.5)
        assert tied.power == pytest.approx(classical.power)
        for k in (0, 5, 500):
            assert alpha(tied, k) == pytest.approx(alpha(classical, k))

    def test_starts_at_alpha0(self):
        for p in P_VALUES:
            assert alpha(Diminishing.from_loss_exponent(0.7, p), 0) == pytest.approx(0.7)

    def test_smaller_p_decays_faster(self):
        """Total shrinkage over a run must increase as ``p`` falls."""
        k = 10_000
        ratios = [
            alpha(Diminishing.from_loss_exponent(1.0, p), k) for p in sorted(P_VALUES)
        ]
        assert ratios == sorted(ratios), (
            f"expected faster decay at smaller p, got {ratios}"
        )

    def test_is_nonincreasing(self):
        for p in P_VALUES:
            s = Diminishing.from_loss_exponent(1.0, p)
            vals = [alpha(s, k) for k in range(50)]
            assert all(a >= b for a, b in pairwise(vals))

    def test_label_records_the_exponent(self):
        label = Diminishing.from_loss_exponent(1e-3, 1.5).label()
        assert "p=1.5" in label and "1/p" in label

    def test_plain_diminishing_label_omits_the_exponent(self):
        assert "p=" not in Diminishing(alpha0=1e-3, power=0.5).label()

    def test_rejects_nonpositive_p(self):
        with pytest.raises(ValueError, match="must be positive"):
            Diminishing.from_loss_exponent(1.0, 0.0)


class TestExponentDiminishingGrid:
    def test_grid_pins_the_power_and_varies_alpha0(self):
        grid = exponent_diminishing_grid(1.5)
        assert len({s.power for s in grid}) == 1
        assert len({s.alpha0 for s in grid}) == len(grid)
        assert all(s.loss_exponent == 1.5 for s in grid)


class TestOtherSchedules:
    def test_constant(self):
        s = Constant(0.3)
        assert alpha(s, 0) == alpha(s, 1000) == 0.3

    def test_geometric_from_total_decay_hits_the_target(self):
        K = 1000
        s = Geometric.from_total_decay(1.0, 1e-3, K)
        assert alpha(s, 0) == pytest.approx(1.0)
        assert alpha(s, K) == pytest.approx(1e-3)

    @pytest.mark.parametrize("bad", [0.0, -1.0, 1.5])
    def test_geometric_rejects_bad_total_decay(self, bad: float):
        with pytest.raises(ValueError, match="total_decay"):
            Geometric.from_total_decay(1.0, bad, 100)
