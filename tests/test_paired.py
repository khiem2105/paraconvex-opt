"""Checks for the paired-comparison statistics."""

from __future__ import annotations

import numpy as np
import pytest

from paraconvex.paired import paired_stats


def constant_offset(offset: float, n: int = 20, base: float = 0.05):
    """An arm that beats the baseline by exactly ``-offset`` in every replication."""
    rng = np.random.default_rng(0)
    baseline = base + rng.normal(0, 0.01, n)  # replication difficulty varies...
    return {2.0: baseline, 1.5: baseline - offset}  # ...but cancels in the difference


class TestPairedStats:
    def test_recovers_a_constant_offset(self):
        table = paired_stats(constant_offset(0.002), baseline_p=2.0)
        row = table.iloc[0]
        assert row["p"] == 1.5
        assert row["median_diff"] == pytest.approx(-0.002)
        assert row["wins"] == row["n"] == 20
        assert row["significant"]

    def test_baseline_is_excluded_from_the_rows(self):
        table = paired_stats(constant_offset(0.001), baseline_p=2.0)
        assert 2.0 not in set(table["p"])
        assert len(table) == 1

    def test_detects_an_effect_that_marginal_spread_would_hide(self):
        """The motivating case: between-replication variation swamps the effect.

        Difficulty varies by ~0.01 across replications while the arm's advantage
        is 0.0005 -- marginal distributions overlap almost completely, yet the
        paired difference is unambiguous.
        """
        data = constant_offset(0.0005)
        spread = data[2.0].max() - data[2.0].min()
        effect = 0.0005
        assert spread > 10 * effect, "test setup should have difficulty dominate"

        row = paired_stats(data, baseline_p=2.0).iloc[0]
        assert row["significant"]
        assert row["wins"] == 20

    def test_no_effect_is_not_flagged(self):
        rng = np.random.default_rng(1)
        base = 0.05 + rng.normal(0, 0.01, 40)
        arm = base + rng.normal(0, 0.01, 40)  # independent noise, zero true effect
        assert not paired_stats({2.0: base, 1.5: arm}, baseline_p=2.0).iloc[0]["significant"]

    def test_sign_convention_negative_is_better(self):
        worse = paired_stats(constant_offset(-0.003), baseline_p=2.0).iloc[0]
        assert worse["median_diff"] > 0
        assert worse["wins"] == 0

    def test_identical_arms_do_not_crash(self):
        """Wilcoxon is undefined on all-zero differences; must degrade gracefully."""
        base = np.full(10, 0.05)
        row = paired_stats({2.0: base, 1.5: base.copy()}, baseline_p=2.0).iloc[0]
        assert row["median_diff"] == 0.0
        assert row["wilcoxon_p"] == 1.0
        assert not row["significant"]

    def test_ci_brackets_the_point_estimate(self):
        row = paired_stats(constant_offset(0.002), baseline_p=2.0).iloc[0]
        assert row["ci_low"] <= row["median_diff"] <= row["ci_high"]

    def test_rows_are_ordered_by_descending_p(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0.05, 0.01, 15)
        vals = {2.0: base, 1.75: base - 1e-3, 1.5: base - 2e-3, 1.25: base - 3e-3}
        assert list(paired_stats(vals, baseline_p=2.0)["p"]) == [1.75, 1.5, 1.25]


class TestValidation:
    def test_missing_baseline(self):
        with pytest.raises(KeyError, match="baseline"):
            paired_stats({1.5: np.zeros(5)}, baseline_p=2.0)

    def test_mismatched_replication_counts(self):
        with pytest.raises(ValueError, match="paired comparison requires"):
            paired_stats({2.0: np.zeros(5), 1.5: np.zeros(6)}, baseline_p=2.0)

    def test_rejects_non_1d_input(self):
        with pytest.raises(ValueError, match="one value per replication"):
            paired_stats({2.0: np.zeros((5, 2)), 1.5: np.zeros((5, 2))}, baseline_p=2.0)

    def test_is_reproducible(self):
        a = paired_stats(constant_offset(0.001), baseline_p=2.0, seed=7)
        b = paired_stats(constant_offset(0.001), baseline_p=2.0, seed=7)
        assert a.equals(b)
