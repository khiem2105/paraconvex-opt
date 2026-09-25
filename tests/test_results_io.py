"""Checks that a finished run can be replotted without re-running it."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from paraconvex.plotting import (
    METHOD_LINTHRESH,
    plot_paired_difference_curve,
)
from paraconvex.results_io import (
    TRAJECTORY_FILENAME,
    load_trajectories,
    save_trajectories,
    truncate_iterations,
)


def fake_results(methods=("subgradient", "prox-linear"), exponents=(2.0, 1.5),
                 T: int = 7, R: int = 4):
    iterations = torch.arange(T) * 10
    rng = torch.Generator().manual_seed(0)
    return {
        (m, p): {
            "iterations": iterations,
            "distance": torch.rand(T, R, generator=rng, dtype=torch.float64),
            "gap": torch.rand(T, R, generator=rng, dtype=torch.float64),
        }
        for m in methods
        for p in exponents
    }


class TestRoundTrip:
    def test_recovers_every_series_exactly(self, tmp_path):
        original = fake_results()
        save_trajectories(original, tmp_path)
        loaded = load_trajectories(tmp_path)

        assert set(loaded) == set(original)
        for key, entry in original.items():
            for quantity in ("iterations", "distance", "gap"):
                torch.testing.assert_close(
                    loaded[key][quantity], entry[quantity], rtol=0, atol=0
                )

    def test_accepts_a_directory_or_the_file(self, tmp_path):
        save_trajectories(fake_results(), tmp_path)
        by_dir = load_trajectories(tmp_path)
        by_file = load_trajectories(tmp_path / TRAJECTORY_FILENAME)
        assert set(by_dir) == set(by_file)

    def test_method_names_with_hyphens_survive(self, tmp_path):
        """`prox-linear` contains a hyphen; the separator must not be one."""
        save_trajectories(fake_results(methods=("prox-linear",)), tmp_path)
        assert {m for m, _ in load_trajectories(tmp_path)} == {"prox-linear"}

    @pytest.mark.parametrize("p", [2.0, 1.75, 1.25, 1.0625])
    def test_exponent_round_trips_through_the_key(self, tmp_path, p: float):
        save_trajectories(fake_results(exponents=(p,)), tmp_path)
        assert {q for _, q in load_trajectories(tmp_path)} == {p}

    def test_single_series(self, tmp_path):
        save_trajectories(fake_results(methods=("subgradient",), exponents=(2.0,)),
                          tmp_path)
        assert set(load_trajectories(tmp_path)) == {("subgradient", 2.0)}


class TestValidation:
    def test_empty_results_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to save"):
            save_trajectories({}, tmp_path)

    def test_mismatched_record_schedules_rejected(self, tmp_path):
        """Series must be on one schedule or the shared `iterations` is a lie."""
        results = fake_results(methods=("subgradient",), exponents=(2.0, 1.5))
        results[("subgradient", 1.5)]["iterations"] = torch.arange(7) * 99
        with pytest.raises(ValueError, match="one record schedule"):
            save_trajectories(results, tmp_path)

    def test_separator_in_method_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="may not contain"):
            save_trajectories(fake_results(methods=("a|b",)), tmp_path)

    def test_missing_file_explains_itself(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="written by the experiment"):
            load_trajectories(tmp_path)

    def test_incomplete_file_rejected(self, tmp_path):
        """An npz missing a *required* quantity must fail loudly, not half-plot."""
        np.savez_compressed(
            tmp_path / TRAJECTORY_FILENAME,
            iterations=np.arange(5),
            **{"gap|subgradient|p=2": np.zeros((5, 3))},
        )
        with pytest.raises(ValueError, match="missing quantities"):
            load_trajectories(tmp_path)

    def test_optional_quantity_may_be_absent(self, tmp_path):
        """`objective` is optional: runs predating it still load and replot.

        Only figures that actually need it should fail, and they raise their own
        error naming the quantity -- see `_plot_quantity`.
        """
        np.savez_compressed(
            tmp_path / TRAJECTORY_FILENAME,
            iterations=np.arange(5),
            **{
                "distance|subgradient|p=2": np.zeros((5, 3)),
                "gap|subgradient|p=2": np.zeros((5, 3)),
            },
        )
        loaded = load_trajectories(tmp_path)
        assert "objective" not in loaded["subgradient", 2.0]


class TestTruncateIterations:
    """Slicing to a shorter horizon, used by `replot.py --max-iterations`."""

    @staticmethod
    def build(n_records=7, step=5000):
        it = torch.arange(0, n_records * step, step)
        return {
            ("subgradient", 2.0): {
                "iterations": it,
                "distance": torch.arange(n_records * 3, dtype=torch.float64).reshape(
                    n_records, 3
                ),
            }
        }

    def test_keeps_records_up_to_and_including_the_cutoff(self):
        out = truncate_iterations(self.build(), 15000)
        entry = out["subgradient", 2.0]
        assert entry["iterations"].tolist() == [0, 5000, 10000, 15000]
        assert entry["distance"].shape == (4, 3)

    def test_slices_every_quantity_along_the_record_axis(self):
        """A quantity left at full length would silently misalign with the axis."""
        out = truncate_iterations(self.build(), 10000)
        entry = out["subgradient", 2.0]
        assert all(
            v.shape[0] == entry["iterations"].shape[0] for v in entry.values()
        )

    def test_cutoff_beyond_the_run_is_a_no_op(self):
        original = self.build()
        out = truncate_iterations(original, 10**9)
        assert torch.equal(
            out["subgradient", 2.0]["iterations"],
            original["subgradient", 2.0]["iterations"],
        )

    def test_cutoff_before_the_first_record_is_rejected(self):
        """Silently emitting an empty figure would be worse than failing."""
        data = {("m", 2.0): {"iterations": torch.tensor([100, 200])}}
        with pytest.raises(ValueError, match="keeps no records"):
            truncate_iterations(data, 50)

    def test_negative_cutoff_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            truncate_iterations(self.build(), -1)


class TestPairedLinthresh:
    """The published figures depend on a fixed threshold for one method.

    These pin the behaviour so a plain re-run reproduces the paper figures
    without anyone having to remember a command-line flag.
    """

    @staticmethod
    def fake():
        """Two arms near 1e-3, plus one two orders of magnitude off scale.

        This is the shape that breaks the automatic rule: the quantile follows
        the off-scale arm and buries the other two.
        """
        T, R = 40, 8
        it = torch.arange(T) * 100
        rng = torch.Generator().manual_seed(0)

        def small():
            return 1e-3 * torch.rand(T, R, generator=rng, dtype=torch.float64)

        return {
            2.0: {"iterations": it, "distance": small()},
            1.5: {"iterations": it, "distance": small()},
            1.25: {
                "iterations": it,
                "distance": 3e-1 * torch.rand(T, R, generator=rng,
                                              dtype=torch.float64),
            },
        }

    def applied_linthresh(self, tmp_path, **kwargs) -> float:
        """The ``linthresh`` the figure was actually drawn with.

        Captured from the ``set_yscale`` call rather than asserted against the
        constant, so the test covers the wiring and not just the table.
        """
        from matplotlib.axes import Axes

        seen = {}
        original = Axes.set_yscale

        def spy(self, value, **kw):
            if value == "symlog":
                seen["linthresh"] = kw["linthresh"]
            return original(self, value, **kw)

        with mock.patch.object(Axes, "set_yscale", spy):
            plot_paired_difference_curve(
                self.fake(), tmp_path / "c.png", baseline_p=2.0, **kwargs
            )
        return seen["linthresh"]

    def test_subgradient_uses_the_pinned_value(self, tmp_path):
        assert self.applied_linthresh(tmp_path, method="subgradient") == 1e-3

    def test_prox_linear_follows_the_automatic_rule(self, tmp_path):
        """Not pinned: forcing it smaller inflates the band of a near-zero arm."""
        assert "prox-linear" not in METHOD_LINTHRESH
        assert self.applied_linthresh(tmp_path, method="prox-linear") > 1e-2

    def test_the_pin_is_what_rescues_the_small_arms(self, tmp_path):
        """The automatic rule really would bury them -- that is why the pin exists."""
        auto = self.applied_linthresh(tmp_path, method="prox-linear")
        assert auto / METHOD_LINTHRESH["subgradient"] > 10

    def test_explicit_argument_beats_the_pin(self, tmp_path):
        got = self.applied_linthresh(
            tmp_path, method="subgradient", linthresh=5e-2
        )
        assert got == 5e-2

    def test_both_formats_are_written(self, tmp_path):
        out = plot_paired_difference_curve(
            self.fake(), tmp_path / "d.png", baseline_p=2.0, method="subgradient"
        )
        assert out.exists() and out.with_suffix(".pdf").exists()

    def test_legacy_key_format_explains_itself(self, tmp_path):
        """Pre-multi-method runs keyed arrays as `distance_p2`, with no method.

        The loader must say so rather than failing on a tuple unpack, so a batch
        replot can skip one stale run instead of aborting.
        """
        np.savez_compressed(
            tmp_path / TRAJECTORY_FILENAME,
            iterations=np.arange(5),
            distance_p2=np.zeros((5, 3)),
        )
        with pytest.raises(ValueError, match="legacy key format"):
            load_trajectories(tmp_path)
