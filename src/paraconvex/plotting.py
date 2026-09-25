"""Figures for the weakly-convex versus paraconvex comparison.

Two conventions are load-bearing here.

*Median and interquartile band, never mean and standard deviation.*  The noise
is Student-t with a deliberately low ``dof``; the sample mean of a heavy-tailed
ensemble is dominated by its worst replication and the sample standard deviation
may not even estimate anything finite.  The median and the 25/75 quantiles are
the honest summaries.

*Series colours are ordered, not categorical.*  ``p`` is an ordered quantity
(``nu = p - 1`` decreases along the series), so the palette runs cool-to-warm to
carry that ordering.  It was validated for colour-vision deficiency rather than
chosen by eye: worst-case normal-vision separation 23.1, worst-case dichromatic
separation 16.1 (OKLab dE x100, floors 15 and 8), minimum contrast 3.20 against
white.  Lines are solid throughout: the exponent is carried by colour and the
method by marker shape, which was judged sufficient without a third channel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from torch import Tensor

from paraconvex.paired import paired_stats

__all__ = [
    "METHOD_LINTHRESH",
    "METHOD_MARKERS",
    "SERIES_COLORS",
    "paired_difference_table",
    "plot_estimator_effects",
    "plot_legend",
    "plot_optimization_gap",
    "plot_paired_difference_curve",
    "plot_paired_difference_points",
    "plot_recovery_error",
    "style_for",
]

#: Cool-to-warm ordered ramp, validated for CVD (see module docstring).
SERIES_COLORS = ["#1b3a6b", "#2a78d6", "#eb6834", "#8c2d04"]

_GRID = {"color": "#d5d5d2", "linewidth": 0.6, "alpha": 0.9}


def _compact(value: float, _pos: int | None = None) -> str:
    """``30000 -> "30k"``; zero and sub-thousand values are left alone."""
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


ITERATION_FORMATTER = FuncFormatter(_compact)

#: Below this horizon the plain numbers are short enough, and abbreviating only
#: some of them would mix units on one axis ("800, 1k, 1.2k").
COMPACT_TICKS_ABOVE = 10_000


def _format_iteration_axis(ax: Axes, k: np.ndarray) -> None:
    """Abbreviate the iteration axis, but only when the run is long enough."""
    if k.max() >= COMPACT_TICKS_ABOVE:
        ax.xaxis.set_major_formatter(ITERATION_FORMATTER)


def style_for(index: int) -> str:
    """Colour for the ``index``-th exponent, assigned in fixed order."""
    if index >= len(SERIES_COLORS):
        raise ValueError(
            f"palette holds {len(SERIES_COLORS)} validated slots; series {index} "
            "would need a re-validated ramp rather than a recycled hue"
        )
    return SERIES_COLORS[index]


def _median_band(values: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band over the replication axis of a ``(T, R)``."""
    arr = values.detach().cpu().numpy()
    return (
        np.median(arr, axis=1),
        np.quantile(arr, 0.25, axis=1),
        np.quantile(arr, 0.75, axis=1),
    )


def _draw_series(
    ax: Axes,
    iterations: Tensor,
    values: Tensor,
    label: str,
    index: int,
    floor: float = 1e-16,
) -> None:
    color = style_for(index)
    k = iterations.detach().cpu().numpy()
    median, lo, hi = _median_band(values)

    # Clamp for the log axis: an exactly-zero gap is possible when an iterate
    # attains the reference minimum, and would silently drop the point.
    median = np.maximum(median, floor)
    lo = np.maximum(lo, floor)
    hi = np.maximum(hi, floor)

    ax.fill_between(k, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(k, median, color=color, linewidth=1.8, label=label)


def _finish(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, which="major", **_GRID)
    ax.grid(True, which="minor", color="#e8e8e6", linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8a86")
    ax.tick_params(colors="#52514e", labelsize=9)


# ---------------------------------------------------------------------------
# Paired differences
#
# The experiment is a paired design: every exponent is run on identical data, so
# replication r differs from replication r' mainly in how hard its draw happens
# to be.  That between-replication variation is an order of magnitude larger
# than the effect of p, and it is *common to all arms*, so marginal summaries
# (medians with interquartile bands) hide the comparison rather than showing it:
# two arms can have near-identical marginal spreads while one wins on nearly
# every individual instance.
#
# Subtracting within a replication cancels the shared difficulty and leaves the
# effect of p alone.  These are the plots that actually answer "is this exponent
# better than the baseline on the same data?".
# ---------------------------------------------------------------------------

_ZERO_LINE = {"color": "#52514e", "linewidth": 1.0, "linestyle": (0, (4, 3))}


def _bootstrap_median_band(
    diff: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise bootstrap CI for the median paired difference; ``diff`` is ``(T, R)``.

    Replications are resampled with the *same* indices at every iteration, so a
    replication enters or leaves a bootstrap sample as a whole trajectory.  That
    is the right unit: the uncertainty being quantified is "which 20 datasets
    did I happen to draw", not independent noise at each recorded step.
    """
    rng = np.random.default_rng(seed)
    R = diff.shape[1]
    idx = rng.integers(0, R, size=(n_boot, R))          # (B, R)
    resampled = diff[:, idx]                             # (T, B, R)
    medians = np.median(resampled, axis=2)               # (T, B)
    lo = np.quantile(medians, alpha / 2, axis=1)
    hi = np.quantile(medians, 1 - alpha / 2, axis=1)
    return np.median(diff, axis=1), lo, hi


def paired_difference_table(
    results: dict[float, dict[str, Tensor]], baseline_p: float = 2.0, seed: int = 0
) -> pd.DataFrame:
    """Final-iteration paired statistics against ``baseline_p``, one row per arm."""
    return paired_stats(
        {p: r["distance"][-1].detach().cpu().numpy() for p, r in results.items()},
        baseline_p=baseline_p,
        seed=seed,
    )


def _paired_axis(ax: Axes) -> None:
    """Shared furniture for a paired-difference axis: zero line, grid, spines."""
    ax.axhline(0.0, **_ZERO_LINE, zorder=1)
    ax.grid(True, which="major", **_GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8a86")
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_paired_difference_curve(
    results: dict[float, dict[str, Tensor]],
    output: Path,
    baseline_p: float,
    method: str,
    exponents: list[float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    figsize: tuple[float, float] = (5.2, 4.0),
    linthresh: float | None = None,
) -> Path:
    """Median paired difference against ``baseline_p`` over iterations.

    Below zero means the exponent beats the baseline on the same data; a
    bootstrap band clear of zero means the effect survives resampling.

    ``results`` is keyed by exponent alone -- one method at a time -- so
    ``method`` names which marker to draw.

    ``linthresh`` sizes symlog's linear zone, and with it how the vertical space
    is split between the decades.  Left ``None`` it follows the default rule
    below.  Pass a value for a panel whose default split is lopsided: when one
    arm runs an order of magnitude above the others, the rule returns a
    threshold near that arm's scale, which spends the axis on the top decade and
    compresses everything under it -- including zero -- into a sliver.
    """
    if baseline_p not in results:
        raise KeyError(f"baseline p={baseline_p} not among {sorted(results)}")
    arms = [p for p in sorted(results, reverse=True) if p != baseline_p]
    if not arms:
        raise ValueError("need at least one arm besides the baseline")

    order = _exponent_order({(method, p): v for p, v in results.items()}, exponents)
    marker = _marker_for(method)
    base = results[baseline_p]["distance"].detach().cpu().numpy()
    k = results[baseline_p]["iterations"].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    spans = []
    for p in arms:
        color = style_for(order.index(p))
        diff = results[p]["distance"].detach().cpu().numpy() - base
        med, lo, hi = _bootstrap_median_band(diff, n_boot=n_boot, seed=seed)
        spans.append(np.abs(med))
        ax.fill_between(k, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.plot(k, med, color=color, linewidth=1.6, marker=marker, markersize=5,
                markevery=max(1, len(k) // 11), markeredgecolor="white",
                markeredgewidth=0.6)

    # Signed-log: an arm that misbehaves early can overshoot its eventual
    # difference many-fold and would otherwise flatten the region that matters.
    # A method with a settled value in METHOD_LINTHRESH uses it; the rest fall
    # back on the quantile rule.
    if linthresh is None:
        linthresh = METHOD_LINTHRESH.get(
            method, max(float(np.quantile(np.concatenate(spans), 0.75)), 1e-5)
        )
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.1)

    _paired_axis(ax)
    _format_iteration_axis(ax, k)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(_paired_ylabel(method, baseline_p, indexed_by_time=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_paired_difference_points(
    results: dict[float, dict[str, Tensor]],
    output: Path,
    baseline_p: float,
    method: str,
    exponents: list[float] | None = None,
    seed: int = 0,
    figsize: tuple[float, float] = (4.2, 4.0),
) -> Path:
    """Final paired difference, one point per replication.

    Shows how *consistent* the sign is, which a median and interval cannot: an
    arm winning narrowly on every replication and one winning hugely on half
    look alike in summary and quite different here.
    """
    if baseline_p not in results:
        raise KeyError(f"baseline p={baseline_p} not among {sorted(results)}")

    all_p = sorted(results, reverse=True)
    order = _exponent_order({(method, p): v for p, v in results.items()}, exponents)
    marker = _marker_for(method)
    base = results[baseline_p]["distance"][-1].detach().cpu().numpy()
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    for i, p in enumerate(all_p):
        if p == baseline_p:
            continue
        color = style_for(order.index(p))
        final = results[p]["distance"][-1].detach().cpu().numpy() - base
        jitter = rng.uniform(-0.13, 0.13, size=final.size)
        ax.scatter(np.full_like(final, i) + jitter, final, s=26, color=color,
                   marker=marker, alpha=0.75, linewidth=0.5,
                   edgecolor="white", zorder=3)
        ax.hlines(np.median(final), i - 0.28, i + 0.28, color=color,
                  linewidth=2.4, zorder=4)

    _paired_axis(ax)
    ax.set_xticks(range(len(all_p)))
    ax.set_xticklabels(
        ["baseline" if p == baseline_p else f"$p={p:g}$" for p in all_p], fontsize=9
    )
    ax.set_xlim(-0.6, len(all_p) - 0.4)
    ax.set_ylabel(_paired_ylabel(method, baseline_p, indexed_by_time=False))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_estimator_effects(
    table: pd.DataFrame,
    output: Path,
    baseline_p: float = 2.0,
) -> Path:
    """Paired estimator effect across a (dof, noise_level) grid.

    One panel per ``dof``; within a panel, the median paired difference against
    the baseline versus noise level, with bootstrap CI whiskers and a zero
    reference line.  Below zero means the exponent recovers ``x*`` more
    accurately than the baseline *on the same data*.

    A forest-plot style rather than a heatmap: the CI is the whole point here --
    the sign of an effect means nothing without knowing whether the interval
    clears zero -- and a heatmap cell cannot show an interval.
    """
    dofs = sorted(table["dof"].unique())
    arms = sorted(table["p"].unique(), reverse=True)
    all_p = sorted(set(arms) | {baseline_p}, reverse=True)

    fig, axes = plt.subplots(
        1, len(dofs), figsize=(1 + 4.4 * len(dofs), 4.4), sharey=True, squeeze=False
    )

    for ax, dof in zip(axes[0], dofs, strict=True):
        sub = table[table["dof"] == dof]
        levels = sorted(sub["noise_level"].unique())
        for p in arms:
            color = style_for(all_p.index(p))
            rows = sub[sub["p"] == p].sort_values("noise_level")
            xs = np.arange(len(levels), dtype=float)
            xs = xs + 0.16 * (arms.index(p) - (len(arms) - 1) / 2)  # dodge
            med = rows["median_diff"].to_numpy()
            err = np.vstack(
                [med - rows["ci_low"].to_numpy(), rows["ci_high"].to_numpy() - med]
            )
            ax.errorbar(
                xs, med, yerr=err, fmt="o", color=color, markersize=6,
                capsize=3, linewidth=1.6, markeredgecolor="white",
                markeredgewidth=0.6, alpha=0.95,
                label=rf"$p={p:g}$",
            )
        ax.axhline(0.0, **_ZERO_LINE, zorder=1)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.set_xlabel("noise level")
        ax.set_title(
            ("Cauchy " if np.isclose(dof, 1.0) else "") + f"dof = {dof:g}",
            fontsize=11, pad=8,
        )
        ax.grid(True, which="major", **_GRID)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8a8a86")
        ax.tick_params(colors="#52514e", labelsize=9)

    axes[0][0].set_ylabel(
        rf"$\mathrm{{dist}}(\hat{{x}}_p) - \mathrm{{dist}}(\hat{{x}}_{{p={baseline_p:g}}})$"
    )
    fig.subplots_adjust(bottom=0.26, wspace=0.08)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Accuracy of the minimizer $\\hat{{x}}_p$ relative to the $p={baseline_p:g}$ "
        "baseline, on identical data; below zero is better",
        fontsize=10.5, y=1.02,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Convergence curves
#
# Two encodings, read independently:
#
#   colour  ->  the exponent p
#   marker  ->  the method (subgradient / prox-linear)
#
# Keeping them orthogonal means a reader can answer "which p?" and "which
# method?" separately rather than decoding a combined key. All lines are solid;
# the palette is CVD-validated (see below), so colour alone carries p.
#
# Each quantity gets its own file and the legend gets a third, so the figures
# can be placed independently in a paper. No titles: captions belong in LaTeX,
# and a title baked into the image duplicates them.
# ---------------------------------------------------------------------------

#: Marker per method, assigned in fixed order.
METHOD_MARKERS = {"subgradient": "o", "prox-linear": "s"}

#: LaTeX superscript naming each method's iterate in axis labels, matching the
#: notation of the paper: x^dagger for the subgradient method, x^natural for
#: prox-linear.  Distinct symbols let a paired difference be read without
#: consulting the caption to learn which method produced it.
METHOD_ITERATE_SYMBOL = {"subgradient": r"\dagger", "prox-linear": r"\natural"}

#: Fixed symlog linear-threshold for a method's paired-difference panel, used in
#: place of the automatic rule.  A method appears here only when the automatic
#: rule misbehaves for it, and the entry records the settled value so a plain
#: re-run reproduces the published figure without anyone remembering a flag.
#:
#: ``subgradient``: at ``p=1.25`` this method runs to ~3e-1, two orders above the
#: other arms, and the automatic rule (a high quantile of ``|median|``) follows
#: it up to ~1.5e-2.  Everything under that lands in symlog's linear zone,
#: including the ~1e-3 endgame that separates the two noise settings -- kappa=1
#: finishes below zero, kappa=2.5 at or above it.  Pinning 1e-3 puts that gap at
#: ~21% of the axis instead of ~2%.  Going lower inflates the zero-straddling
#: confidence band faster than it grows the gap.
#:
#: ``prox-linear`` is deliberately absent: ``p=1.25`` stays on scale there, the
#: automatic rule already returns ~1e-3 to 1e-2, and forcing it smaller blows up
#: the band of an arm whose interval genuinely contains zero.
METHOD_LINTHRESH: dict[str, float] = {"subgradient": 1e-3}

SeriesKey = tuple[str, float]


def _marker_for(method: str) -> str:
    try:
        return METHOD_MARKERS[method]
    except KeyError:
        raise KeyError(
            f"no marker assigned to method {method!r}; "
            f"known: {sorted(METHOD_MARKERS)}"
        ) from None


def _paired_ylabel(method: str, baseline_p: float, indexed_by_time: bool) -> str:
    """Axis label for a paired distance difference against the baseline.

    ``indexed_by_time`` adds the iteration subscript ``t``: the curve figure
    plots the difference at every iterate, whereas the per-replication figure
    shows only the final one, where a ``t`` would be misleading.
    """
    try:
        sym = METHOD_ITERATE_SYMBOL[method]
    except KeyError:
        raise KeyError(
            f"no iterate symbol assigned to method {method!r}; "
            f"known: {sorted(METHOD_ITERATE_SYMBOL)}"
        ) from None
    t = ",t" if indexed_by_time else ""
    return (
        rf"$\mathrm{{dist}}(x^{{{sym}}}_{{p{t}}}, x^*)"
        rf" - \mathrm{{dist}}(x^{{{sym}}}_{{{baseline_p:g}{t}}}, x^*)$"
    )


def _exponent_order(
    results: dict[SeriesKey, dict[str, Tensor]],
    exponents: list[float] | None = None,
) -> list[float]:
    """Exponents in fixed descending order; colours are assigned by position.

    Pass ``exponents`` to pin that assignment when plotting a *subset*.  Colour
    follows the exponent, not its rank within whatever happens to be plotted, so
    a figure showing only ``p in {2, 1.25}`` must still colour them slot 0 and
    slot 3 -- otherwise the same ``p`` changes colour between figures in the
    same paper.  Filtering by *method* is safe without this, since the exponents
    present do not change.
    """
    present = sorted({p for _, p in results}, reverse=True)
    if exponents is None:
        return present
    order = sorted(set(exponents), reverse=True)
    missing = set(present) - set(order)
    if missing:
        raise ValueError(
            f"`exponents` must cover everything plotted; missing {sorted(missing)}"
        )
    return order


def _plot_quantity(
    results: dict[SeriesKey, dict[str, Tensor]],
    key: str,
    ylabel: str,
    output: Path,
    figsize: tuple[float, float] = (5.2, 4.0),
    floor: float = 1e-16,
    n_markers: int = 11,
    exponents: list[float] | None = None,
) -> Path:
    """One quantity, one file, no title."""
    absent = sorted({f"{m} p={p:g}" for (m, p), e in results.items() if key not in e})
    if absent:
        raise KeyError(
            f"no {key!r} recorded for: {absent}. Runs written before this "
            f"quantity was persisted only carry "
            f"{sorted(set().union(*(set(e) for e in results.values())) - {'iterations'})}; "
            "re-run the experiment to record it."
        )

    exponents = _exponent_order(results, exponents)
    fig, ax = plt.subplots(figsize=figsize)

    for (method, p), entry in sorted(
        results.items(), key=lambda kv: (kv[0][0], -kv[0][1])
    ):
        color = style_for(exponents.index(p))
        k = entry["iterations"].detach().cpu().numpy()
        median = np.maximum(_median_band(entry[key])[0], floor)
        ax.plot(
            k,
            median,
            color=color,
            linewidth=1.6,
            marker=_marker_for(method),
            markersize=5,
            markevery=max(1, len(k) // n_markers),
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(ylabel)
    _format_iteration_axis(ax, k)
    ax.grid(True, which="major", **_GRID)
    ax.grid(True, which="minor", color="#e8e8e6", linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8a86")
    ax.tick_params(colors="#52514e", labelsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_optimization_gap(
    results: dict[SeriesKey, dict[str, Tensor]], output: Path, **kwargs
) -> Path:
    """Raw full-sample ``f(x_t)`` against iteration.

    Deliberately *not* the gap ``f(x_t) - f_min``: the reference minimum is only
    an estimate, obtained from a handful of full-batch descents, and subtracting
    it puts that estimate's error into every curve.  Plotting the objective
    itself keeps the figure free of any quantity the code had to guess at.  The
    curves therefore flatten at each arm's own minimum rather than descending
    indefinitely.

    One caveat for reading it: ``f_p`` is a *different function* for each ``p``,
    so the vertical offsets between exponents are not a performance comparison.
    Comparing methods at a fixed ``p`` is meaningful; comparing levels across
    ``p`` is not.
    """
    return _plot_quantity(results, "objective", "Function value", output, **kwargs)


def plot_recovery_error(
    results: dict[SeriesKey, dict[str, Tensor]], output: Path, **kwargs
) -> Path:
    """Distance to the truth against iteration; plateaus at the statistical floor.

    The axis reads "Distance to $x^*$" while the quantity is
    ``min(||x_t - x*||, ||x_t + x*||) / ||x*||``.  The normalisation is a no-op
    numerically (``x*`` is drawn with unit norm), but the minimum over ``+-x*``
    is substantive -- phase retrieval determines ``x*`` only up to sign -- and
    the short label leaves it to the caption.
    """
    return _plot_quantity(results, "distance", r"Distance to $x^*$", output, **kwargs)


def plot_legend(
    results: dict[SeriesKey, dict[str, Tensor]],
    output: Path,
    ncol: int | None = None,
    exponents: list[float] | None = None,
    fontsize: float = 13.0,
    handlelength: float = 1.8,
) -> Path:
    """Standalone legend for the curve figures.

    Split into two blocks -- one for the exponent, one for the method -- because
    the encoding is two independent factors.  A combined legend would need one
    entry per (method, p) pair, which grows multiplicatively and obscures that
    colour and marker mean different things.

    Each block's handle shows only the channel it encodes.  Exponent entries are
    a coloured line, since colour is what identifies ``p``.  Method entries are a
    **bare marker with no line through it**: the method is encoded by marker
    shape alone, so a line in that handle would suggest a line style that
    carries no meaning here.

    ``fontsize`` is sized for print rather than for screen -- these are dropped
    into a paper at a fraction of their rendered width, so legend text set at a
    typical on-screen 9-10pt ends up unreadably small on the page.  Marker and
    line sizes track it so the handles stay legible at the same scale.

    ``handlelength`` applies to *every* entry, and a marker is centred in its
    handle box, so a length chosen for the line entries strands the bare markers
    in dead space.  1.8 is the compromise: still long enough to read a line's
    colour, short enough to keep each marker beside its own label.

    Emitted as its own file so a paper can place one legend beside a row of
    subfigures instead of repeating it in each.
    """
    order = _exponent_order(results, exponents)
    present = sorted({p for _, p in results}, reverse=True)
    methods = sorted({m for m, _ in results})
    scale = fontsize / 9.5

    handles, labels = [], []
    for p in present:
        handles.append(
            Line2D([], [], color=style_for(order.index(p)), linewidth=1.8 * scale)
        )
        labels.append(rf"$p={p:g}$")
    for method in methods:
        handles.append(
            Line2D(
                [], [], color="#52514e", linestyle="none",
                marker=_marker_for(method), markersize=6 * scale,
                markeredgecolor="white", markeredgewidth=0.6 * scale,
            )
        )
        labels.append(method)

    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol or len(handles),
        frameon=False,
        fontsize=fontsize,
        handlelength=handlelength,
        handletextpad=0.7,
        columnspacing=1.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output
