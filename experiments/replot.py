"""Redraw the figures for a finished run, without re-running it.

Everything the curve figures need is stored in ``trajectories.npz``, so a change
to styling, labelling or layout costs a second rather than the minutes an
experiment takes.

Usage
-----
    uv run python experiments/replot.py results/cauchy_nl0.01_long
    uv run python experiments/replot.py results/* --into figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paraconvex.plotting import (
    paired_difference_table,
    plot_legend,
    plot_optimization_gap,
    plot_paired_difference_curve,
    plot_paired_difference_points,
    plot_recovery_error,
)
from paraconvex.results_io import (
    TRAJECTORY_FILENAME,
    load_trajectories,
    truncate_iterations,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "runs", type=Path, nargs="+",
        help=f"run directories, each containing {TRAJECTORY_FILENAME}",
    )
    ap.add_argument(
        "--into", type=Path, default=None,
        help="write figures here instead of back into each run directory; "
             "filenames are prefixed with the run name to avoid collisions",
    )
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument(
        "--method", nargs="+", default=None,
        help="plot only these methods (default: all present)",
    )
    ap.add_argument(
        "--p", type=float, nargs="+", default=None,
        help="plot only these exponents. Colours stay pinned to the run's full "
             "set, so a subset figure keeps the same colour per p",
    )
    ap.add_argument(
        "--linthresh", type=float, default=None,
        help=(
            "override symlog's linear-threshold on the paired-curve figures. "
            "Use with --paired-only --method M to retune a single panel whose "
            "default split is lopsided, without touching the others"
        ),
    )
    ap.add_argument(
        "--paired-only", action="store_true",
        help=(
            "write only the paired figures. Pair with --method to retune one "
            "panel: without it, --method would also redraw the shared "
            "optimization-gap and recovery figures using that method alone"
        ),
    )
    ap.add_argument(
        "--max-iterations", type=int, default=None,
        help=(
            "drop records past this iteration. The data is sliced, not just the "
            "axis clipped, so the y-range fits what is shown. This also moves "
            "the last record, so the paired-final figure reports the iterate at "
            "this cut-off rather than at the end of the run -- and those numbers "
            "will then differ from summary.csv, which is written for the full run"
        ),
    )
    ap.add_argument(
        "--skip-paired", action="store_true",
        help="curve figures only; skip the paired-difference figures",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    for run in args.runs:
        if not (run / TRAJECTORY_FILENAME).exists():
            print(f"[skip] {run}: no {TRAJECTORY_FILENAME}")
            continue

        try:
            results = load_trajectories(run)
        except ValueError as exc:
            # A legacy or malformed file should cost one run, not the batch --
            # `replot.py results/*` is the normal way to redraw everything.
            print(f"[skip] {run}: {exc}")
            continue
        # Pin colours to everything in the file, before any filtering.
        all_exponents = sorted({p for _, p in results}, reverse=True)
        if args.method:
            results = {k: v for k, v in results.items() if k[0] in args.method}
        if args.p:
            keep = {float(v) for v in args.p}
            results = {k: v for k, v in results.items() if k[1] in keep}
        if not results:
            print(f"[skip] {run}: no series left after filtering")
            continue
        if args.max_iterations is not None:
            results = truncate_iterations(results, args.max_iterations)
        out = args.into or run
        prefix = f"{run.name}_" if args.into else ""
        methods = sorted({m for m, _ in results})
        exponents = sorted({p for _, p in results}, reverse=True)
        print(f"[{run.name}] methods={methods} p={exponents}")

        written = [] if args.paired_only else [
            plot_optimization_gap(results, out / f"{prefix}optimization_gap.png",
                                  exponents=all_exponents),
            plot_recovery_error(results, out / f"{prefix}recovery_error.png",
                                exponents=all_exponents),
            plot_legend(results, out / f"{prefix}legend.png",
                        exponents=all_exponents),
        ]

        if not args.skip_paired:
            for method in methods:
                per_method = {p: v for (mm, p), v in results.items() if mm == method}
                if len(per_method) < 2:
                    continue
                baseline = max(per_method)
                written += [
                    plot_paired_difference_curve(
                        per_method, out / f"{prefix}paired_curve_{method}.png",
                        baseline_p=baseline, method=method,
                        exponents=all_exponents, seed=args.seed,
                        linthresh=args.linthresh,
                    ),
                    plot_paired_difference_points(
                        per_method, out / f"{prefix}paired_final_{method}.png",
                        baseline_p=baseline, method=method,
                        exponents=all_exponents, seed=args.seed,
                    ),
                ]
                table = paired_difference_table(
                    per_method, baseline_p=baseline, seed=args.seed
                )
                table.insert(0, "method", method)
                table.to_csv(out / f"{prefix}paired_vs_baseline_{method}.csv",
                             index=False)

        for path in written:
            print(f"    {path}")


if __name__ == "__main__":
    main()
