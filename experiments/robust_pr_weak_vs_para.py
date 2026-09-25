"""Weakly convex (p=2) versus paraconvex (p<2) robust phase retrieval.

**One dataset, several estimators.**  The measurements are fixed,

    b_i = <a_i, x*>^2 + sigma * t(dof),

and the exponent enters only through the loss,

    min_x (1/m) sum_i | |<a_i,x>|^p - T_p(b_i) |,      T_p(b) = sign(b)|b|^{p/2}.

Since ``T_p(<a,x*>^2) = |<a,x*>|^p`` exactly, ``x*`` solves every model, so
varying ``p`` varies the *estimator applied to identical data* -- which is what
makes this a robustness comparison.  ``p=2`` is the unmodified classical
problem (``T_2`` is the identity); ``p<2`` compresses outliers as
``|xi|^{p/2}``.

Experimental controls, and why each is there:

* **Shared data.**  One sensing matrix, one ground truth, one noise draw, one
  ``b`` -- reused by every exponent.  Arms differ only in the loss.
* **Per-exponent tuning.**  The subgradient scale carries a factor
  ``p|<a,x>|^{p-1}`` that varies with p, so a shared step-size would confound
  "which p is better" with "which p suited this alpha_0".  Each arm is tuned on
  its own grid and reported at its own best.
* **Disjoint tuning data.**  Tuning runs on a design drawn from a different
  seed, so the reported curves are not selected on the randomness they score on.
* **Median, not mean.**  Heavy tails make the ensemble mean meaningless.

Usage
-----
    uv run python experiments/robust_pr_weak_vs_para.py
    uv run python experiments/robust_pr_weak_vs_para.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paraconvex.algorithms import stochastic_prox_linear, stochastic_subgradient
from paraconvex.data import make_design
from paraconvex.plotting import (
    paired_difference_table,
    plot_legend,
    plot_optimization_gap,
    plot_paired_difference_curve,
    plot_paired_difference_points,
    plot_recovery_error,
)
from paraconvex.problems import RobustPhaseRetrieval
from paraconvex.problems.phase_retrieval import uniform_in_ball
from paraconvex.reference import estimate_minimum
from paraconvex.results_io import save_trajectories
from paraconvex.tuning import (
    exponent_diminishing_grid,
    geometric_grid,
    tune_stepsize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="signal dimension")
    ap.add_argument("--m", type=int, default=300, help="number of measurements")
    ap.add_argument("--replications", type=int, default=20)
    ap.add_argument("--tune-replications", type=int, default=8)
    ap.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[2.0, 1.75, 1.5, 1.25],
        help="exponents; nu = p - 1",
    )
    ap.add_argument(
        "--dof",
        type=float,
        default=2.5,
        help=(
            "Student-t degrees of freedom. An outlier reaches the objective as "
            "|xi|^(p/2), so integrability needs dof > max(p)/2. Values in "
            "(1, 2] are where the p=2 arm starts to suffer while p<2 arms stay "
            "integrable -- the sharpest setting for the comparison."
        ),
    )
    ap.add_argument(
        "--allow-nonintegrable",
        action="store_true",
        help=(
            "run arms whose objective has infinite expectation (dof <= p/2). "
            "Needed for Cauchy (dof=1), where the p=2 loss sits exactly at the "
            "boundary while every p<2 arm stays integrable -- the sharpest "
            "setting for the comparison, and worth probing on purpose"
        ),
    )
    ap.add_argument(
        "--target-transform",
        choices=("signed", "clip"),
        default="signed",
        help=(
            "how to map a possibly-negative squared measurement onto the p-th "
            "power scale; 'signed' leaves the p=2 arm as the classical problem"
        ),
    )
    ap.add_argument("--noise-level", type=float, default=0.1)
    ap.add_argument(
        "--method",
        nargs="+",
        choices=("subgradient", "prox-linear"),
        default=["subgradient"],
        help=(
            "which members of the stochastic model-based family to run; give "
            "several to overlay them in one figure. 'prox-linear' keeps the "
            "outer |.| exact and linearises only the inner map, giving a step "
            "with an automatic trust region; it requires batch size 1, where "
            "its subproblem is closed-form"
        ),
    )
    ap.add_argument(
        "--schedule",
        choices=("geometric", "exponent-diminishing"),
        default="geometric",
        help=(
            "step-size family. 'geometric' is alpha_0 lam^k (theory-matched for "
            "a linear rate under a Holderian error bound). "
            "'exponent-diminishing' is alpha_0 (k+1)^(-1/p), whose decay is "
            "pinned by the loss exponent and which reduces to the classical "
            "k^(-1/2) rule at p=2"
        ),
    )
    ap.add_argument(
        "--alpha0-log-range",
        type=float,
        nargs=2,
        default=(-5.0, -1.0),
        metavar=("LO", "HI"),
        help=(
            "log10 range of the alpha_0 tuning grid, at a fixed density of two "
            "points per decade. Applied to every arm: widening it for one "
            "exponent only would give that arm a larger search than the others "
            "and make the comparison unfair. Check the reported grid edges -- a "
            "winner pinned at LO or HI means the range is too narrow"
        ),
    )
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--relative-start", type=float, default=0.1)
    ap.add_argument("--record-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-seed", type=int, default=9999)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small, fast configuration for smoke-testing the pipeline",
    )
    args = ap.parse_args()

    if args.quick:
        args.n, args.m = 40, 120
        args.replications, args.tune_replications = 6, 4
        args.iterations, args.record_every = 1500, 50
        args.p = [2.0, 1.5]

    if "prox-linear" in args.method and args.batch_size != 1:
        raise SystemExit(
            f"--method prox-linear requires --batch-size 1 (its subproblem is "
            f"closed-form only for a scalar residual); got {args.batch_size}"
        )
    max_p = max(args.p)
    if args.dof <= max_p / 2.0 and not args.allow_nonintegrable:
        raise SystemExit(
            f"dof={args.dof:g} makes the p={max_p:g} objective non-integrable "
            f"(need dof > max(p)/2 = {max_p / 2:g}); try --dof "
            f"{max_p / 2 + 0.5:g}, or pass --allow-nonintegrable to probe this "
            f"regime on purpose"
        )
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Design: n={args.n} m={args.m} R={args.replications} "
          f"dof={args.dof:g} noise={args.noise_level:g} "
          f"iters={args.iterations} batch={args.batch_size}")
    print(f"Exponents: {args.p}  (nu = p-1)\n")

    # One dataset for every exponent: the loss changes, the data does not.
    eval_data = make_design(
        n=args.n, m=args.m, n_replications=args.replications,
        dof=args.dof, seed=args.seed,
    ).instantiate(args.noise_level)
    tune_data = make_design(
        n=args.n, m=args.m, n_replications=args.tune_replications,
        dof=args.dof, seed=args.tune_seed,
    ).instantiate(args.noise_level)

    # One start point per dataset, shared by every exponent. Drawn here rather
    # than inside the loop: the arms must begin from the same place for the
    # paired comparison to isolate the effect of p, and hoisting it makes that
    # structural instead of depending on the loop re-seeding identically.
    eval_x0 = uniform_in_ball(
        eval_data.x_true, torch.Generator().manual_seed(args.seed),
        args.relative_start,
    )
    tune_x0 = uniform_in_ball(
        tune_data.x_true, torch.Generator().manual_seed(args.tune_seed),
        args.relative_start,
    )

    print(f"Data: {eval_data.describe()}")
    print(f"Methods: {', '.join(args.method)} | "
          f"target transform: {args.target_transform}\n")

    runners = {
        "subgradient": stochastic_subgradient,
        "prox-linear": stochastic_prox_linear,
    }

    lo, hi = args.alpha0_log_range
    if hi <= lo:
        raise SystemExit(f"--alpha0-log-range needs HI > LO, got {lo} {hi}")
    alpha0_values = np.logspace(lo, hi, round(2 * (hi - lo)) + 1)

    def build_grid(p: float):
        """Candidate schedules for one exponent.

        The exponent-diminishing family pins its decay to ``p``, so its grid is
        one-dimensional (``alpha_0`` only) and must be rebuilt per arm; the
        geometric family searches ``alpha_0`` and the total decay jointly.
        """
        if args.schedule == "exponent-diminishing":
            return exponent_diminishing_grid(p, alpha0_values=alpha0_values)
        return geometric_grid(
            n_iterations=args.iterations,
            alpha0_values=alpha0_values,
            total_decay_values=(1e-1, 1e-2, 1e-3),
        )

    results: dict[tuple[str, float], dict[str, torch.Tensor]] = {}
    summary_rows = []
    tuning_frames = []

    # f_min depends only on the problem, not the method, so it is computed once
    # per exponent and shared -- both methods must be measured against the same
    # reference or their gaps are not comparable.
    reference_minimum: dict[float, torch.Tensor] = {}

    for method in args.method:
        for p in sorted(args.p, reverse=True):
            t0 = time.time()
            grid = build_grid(p)
            print(f"[{method}, p={p:g}, nu={p - 1:g}] tuning over {len(grid)} "
                  f"schedules ...", flush=True)

            tune_problem = RobustPhaseRetrieval(
                tune_data, p=p, target_transform=args.target_transform,
                check_moment=not args.allow_nonintegrable,
            )
            outcome = tune_stepsize(
                tune_problem,
                candidates=grid,
                n_iterations=args.iterations,
                batch_size=args.batch_size,
                relative_start=args.relative_start,
                seed=args.tune_seed,
                x0=tune_x0,
                runner=runners[method],
            )
            best_a0 = getattr(outcome.best, "alpha0", None)
            edge = ""
            if best_a0 is not None and (
                best_a0 <= alpha0_values[0] or best_a0 >= alpha0_values[-1]
            ):
                edge = "  <-- AT GRID EDGE, widen --alpha0-log-range"
            print(f"    best: {outcome.summary()}{edge}")
            frame = outcome.table.copy()
            frame.insert(0, "p", p)
            frame.insert(0, "method", method)
            tuning_frames.append(frame)

            problem = RobustPhaseRetrieval(
                eval_data, p=p, target_transform=args.target_transform,
                check_moment=not args.allow_nonintegrable,
            )
            result = runners[method](
                problem,
                x0=eval_x0,
                stepsize=outcome.best,
                n_iterations=args.iterations,
                batch_size=args.batch_size,
                seed=args.seed,
                record_every=args.record_every,
            )

            # Reference minimum, folding in every value any run achieved so the
            # plotted gap can never go negative.
            observed = result.objective.min(dim=0).values
            if p in reference_minimum:
                f_min = torch.minimum(reference_minimum[p], observed)
            else:
                f_min = estimate_minimum(problem, seed=args.seed + 500,
                                         observed=observed)
            reference_minimum[p] = f_min
            gap = (result.objective - f_min.unsqueeze(0)).clamp_min(0.0)

            results[method, p] = {
                "iterations": result.iterations,
                # The raw objective is stored alongside the gap, not derived
                # from it: `f_min` is an estimate, and keeping `objective` on
                # disk means a figure can be drawn without depending on it.
                "objective": result.objective,
                "gap": gap,
                "distance": result.distance,
            }

            final_dist = result.final_distance()
            summary_rows.append({
                "method": method,
                "p": p,
                "nu": p - 1.0,
                "schedule": outcome.best.label(),
                "median_final_distance": final_dist.median().item(),
                "q25_final_distance": final_dist.quantile(0.25).item(),
                "q75_final_distance": final_dist.quantile(0.75).item(),
                "median_final_gap": gap[-1].median().item(),
                "median_f_min": f_min.median().item(),
                "diverged": int(result.diverged.sum().item()),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"    final distance (median): {final_dist.median().item():.4e} "
                  f"| final gap: {gap[-1].median().item():.4e} "
                  f"| {time.time() - t0:.0f}s\n", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "summary.csv", index=False)

    # Per-replication finals. The design is paired -- every arm sees the same
    # data -- so the powerful comparison is the within-replication difference
    # between exponents, which cancels the (large) variation in instance
    # difficulty. Marginal IQRs cannot show that, so the raw values are saved.
    pd.DataFrame(
        {
            f"{method}|p={p:g}": entry["distance"][-1].cpu().numpy()
            for (method, p), entry in sorted(results.items())
        }
    ).rename_axis("replication").to_csv(args.out / "final_distance_by_replication.csv")

    # Full (T, R) trajectories, so figures can be redrawn without re-running the
    # experiment; see experiments/replot.py.
    save_trajectories(results, args.out)

    pd.concat(tuning_frames, ignore_index=True).to_csv(
        args.out / "tuning.csv", index=False
    )

    figures = [
        plot_optimization_gap(results, args.out / "optimization_gap.png"),
        plot_recovery_error(results, args.out / "recovery_error.png"),
        plot_legend(results, args.out / "legend.png"),
    ]

    # The comparison the paired design was built for, done within each method:
    # marginal medians are dominated by how hard each replication's draw happens
    # to be, which is shared across arms, and differencing removes it.
    paired_tables = []
    for method in args.method:
        per_method = {p: v for (mm, p), v in results.items() if mm == method}
        if len(per_method) < 2:
            continue
        baseline = max(per_method)
        table = paired_difference_table(per_method, baseline_p=baseline,
                                        seed=args.seed)
        table.insert(0, "method", method)
        paired_tables.append(table)
        figures += [
            plot_paired_difference_curve(
                per_method, args.out / f"paired_curve_{method}.png",
                baseline_p=baseline, method=method, seed=args.seed,
            ),
            plot_paired_difference_points(
                per_method, args.out / f"paired_final_{method}.png",
                baseline_p=baseline, method=method, seed=args.seed,
            ),
        ]
    paired = pd.concat(paired_tables, ignore_index=True) if paired_tables else None
    if paired is not None:
        paired.to_csv(args.out / "paired_vs_baseline.csv", index=False)

    (args.out / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}, indent=2)
    )

    print(summary.to_string(index=False))
    if paired is not None:
        print("\nPaired vs the largest p, within each method "
              "(negative = better, on identical data):")
        print(paired.to_string(index=False))
    print("\nFigures:")
    for f in figures:
        print(f"  {f}")
    print(f"Tables:  {args.out}/summary.csv, {args.out}/tuning.csv, "
          f"{args.out}/final_distance_by_replication.csv")


if __name__ == "__main__":
    main()
