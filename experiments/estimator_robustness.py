"""Is the paraconvex loss a more robust *estimator*?

This asks a purely statistical question and deliberately removes the optimizer
from it:

    is  argmin f_p  closer to  x*  than  argmin f_2,  on identical data?

Robustness is a property of the estimator, not of a trajectory.  Comparing
stochastic iterates at a fixed ``k`` confounds the two, and can actively
mislead: measured on 2026-08-07 under Cauchy noise, the ``p=2`` iterate at
``k=40000`` sat *closer* to ``x*`` (0.01369) than its own minimizer (0.01416),
so early stopping flattered the baseline and understated the true effect.

Each cell of the ``dof`` x ``noise_level`` grid drives full-batch descent to a
converged minimizer for every exponent, then reports the **paired** difference
against the ``p=2`` baseline -- paired because all exponents share one dataset,
and between-replication variation in draw difficulty is roughly 20x the effect
being measured.

Cheaper than the stochastic sweep: no step-size tuning is needed, since the
answer does not depend on the optimizer.

Usage
-----
    uv run python experiments/estimator_robustness.py
    uv run python experiments/estimator_robustness.py --quick
    uv run python experiments/estimator_robustness.py --dof 1 1.5 2.5 \
        --noise-levels 0.01 0.03 0.1 0.3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from paraconvex.data import make_design
from paraconvex.paired import paired_stats
from paraconvex.plotting import plot_estimator_effects
from paraconvex.problems import RobustPhaseRetrieval
from paraconvex.reference import converged_minimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "estimator_robustness"

#: Above this, the polish stage was still making real progress and the reported
#: minimizer should not be trusted as converged.
POLISH_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--replications", type=int, default=20)
    ap.add_argument("--p", type=float, nargs="+", default=[2.0, 1.75, 1.5, 1.25])
    ap.add_argument(
        "--dof", type=float, nargs="+", default=[1.0, 2.5],
        help="tail indices; 1.0 is Cauchy",
    )
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--baseline-p", type=float, default=2.0)
    ap.add_argument("--coarse-iterations", type=int, default=15000)
    ap.add_argument("--polish-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n, args.m, args.replications = 40, 120, 8
        args.p = [2.0, 1.5]
        args.dof = [1.0, 2.5]
        args.noise_levels = [0.01, 0.1]
        args.coarse_iterations, args.polish_iterations = 3000, 1000

    if args.baseline_p not in args.p:
        raise SystemExit(f"--baseline-p {args.baseline_p:g} must be among --p {args.p}")
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Grid: dof={args.dof} x noise={args.noise_levels}, exponents {args.p}")
    print(f"n={args.n} m={args.m} R={args.replications}, baseline p={args.baseline_p:g}\n")

    rows, raw_rows = [], []
    worst_polish = 0.0

    for dof in args.dof:
        for nl in args.noise_levels:
            t0 = time.time()
            data = make_design(
                n=args.n, m=args.m, n_replications=args.replications,
                dof=dof, seed=args.seed,
            ).instantiate(nl)

            distances: dict[float, torch.Tensor] = {}
            for p in args.p:
                problem = RobustPhaseRetrieval(data, p=p, check_moment=False)
                # Start at x* so the run converges to the minimizer of the basin
                # containing the truth. The question is how far that minimizer
                # has been displaced by noise, not whether descent can find it.
                x_hat, _, polish_gain = converged_minimizer(
                    problem,
                    data.x_true.clone(),
                    coarse_iterations=args.coarse_iterations,
                    polish_iterations=args.polish_iterations,
                )
                worst_polish = max(worst_polish, polish_gain)
                distances[p] = problem.distance_to_truth(x_hat).cpu()

            table = paired_stats(
                {p: d.numpy() for p, d in distances.items()},
                baseline_p=args.baseline_p,
                seed=args.seed,
            )
            table.insert(0, "noise_level", nl)
            table.insert(0, "dof", dof)
            rows.append(table)

            for p, d in distances.items():
                for r, v in enumerate(d.numpy()):
                    raw_rows.append(
                        {"dof": dof, "noise_level": nl, "p": p,
                         "replication": r, "distance": float(v)}
                    )

            base_med = distances[args.baseline_p].median().item()
            flags = " ".join(
                f"p={row.p:g}:{row.median_diff:+.5f}{'*' if row.significant else ''}"
                for row in table.itertuples()
            )
            print(f"  dof={dof:<4g} noise={nl:<5g} baseline={base_med:.5f}  "
                  f"{flags}   [{time.time() - t0:.0f}s]", flush=True)

    effects = pd.concat(rows, ignore_index=True)
    effects.to_csv(args.out / "estimator_effects.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(args.out / "minimizer_distances.csv", index=False)

    figure = plot_estimator_effects(
        effects, output=args.out / "estimator_effects.png", baseline_p=args.baseline_p
    )
    (args.out / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}, indent=2)
    )

    print(f"\nWorst polish-stage improvement: {worst_polish:.2e} ", end="")
    if worst_polish > POLISH_TOLERANCE:
        print(f"-- EXCEEDS {POLISH_TOLERANCE:.0e}: minimizers may not be converged; "
              "raise --coarse-iterations")
    else:
        print(f"(< {POLISH_TOLERANCE:.0e}, converged)")

    print("\nPaired effect vs baseline (negative = more accurate; * = CI excludes 0):")
    show = effects[["dof", "noise_level", "p", "median_diff", "ci_low", "ci_high",
                    "wins", "n", "wilcoxon_p", "significant"]]
    print(show.to_string(index=False))
    print(f"\nFigure: {figure}\nTables: {args.out}/estimator_effects.csv, "
          f"{args.out}/minimizer_distances.csv")


if __name__ == "__main__":
    main()
