"""Add the raw objective to finished runs, without re-running them.

Older runs stored only ``gap = f(x_t) - f_min`` and never the objective itself,
so the function-value figure cannot be drawn from them.  Re-running is the
obvious fix and costs tens of minutes per run, almost all of it re-doing the
step-size grid.  This script reconstructs ``objective = gap + f_min`` instead,
in about a minute.

Why that is exact rather than an approximation
----------------------------------------------
The driver sets ``f_min = min(reference_estimate, observed)`` where ``observed``
is the lowest objective the stochastic run itself reached.  Only the first
branch is reproducible from a seed, so the reconstruction is exact only when
``f_min == reference_estimate``.  That holds precisely when ``min_t gap > 0``:

    min_t gap = observed - f_min,

so a strictly positive minimum gap means ``f_min`` came from the reference and
not from the trajectory.  The script **verifies this per replication** and
refuses to touch a run where any series has ``min_t gap == 0``.

The reference estimate is then recomputed by rebuilding the problem from
``config.json`` -- the design is a pure function of the recorded seeds -- and
calling :func:`estimate_minimum` with the driver's seed offset.  As a second,
independent check, the recomputed median is compared against ``median_f_min``
recorded in ``summary.csv`` at the time of the run.

Usage
-----
    uv run python experiments/backfill_objective.py results/dimin_dof1_nl0.01
    uv run python experiments/backfill_objective.py results/* --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paraconvex.data import make_design
from paraconvex.problems import RobustPhaseRetrieval
from paraconvex.reference import estimate_minimum
from paraconvex.results_io import TRAJECTORY_FILENAME

#: The driver derives the reference-minimum seed this way; mirrored here.
_FMIN_SEED_OFFSET = 500

#: Relative tolerance when checking the recomputed median against the recorded
#: one.  Exact equality is not required: `summary.csv` stores a rounded float
#: and the check is only meant to catch a wholesale mismatch.
_MEDIAN_RTOL = 1e-6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="verify and report, but do not modify any file",
    )
    return ap.parse_args()


def backfill(run: Path, dry_run: bool) -> str:
    npz_path = run / TRAJECTORY_FILENAME
    if not npz_path.exists():
        return "skip: no trajectories.npz"

    raw = dict(np.load(npz_path))
    gap_keys = [k for k in raw if k.startswith("gap|")]
    if not gap_keys:
        return "skip: no gap series (pre-multi-method format)"
    if any(k.startswith("objective|") for k in raw):
        return "skip: already has objective"

    # -- precondition: f_min must have come from the reference, not the run ---
    touching = [k for k in gap_keys if raw[k].min(axis=0).min() <= 0.0]
    if touching:
        return f"REFUSED: min_t gap == 0 for {len(touching)} series; f_min is not recoverable"

    cfg = json.loads((run / "config.json").read_text())
    summary = pd.read_csv(run / "summary.csv")

    data = make_design(
        n=cfg["n"], m=cfg["m"], n_replications=cfg["replications"],
        dof=cfg["dof"], seed=cfg["seed"],
    ).instantiate(cfg["noise_level"])

    notes = []
    for p in sorted(cfg["p"], reverse=True):
        problem = RobustPhaseRetrieval(
            data, p=p, target_transform=cfg["target_transform"],
            check_moment=not cfg.get("allow_nonintegrable", False),
        )
        f_min = estimate_minimum(problem, seed=cfg["seed"] + _FMIN_SEED_OFFSET)

        # Cross-check against what the run itself recorded.
        rows = summary[np.isclose(summary["p"], p)]
        if not rows.empty and "median_f_min" in rows:
            recorded = float(rows["median_f_min"].iloc[0])
            got = float(f_min.median())
            if not np.isclose(got, recorded, rtol=_MEDIAN_RTOL):
                return (
                    f"REFUSED at p={p:g}: recomputed median f_min {got:.8g} != "
                    f"recorded {recorded:.8g}; the run's configuration is not "
                    "reproducible from config.json"
                )
            notes.append(f"p={p:g} ok")

        fm = f_min.cpu().numpy()
        for key in [k for k in gap_keys if k.endswith(f"|p={p:g}")]:
            raw[key.replace("gap|", "objective|", 1)] = raw[key] + fm[None, :]

    if dry_run:
        return f"OK (dry run): would add {len(gap_keys)} objective series; " + ", ".join(notes)

    np.savez_compressed(npz_path, **raw)
    return f"written: +{len(gap_keys)} objective series; " + ", ".join(notes)


def main() -> None:
    args = parse_args()
    for run in args.runs:
        if not run.is_dir():
            continue
        print(f"{run.name:<32} {backfill(run, args.dry_run)}", flush=True)


if __name__ == "__main__":
    main()
