"""Paired comparison against a baseline arm.

Every exponent is run on identical data, so the arms form a *paired* design and
the comparison must be the within-replication difference.  Between-replication
variation -- how hard a given random draw happens to be -- is roughly 20x larger
than the effect of ``p`` and is common to all arms, so marginal medians with
interquartile bands are dominated by it: two arms can look indistinguishable
while one wins on nearly every individual instance.  Differencing within a
replication cancels the shared difficulty and leaves the effect alone.

Overlapping marginal bands are therefore **not** evidence of no effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["paired_stats"]


def paired_stats(
    values: dict[float, np.ndarray],
    baseline_p: float,
    n_boot: int = 20000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired difference of each arm against ``baseline_p``; one row per arm.

    Parameters
    ----------
    values:
        ``{p: array of shape (R,)}`` -- one scalar per replication per arm, with
        replication ``r`` meaning the same underlying dataset in every arm.
    n_boot:
        Bootstrap resamples for the CI on the *median* difference.  The median
        rather than the mean because the ensemble is heavy-tailed.

    Returns
    -------
    Columns: ``p``, ``nu``, ``median``, ``median_diff``, ``ci_low``,
    ``ci_high``, ``wins`` (replications strictly better than baseline), ``n``,
    ``wilcoxon_p``, ``significant`` (CI excludes zero).

    Negative ``median_diff`` means the arm beats the baseline.
    """
    if baseline_p not in values:
        raise KeyError(f"baseline p={baseline_p} not among arms {sorted(values)}")

    base = np.asarray(values[baseline_p], dtype=float)
    if base.ndim != 1:
        raise ValueError(f"expected one value per replication, got shape {base.shape}")

    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(values, reverse=True):
        if p == baseline_p:
            continue
        arm = np.asarray(values[p], dtype=float)
        if arm.shape != base.shape:
            raise ValueError(
                f"arm p={p} has shape {arm.shape}, baseline has {base.shape}; "
                "paired comparison requires matching replications"
            )
        diff = arm - base

        idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
        boot = np.median(diff[idx], axis=1)
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

        # Wilcoxon is undefined when every difference is zero; report p=1 there.
        wilcoxon_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(diff).pvalue)

        rows.append(
            {
                "p": p,
                "nu": p - 1.0,
                "median": float(np.median(arm)),
                "median_diff": float(np.median(diff)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "wins": int((diff < 0).sum()),
                "n": int(diff.size),
                "wilcoxon_p": wilcoxon_p,
                "significant": bool(hi < 0 or lo > 0),
            }
        )
    return pd.DataFrame(rows)
