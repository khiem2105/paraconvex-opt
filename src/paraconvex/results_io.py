"""Persist and reload recorded trajectories.

A 40k-iteration sweep across several exponents and methods costs minutes;
redrawing a figure should not.  Everything the curve figures need is the
``(T, R)`` arrays plus the record schedule, so a run writes them once and any
later replot reads them back.

Save and load live together deliberately.  The key format
``"<quantity>|<method>|p=<p>"`` is an implementation detail of this module, and
splitting the two halves across the driver and a plotting script is how the
writer and the reader drift apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "TRAJECTORY_FILENAME",
    "load_trajectories",
    "save_trajectories",
    "truncate_iterations",
]

TRAJECTORY_FILENAME = "trajectories.npz"

#: Quantities recorded per (method, exponent).  ``objective`` is the raw
#: full-sample ``f(x_t)``; ``gap`` subtracts an *estimated* minimum, so the two
#: are kept side by side rather than one being derived from the other -- the
#: estimate is not reliable enough to be the only thing on disk, and without the
#: raw values a run cannot be replotted without it.
_QUANTITIES = ("distance", "gap", "objective")

#: What a file must contain to load at all.  ``objective`` is absent from runs
#: written before it was recorded, and those should still replot their distance
#: and gap figures rather than failing outright.
_REQUIRED_QUANTITIES = ("distance",)

_SEPARATOR = "|"


def _key(quantity: str, method: str, p: float) -> str:
    if _SEPARATOR in method:
        raise ValueError(f"method name may not contain {_SEPARATOR!r}: {method!r}")
    return f"{quantity}{_SEPARATOR}{method}{_SEPARATOR}p={p:g}"


def save_trajectories(
    results: dict[tuple[str, float], dict[str, Tensor]], directory: Path
) -> Path:
    """Write every recorded trajectory to ``directory/trajectories.npz``.

    ``results`` is keyed by ``(method, p)``.  The record schedule is stored once,
    since all series share it -- the solvers run in lockstep by construction.
    """
    if not results:
        raise ValueError("nothing to save: results is empty")

    schedules = {tuple(e["iterations"].tolist()) for e in results.values()}
    if len(schedules) != 1:
        raise ValueError(
            "all series must share one record schedule; got "
            f"{len(schedules)} distinct ones"
        )

    arrays = {"iterations": next(iter(results.values()))["iterations"].cpu().numpy()}
    for (method, p), entry in results.items():
        for quantity in _QUANTITIES:
            if quantity in entry:
                arrays[_key(quantity, method, p)] = entry[quantity].cpu().numpy()

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAJECTORY_FILENAME
    np.savez_compressed(path, **arrays)
    return path


def load_trajectories(path: Path) -> dict[tuple[str, float], dict[str, Tensor]]:
    """Read back what :func:`save_trajectories` wrote.

    ``path`` may be the ``.npz`` itself or the directory containing it.  The
    result plugs straight into the plotting functions.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TRAJECTORY_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; it is written by the experiment drivers, and "
            "runs predating the multi-method key format will not load"
        )

    raw = np.load(path)
    iterations = torch.as_tensor(raw["iterations"])

    results: dict[tuple[str, float], dict[str, Tensor]] = {}
    for name in raw.files:
        if name == "iterations":
            continue
        parts = name.split(_SEPARATOR)
        if len(parts) != 3:
            # Runs predating multi-method support keyed arrays as e.g.
            # "distance_p2", with no method field.  Say so plainly instead of
            # failing on a tuple unpack, so callers can skip such a run rather
            # than aborting a whole batch.
            raise ValueError(
                f"{path} uses a legacy key format ({name!r}); it predates "
                "multi-method support and cannot be replotted. Re-run the "
                "experiment to regenerate it."
            )
        quantity, method, p_field = parts
        key = (method, float(p_field.removeprefix("p=")))
        entry = results.setdefault(key, {"iterations": iterations})
        entry[quantity] = torch.as_tensor(raw[name])

    missing = {
        key: sorted(set(_REQUIRED_QUANTITIES) - set(entry))
        for key, entry in results.items()
        if not set(_REQUIRED_QUANTITIES) <= set(entry)
    }
    if missing:
        raise ValueError(f"{path} is missing quantities for some series: {missing}")

    return results


def truncate_iterations(
    results: dict[tuple[str, float], dict[str, Tensor]], max_iterations: int
) -> dict[tuple[str, float], dict[str, Tensor]]:
    """Keep only records at iteration ``<= max_iterations``.

    The data is sliced rather than the axis merely being clipped, so the y-range
    autoscales to what is actually shown; an ``xlim`` alone would leave the
    vertical extent set by iterates the reader cannot see.

    Applied to the whole result set at once, which keeps every iteration-indexed
    figure telling the same story.  Note this also moves the *last* record, so
    companion figures that report "the final iterate" report it at
    ``max_iterations`` -- that consistency is the point, but it does mean the
    numbers in a truncated figure will not match a ``summary.csv`` written for
    the full run.
    """
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")

    truncated: dict[tuple[str, float], dict[str, Tensor]] = {}
    for key, entry in results.items():
        keep = entry["iterations"] <= max_iterations
        if not bool(keep.any()):
            raise ValueError(
                f"max_iterations={max_iterations} keeps no records for {key}; "
                f"the first is at {int(entry['iterations'][0])}"
            )
        # `keep` is (T,), and every stored array -- (T,) iterations and (T, R)
        # quantities alike -- is indexed along that same leading axis.
        truncated[key] = {name: value[keep] for name, value in entry.items()}
    return truncated
