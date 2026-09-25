# paraconvex

Numerical experiments for a working paper on **stochastic model-based methods
for minimizing paraconvex functions**.

A function `h` is *ν-paraconvex* on a convex set `S` (for `ν ∈ (0,1]`, `ρ > 0`) if

```
h(λx + (1-λ)y) ≤ λ h(x) + (1-λ) h(y) + ρ · min{λ, 1-λ} · ‖x - y‖^(1+ν)
```

for all `x, y ∈ S` and `λ ∈ [0,1]`.  Taking `ν = 1` recovers ρ-weak convexity, so
paraconvexity interpolates between weak convexity and convexity.

The prior literature on paraconvex minimization is deterministic and covers only
the projected subgradient method; this project targets the **stochastic** setting.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) and pinned to
Python 3.12.

```bash
uv sync                                   # local: CPU-only torch (the default)
uv sync --no-group cpu --group cu126      # GPU cluster: CUDA 12.6 torch
```

`cpu` and `cu126` are mutually exclusive dependency-groups, with `cpu` in
`default-groups` so a bare `uv sync` does the right thing locally. `uv.lock`
records **both** resolutions, so the same committed lock file reproduces the
environment on either machine, with identical torch versions (2.13.0) on both.

If a cluster needs a different CUDA, change the index URL in `pyproject.toml`
and re-lock — see the comment block there for which indexes carry which
versions.

Verify and check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
uv run ruff check
```

## Layout

```
src/paraconvex/   importable library (method + problem implementations)
experiments/      driver scripts, one per figure/table
tests/            unit tests
data/             datasets (gitignored)
results/          generated figures and raw runs (gitignored)
```

`results/` and `data/` are deliberately not versioned: a figure is reproducible
from `uv.lock` + the driver script + its seed, which are all tracked.

## Reference

The deterministic predecessor, used as a baseline and comparison source:

> M. Rahimi, S. Ghaderi, Y. Moreau, M. Ahookhosh.
> *Projected subgradient methods for paraconvex optimization: Application to
> robust low-rank matrix recovery.* arXiv:2501.00427.
