"""GEMM launch geometry: the tile shape and CTA count PipeWeave's features need.

Their GEMM feature vector is not a function of ``(M, N, K)``. It also needs
``tile_M/N/K``, ``cta_count`` and ``is_split_k`` -- the launch parameters cuBLAS or
CUTLASS actually chose. Those are observable with a profiler and not otherwise
derivable, so upstream's ``aggregator.py`` approximates them at inference time by
nearest-neighbour lookup in ``(M, N, K)`` space against its own training set.

This module does the same lookup, and then differs from upstream in one place, for a
reason their own data supports.

What their data says about ``cta_count``
----------------------------------------
Counting over all 118,800 rows of ``gemm_test.csv``, with ``tiles = ceil(M/tile_M) *
ceil(N/tile_N)``:

======================  =====  ===================  =================  ===============
architecture            rows   ``cta == num_sms``   ``cta == tiles``   ``is_split_k``
======================  =====  ===================  =================  ===============
hopper                  43200  56%                  35%                0 for 94% of rows
ampere                  32400  2%                   80%                1 for every row
ada                     32400  0%                   50%                1 for every row
blackwell               10800  0%                   79%                1 for every row
======================  =====  ===================  =================  ===============

Two different launch models, split cleanly by architecture:

* **Hopper** runs a *persistent* kernel: one CTA per SM, each walking several tiles.
  ``cta_count == min(tiles, num_sms)`` holds for 86% of Hopper rows, and ``is_split_k``
  is off. Upstream's ``gemm9`` non-split-K branch is written for exactly this -- it
  derives ``tiles_per_cta`` and scales the per-SM work by it.
* **Everything else** runs one CTA per output tile, with ``is_split_k`` always on and
  ``cta_count = tiles * split_k_slices``. The median ratio is 1 (no K splitting), with
  a long tail to 56x on the skinny-M shapes where splitting K is the only way to fill
  the machine.

So ``cta_count`` and ``is_split_k`` are *structural*: given the tile shape and the
architecture, the default launch follows. Only the tile shape genuinely needs a lookup.

``mode="structural"`` (the default) looks up the tile shape alone and derives the rest.
``mode="upstream"`` copies all five fields off the neighbour, reproducing
``aggregator.py`` exactly -- which for a neighbour with different ``M`` and ``N`` means
importing a ``cta_count`` that does not correspond to any grid the query shape would
launch. Use it to reproduce their published numbers, not to predict.

Reference table
---------------
``data/gemm_tile_reference.csv.gz`` is the launch-geometry columns of upstream's
``gemm_test.csv``: 118,800 rows over all eleven of their GPUs. Their ``aggregator.py``
reads ``gemm_train.csv`` instead, which is a 131 MB Git-LFS object that this
environment's proxy will not serve. The test split runs the same generator over the
same grid, so it is an equivalent source of tile shapes; if you have the train file,
point ``load_reference`` at it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kernelenergy.pipeweave.hardware import PW_HARDWARE

__all__ = ["GemmLaunch", "choose_launch", "load_reference", "REFERENCE_PATH"]

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "gemm_tile_reference.csv.gz"


@dataclass(frozen=True)
class GemmLaunch:
    tile_m: int
    tile_n: int
    tile_k: int
    cta_count: int
    is_split_k: bool
    source: str  # how it was chosen, for the audit trail
    neighbour_distance: float = float("nan")


@functools.lru_cache(maxsize=1)
def load_reference(path: str | Path | None = None) -> pd.DataFrame:
    p = REFERENCE_PATH if path is None else Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"the GEMM tile reference is missing from {p}.\n"
            f"\n"
            f"Without it no GEMM or conv kernel can be given a tile shape, so every "
            f"one of those rows fails and only the norm and elementwise operators "
            f"survive.\n"
            f"\n"
            f"If this repo was cloned, the likely cause is a .gitignore rule: an "
            f"unanchored 'data/' matches a directory named data at ANY depth, not "
            f"just the top-level measurement output directory. Check with\n"
            f"    git check-ignore -v kernelenergy/pipeweave/data/{REFERENCE_PATH.name}\n"
            f"and anchor the rule to '/data/' if that is what it reports."
        )
    return pd.read_csv(p)


def _reference_for(gpu_key: str) -> tuple[pd.DataFrame, str]:
    """Rows to search for this card, and a label saying how they were found.

    Prefer the same GPU by name; fall back to the same architecture. Tile choice is a
    property of the kernel library and the SM count far more than of the memory system,
    so an architecture-level fallback is sound -- and it is the only option for the four
    fleet cards upstream never measured.
    """
    row = PW_HARDWARE[gpu_key]
    ref = load_reference()
    same = ref[ref["hardware"] == row.name]
    if len(same):
        return same, "exact"
    arch = ref[ref["arch"] == row.architecture]
    if not len(arch):
        raise KeyError(
            f"no reference GEMM launches for {gpu_key} "
            f"(architecture {row.architecture!r})"
        )
    # Among same-architecture cards, prefer the one closest in SM count: the tile ladder
    # is chosen to fill the machine, so SM count is what actually drives it.
    by_sm = arch.assign(d=(arch["num_sms"] - row.num_sms).abs())
    best = by_sm.loc[by_sm["d"].idxmin(), "hardware"]
    return arch[arch["hardware"] == best], f"arch:{best}"


def choose_launch(
    m: int, n: int, k: int, gpu_key: str, mode: str = "structural"
) -> GemmLaunch:
    """Pick the launch geometry for one GEMM on one card.

    ``mode='structural'`` -- nearest-neighbour tile shape, architecture-derived
    ``cta_count`` and ``is_split_k``. The default.

    ``mode='upstream'`` -- all five fields copied from the neighbour, reproducing
    ``aggregator.py``.
    """
    if mode not in ("structural", "upstream"):
        raise ValueError(f"mode must be 'structural' or 'upstream', got {mode!r}")

    ref, how = _reference_for(str(gpu_key).upper())
    # Upstream's metric, verbatim: unweighted Euclidean in raw (M, N, K). It is
    # scale-blind -- K dominates because K is usually largest -- but changing it would
    # stop reproducing their choice, and their choice is what their weights saw.
    d = np.sqrt(
        (ref["M"].to_numpy(float) - m) ** 2
        + (ref["N"].to_numpy(float) - n) ** 2
        + (ref["K"].to_numpy(float) - k) ** 2
    )
    i = int(np.argmin(d))
    nb = ref.iloc[i]
    tm, tn, tk = int(nb["tile_M"]), int(nb["tile_N"]), int(nb["tile_K"])

    if mode == "upstream":
        return GemmLaunch(
            tm, tn, tk, int(nb["cta_count"]), bool(nb["is_split_k"]),
            source=f"upstream/{how}", neighbour_distance=float(d[i]),
        )

    row = PW_HARDWARE[str(gpu_key).upper()]
    tiles = math.ceil(m / tm) * math.ceil(n / tn)
    if row.architecture == "hopper":
        # Persistent: one CTA per SM, each sweeping several tiles.
        return GemmLaunch(
            tm, tn, tk, max(1, min(tiles, row.num_sms)), False,
            source=f"structural/hopper/{how}", neighbour_distance=float(d[i]),
        )
    # One CTA per tile, split-K enabled but not used by default. slices=1 makes
    # tile_split_k collapse to k_padded, i.e. each CTA does a full-K tile -- which is
    # what the median row in their non-Hopper data does.
    return GemmLaunch(
        tm, tn, tk, max(1, tiles), True,
        source=f"structural/tiled/{how}", neighbour_distance=float(d[i]),
    )
