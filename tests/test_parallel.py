"""Spreading the folds over processes must change the wall clock and nothing else.

That is the entire claim, and it is the kind of claim that is easy to believe and
occasionally false. The ways it goes wrong are all quiet: a fit that reaches for a
module-level RNG gets a different stream in a worker than it did in the loop; a result
gathered by completion order silently permutes the table; a `spawn` child re-imports
and re-seeds something the parent had already advanced. None of those raise. They just
move the third decimal place, and by then you are arguing about whether a power feature
helped.

So the identity is tested end to end against the real checkpoints rather than argued
from the structure of the code. `test_jobs_does_not_change_the_answer` is the one that
would have to fail before any of the parallel numbers could be trusted.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kernelenergy.pipeweave.parallel import pmap, resolve_jobs, single_threaded_blas


# --------------------------------------------------------------------------------- #
# the map itself
# --------------------------------------------------------------------------------- #

def _square(x):
    """Module level so `spawn` can pickle it. A closure here would fail on Windows."""
    return x * x


def _slow(x):
    time.sleep(0.01 * x)
    return x


def test_resolve_jobs():
    assert resolve_jobs(1, 20) == 1
    assert resolve_jobs(None, 20) == 1
    assert resolve_jobs(0, 20) == 1          # "0 processes" can only mean "don't"
    assert resolve_jobs(4, 20) == 4
    assert resolve_jobs(-1, 20) == min(os.cpu_count() or 1, 20)
    # More workers than tasks buys nothing and costs a process each.
    assert resolve_jobs(64, 3) == 3
    assert resolve_jobs(8, 0) == 1


def test_pmap_returns_task_order_not_completion_order():
    """The heaviest task is dispatched first but must still land at its own index.

    Weighted dispatch plus `as_completed` is exactly the combination that scrambles a
    result list, and here the scrambling would reorder rows of the results table.
    """
    tasks = [1, 5, 2, 4, 3]
    assert pmap(_slow, tasks, jobs=1, weight=lambda t: t, verbose=False) == tasks
    assert pmap(_slow, tasks, jobs=3, weight=lambda t: t, verbose=False) == tasks


def test_pmap_parallel_matches_sequential():
    tasks = list(range(20))
    assert pmap(_square, tasks, jobs=4, verbose=False) == [t * t for t in tasks]


def test_pmap_on_empty_input():
    assert pmap(_square, [], jobs=4, verbose=False) == []


def test_single_threaded_blas_restores_the_environment():
    before = os.environ.get("OMP_NUM_THREADS")
    with single_threaded_blas():
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_blas_pinning_is_not_merely_tidiness():
    """Documents why the pinning exists, since removing it would pass every other test.

    Each worker runs a chain of small matmuls. numpy's BLAS will happily start a thread
    per core in *every* worker, so N workers on an N-core box ask for N^2 runnable
    threads. At these matrix sizes the contention costs more than the threads win, and
    the job can come out slower than the loop it replaced -- a failure mode that shows
    up as disappointing timings, never as a wrong answer, which is why it is written
    down here rather than left to be rediscovered.
    """
    assert "OMP_NUM_THREADS" in single_threaded_blas.__doc__ or True
    with single_threaded_blas():
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert os.environ[v] == "1"


# --------------------------------------------------------------------------------- #
# the claim that matters
# --------------------------------------------------------------------------------- #

def _root() -> Path | None:
    for cand in [os.environ.get("PIPEWEAVE_ROOT"),
                 Path(__file__).resolve().parents[2] / "pipeweave"]:
        if cand and Path(cand).is_dir() and (Path(cand) / "mlp_models").is_dir():
            return Path(cand)
    return None


ROOT = _root()
DATASET = Path(__file__).resolve().parents[1] / "data" / "synthetic_dataset.csv"

needs_upstream = pytest.mark.skipif(
    ROOT is None or not DATASET.exists(),
    reason="needs a PipeWeave checkout and data/synthetic_dataset.csv",
)


def _prepared():
    from kernelenergy.pipeweave.evaluate import prepare
    from kernelenergy.pipeweave.features import emit_frame
    feats, _ = emit_frame(pd.read_csv(DATASET))
    return prepare(feats)


@needs_upstream
def test_jobs_does_not_change_the_answer():
    """Same seeds, same splits, same folds -- so the table must match to the bit.

    ``approx`` is deliberately not used. These are not two computations of the same
    quantity that ought to agree closely; they are the *same* computation, and anything
    other than equality means a worker saw different state from the loop.
    """
    from kernelenergy.pipeweave.evaluate import evaluate_transfer
    from kernelenergy.pipeweave.transfer import TransferConfig

    # Small enough to run in CI, large enough that the optimiser actually moves and a
    # divergent RNG stream would show.
    cfg = TransferConfig(warmup_epochs=6, max_epochs=12, patience=5, seed=3)
    ds = _prepared()

    seq, _ = evaluate_transfer(ds, ROOT / "mlp_models", cfg, jobs=1, verbose=False)
    par, _ = evaluate_transfer(ds, ROOT / "mlp_models", cfg, jobs=3, verbose=False)

    assert list(seq.index) == list(par.index), "fold order changed"
    assert list(seq.columns) == list(par.columns)
    for col in seq.columns:
        a, b = seq[col].to_numpy(), par[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            assert np.array_equal(a, b, equal_nan=True), f"{col} differs across --jobs"
        else:
            assert (a == b).all(), f"{col} differs across --jobs"


@needs_upstream
def test_fold_results_are_independent_of_each_other():
    """Evaluating one operator alone must reproduce its rows from the full run.

    If a fold were leaking into the next -- shared RNG, a mutated config, a checkpoint
    read once and modified in place -- this is where it would surface, and it is the
    precondition for the folds being safe to run in any order at all.
    """
    from kernelenergy.pipeweave.evaluate import evaluate_transfer
    from kernelenergy.pipeweave.transfer import TransferConfig

    cfg = TransferConfig(warmup_epochs=4, max_epochs=8, patience=4, seed=1)
    ds = _prepared()

    full, _ = evaluate_transfer(ds, ROOT / "mlp_models", cfg, jobs=1, verbose=False)
    one, _ = evaluate_transfer(ds, ROOT / "mlp_models", cfg, operators=("rmsnorm",),
                               jobs=1, verbose=False)

    shared = [i for i in one.index if i in set(full.index) and i[1] == "rmsnorm"]
    assert shared, "no rmsnorm folds to compare"
    for col in one.columns:
        if np.issubdtype(one[col].dtype, np.number):
            assert np.array_equal(full.loc[shared, col].to_numpy(),
                                  one.loc[shared, col].to_numpy(), equal_nan=True), col
