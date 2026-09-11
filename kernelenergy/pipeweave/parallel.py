"""Run independent fits across processes.

The transfer evaluation is leave-one-GPU-out, one network per (fold, operator), and
every one of those fits is independent of every other: its own checkpoint copy, its own
seeded validation split, its own initialisation draw. Nothing is shared and nothing is
accumulated. So the whole evaluation is a map, and the only reason it takes a quarter of
an hour is that it has been running as a `for` loop.

A worked measurement, on this project's own model (256-128-64 on 11-15 inputs, about 44k
parameters):

    fold of 900 rows, dim 11      3.8 s
    fold of 400 rows, dim 15      1.7 s
    fold of 1400 rows, dim 11    14.2 s

Twenty folds a run, ten runs for a five-seed A/B, so two hundred fits and roughly
thirteen minutes on one core. Spread over cores that is under a minute, and the numbers
do not move: each fit already draws from `numpy.random.default_rng(seed + ...)`, so the
process it happens to run in is not an input to it. That last point is the whole reason
to parallelise here rather than reach for a GPU. A batch step of this model is about
34 MFLOP -- under two microseconds of A100 -- against a launch and interpreter overhead
two orders of magnitude larger, so a card would sit at about one percent utilisation and
lose to a handful of cores.

Two details that are easy to get wrong and expensive when you do:

**BLAS oversubscription.** numpy's BLAS threads by default. Eight worker processes each
starting eight BLAS threads on an eight-core machine is sixty-four runnable threads
fighting over eight cores, and small matmuls -- which is all of these -- lose badly to
the contention. It is entirely possible to make the job slower by parallelising it.
:func:`pmap` therefore pins the workers to one BLAS thread each. The variables are set
in the *parent* before the pool starts, because a child reads them when it imports numpy
and a pool initializer runs too late to be sure of beating that import.

**Windows spawns rather than forks.** A child re-imports `__main__`, so a script that
calls :func:`pmap` at module level recurses into itself. Console entry points carry the
`if __name__ == "__main__"` guard that prevents this and the CLI is one, but a bare
`python -c` is not, and the error Python raises for it says nothing useful. It is caught
below and re-raised with the fix.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager

__all__ = ["resolve_jobs", "pmap", "single_threaded_blas", "spawn_obstacle"]

_GUARD_HELP = (
    "Run this through the 'kernelenergy' command, which is guarded, or put your own "
    "call inside\n"
    "    if __name__ == '__main__':\n"
    "...or pass --jobs 1 to stay in one process."
)

#: Every environment variable a BLAS might read for its thread count. Setting all of
#: them is cheaper than discovering which one this wheel was built against.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def resolve_jobs(jobs: int | None, n_tasks: int) -> int:
    """Turn a ``--jobs`` value into a worker count.

    ``None`` or ``1`` means stay in this process -- and that is a real distinction, not
    a pool of size one: the sequential path keeps tracebacks intact and lets a debugger
    work, which matters more often than the last core does.

    ``-1`` (or any negative) means every core. More workers than tasks is pointless, so
    the count is capped at ``n_tasks``.
    """
    if jobs is None or jobs == 0:
        jobs = 1
    jobs = int(jobs)
    if jobs < 0:
        jobs = os.cpu_count() or 1
    return max(1, min(jobs, max(n_tasks, 1)))


def spawn_obstacle() -> str:
    """Why a spawned worker could not start here, or ``""`` if it can.

    Checked before the pool rather than after, because the symptom otherwise is
    ``BrokenProcessPool`` -- an error that names nothing and points nowhere, arriving
    only once the work has already been dispatched.

    A spawned child bootstraps by re-importing the parent's ``__main__``. That is fine
    for a real script (the ``if __name__`` guard stops the recursion) and fine for an
    interactive interpreter (there is no file, so nothing is re-imported). It breaks in
    the one case between the two: ``__main__`` claims a filename that does not exist on
    disk -- code piped in on stdin, most often -- and the child dies trying to open it.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if path and not os.path.exists(path):
        return (f"the main module reports its file as {path!r}, which does not exist, "
                f"so a spawned worker cannot re-import it (code piped to the "
                f"interpreter does this).")
    return ""


@contextmanager
def single_threaded_blas():
    """Pin BLAS to one thread for the duration, then put the environment back.

    Affects processes started inside the block. This one has already imported numpy, so
    its own threading is unchanged -- which is what we want: the parent is not doing the
    arithmetic.
    """
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        yield
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


def pmap(fn, tasks, jobs: int = 1, weight=None, label: str = "",
         verbose: bool = True) -> list:
    """``[fn(t) for t in tasks]``, over ``jobs`` processes, in task order.

    ``fn`` and every task must be picklable, which on Windows means ``fn`` has to be a
    module-level function rather than a closure or a lambda.

    ``weight`` is an optional cost estimate per task. Tasks are dispatched heaviest
    first, because the gemm folds here run about four times as long as the rmsnorm ones
    and a pool that starts a fourteen-second fit last spends thirteen seconds with one
    core busy and the rest idle. Results still come back in the original order.
    """
    tasks = list(tasks)
    if not tasks:
        return []
    n_workers = resolve_jobs(jobs, len(tasks))

    order = list(range(len(tasks)))
    if weight is not None:
        order.sort(key=lambda i: -float(weight(tasks[i])))

    if n_workers == 1:
        out = [None] * len(tasks)
        for n, i in enumerate(order, 1):
            out[i] = fn(tasks[i])
            if verbose:
                _tick(n, len(tasks), label)
        return out

    obstacle = spawn_obstacle()
    if obstacle:
        raise RuntimeError(f"cannot run {n_workers} workers: {obstacle}\n{_GUARD_HELP}")

    out: list = [None] * len(tasks)
    t0 = time.perf_counter()
    # "spawn" everywhere, not just where it is forced. A forked child inherits the
    # parent's BLAS thread pool in an undefined state, and the resulting hangs are
    # intermittent and miserable to chase. Paying the import cost once per worker is a
    # far better trade than a job that deadlocks one run in twenty.
    ctx = mp.get_context("spawn")
    try:
        with single_threaded_blas(), ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(fn, tasks[i]): i for i in order}
            for n, fut in enumerate(as_completed(futures), 1):
                out[futures[fut]] = fut.result()
                if verbose:
                    _tick(n, len(tasks), label)
    except BrokenExecutor as e:
        # A worker died before it could report anything. Almost always the bootstrap
        # rather than the work: an unguarded entry point, or the OS killing a child for
        # memory. Say both, because the traceback says neither.
        raise RuntimeError(
            f"a worker process died during startup ({type(e).__name__}). The usual "
            f"causes are an unguarded entry point -- a spawned worker re-imports the "
            f"main module -- or the machine running out of memory with {n_workers} "
            f"copies of the dataset resident.\n{_GUARD_HELP}"
        ) from e
    except RuntimeError as e:
        if "freeze_support" in str(e) or "spawn" in str(e).lower():
            raise RuntimeError(
                f"parallel execution needs the entry point guarded, because a worker "
                f"starts by re-importing the main module.\n{_GUARD_HELP}"
            ) from e
        raise
    if verbose:
        print(f"    {label or 'done'}: {len(tasks)} tasks on {n_workers} workers "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def _tick(n: int, total: int, label: str) -> None:
    end = "\n" if n == total else "\r"
    sys.stdout.write(f"    {label or 'fit'} {n}/{total}   ")
    sys.stdout.write(end)
    sys.stdout.flush()
