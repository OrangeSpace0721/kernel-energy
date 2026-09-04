"""SLURM integration: job identity, work sharding, walltime and preemption.

Three problems a scheduler introduces that a workstation does not.

**The sweep will be interrupted.** Walltime expires, the job is preempted, the node
drains. The measure stage is already resumable, but resumability is only useful if the
process stops *cleanly* -- killed mid-window it may leave a half-written row, and killed
without warning it wastes whatever the current kernel had already spent. So the sweep
takes a signal, finishes the kernel in flight, flushes, and exits with a status that says
whether there is more to do.

**One job is the wrong unit of work.** A few hundred configs at ~12 s each is hours per
card, which is a long time to hold an exclusive node and a long time to lose to one
preemption. Array tasks split the catalogue; the split has to be deterministic, so a
requeued task picks up its own shard and not somebody else's, and balanced, so the array
finishes when the last task does rather than when the unlucky one does.

**Walltime is a hard edge.** Starting a 12-second measurement with 4 seconds left wastes
the measurement and, worse, may write a truncated window as if it were a real one. The
sweep stops when the next kernel does not fit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

__all__ = [
    "SlurmContext",
    "slurm_context",
    "shard_configs",
    "InterruptGuard",
    "EXIT_INCOMPLETE",
]

#: Exit status meaning "I stopped cleanly but there is work left". The sbatch wrapper
#: turns this into a requeue rather than treating it as success or as failure.
EXIT_INCOMPLETE = 64


@dataclass
class SlurmContext:
    """What the scheduler told us about this job."""

    in_slurm: bool
    job_id: str = ""
    array_job_id: str = ""
    task_id: int = 0
    task_count: int = 1
    node: str = ""
    partition: str = ""
    cpus: int = 0
    gpus_env: str = ""
    end_time: float | None = None  # unix timestamp of walltime expiry
    started: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        if not self.in_slurm:
            return "local"
        if self.task_count > 1:
            return f"{self.array_job_id or self.job_id}_{self.task_id}"
        return self.job_id

    def seconds_remaining(self) -> float | None:
        if self.end_time is None:
            return None
        return max(0.0, self.end_time - time.time())

    def describe(self) -> str:
        if not self.in_slurm:
            return "not running under SLURM"
        rem = self.seconds_remaining()
        rem_s = f"{rem / 60:.0f} min left" if rem is not None else "walltime unknown"
        return (
            f"SLURM job {self.label} on {self.node or '?'} "
            f"(partition {self.partition or '?'}, task {self.task_id + 1}/"
            f"{self.task_count}, {rem_s})"
        )


def _job_end_time(job_id: str) -> float | None:
    """When walltime expires, as a unix timestamp.

    ``SLURM_JOB_END_TIME`` is present on recent SLURM and is the cheap path. Older
    versions need ``scontrol``, which is a fork per call, so it is done once at startup
    and cached in the context.
    """
    raw = os.environ.get("SLURM_JOB_END_TIME", "").strip()
    if raw.isdigit():
        return float(raw)
    if not job_id:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", job_id, "--oneliner"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return None
    for field_name in ("EndTime=",):
        for tok in out.split():
            if tok.startswith(field_name):
                stamp = tok[len(field_name):]
                for fmt in ("%Y-%m-%dT%H:%M:%S",):
                    try:
                        return time.mktime(time.strptime(stamp, fmt))
                    except ValueError:
                        pass
    return None


def slurm_context() -> SlurmContext:
    env = os.environ
    job_id = env.get("SLURM_JOB_ID", "") or env.get("SLURM_JOBID", "")
    if not job_id:
        return SlurmContext(in_slurm=False)

    task_id = int(env.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    count = env.get("SLURM_ARRAY_TASK_COUNT", "")
    if count.isdigit():
        task_count = int(count)
    else:
        # SLURM_ARRAY_TASK_COUNT is not set on every version; derive it from the min/max
        # pair, which is.
        lo = env.get("SLURM_ARRAY_TASK_MIN", "")
        hi = env.get("SLURM_ARRAY_TASK_MAX", "")
        task_count = (int(hi) - int(lo) + 1) if lo.isdigit() and hi.isdigit() else 1

    # Array indices usually start at 0, but a 1-based array is common too; normalise so
    # shard arithmetic never depends on which convention the submitter used.
    lo = env.get("SLURM_ARRAY_TASK_MIN", "")
    if lo.isdigit():
        task_id -= int(lo)

    return SlurmContext(
        in_slurm=True,
        job_id=job_id,
        array_job_id=env.get("SLURM_ARRAY_JOB_ID", ""),
        task_id=max(task_id, 0),
        task_count=max(task_count, 1),
        node=env.get("SLURMD_NODENAME", "") or env.get("SLURM_NODELIST", ""),
        partition=env.get("SLURM_JOB_PARTITION", ""),
        cpus=int(env.get("SLURM_CPUS_PER_TASK", "0") or 0),
        gpus_env=env.get("SLURM_JOB_GPUS", "") or env.get("SLURM_STEP_GPUS", ""),
        end_time=_job_end_time(job_id),
    )


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #


def estimate_cost_s(config, replay_cfg) -> float:
    """Roughly how long one config takes to measure.

    Mostly constant, because the harness targets a fixed *window* rather than a fixed
    iteration count -- a fast kernel simply runs more iterations in the same three
    seconds. What does vary is setup: allocating and filling several copies of a large
    working set costs real time, and the biggest attention configs allocate gigabytes.
    """
    fixed = (
        replay_cfg.warmup_s
        + replay_cfg.repeats * replay_cfg.target_window_s
        + replay_cfg.repeats * replay_cfg.inter_repeat_sleep_s
        + 1.5  # calibration probe, allocation, teardown
    )
    try:
        from kernelenergy.kernels.registry import make_kernel

        gib = make_kernel(config).working_set_bytes() * replay_cfg.n_buffers / 2**30
    except Exception:
        gib = 0.5
    return fixed + 0.4 * gib  # ~0.4 s per GiB to allocate and fill


def shard_configs(configs, task_id: int, task_count: int, replay_cfg=None) -> list:
    """Split the catalogue across array tasks, balanced by estimated cost.

    Longest-processing-time-first greedy: sort by descending cost and repeatedly give the
    next config to the shard with the least work so far. It is the standard makespan
    heuristic and is within 4/3 of optimal, which is far better than the round-robin the
    obvious ``configs[i::n]`` would give when costs are uneven.

    Deterministic, and independent of the order the configs arrived in -- ties break on
    the signature. That matters for resumption: a requeued task must reconstruct exactly
    the shard it had before, or it will re-measure work another task already did and skip
    work nobody did.
    """
    if task_count <= 1:
        return list(configs)
    if not 0 <= task_id < task_count:
        raise ValueError(f"task_id {task_id} outside 0..{task_count - 1}")

    if replay_cfg is None:
        from kernelenergy.measure.replay import ReplayConfig

        replay_cfg = ReplayConfig()

    scored = sorted(
        ((estimate_cost_s(c, replay_cfg), c.signature(), c) for c in configs),
        key=lambda t: (-t[0], t[1]),
    )
    loads = [0.0] * task_count
    shards: list[list] = [[] for _ in range(task_count)]
    for cost, _sig, cfg in scored:
        j = min(range(task_count), key=lambda k: (loads[k], k))
        shards[j].append(cfg)
        loads[j] += cost
    return shards[task_id]


# --------------------------------------------------------------------------- #
# Interruption
# --------------------------------------------------------------------------- #


class InterruptGuard:
    """Catch scheduler signals and stop between kernels rather than during one.

    SLURM can be asked to warn before it kills (``--signal=B:USR1@300`` sends SIGUSR1
    five minutes ahead of walltime). Preemption sends SIGTERM. Both are handled the same
    way: raise a flag, let the sweep finish the kernel it is measuring, and exit cleanly
    so the partial results file stays valid and the next run resumes from it.

    Handlers do nothing but set a flag, which is the only thing that is safe to do in a
    signal handler while NVML and CUDA calls are in flight on other threads.
    """

    def __init__(self, signals=(signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)):
        self.signals = signals
        self.triggered_by: str | None = None
        self._previous: dict = {}

    @property
    def interrupted(self) -> bool:
        return self.triggered_by is not None

    def _handle(self, signum, _frame) -> None:
        if self.triggered_by is None:
            self.triggered_by = signal.Signals(signum).name

    def __enter__(self) -> "InterruptGuard":
        for s in self.signals:
            try:
                self._previous[s] = signal.getsignal(s)
                signal.signal(s, self._handle)
            except (ValueError, OSError):
                pass  # not the main thread, or signal unavailable on this platform
        return self

    def __exit__(self, *exc) -> None:
        for s, prev in self._previous.items():
            try:
                signal.signal(s, prev)
            except (ValueError, OSError):
                pass
