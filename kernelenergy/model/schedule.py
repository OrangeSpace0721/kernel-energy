"""Scheduling Simulator -- PipeWeave IV-B.

Turns an unordered set of tasks into a *task distribution*: which SM gets what. This is
the step that separates PipeWeave from tile-level predictors, and it exists to capture
one thing they miss -- load imbalance. A kernel whose tasks divide evenly across SMs and
one whose tasks leave a ragged final wave have the same total work and different
execution behaviour, and the gap shows up as the difference between the *mean* SM and the
*max* SM, which is what the feature vector then carries.

Two paradigms, as in the paper:

* **Hardware scheduler** (round-robin). The GigaThread engine hands CTAs to SMs in
  rounds, filling every SM once before starting a second round, and issues a replacement
  CTA when a resident one retires. Its exact policy is undocumented; round-robin is the
  standard empirical inference and is what PipeWeave uses.

* **Software scheduler** (persistent kernels). A fixed grid of long-lived CTAs pulls
  tiles off a shared queue. Assignment is dynamic, so the distribution is near-perfectly
  balanced regardless of how the tile count divides -- the tail effect largely disappears
  and only the last tile's residue remains.

Which one applies is a property of the kernel implementation, not the shape. Modern
cuBLAS and CUTLASS GEMMs and FlashAttention-3 use persistent designs; FlashAttention-2
and most elementwise kernels do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kernelenergy.kernels.base import Task

__all__ = ["TaskDistribution", "simulate", "SM_LOAD_FIELDS"]

SM_LOAD_FIELDS = (
    "n_mma", "n_fma", "n_xu", "bytes_global", "bytes_l2", "bytes_shared",
)

#: How many CTAs of a category are typically resident on one SM at once. Real occupancy
#: depends on register and shared-memory use, which we do not have without Nsight
#: Compute; these are the defaults observed for well-tuned kernels of each shape. They
#: matter only through wave counting, not through the totals.
DEFAULT_CTAS_PER_SM = {
    "gemm": 2,
    "conv": 2,
    "attention": 2,
    "norm": 8,
    "elementwise": 16,
}


@dataclass
class SMLoad:
    """Aggregate demand landing on one SM."""

    n_mma: float = 0.0
    n_fma: float = 0.0
    n_xu: float = 0.0
    bytes_global: float = 0.0
    bytes_l2: float = 0.0
    bytes_shared: float = 0.0
    n_tasks: int = 0

    def add(self, task: Task, k: int = 1) -> None:
        self.n_mma += task.n_mma * k
        self.n_fma += task.n_fma * k
        self.n_xu += task.n_xu * k
        self.bytes_global += task.bytes_global * k
        self.bytes_l2 += task.bytes_l2 * k
        self.bytes_shared += task.bytes_shared * k
        self.n_tasks += k


@dataclass
class TaskDistribution:
    """The partition ``{T_1, ..., T_Nsm}`` of PipeWeave eq. 2."""

    n_sms: int
    loads: list[SMLoad]
    n_tasks_total: int
    n_waves: float
    tail_fraction: float  # how much of the last wave is idle, 0 = perfectly even
    scheduler: str

    # -- aggregates the Feature Analyzer consumes --------------------------- #

    def total(self, field_name: str) -> float:
        return sum(getattr(l, field_name) for l in self.loads)

    def max_sm(self, field_name: str) -> float:
        return max((getattr(l, field_name) for l in self.loads), default=0.0)

    def mean_sm(self, field_name: str) -> float:
        return self.total(field_name) / max(self.n_sms, 1)

    def imbalance(self, field_name: str) -> float:
        """Max over mean. 1.0 is perfect balance; the tail shows up above it."""
        mean = self.mean_sm(field_name)
        return self.max_sm(field_name) / mean if mean > 0 else 1.0

    @property
    def sm_occupancy(self) -> float:
        """Fraction of SMs that receive any work at all."""
        return sum(1 for l in self.loads if l.n_tasks > 0) / max(self.n_sms, 1)


def simulate(
    tasks: list[Task],
    n_sms: int,
    scheduler: str = "hardware",
    ctas_per_sm: int = 2,
) -> TaskDistribution:
    """Map tasks onto SMs.

    ``tasks`` is the decomposer's output: entries with ``count > 1`` stand for that many
    identical tasks, which is the common case and is expanded here rather than in the
    decomposer so a kernel with a million tiles costs a million integers, not a million
    objects.
    """
    if n_sms <= 0:
        raise ValueError("n_sms must be positive")
    loads = [SMLoad() for _ in range(n_sms)]
    total = sum(int(t.count) for t in tasks)
    if total == 0:
        return TaskDistribution(n_sms, loads, 0, 0.0, 0.0, scheduler)

    if scheduler == "software":
        # Persistent CTAs pull from a queue, so work splits as evenly as the arithmetic
        # allows and the identity of the task no longer matters to placement.
        _distribute_even(tasks, loads, n_sms)
    elif scheduler == "hardware":
        _distribute_round_robin(tasks, loads, n_sms)
    else:
        raise ValueError(f"scheduler must be 'hardware' or 'software', got {scheduler!r}")

    slots = n_sms * max(ctas_per_sm, 1)
    n_waves = total / slots
    tail = math.ceil(n_waves) - n_waves  # 0 when the last wave is full

    return TaskDistribution(
        n_sms=n_sms,
        loads=loads,
        n_tasks_total=total,
        n_waves=n_waves,
        tail_fraction=tail,
        scheduler=scheduler,
    )


def _distribute_round_robin(tasks: list[Task], loads: list[SMLoad], n_sms: int) -> None:
    """Deal tasks to SMs in rounds, in decomposition order.

    Expanded in bulk: a run of ``c`` identical tasks starting at cursor ``i`` gives
    ``c // n_sms`` to every SM and one extra to the first ``c % n_sms`` SMs it reaches.
    Same answer as dealing one at a time, without the loop.
    """
    cursor = 0
    for t in tasks:
        c = int(t.count)
        if c <= 0:
            continue
        base, rem = divmod(c, n_sms)
        if base:
            for l in loads:
                l.add(t, base)
        for j in range(rem):
            loads[(cursor + j) % n_sms].add(t, 1)
        cursor = (cursor + c) % n_sms


def _distribute_even(tasks: list[Task], loads: list[SMLoad], n_sms: int) -> None:
    """Balance total demand rather than task count, as a work queue does.

    Greedy: hand each block of identical tasks to the SMs with the least MMA work so
    far. With homogeneous tasks this reduces to the same floor/ceil split as
    round-robin, which is the correct behaviour -- persistent scheduling only helps when
    the tasks are heterogeneous.
    """
    order = sorted(range(n_sms), key=lambda j: loads[j].n_mma)
    for t in sorted(tasks, key=lambda x: -x.n_mma):
        c = int(t.count)
        if c <= 0:
            continue
        base, rem = divmod(c, n_sms)
        if base:
            for l in loads:
                l.add(t, base)
        order = sorted(range(n_sms), key=lambda j: loads[j].n_mma)
        for j in order[:rem]:
            loads[j].add(t, 1)
