"""Append-safe CSV writer, and the sweep driver.

A sweep across six cards takes hours and will be interrupted -- by a preemption, a
driver hiccup, an OOM on the largest config, or someone needing the card. So results are
appended row by row and the sweep is resumable: on restart it reads what is already
there and skips those ``(kernel_sig, gpu_key)`` pairs. Nothing is held in memory that
would be lost.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from kernelenergy.hardware import GPU
from kernelenergy.kernels.base import KernelConfig
from kernelenergy.measure.replay import (
    OutOfMemoryError,
    ReplayConfig,
    measure_kernel,
    measure_launch_overhead,
)
from kernelenergy.measure.schema import MeasurementRow

__all__ = ["ResultWriter", "SweepResult", "run_sweep", "load_results", "merge_results"]


class ResultWriter:
    """Row-at-a-time CSV appender that tolerates a growing column set.

    Kernel parameters arrive as ``p_*`` columns and differ by category, so the header is
    not known until the first row of each category. Rather than rewrite the file every
    time a new column appears, each category gets its own file and ``merge_results``
    joins them.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._writer = None
        self._columns: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="") as f:
                header = next(csv.reader(f), None)
            self._columns = header

    def __enter__(self) -> "ResultWriter":
        self._fh = self.path.open("a", newline="")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, row: MeasurementRow) -> None:
        d = row.as_row()
        if self._columns is None:
            self._columns = list(d)
            self._writer = csv.DictWriter(self._fh, fieldnames=self._columns)
            self._writer.writeheader()
        elif self._writer is None:
            self._writer = csv.DictWriter(
                self._fh, fieldnames=self._columns, extrasaction="ignore"
            )
        extra = set(d) - set(self._columns)
        if extra:
            # A new parameter appeared mid-file. Dropping it silently would lose data,
            # so say so loudly rather than writing a subtly incomplete dataset.
            raise ValueError(
                f"{self.path.name}: row introduces columns {sorted(extra)} not in the "
                f"existing header. Write this category to its own file, or delete the "
                f"partial file and restart the sweep."
            )
        self._writer.writerow({c: d.get(c, "") for c in self._columns})
        self._fh.flush()
        os.fsync(self._fh.fileno())


def _already_done(out_dir: Path, gpu_key: str) -> set[tuple[str, str]]:
    """Every (kernel_sig, gpu_key) already recorded for this card, across all shards.

    Scanning *all* of the card's files rather than only the one this task writes is what
    makes an array resumable after the shard count changes. Resubmit a 4-way array as an
    8-way one and each new task still skips whatever any earlier task measured.
    """
    done: set[tuple[str, str]] = set()
    for path in out_dir.glob(f"{gpu_key}__*.csv"):
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, usecols=["kernel_sig", "gpu_key"])
        except Exception:
            continue
        done |= set(map(tuple, df[["kernel_sig", "gpu_key"]].astype(str).to_numpy()))
    return done


@dataclass
class SweepResult:
    """What happened, and whether there is more to do."""

    measured: int = 0
    skipped_done: int = 0
    skipped_oom: int = 0
    failed: int = 0
    remaining: int = 0
    stopped_reason: str = "complete"  # complete | signal | walltime
    idle_power_w: float = float("nan")
    overhead_s: float = float("nan")
    overhead_j: float = float("nan")
    elapsed_s: float = 0.0

    @property
    def incomplete(self) -> bool:
        return self.remaining > 0

    def summary(self) -> str:
        return (
            f"{self.measured} measured, {self.skipped_done} already done, "
            f"{self.skipped_oom} skipped (oom), {self.failed} failed, "
            f"{self.remaining} remaining -- stopped: {self.stopped_reason} "
            f"after {self.elapsed_s / 60:.1f} min"
        )


def run_sweep(
    configs: Iterable[KernelConfig],
    gpu: GPU,
    out_dir: str | Path,
    cfg: ReplayConfig | None = None,
    measure_idle_first: bool = True,
    resume: bool = True,
    progress: bool = True,
    shard: tuple[int, int] | None = None,
    time_budget_s: float | None = None,
    interrupt=None,
    on_row=None,
) -> SweepResult:
    """Measure every config on the local GPU, writing one CSV per (category, shard).

    Under a scheduler three things differ from a workstation run:

    * **Output files are per shard.** Array tasks on a shared filesystem must not append
      to the same file -- concurrent appends on Lustre or GPFS interleave at unpredictable
      boundaries and corrupt rows. Separate files, merged later, avoids the problem
      instead of trying to lock around it.
    * **The sweep stops before walltime.** Starting a 12-second measurement with 4
      seconds left wastes it and risks writing a truncated window as though it were real.
    * **Signals stop it cleanly.** ``interrupt`` is an ``InterruptGuard``; when it fires,
      the kernel in flight completes and the sweep returns with ``remaining > 0`` so the
      caller can requeue.
    """
    from kernelenergy.hpc.slurm import estimate_cost_s, slurm_context
    from kernelenergy.nvml import measure_idle

    cfg = cfg or ReplayConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = slurm_context()
    nvml_index = cfg.resolved_nvml_index()

    started = time.monotonic()
    deadline = started + time_budget_s if time_budget_s else None
    shard_label = f"{shard[0]}of{shard[1]}" if shard else ""
    result = SweepResult()

    idle = None
    if measure_idle_first:
        if progress:
            print("measuring idle power (~25 s; the card must be quiet)")
        idle = measure_idle(nvml_index).power_w
        result.idle_power_w = idle
        if progress:
            print(f"  idle = {idle:.1f} W")

    if progress:
        print("pricing launch overhead")
    overhead = measure_launch_overhead(cfg)
    result.overhead_s, result.overhead_j = overhead
    if progress:
        print(f"  {overhead[0] * 1e6:.1f} us, {overhead[1] * 1e6:.2f} uJ per launch")

    configs = list(configs)
    done = _already_done(out_dir, gpu.gpu_key) if resume else set()
    writers: dict[str, ResultWriter] = {}

    def _writer(cat: str) -> ResultWriter:
        if cat not in writers:
            name = f"{gpu.gpu_key}__{cat}"
            if shard_label:
                name += f"__{shard_label}"
            writers[cat] = ResultWriter(out_dir / f"{name}.csv").__enter__()
        return writers[cat]

    try:
        for i, config in enumerate(configs, 1):
            if (config.signature(), gpu.gpu_key) in done:
                result.skipped_done += 1
                continue

            if interrupt is not None and interrupt.interrupted:
                result.stopped_reason = f"signal ({interrupt.triggered_by})"
                result.remaining = len(configs) - i + 1
                break

            if deadline is not None:
                need = estimate_cost_s(config, cfg)
                if time.monotonic() + need > deadline:
                    # Do not start what cannot finish. A truncated window is worse than
                    # a missing row: the row would look real.
                    result.stopped_reason = "walltime"
                    result.remaining = len(configs) - i + 1
                    break

            try:
                row, res = measure_kernel(config, gpu, cfg, idle_power_w=idle,
                                          overhead=overhead)
            except OutOfMemoryError as e:
                result.skipped_oom += 1
                if progress:
                    print(f"[{i}/{len(configs)}] skip (oom): {e}")
                continue
            except Exception as e:  # keep the sweep alive; one bad config is not fatal
                result.failed += 1
                if progress:
                    print(f"[{i}/{len(configs)}] FAILED {config.signature()}: "
                          f"{type(e).__name__}: {e}")
                continue

            row.slurm_job = ctx.label
            row.shard = f"{shard[0]}/{shard[1]}" if shard else ""
            _writer(config.category).write(row)
            done.add((config.signature(), gpu.gpu_key))
            result.measured += 1

            if progress:
                flag = "  CONTENDED" if row.contended else ""
                print(
                    f"[{i}/{len(configs)}] {config.category:12s} {config.signature()} "
                    f"{res.latency_s * 1e6:9.1f} us  {res.energy_j * 1e3:8.3f} mJ  "
                    f"{res.power_avg_w:6.1f} W  (sd {res.energy_sd_rel:.1%}){flag}"
                )
            if on_row is not None:
                on_row(row, res)
    finally:
        for w in writers.values():
            w.__exit__(None, None, None)
        result.elapsed_s = time.monotonic() - started

    return result


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_results(out_dir: str | Path, pattern: str = "*.csv",
                  drop_contended: bool = False) -> pd.DataFrame:
    """Concatenate every per-(GPU, category, shard) file into one frame.

    Columns present in some categories and not others become NaN, which is correct: a
    GEMM has no ``p_s_kv``.

    Duplicates arise routinely on a cluster -- a requeued array task, an overlapping
    resubmission, a shard count that changed between runs. When the same
    ``(kernel_sig, gpu_key)`` appears more than once the *uncontended* row wins, and
    among equals the later one does. Preferring uncontended matters: a row measured while
    a neighbour shared the board carries their joules too, so if a clean measurement of
    the same kernel exists anywhere it is unambiguously the better one.
    """
    out_dir = Path(out_dir)
    files = sorted(p for p in out_dir.glob(pattern) if p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"no result CSVs in {out_dir}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True, sort=False)

    if "contended" in df.columns:
        df["contended"] = df["contended"].fillna(0).astype(int)
        if drop_contended:
            n = int(df["contended"].sum())
            if n:
                print(f"merge_results: dropped {n} contended rows")
            df = df[df["contended"] == 0]
        else:
            # Sort contended first so keep='last' prefers the clean row.
            df = df.sort_values("contended", ascending=False, kind="stable")

    return df.drop_duplicates(subset=["kernel_sig", "gpu_key"], keep="last")
