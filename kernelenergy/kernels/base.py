"""Kernel abstraction: one class per kernel category, replayable and decomposable.

Each ``KernelSpec`` does three jobs from a single config:

* ``build`` / ``run``   -- reconstruct the kernel in isolation so the replay harness can
                           execute it in a tight loop and measure its energy.
* ``decompose``         -- PipeWeave's Kernel Decomposer (IV-A): split the kernel into
                           *tasks*, the schedulable packets of work an SM takes on, each
                           carrying its own pipeline demands.
* ``roofline``          -- total FLOPs and bytes, used for sanity checks and baselines.

The task is the unit that matters. PipeWeave's whole argument is that a tile-level
descriptor fed straight to an ML model is too coarse, because it collapses heterogeneous
pipeline activity -- tensor, FMA, special-function, memory -- into one aggregate number
and treats the SM as a black box. A ``Task`` here keeps those demands separate, and the
Feature Analyzer aggregates them without ever mixing them.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar

__all__ = [
    "Task",
    "KernelConfig",
    "KernelSpec",
    "dtype_bytes",
    "DTYPES",
    "choose_gemm_tile",
    "TileChoice",
]

# --------------------------------------------------------------------------- #
# dtypes
# --------------------------------------------------------------------------- #

DTYPES = {
    "fp32": 4,
    "tf32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
}


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPES)}") from None


def uses_tensor_core(dtype: str) -> bool:
    return dtype in ("tf32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8")


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """One schedulable packet of work, with its demand on each instruction pipeline.

    Counts are *operations issued to that pipeline*, not FLOPs.

    ``n_mma`` counts multiply-accumulate output-element updates on the Tensor pipe, so
    a GEMM tile contributes ``alpha * tile_M * tile_N * tile_K`` with alpha = 2 for a
    plain matmul and 4 for a FlashAttention task, which does two chained matmuls per
    task (PipeWeave eq. 3).

    Byte counts are split by the level they are served from, because a byte from shared
    memory and a byte from HBM cost different amounts of both time and energy -- and
    energy is where the split matters most, since off-chip movement dominates the
    picture in a way it does not for latency alone.
    """

    n_mma: float = 0.0  # Tensor pipe
    n_fma: float = 0.0  # FMA pipe (FP32 add/mul/fma)
    n_xu: float = 0.0  # XU pipe (exp2, rcp, rsqrt, sin, ...)
    bytes_global: float = 0.0  # traffic that reaches HBM
    bytes_l2: float = 0.0  # traffic that reaches L2 (>= global)
    bytes_shared: float = 0.0  # traffic through shared memory / SMEM
    count: int = 1  # multiplicity of identical tasks
    label: str = ""

    def scaled(self, k: float) -> "Task":
        return Task(
            n_mma=self.n_mma * k,
            n_fma=self.n_fma * k,
            n_xu=self.n_xu * k,
            bytes_global=self.bytes_global * k,
            bytes_l2=self.bytes_l2 * k,
            bytes_shared=self.bytes_shared * k,
            count=self.count,
            label=self.label,
        )

    @property
    def total_ops(self) -> float:
        return self.n_mma + self.n_fma + self.n_xu


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def _canonical(v: Any) -> Any:
    """Reduce a parameter value to a plain Python type with a stable JSON encoding.

    ``bool`` is checked before ``int`` because ``bool`` subclasses ``int``, and a CSV
    round trip must not turn ``causal=False`` into ``0``. Whole-valued floats become
    ints for the same reason: pandas reads an integer column containing any NaN as
    float, so ``m=4608`` comes back as ``4608.0``.
    """
    import numbers

    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, (str, bytes)):
        s = v.decode() if isinstance(v, bytes) else v
        # pandas writes booleans as the strings "True"/"False".
        if s in ("True", "False"):
            return s == "True"
        return s
    return v


@dataclass
class KernelConfig:
    """A kernel plus the input parameters that determine its work.

    This is the row key of the whole dataset: one measured energy per
    ``(KernelConfig, GPU)`` pair.
    """

    category: str  # gemm | attention | conv | norm | elementwise
    dtype: str
    params: dict[str, Any] = field(default_factory=dict)
    # provenance -- which pipeline and module this was captured from
    source_model: str = ""
    source_module: str = ""
    op_name: str = ""

    def __post_init__(self) -> None:
        # Params arrive from three places -- literals in code, NumPy scalars out of a
        # capture, and whatever pandas infers when a catalogue CSV is read back. Those
        # must hash identically or a saved catalogue reloads as a different set of
        # kernels and every (kernel_sig, gpu_key) join silently misses. Normalising
        # here, once, is what makes the signature stable across a round trip.
        object.__setattr__(self, "params",
                           {k: _canonical(v) for k, v in self.params.items()})

    def signature(self) -> str:
        """Stable identity, ignoring provenance.

        Two configs with the same signature are the same measurement, whichever model
        they were captured from -- so a GEMM shape shared by FLUX and SD3.5 is measured
        once and carries both provenance tags.
        """
        payload = {
            "category": self.category,
            "dtype": self.dtype,
            "params": {k: self.params[k] for k in sorted(self.params)},
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def as_row(self) -> dict:
        row = {
            "kernel_sig": self.signature(),
            "category": self.category,
            "dtype": self.dtype,
            "source_model": self.source_model,
            "source_module": self.source_module,
            "op_name": self.op_name,
        }
        for k, v in self.params.items():
            row[f"p_{k}"] = v
        return row

    @classmethod
    def from_row(cls, row: dict) -> "KernelConfig":
        params = {
            k[2:]: v for k, v in row.items()
            if k.startswith("p_") and v == v and v is not None  # drop NaN
        }
        return cls(
            category=row["category"],
            dtype=row["dtype"],
            params=params,
            source_model=row.get("source_model", "") or "",
            source_module=row.get("source_module", "") or "",
            op_name=row.get("op_name", "") or "",
        )


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TileChoice:
    m: int
    n: int
    k: int
    persistent: bool = False
    source: str = "heuristic"


# cuBLAS is closed-source, so PipeWeave recovers its tiling empirically: profile the
# kernel over many (M,N,K) and correlate kernel name, CTA count and input size to
# reverse-engineer the partitioning (Sec. V-A). The table below is the surrogate F --
# a defensible default, not ground truth. `calibrate_tiles.py` replaces it with values
# fitted to observed grid dimensions on your own cards, which is what you should run
# before trusting the analytical features on a new architecture.
_TILE_LADDER: tuple[TileChoice, ...] = (
    TileChoice(256, 128, 64),
    TileChoice(128, 256, 64),
    TileChoice(128, 128, 64),
    TileChoice(128, 64, 64),
    TileChoice(64, 128, 64),
    TileChoice(64, 64, 32),
    TileChoice(32, 32, 32),
)


def choose_gemm_tile(
    m: int, n: int, k: int, n_sms: int, dtype: str = "bf16",
    persistent_threshold: float = 4.0,
) -> TileChoice:
    """Pick the tile cuBLAS would plausibly pick.

    The rule that matters is wave quantisation: a tile so large that the grid does not
    fill the SMs wastes most of the machine, so the heuristic walks down the ladder
    until the CTA count covers the SMs at least twice, then stops. Kernels whose grid
    greatly exceeds the SM count are marked persistent, since modern cuBLAS and
    CUTLASS use stream-K / persistent designs there and PipeWeave schedules those
    differently (IV-B).
    """
    best = _TILE_LADDER[-1]
    for tile in _TILE_LADDER:
        ctas = math.ceil(m / tile.m) * math.ceil(n / tile.n)
        if ctas >= 2 * n_sms:
            best = tile
            break
        best = tile if ctas >= n_sms else best
    ctas = math.ceil(m / best.m) * math.ceil(n / best.n)
    persistent = ctas > persistent_threshold * n_sms
    return TileChoice(best.m, best.n, best.k, persistent=persistent, source="heuristic")


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


class KernelSpec:
    """Base class. One subclass per kernel category."""

    category: ClassVar[str] = ""
    #: parameter names required in ``KernelConfig.params``
    required: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: KernelConfig) -> None:
        missing = [p for p in self.required if p not in config.params]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: config is missing {missing}; "
                f"has {sorted(config.params)}"
            )
        self.config = config
        self.p = config.params
        self.dtype = config.dtype
        self.dsize = dtype_bytes(config.dtype)

    # -- identity ----------------------------------------------------------- #

    def describe(self) -> str:
        ps = ",".join(f"{k}={self.p[k]}" for k in sorted(self.p))
        return f"{self.category}[{self.dtype}]({ps})"

    # -- replay (torch, GPU only) ------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        """Allocate the inputs. Returns an opaque state passed to ``run``."""
        raise NotImplementedError

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        """Execute the kernel once. Must launch work and must not synchronise."""
        raise NotImplementedError

    def working_set_bytes(self) -> float:
        """Total bytes resident, used to check the config fits in memory."""
        raise NotImplementedError

    # -- analysis ----------------------------------------------------------- #

    def decompose(self, gpu) -> list[Task]:
        """PipeWeave IV-A. Split into tasks with per-pipeline demands."""
        raise NotImplementedError

    def flops(self) -> float:
        raise NotImplementedError

    def bytes_global(self) -> float:
        raise NotImplementedError

    def arithmetic_intensity(self) -> float:
        b = self.bytes_global()
        return self.flops() / b if b > 0 else float("inf")

    # -- shared helpers ----------------------------------------------------- #

    @staticmethod
    def _gemm_tasks(
        m: int, n: int, k: int, dsize: int, tile: TileChoice,
        alpha: float = 2.0, accum_dsize: int = 4,
    ) -> list[Task]:
        """Task decomposition shared by GEMM and implicit-GEMM convolution.

        One task per output tile. Each computes ``alpha * tile_M * tile_N * K``
        MMA ops, streams its A panel and B panel through shared memory, and writes
        its output tile once.

        The global/L2 split is where the tiling shows up. In the ideal case each input
        element crosses HBM once, so global traffic is ``(MK + KN + MN) * dsize``.
        But an A panel is re-read once per output column-block and a B panel once per
        output row-block, and those re-reads are what L2 is for -- so L2 traffic carries
        the ``ceil(N/tile_N)`` and ``ceil(M/tile_M)`` multipliers. The ratio between the
        two is the reuse factor, and it is the single most informative memory feature
        for a GEMM.
        """
        tm, tn, tk = tile.m, tile.n, tile.k
        n_m = math.ceil(m / tm)
        n_n = math.ceil(n / tn)
        n_tiles = n_m * n_n

        per_tile_mma = alpha * tm * tn * k
        # Per tile: read A panel (tm x k) and B panel (k x tn), write C tile.
        per_tile_l2 = (tm * k + k * tn) * dsize
        per_tile_out = tm * tn * accum_dsize
        # Everything a tile reads passes through shared memory, staged tk at a time.
        per_tile_smem = (tm * k + k * tn) * dsize

        # Ideal HBM traffic, divided evenly over tiles so per-task numbers stay
        # meaningful when the scheduler splits them across SMs.
        global_total = (m * k + k * n) * dsize + m * n * accum_dsize
        per_tile_global = global_total / max(n_tiles, 1)

        return [
            Task(
                n_mma=per_tile_mma,
                n_fma=0.0,
                n_xu=0.0,
                bytes_global=per_tile_global,
                bytes_l2=per_tile_l2 + per_tile_out,
                bytes_shared=per_tile_smem,
                count=n_tiles,
                label=f"gemm_tile_{tm}x{tn}x{tk}",
            )
        ]
