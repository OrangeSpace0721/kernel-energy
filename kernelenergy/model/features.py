"""Feature Analyzer -- PipeWeave IV-C, plus the extension energy needs.

PipeWeave turns a task distribution into two things per instruction pipeline: the
**demand** (how many operations or bytes) and the **theoretical cycles** (what that demand
would cost if that pipeline alone were the bottleneck). Each is reported at two
granularities, whole-GPU and worst-SM, giving the feature vector of its Table IV. The
point of keeping the pipelines separate rather than collapsing them into one roofline is
that their interaction is exactly what the MLP is there to learn.

Where this departs from the paper
---------------------------------
The latency features are hardware-aware only through the throughputs in the denominators
-- an A100 and an H100 differ because their cycles differ. That is sufficient when the
target is time. It is not sufficient when the target is energy, because two cards can
have identical theoretical cycles and very different power draw: TDP is not a function of
throughput. So ``power_features`` adds an explicit power-domain block: TDP, the enforced
limit, memory technology, and the energy-relevant *ratios* -- joules of budget per unit of
peak throughput, and bytes moved per flop.

The bytes-per-flop ratio deserves its own mention. It is a weak feature for latency,
where a memory-bound kernel and a compute-bound one of equal duration cost the same. It
is a strong one for energy, where moving a byte off-chip costs orders of magnitude more
than the arithmetic consuming it, so two kernels of equal duration can differ severalfold
in joules purely by where their bytes came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kernelenergy.hardware import GPU
from kernelenergy.kernels.base import KernelSpec
from kernelenergy.kernels.registry import make_kernel
from kernelenergy.model.schedule import DEFAULT_CTAS_PER_SM, TaskDistribution, simulate

__all__ = [
    "analyse",
    "feature_row",
    "FEATURE_COLUMNS",
    "MATH_PIPES",
    "AnalysisResult",
]

MATH_PIPES = ("tensor", "fma", "xu")

_EPS = 1e-30


def _log(x: float) -> float:
    """Log with a floor, so an all-zero pipeline gives a finite, constant feature."""
    return math.log(max(float(x), _EPS))


@dataclass
class AnalysisResult:
    """Everything the analytical stack derives for one (kernel, GPU) pair."""

    features: dict[str, float]
    distribution: TaskDistribution
    theoretical_time_s: float  # the roofline floor: max over pipelines
    bottleneck: str
    flops: float
    bytes_global: float


def _pipe_throughputs(gpu: GPU) -> dict[str, float]:
    """Ops per clock per SM for each math pipeline (PipeWeave Table II)."""
    return {
        "tensor": float(gpu.ops_per_clk),
        "fma": float(gpu.fma_per_clk),
        "xu": float(gpu.xu_per_clk),
    }


def _pipe_demand(dist: TaskDistribution, pipe: str, which: str) -> float:
    field = {"tensor": "n_mma", "fma": "n_fma", "xu": "n_xu"}[pipe]
    return dist.total(field) if which == "gpu" else dist.max_sm(field)


def analyse(
    kernel: KernelSpec,
    gpu: GPU,
    clock: str = "tensor",
    scheduler: str | None = None,
    ctas_per_sm: int | None = None,
) -> AnalysisResult:
    """Run decompose -> schedule -> features for one kernel on one GPU."""
    tasks = kernel.decompose(gpu)

    if scheduler is None:
        # cuBLAS GEMM and FA3 are persistent on Hopper; treat everything else as
        # hardware-scheduled. See PipeWeave Sec. V-A.
        persistent = gpu.architecture == "Hopper" and kernel.category in ("gemm", "attention")
        scheduler = "software" if persistent else "hardware"
    if ctas_per_sm is None:
        ctas_per_sm = DEFAULT_CTAS_PER_SM.get(kernel.category, 2)

    dist = simulate(tasks, gpu.sms, scheduler=scheduler, ctas_per_sm=ctas_per_sm)

    clock_hz = (gpu.tensor_clock_mhz if clock == "tensor" else gpu.boost_clock_mhz) * 1e6
    th = _pipe_throughputs(gpu)

    f: dict[str, float] = {}

    # ---- math pipelines ---------------------------------------------------- #
    pipe_times: dict[str, float] = {}
    for p in MATH_PIPES:
        n_gpu = _pipe_demand(dist, p, "gpu")
        n_sm = _pipe_demand(dist, p, "sm")
        # PipeWeave eq. 4/5: cycles = ops / throughput, aggregated at each granularity.
        c_gpu = n_gpu / (gpu.sms * th[p]) if th[p] > 0 else 0.0
        c_sm = n_sm / th[p] if th[p] > 0 else 0.0
        f[f"log_ops_{p}_gpu"] = _log(n_gpu)
        f[f"log_ops_{p}_sm_max"] = _log(n_sm)
        f[f"log_cycles_{p}_gpu"] = _log(c_gpu)
        f[f"log_cycles_{p}_sm_max"] = _log(c_sm)
        # The worst SM is what the kernel actually waits for.
        pipe_times[p] = c_sm / clock_hz

    # ---- MIO pipelines ----------------------------------------------------- #
    bw = {
        "global": gpu.mem_bandwidth_gbs * 1e9,
        "l2": gpu.l2_bandwidth_gbs * 1e9,
        "shared": gpu.smem_bandwidth_gbs(clock) * 1e9,
    }
    per_sm_bw = {k: v / gpu.sms for k, v in bw.items()}

    for level, attr in (("global", "bytes_global"), ("l2", "bytes_l2"),
                        ("shared", "bytes_shared")):
        b_gpu = dist.total(attr)
        b_sm = dist.max_sm(attr)
        t_gpu = b_gpu / bw[level] if bw[level] > 0 else 0.0
        t_sm = b_sm / per_sm_bw[level] if per_sm_bw[level] > 0 else 0.0
        f[f"log_bytes_{level}_gpu"] = _log(b_gpu)
        f[f"log_bytes_{level}_sm_max"] = _log(b_sm)
        f[f"log_cycles_mem_{level}_gpu"] = _log(t_gpu * clock_hz)
        f[f"log_cycles_mem_{level}_sm_max"] = _log(t_sm * clock_hz)
        # Both are kept as features, but only the GPU-level one is a valid *floor* --
        # see the note on floor_pipes below.
        pipe_times[f"mem_{level}"] = t_gpu
        pipe_times[f"mem_{level}_sm"] = t_sm

    # ---- cache residency, and the compulsory-miss floor --------------------- #
    # HBM bandwidth bounds only the traffic that actually reaches HBM. A kernel whose
    # working set fits in L2 never goes there -- and in a replay loop it certainly does
    # not, because the same buffers are touched thousands of times in a row.
    #
    # This is not a small correction on this fleet. The L40S carries 96 MB of L2 and a
    # 4608x3072 bf16 norm moves 57 MB, so the whole thing is resident; charging it
    # 864 GB/s says it takes 66 us when it measured under 20, and eta sails past 3.
    # The cards where it bites are exactly the ones with the largest L2 relative to their
    # HBM bandwidth -- L40S and L4 -- which is the signature the first two dataset builds
    # showed.
    #
    # The honest floor is the *compulsory* traffic: the share of the working set that
    # cannot fit in cache and must therefore cross the memory bus at least once. Below
    # capacity that share is zero and memory imposes no bound at all, leaving the math
    # pipes to set the floor -- conservative by construction, since it can only lower the
    # floor, never raise it above what was measured.
    l2_capacity = gpu.l2_cache_mb * 2**20
    working_set = dist.total("bytes_global")
    residency = working_set / max(l2_capacity, 1.0)
    miss_fraction = max(0.0, 1.0 - 1.0 / residency) if residency > 1.0 else 0.0

    f["log_l2_residency"] = _log(residency)
    f["compulsory_miss_fraction"] = miss_fraction
    f["fits_in_l2"] = 1.0 if residency <= 1.0 else 0.0
    pipe_times["mem_compulsory"] = (
        working_set * miss_fraction / bw["global"] if bw["global"] > 0 else 0.0
    )

    # ---- the roofline floor ------------------------------------------------ #
    # Only the math pipelines and HBM set the floor. L2 and shared-memory times are
    # kept as features and as slack terms but deliberately excluded here, because their
    # bandwidth constants are estimates while the tensor throughput and HBM bandwidth
    # are spec-sheet exact. An overstated floor is not a harmless error: efficiency is
    # defined as floor/measured, so a floor above the measured time gives eta > 1, which
    # a sigmoid-bounded head cannot represent and which would quietly cap the model.
    #
    # The memory term is the GPU-level one, NOT the worst-SM one, and the asymmetry with
    # the math pipes is deliberate. A per-SM *math* time is a genuine lower bound: the
    # busiest SM really must issue that many ops at that pipe's rate, so the kernel
    # cannot finish sooner. A per-SM *bandwidth* time is not, because bandwidth is a
    # shared global resource rather than a per-SM allocation -- dividing it by the SM
    # count assumes every SM is active and drawing an equal slice. A GEMM occupying four
    # tiles on a 142-SM L40S has the whole 864 GB/s available, not 864/142 = 6.1 GB/s,
    # and treating it otherwise overstates its time by more than an order of magnitude.
    #
    # That error is not uniform across the fleet: it scales with bandwidth-per-SM, so it
    # is worst exactly on the cards with the least (L4 5.0 GB/s/SM, L40S 6.1) and mild on
    # the ones with the most (H200 36.4). The first run of this dataset showed the
    # signature clearly -- eta > 1 on 25-34% of L40S and L4 rows and 0% for the same
    # categories on H100/H200/A100.
    #
    # total_bytes / total_bandwidth is a true floor regardless of occupancy: the kernel
    # cannot move its bytes faster than the memory system delivers them.
    floor_pipes = {k: v for k, v in pipe_times.items()
                   if k in MATH_PIPES or k == "mem_compulsory"}
    bottleneck = max(floor_pipes, key=lambda k: floor_pipes[k])
    t_theory = max(floor_pipes[bottleneck], _EPS)
    f["log_theoretical_time"] = _log(t_theory)
    for k, v in pipe_times.items():
        # How close each pipeline is to being the binding one. The MLP needs this
        # because contention is a function of the *runners-up*, not just the winner.
        f[f"slack_{k}"] = math.log(max(v, _EPS) / t_theory)

    # ---- scheduling -------------------------------------------------------- #
    f["log_n_tasks"] = _log(dist.n_tasks_total)
    f["log_n_waves"] = _log(dist.n_waves)
    f["tail_fraction"] = dist.tail_fraction
    f["sm_occupancy"] = dist.sm_occupancy
    f["imbalance_mma"] = dist.imbalance("n_mma")
    f["imbalance_bytes"] = dist.imbalance("bytes_global")
    f["is_persistent"] = 1.0 if dist.scheduler == "software" else 0.0

    # ---- intensity --------------------------------------------------------- #
    flops = kernel.flops()
    b_glob = dist.total("bytes_global")
    b_l2 = dist.total("bytes_l2")
    f["log_flops"] = _log(flops)
    f["log_bytes_per_flop"] = _log(b_glob / max(flops, _EPS))
    f["log_l2_reuse"] = _log(b_l2 / max(b_glob, _EPS))
    f["log_arithmetic_intensity"] = _log(flops / max(b_glob, _EPS))

    # ---- hardware, throughput domain (implicit in PipeWeave) ---------------- #
    peak = gpu.peak_tensor_flops(clock)
    f["log_peak_flops"] = _log(peak)
    f["log_mem_bw"] = _log(gpu.mem_bandwidth_gbs)
    f["log_sms"] = _log(gpu.sms)
    f["log_ops_per_clk"] = _log(gpu.ops_per_clk)
    f["log_clock"] = _log(clock_hz)
    f["log_smem_kb"] = _log(gpu.smem_size_kb_sm)
    f["log_l2_bw"] = _log(gpu.l2_bandwidth_gbs)

    # ---- hardware, power domain (the energy extension) --------------------- #
    f["log_tdp"] = _log(gpu.tdp_w)
    f["log_idle_power"] = _log(gpu.idle_power_w)
    f["idle_fraction"] = gpu.idle_power_w / gpu.tdp_w
    # Joules of power budget per unit of peak throughput -- the card's nominal energy
    # efficiency, and the thing that most directly sets the level of E for a given kernel.
    f["log_watt_per_tflop"] = _log(gpu.tdp_w / max(peak / 1e12, _EPS))
    f["log_watt_per_gbs"] = _log(gpu.tdp_w / max(gpu.mem_bandwidth_gbs, _EPS))
    f["is_hbm"] = 1.0 if gpu.is_hbm else 0.0
    f["compute_capability"] = gpu.compute_capability

    # ---- kernel identity --------------------------------------------------- #
    f["log_dsize"] = _log(kernel.dsize)
    f["uses_tensor_core"] = 1.0 if dist.total("n_mma") > 0 else 0.0

    return AnalysisResult(
        features=f,
        distribution=dist,
        theoretical_time_s=t_theory,
        bottleneck=bottleneck,
        flops=flops,
        bytes_global=b_glob,
    )


def feature_row(config, gpu: GPU, **kw) -> dict:
    """Convenience: config dict/KernelConfig + GPU -> flat feature dict."""
    from kernelenergy.kernels.base import KernelConfig

    if isinstance(config, dict):
        config = KernelConfig.from_row(config)
    res = analyse(make_kernel(config), gpu, **kw)
    row = dict(res.features)
    row["theoretical_time_s"] = res.theoretical_time_s
    row["bottleneck"] = res.bottleneck
    row["analytic_flops"] = res.flops
    row["analytic_bytes_global"] = res.bytes_global
    return row


def _probe_columns() -> list[str]:
    """The feature names, derived by running the analyser once on a trivial kernel."""
    from kernelenergy.hardware import GPUS
    from kernelenergy.kernels.base import KernelConfig

    cfg = KernelConfig(category="gemm", dtype="bf16", params={"m": 128, "n": 128, "k": 128})
    res = analyse(make_kernel(cfg), next(iter(GPUS.values())))
    return list(res.features)


FEATURE_COLUMNS: list[str] = _probe_columns()


def stack_features(rows: list[dict]) -> np.ndarray:
    """Feature dicts -> matrix in ``FEATURE_COLUMNS`` order, raising on gaps."""
    out = np.empty((len(rows), len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(FEATURE_COLUMNS):
            try:
                out[i, j] = r[c]
            except KeyError:
                raise KeyError(f"row {i} is missing feature {c!r}") from None
    return out
