"""The replay harness: isolate a kernel, run it in a loop, measure the joules.

Why replay at all
-----------------
The obvious approach -- run the pipeline once, capture the kernel timeline, attribute
energy to kernels by timestamp -- does not work, and it is worth being explicit about
why, because it looks like it should. A GPU kernel in a diffusion pipeline runs for tens
to hundreds of microseconds. The board's power telemetry updates on the order of tens of
milliseconds and is itself a filtered quantity. Two or three orders of magnitude separate
them. Any per-kernel number recovered by correlating those two timelines is a
deconvolution of a heavily smoothed signal against a rapidly switching one, and the
result is dominated by the assumptions of the deconvolution rather than by the data.

Replaying the kernel in a tight loop inverts the problem. The loop runs for seconds, so
the sensor has thousands of update intervals to track it, and the driver's own energy
counter integrates exactly over the window. What you measure is unambiguous.

What replay costs you
---------------------
Three things, all real, none fatal, and all worth stating rather than discovering later:

1. **Cache state.** A kernel replayed back-to-back finds its inputs warm in L2. In the
   pipeline they may not be. ``n_buffers`` mitigates this by rotating through several
   input copies so the working set exceeds L2, which is the right default for anything
   whose inputs are smaller than the cache.

2. **No overlap.** In the real pipeline a kernel may overlap with others on separate
   streams, sharing the power budget. Replayed alone it has the whole budget. This
   inflates per-kernel power relative to its in-pipeline share, and is the main reason
   the sum of replayed kernel energies will exceed a measured end-to-end run. That
   discrepancy is a measurement to make, not a bug to hide -- ``scripts/05_validate_e2e.py``
   quantifies it.

3. **Steady-state clocks.** Several seconds of one kernel drives the card to whatever
   clock that kernel's power draw sustains. A single instance in a mixed pipeline sees
   the clock the *mixture* sustains. The recorded ``sm_clock_mean_mhz`` and
   ``frac_sw_power_cap`` are there so this is visible per row rather than assumed away.

The warmup exists for the same reason. Power ramps over roughly a second on a large card
and clocks settle after that; measuring from cold gives a number that is neither the
transient nor the steady state.
"""

from __future__ import annotations

import platform
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from kernelenergy import __version__
from kernelenergy.hardware import GPU
from kernelenergy.kernels.base import KernelConfig, KernelSpec
from kernelenergy.kernels.registry import make_kernel
from kernelenergy.measure.schema import MeasurementRow
from kernelenergy.nvml import PowerSampler, power_limit_w, supports_energy_counter

__all__ = ["ReplayConfig", "ReplayResult", "measure_kernel", "OutOfMemoryError"]


class OutOfMemoryError(RuntimeError):
    """The config does not fit on this card at the requested buffer count."""


@dataclass
class ReplayConfig:
    target_window_s: float = 3.0  # length of one measurement window
    warmup_s: float = 2.0  # discarded, lets clocks and power settle
    repeats: int = 3
    min_iters: int = 8
    max_iters: int = 2_000_000
    sample_interval_s: float = 0.005
    inter_repeat_sleep_s: float = 0.25
    n_buffers: int = 4  # input copies to rotate through, defeats L2 warmth
    subtract_launch_overhead: bool = True
    max_working_set_frac: float = 0.6  # of free memory
    notes: str = ""

    # --- device indices: two of them, and they are not the same number ------ #
    #: The CUDA ordinal, i.e. what ``torch.device(f"cuda:{i}")`` means. Filtered by
    #: CUDA_VISIBLE_DEVICES.
    device_index: int = 0
    #: The NVML ordinal for the *same physical card*. NVML is not filtered by
    #: CUDA_VISIBLE_DEVICES, so under a scheduler this is usually a different number --
    #: given GPU 2 of a node, torch calls it cuda:0 while NVML calls it 2. Left None it
    #: is resolved by UUID at measurement time; setting it equal to ``device_index``
    #: without checking is how you end up reading an idle neighbour's energy counter for
    #: an entire sweep. See kernelenergy.hpc.device.
    nvml_index: int | None = None
    #: Proceed even when another process holds the GPU. Rows are stamped contended=1.
    allow_shared_gpu: bool = False

    def resolved_nvml_index(self) -> int:
        if self.nvml_index is not None:
            return self.nvml_index
        from kernelenergy.hpc.device import resolve_device

        return resolve_device(self.device_index).nvml_index


@dataclass
class ReplayResult:
    latency_s: float
    energy_j: float
    energy_raw_j: float
    power_avg_w: float
    latency_sd_rel: float
    energy_sd_rel: float
    n_iters: int
    n_repeats: int
    window_s: float
    energy_check_ratio: float
    energy_counter_used: bool
    summaries: list = None  # type: ignore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sync(device_index: int) -> None:  # pragma: no cover - needs a GPU
    import torch

    torch.cuda.synchronize(device_index)


def _time_one_iteration(kernel: KernelSpec, states: list, device_index: int,
                        probe: int = 5) -> float:  # pragma: no cover
    """Rough per-iteration latency, used only to size the loop."""
    for _ in range(3):
        kernel.run(states[0])
    _sync(device_index)
    t0 = time.perf_counter()
    for i in range(probe):
        kernel.run(states[i % len(states)])
    _sync(device_index)
    return (time.perf_counter() - t0) / probe


def _build_buffers(kernel: KernelSpec, cfg: ReplayConfig, device: str) -> list:
    """Allocate ``n_buffers`` independent input sets, shrinking if memory is tight.

    Falls back to one buffer rather than failing: a single buffer with a warm cache is a
    worse measurement than several with a cold one, but it is a measurement.
    """
    import torch

    free, _total = torch.cuda.mem_get_info(cfg.device_index)
    budget = free * cfg.max_working_set_frac
    per = kernel.working_set_bytes()
    if per > budget:
        raise OutOfMemoryError(
            f"{kernel.describe()} needs {per / 2**30:.2f} GiB, budget is "
            f"{budget / 2**30:.2f} GiB"
        )
    n = max(1, min(cfg.n_buffers, int(budget // max(per, 1))))
    states = []
    for _ in range(n):
        try:
            states.append(kernel.build(device))
        except torch.cuda.OutOfMemoryError:
            break
    if not states:
        raise OutOfMemoryError(f"could not allocate even one buffer for {kernel.describe()}")
    return states


def _run_window(kernel: KernelSpec, states: list, n_iters: int, cfg: ReplayConfig,
                nvml_index: int | None = None):
    """One measured window. Returns (elapsed_s, PowerSampler).

    Note the two different indices: the kernel runs on the CUDA ordinal, the sampler
    reads the NVML ordinal for that same card.
    """
    nb = len(states)
    nvml_index = cfg.resolved_nvml_index() if nvml_index is None else nvml_index
    _sync(cfg.device_index)
    with PowerSampler(nvml_index, cfg.sample_interval_s) as sampler:
        t0 = time.perf_counter()
        for i in range(n_iters):
            kernel.run(states[i % nb])
        _sync(cfg.device_index)
        elapsed = time.perf_counter() - t0
    return elapsed, sampler


class _NullKernel(KernelSpec):
    """A launch that does almost nothing, to price the launch itself.

    Subtracting this removes the CPU-side dispatch and the driver's fixed per-launch
    energy from short kernels, where it can otherwise be a large share of the total. It
    is not free of assumptions -- a real launch also pays for its own arguments -- but it
    is much better than ignoring the overhead, and for kernels above ~50 microseconds it
    is negligible either way.
    """

    category = "null"
    required = ()

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        return {"x": torch.zeros(1, device=device)}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        state["x"].add_(0.0)

    def working_set_bytes(self) -> float:
        return 4.0

    def decompose(self, gpu):
        return []

    def flops(self) -> float:
        return 0.0

    def bytes_global(self) -> float:
        return 0.0


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def measure_kernel(
    config: KernelConfig,
    gpu: GPU,
    cfg: ReplayConfig | None = None,
    idle_power_w: float | None = None,
    overhead: tuple[float, float] | None = None,
) -> tuple[MeasurementRow, ReplayResult]:
    """Measure one kernel config on the local GPU.

    ``overhead`` is ``(seconds_per_launch, joules_per_launch)`` from
    :func:`measure_launch_overhead`. Measure it once per session and pass it in; it does
    not depend on the kernel.
    """
    import torch  # noqa: F401  - fail here with a clear message if missing

    cfg = cfg or ReplayConfig()
    device = f"cuda:{cfg.device_index}"
    nvml_index = cfg.resolved_nvml_index()
    kernel = make_kernel(config)

    # A co-tenant process adds its joules to the board counter with no way to separate
    # them afterwards, so check immediately before the window rather than only at job
    # start -- on a shared node a neighbour can arrive at any point during a sweep.
    contended = 0
    try:
        from kernelenergy.hpc.device import ResolvedDevice, assert_exclusive_gpu

        probe = ResolvedDevice(cfg.device_index, nvml_index, "", "", gpu.name, "given")
        foreign = assert_exclusive_gpu(probe, allow_shared=cfg.allow_shared_gpu)
        contended = int(bool(foreign))
    except ImportError:
        pass

    states = _build_buffers(kernel, cfg, device)

    # --- size the loop --------------------------------------------------- #
    t_iter = _time_one_iteration(kernel, states, cfg.device_index)
    n_iters = int(np.clip(round(cfg.target_window_s / max(t_iter, 1e-9)),
                          cfg.min_iters, cfg.max_iters))

    # --- warm up ----------------------------------------------------------- #
    n_warm = max(1, int(cfg.warmup_s / max(t_iter, 1e-9)))
    for i in range(min(n_warm, cfg.max_iters)):
        kernel.run(states[i % len(states)])
    _sync(cfg.device_index)

    # --- measure ----------------------------------------------------------- #
    lats, energies, summaries = [], [], []
    for r in range(cfg.repeats):
        elapsed, sampler = _run_window(kernel, states, n_iters, cfg, nvml_index)
        s = sampler.summary()
        summaries.append(s)
        e_total = s.energy_counter_j
        counter_used = e_total is not None
        if not counter_used:
            e_total = s.energy_integrated_j or (s.power_mean_w * elapsed)
        lats.append(elapsed / n_iters)
        energies.append(e_total / n_iters)
        if r + 1 < cfg.repeats:
            time.sleep(cfg.inter_repeat_sleep_s)

    lat = float(np.median(lats))
    en_raw = float(np.median(energies))
    lat_sd = float(np.std(lats) / max(np.mean(lats), 1e-30))
    en_sd = float(np.std(energies) / max(np.mean(energies), 1e-30))

    # --- remove launch overhead -------------------------------------------- #
    lat_net, en_net = lat, en_raw
    if cfg.subtract_launch_overhead and overhead is not None:
        oh_t, oh_e = overhead
        lat_net = max(lat - oh_t, 1e-9)
        en_net = max(en_raw - oh_e, 1e-12)

    last = summaries[-1]
    ratio = float("nan")
    if last.energy_counter_j and last.energy_integrated_j:
        ratio = last.energy_counter_j / last.energy_integrated_j

    row = MeasurementRow(
        kernel_sig=config.signature(),
        category=config.category,
        dtype=config.dtype,
        gpu_key=gpu.gpu_key,
        source_model=config.source_model,
        source_module=config.source_module,
        op_name=config.op_name,
        latency_s=lat_net,
        energy_j=en_net,
        energy_raw_j=en_raw,
        power_avg_w=en_net / max(lat_net, 1e-12),
        latency_sd_rel=lat_sd,
        energy_sd_rel=en_sd,
        n_iters=n_iters,
        n_repeats=cfg.repeats,
        window_s=float(np.median([s.duration_s for s in summaries])),
        idle_power_w=idle_power_w if idle_power_w is not None else gpu.idle_power_w,
        sm_clock_mean_mhz=last.sm_clock_mean_mhz,
        sm_clock_median_mhz=last.sm_clock_median_mhz,
        mem_clock_mean_mhz=last.mem_clock_mean_mhz,
        temperature_mean_c=last.temperature_mean_c,
        temperature_max_c=last.temperature_max_c,
        frac_sw_power_cap=last.frac_sw_power_cap,
        frac_hw_slowdown=last.frac_hw_slowdown,
        frac_sw_thermal=last.frac_sw_thermal,
        frac_hw_thermal=last.frac_hw_thermal,
        power_limit_w=_safe_power_limit(nvml_index),
        power_limit_default_w=_safe_default_power_limit(nvml_index),
        energy_counter_used=int(bool(summaries[-1].energy_counter_j is not None)),
        energy_check_ratio=ratio,
        n_buffers=len(states),
        cuda_index=cfg.device_index,
        nvml_index=nvml_index,
        gpu_uuid=_safe_uuid(nvml_index),
        contended=contended,
        gpu_name=gpu.name,
        architecture=gpu.architecture,
        compute_capability=gpu.compute_capability,
        sms=gpu.sms,
        ops_per_clk=gpu.ops_per_clk,
        tensor_clock_mhz=gpu.tensor_clock_mhz,
        boost_clock_mhz=gpu.boost_clock_mhz,
        peak_tensor_flops=gpu.peak_tensor_flops("tensor"),
        mem_bandwidth_gbs=gpu.mem_bandwidth_gbs,
        l2_bandwidth_gbs=gpu.l2_bandwidth_gbs,
        l2_cache_mb=gpu.l2_cache_mb,
        tdp_w=gpu.tdp_w,
        is_hbm=int(gpu.is_hbm),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(),
        driver_version=_safe_driver_version(),
        torch_version=_safe_torch_version(),
        cuda_version=_safe_cuda_version(),
        kernelenergy_version=__version__,
        notes=cfg.notes,
        params=dict(config.params),
    )

    _free(states)
    return row, ReplayResult(
        latency_s=lat_net,
        energy_j=en_net,
        energy_raw_j=en_raw,
        power_avg_w=row.power_avg_w,
        latency_sd_rel=lat_sd,
        energy_sd_rel=en_sd,
        n_iters=n_iters,
        n_repeats=cfg.repeats,
        window_s=row.window_s,
        energy_check_ratio=ratio,
        energy_counter_used=bool(row.energy_counter_used),
        summaries=summaries,
    )


def measure_launch_overhead(cfg: ReplayConfig | None = None) -> tuple[float, float]:
    """Seconds and joules attributable to one kernel launch on this machine.

    Run once per session, before the sweep.
    """
    cfg = cfg or ReplayConfig()
    nvml_index = cfg.resolved_nvml_index()
    null = _NullKernel(KernelConfig(category="null", dtype="fp32", params={}))
    states = [null.build(f"cuda:{cfg.device_index}")]
    t_iter = _time_one_iteration(null, states, cfg.device_index, probe=50)
    n_iters = int(np.clip(round(cfg.target_window_s / max(t_iter, 1e-9)),
                          cfg.min_iters, cfg.max_iters))
    for i in range(n_iters // 4):
        null.run(states[0])
    _sync(cfg.device_index)

    ts, es = [], []
    for _ in range(cfg.repeats):
        elapsed, sampler = _run_window(null, states, n_iters, cfg, nvml_index)
        s = sampler.summary()
        e = s.energy_counter_j if s.energy_counter_j is not None else s.energy_integrated_j
        ts.append(elapsed / n_iters)
        es.append((e or 0.0) / n_iters)
    _free(states)
    return float(np.median(ts)), float(np.median(es))


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def _free(states: list) -> None:
    try:
        import torch

        states.clear()
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _safe_power_limit(nvml_index: int) -> float:
    try:
        return power_limit_w(nvml_index)
    except Exception:
        return float("nan")


def _safe_default_power_limit(nvml_index: int) -> float:
    """The card's own default limit, to compare against the enforced one.

    Sites cap cards below their datasheet TDP for facility power reasons. When they do,
    ``pi = P_avg / TDP`` has the wrong denominator and is compressed by a factor nobody
    downstream can recover -- unless both numbers are recorded, which is why they are.
    """
    try:
        from kernelenergy.hpc.device import power_limits

        return power_limits(nvml_index)["default_w"]
    except Exception:
        return float("nan")


def _safe_uuid(nvml_index: int) -> str:
    """The card's own identity, so a run can be traced to a physical GPU after the fact."""
    try:
        from kernelenergy.nvml import nvml_session

        with nvml_session() as nv:
            u = nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(nvml_index))
            return u.decode() if isinstance(u, bytes) else str(u)
    except Exception:
        return ""


def _safe_driver_version() -> str:
    try:
        from kernelenergy.nvml import nvml_session

        with nvml_session() as nv:
            v = nv.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return ""


def _safe_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return ""


def _safe_cuda_version() -> str:
    try:
        import torch

        return torch.version.cuda or ""
    except Exception:
        return ""
