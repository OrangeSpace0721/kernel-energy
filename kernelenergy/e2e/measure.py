"""Measure one real generation: energy, latency, kernel counts, busy time.

Three passes over the same generation, deliberately separate because they interfere with
each other:

1. **Timing / energy.** Clean runs with nothing attached but the NVML sampler, repeated
   and taken as a median. This is the ground truth and it must not be perturbed.
2. **Profiler.** One run under ``torch.profiler`` for total device time and per-kernel
   durations. The profiler adds meaningful overhead and inflates wall time, so its
   *timing* is used only for the busy/gap split, never as the generation's latency.
3. **Capture.** One run under the functional-level capture to get the call count of every
   kernel configuration at *this* step count and resolution.

Pass 3 is what removes the extrapolation. The predecessor to this module multiplied
capture counts taken at one denoising step by the step count, with a rule that
convolutions belong to the VAE and therefore should not be scaled. That holds for these
three pipelines today and is silently wrong for any pipeline with a convolution in its
denoiser -- and it is free to just count properly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

__all__ = ["E2EMeasurement", "measure_generation"]


@dataclass
class E2EMeasurement:
    """One (model, gpu, steps, resolution) cell."""

    gpu_key: str
    model: str
    steps: int
    height: int
    width: int

    # -- ground truth, from clean unprofiled runs ---------------------------- #
    latency_s: float
    energy_j: float
    power_avg_w: float
    latency_sd_rel: float
    energy_sd_rel: float
    n_repeats: int
    energy_counter_used: bool

    # -- card state ---------------------------------------------------------- #
    idle_power_w: float
    sm_clock_median_mhz: float
    frac_sw_power_cap: float
    frac_sw_thermal: float
    temperature_max_c: float

    # -- from the profiler pass ---------------------------------------------- #
    #: Total CUDA device time summed over every kernel. The GPU-busy share of the run.
    device_time_s: float
    #: Device time in kernels whose name matches a modelled category. The rest is real
    #: work that no per-kernel prediction will ever account for.
    device_time_covered_s: float
    #: Wall time of the profiled run. Inflated by profiling; kept only to show by how
    #: much, so nobody mistakes it for the latency.
    profiled_latency_s: float

    #: config signature -> invocations in this generation
    calls: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @property
    def gap_s(self) -> float:
        """Wall time with no kernel resident. Charged at idle power, not zero.

        Negative would mean kernels overlapped in a way that sums past the wall clock,
        which on a single stream should not happen; it is clamped at zero in the
        reconciliation and reported here as-is so it can be seen.
        """
        return self.latency_s - self.device_time_s

    @property
    def busy_fraction(self) -> float:
        return self.device_time_s / max(self.latency_s, 1e-12)

    @property
    def coverage_fraction(self) -> float:
        """Share of device time in kernels this project models at all."""
        return self.device_time_covered_s / max(self.device_time_s, 1e-12)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("calls")
        d["n_configs"] = len(self.calls)
        d["total_calls"] = int(sum(self.calls.values()))
        d["gap_s"] = self.gap_s
        d["busy_fraction"] = self.busy_fraction
        d["coverage_fraction"] = self.coverage_fraction
        return d


def _clean_runs(pipe, spec, *, steps, height, width, device, repeats, prompt):
    """Pass 1: energy and latency, with nothing attached but the sampler."""
    import torch

    from kernelenergy.nvml import PowerSampler

    def _gen():
        with torch.no_grad():
            pipe(prompt=prompt, num_inference_steps=steps, height=height,
                 width=width, **spec.extra_call_kwargs)

    # One discarded generation: the first call compiles autotuning caches, allocates
    # workspaces and picks cuBLAS algorithms, none of which recurs.
    _gen()
    torch.cuda.synchronize()

    lats, energies, summaries = [], [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        with PowerSampler(device, 0.005) as sampler:
            t0 = time.perf_counter()
            _gen()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        summ = sampler.summary()
        e = summ.energy_counter_j
        if e is None:
            e = summ.energy_integrated_j
        lats.append(elapsed)
        energies.append(float(e))
        summaries.append(summ)
        time.sleep(1.0)  # let the card settle so repeats are independent

    return np.array(lats), np.array(energies), summaries


def _profiler_pass(pipe, spec, *, steps, height, width, prompt):
    """Pass 2: total and per-category device time."""
    import torch

    from kernelenergy.trace.profile import categorise_kernel, is_device_kernel

    def _gen():
        with torch.no_grad():
            pipe(prompt=prompt, num_inference_steps=steps, height=height,
                 width=width, **spec.extra_call_kwargs)

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        _gen()
        torch.cuda.synchronize()
    profiled = time.perf_counter() - t0

    total_us = 0.0
    covered_us = 0.0
    for ev in prof.key_averages():
        if not is_device_kernel(ev):
            continue
        us = float(getattr(ev, "self_device_time_total", 0)
                   or getattr(ev, "self_cuda_time_total", 0))
        total_us += us
        if categorise_kernel(ev.key) != "other":
            covered_us += us
    return total_us / 1e6, covered_us / 1e6, profiled


def measure_generation(
    pipe,
    spec,
    *,
    model: str,
    gpu_key: str,
    steps: int,
    height: int = 1024,
    width: int = 1024,
    device: int = 0,
    repeats: int = 3,
    prompt: str = "a photograph of a city street",
    idle_seconds: float = 10.0,
    with_capture: bool = True,
) -> E2EMeasurement:
    """Measure one generation three ways. See the module docstring for why three."""
    from kernelenergy.nvml import measure_idle
    from kernelenergy.trace.capture import capture_pipeline

    lats, energies, summaries = _clean_runs(
        pipe, spec, steps=steps, height=height, width=width,
        device=device, repeats=repeats, prompt=prompt,
    )
    device_s, covered_s, profiled = _profiler_pass(
        pipe, spec, steps=steps, height=height, width=width, prompt=prompt)

    calls: dict[str, int] = {}
    if with_capture:
        # The whole point: counts at THIS step count, not one step scaled up.
        configs, counts = capture_pipeline(
            pipe, model=model, steps=steps, height=height, width=width,
            **spec.extra_call_kwargs,
        )
        by_sig = {c.signature(): c for c in configs}
        for sig in by_sig:
            calls[sig] = int(counts.get(sig, 0)) if hasattr(counts, "get") else 0
        # ``counts`` is keyed by signature in the current capture implementation; if a
        # future version keys it differently, fall back to one call per config rather
        # than silently reporting zeros.
        if calls and max(calls.values()) == 0:
            calls = {sig: 1 for sig in by_sig}

    idle = measure_idle(device, seconds=idle_seconds)
    last = summaries[-1]

    return E2EMeasurement(
        gpu_key=gpu_key, model=model, steps=steps, height=height, width=width,
        latency_s=float(np.median(lats)),
        energy_j=float(np.median(energies)),
        power_avg_w=float(np.median(energies) / np.median(lats)),
        latency_sd_rel=float(np.std(lats) / max(np.mean(lats), 1e-12)),
        energy_sd_rel=float(np.std(energies) / max(np.mean(energies), 1e-12)),
        n_repeats=int(len(lats)),
        energy_counter_used=summaries[-1].energy_counter_j is not None,
        idle_power_w=float(idle.power_w),
        sm_clock_median_mhz=float(last.sm_clock_median_mhz),
        frac_sw_power_cap=float(last.frac_sw_power_cap),
        frac_sw_thermal=float(getattr(last, "frac_sw_thermal", 0.0)),
        temperature_max_c=float(getattr(last, "temperature_max_c", 0.0)),
        device_time_s=device_s,
        device_time_covered_s=covered_s,
        profiled_latency_s=profiled,
        calls=calls,
    )


def calls_frame(m: E2EMeasurement) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kernel_sig": s, "calls": c, "gpu_key": m.gpu_key, "model": m.model,
          "steps": m.steps, "height": m.height, "width": m.width}
         for s, c in m.calls.items()]
    )
