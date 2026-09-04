"""NVML instrumentation: energy counter, power sampling, clocks, throttle reasons.

The primary instrument is ``nvmlDeviceGetTotalEnergyConsumption``, a monotonic
millijoule counter maintained by the driver since Volta. It is the right tool and not
the obvious one: integrating polled power over a window is a strictly worse estimator of
the same quantity, because the power sensor updates on its own schedule (tens of
milliseconds, card-dependent) and polling it faster just resamples the same held value.
The counter integrates on the card, so a window boundary lands where you put it.

Polled power is still collected, for three reasons: it is the fallback on cards without
the counter, it gives the mean/peak/variance the counter cannot, and the clock and
throttle-reason samples that come with it are what let you tell a power-capped run from a
thermally throttled one -- a distinction the run-level work in this project found
separates the fleet into two regimes.
"""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import shape depends on which package is installed
    import pynvml as _nvml
except ImportError:  # nvidia-ml-py installs as `pynvml` too, but be explicit
    try:
        from nvidia import nvml as _nvml  # type: ignore
    except ImportError:  # pragma: no cover
        _nvml = None

__all__ = [
    "nvml_session",
    "device_name",
    "device_handle",
    "supports_energy_counter",
    "read_energy_mj",
    "PowerSampler",
    "SampleSummary",
    "measure_idle",
    "IdleResult",
    "power_limit_w",
]

_INIT_LOCK = threading.Lock()
_INIT_COUNT = 0


def _require() -> Any:
    if _nvml is None:
        raise RuntimeError(
            "NVML bindings not found. Install with: pip install nvidia-ml-py"
        )
    return _nvml


@contextmanager
def nvml_session():
    """Reference-counted NVML init/shutdown, safe to nest."""
    global _INIT_COUNT
    nv = _require()
    with _INIT_LOCK:
        if _INIT_COUNT == 0:
            nv.nvmlInit()
        _INIT_COUNT += 1
    try:
        yield nv
    finally:
        with _INIT_LOCK:
            _INIT_COUNT -= 1
            if _INIT_COUNT == 0:
                try:
                    nv.nvmlShutdown()
                except Exception:  # pragma: no cover
                    pass


def device_handle(index: int = 0):
    nv = _require()
    return nv.nvmlDeviceGetHandleByIndex(index)


def device_name(index: int = 0) -> str:
    with nvml_session() as nv:
        name = nv.nvmlDeviceGetName(device_handle(index))
        return name.decode() if isinstance(name, bytes) else str(name)


def power_limit_w(index: int = 0) -> float:
    """The enforced power limit, which is what the cap actually binds against.

    Not the same as the datasheet TDP when the card has been configured with a lower
    limit -- worth recording per run so a capped card is not mistaken for a slow one.
    """
    with nvml_session() as nv:
        return nv.nvmlDeviceGetEnforcedPowerLimit(device_handle(index)) / 1000.0


def supports_energy_counter(index: int = 0) -> bool:
    with nvml_session() as nv:
        try:
            nv.nvmlDeviceGetTotalEnergyConsumption(device_handle(index))
            return True
        except Exception:
            return False


def read_energy_mj(handle) -> int:
    """Total energy drawn since driver load, in millijoules."""
    nv = _require()
    return int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))


# --------------------------------------------------------------------------- #
# Throttle reasons
# --------------------------------------------------------------------------- #

_THROTTLE_BITS = {
    "gpu_idle": 0x0000000000000001,
    "applications_clocks_setting": 0x0000000000000002,
    "sw_power_cap": 0x0000000000000004,
    "hw_slowdown": 0x0000000000000008,
    "sync_boost": 0x0000000000000010,
    "sw_thermal_slowdown": 0x0000000000000020,
    "hw_thermal_slowdown": 0x0000000000000040,
    "hw_power_brake_slowdown": 0x0000000000000080,
    "display_clock_setting": 0x0000000000000100,
}


def decode_throttle(mask: int) -> dict[str, bool]:
    return {name: bool(mask & bit) for name, bit in _THROTTLE_BITS.items()}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


@dataclass
class SampleSummary:
    """Aggregate of a sampling window."""

    n_samples: int
    duration_s: float

    power_mean_w: float
    power_median_w: float
    power_max_w: float
    power_sd_w: float

    sm_clock_mean_mhz: float
    sm_clock_median_mhz: float
    sm_clock_max_mhz: float
    mem_clock_mean_mhz: float

    temperature_mean_c: float
    temperature_max_c: float

    util_gpu_mean_pct: float
    util_mem_mean_pct: float

    frac_sw_power_cap: float
    frac_hw_slowdown: float
    frac_sw_thermal: float
    frac_hw_thermal: float

    energy_counter_j: float | None = None  # from the mJ counter, if available
    energy_integrated_j: float | None = None  # trapezoid over polled power

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


@dataclass
class _Sample:
    t: float
    power_w: float
    sm_clock: float
    mem_clock: float
    temp: float
    util_gpu: float
    util_mem: float
    throttle: int


class PowerSampler:
    """Background NVML sampler.

    Use as a context manager around the work you want to characterise::

        with PowerSampler(index=0, interval_s=0.005) as s:
            run_the_kernel_loop()
        summary = s.summary()
        print(summary.energy_counter_j, summary.frac_sw_power_cap)

    ``energy_counter_j`` is the counter delta across the window and is the number to
    trust. ``energy_integrated_j`` is the trapezoid integral of polled power over the
    same window; the two agreeing to a few percent is a useful sanity check that the
    window was long enough for the sensor to track it, and their disagreeing on short
    windows is expected rather than alarming.
    """

    def __init__(
        self,
        index: int = 0,
        interval_s: float = 0.005,
        collect_clocks: bool = True,
    ) -> None:
        self.index = index
        self.interval_s = interval_s
        self.collect_clocks = collect_clocks
        self._samples: list[_Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._e0: int | None = None
        self._e1: int | None = None
        self._t0: float = 0.0
        self._t1: float = 0.0
        self._session = None

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> "PowerSampler":
        self._session = nvml_session()
        nv = self._session.__enter__()
        self._handle = nv.nvmlDeviceGetHandleByIndex(self.index)
        self._samples.clear()
        self._stop.clear()
        try:
            self._e0 = read_energy_mj(self._handle)
        except Exception:
            self._e0 = None
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._t1 = time.perf_counter()
        try:
            self._e1 = read_energy_mj(self._handle)
        except Exception:
            self._e1 = None
        if self._session is not None:
            self._session.__exit__(*exc)
            self._session = None

    # -- sampling ----------------------------------------------------------- #

    def _loop(self) -> None:
        nv = _require()
        h = self._handle
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                if self.collect_clocks:
                    sm = float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
                    mem = float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))
                    temp = float(
                        nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                    )
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    ug, um = float(util.gpu), float(util.memory)
                    try:
                        thr = int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
                    except Exception:
                        thr = int(
                            getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons")(h)
                        )
                else:
                    sm = mem = temp = ug = um = 0.0
                    thr = 0
                self._samples.append(
                    _Sample(time.perf_counter(), power, sm, mem, temp, ug, um, thr)
                )
            except Exception:
                pass
            next_t += self.interval_s
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:  # we fell behind; resync rather than spin
                next_t = time.perf_counter()

    # -- results ------------------------------------------------------------ #

    @property
    def energy_counter_j(self) -> float | None:
        if self._e0 is None or self._e1 is None:
            return None
        return (self._e1 - self._e0) / 1000.0

    def summary(self) -> SampleSummary:
        s = self._samples
        dur = max(self._t1 - self._t0, 1e-9)
        if not s:
            zeros = dict.fromkeys(
                [f for f in SampleSummary.__dataclass_fields__ if f
                 not in ("n_samples", "duration_s", "energy_counter_j",
                         "energy_integrated_j")],
                float("nan"),
            )
            return SampleSummary(
                n_samples=0,
                duration_s=dur,
                energy_counter_j=self.energy_counter_j,
                energy_integrated_j=None,
                **zeros,
            )

        p = [x.power_w for x in s]
        sm = [x.sm_clock for x in s]
        mem = [x.mem_clock for x in s]
        tp = [x.temp for x in s]

        # trapezoid over the actual sample timestamps
        integ = 0.0
        for a, b in zip(s, s[1:]):
            integ += 0.5 * (a.power_w + b.power_w) * (b.t - a.t)

        def frac(bit_name: str) -> float:
            bit = _THROTTLE_BITS[bit_name]
            return sum(1 for x in s if x.throttle & bit) / len(s)

        return SampleSummary(
            n_samples=len(s),
            duration_s=dur,
            power_mean_w=statistics.fmean(p),
            power_median_w=statistics.median(p),
            power_max_w=max(p),
            power_sd_w=statistics.pstdev(p) if len(p) > 1 else 0.0,
            sm_clock_mean_mhz=statistics.fmean(sm),
            sm_clock_median_mhz=statistics.median(sm),
            sm_clock_max_mhz=max(sm),
            mem_clock_mean_mhz=statistics.fmean(mem),
            temperature_mean_c=statistics.fmean(tp),
            temperature_max_c=max(tp),
            util_gpu_mean_pct=statistics.fmean(x.util_gpu for x in s),
            util_mem_mean_pct=statistics.fmean(x.util_mem for x in s),
            frac_sw_power_cap=frac("sw_power_cap"),
            frac_hw_slowdown=frac("hw_slowdown"),
            frac_sw_thermal=frac("sw_thermal_slowdown"),
            frac_hw_thermal=frac("hw_thermal_slowdown"),
            energy_counter_j=self.energy_counter_j,
            energy_integrated_j=integ,
        )


# --------------------------------------------------------------------------- #
# Idle
# --------------------------------------------------------------------------- #


@dataclass
class IdleResult:
    power_w: float
    power_sd_w: float
    sm_clock_mhz: float
    temperature_c: float
    seconds: float
    samples: int
    raw: SampleSummary = field(repr=False, default=None)  # type: ignore


def measure_idle(index: int = 0, seconds: float = 20.0,
                 settle_s: float = 5.0) -> IdleResult:
    """Measure resting power with nothing running.

    ``P_idle`` is subtracted nowhere by default -- the dataset records total board
    energy, and what you do with the floor is a modelling choice, not a measurement one.
    But it has to be *known*, because for a short kernel on a 700 W card the idle floor
    can be most of the joules, and a model fit to unsubtracted energy on one card and
    subtracted on another is comparing two different quantities.

    ``settle_s`` is discarded before the window opens: after a workload finishes, clocks
    and fans take several seconds to come down and sampling immediately gives a floor
    that is too high.
    """
    time.sleep(settle_s)
    with PowerSampler(index=index, interval_s=0.02) as s:
        time.sleep(seconds)
    summ = s.summary()
    return IdleResult(
        power_w=summ.power_median_w,
        power_sd_w=summ.power_sd_w,
        sm_clock_mhz=summ.sm_clock_median_mhz,
        temperature_c=summ.temperature_mean_c,
        seconds=summ.duration_s,
        samples=summ.n_samples,
        raw=summ,
    )
