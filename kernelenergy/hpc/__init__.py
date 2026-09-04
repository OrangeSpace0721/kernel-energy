"""HPC integration: device resolution, scheduler awareness, preflight checks.

Nothing in here changes what is measured. It changes what is *trusted*: under a scheduler
the same code can measure the wrong card, share a board with a stranger, or be killed
mid-window, and none of those announce themselves in the data.
"""

from kernelenergy.hpc.device import (
    DeviceResolutionError,
    GpuContendedError,
    ResolvedDevice,
    assert_exclusive_gpu,
    assert_no_mig,
    gpu_processes,
    power_limits,
    resolve_device,
)
from kernelenergy.hpc.preflight import Check, format_report, run_preflight
from kernelenergy.hpc.slurm import (
    EXIT_INCOMPLETE,
    InterruptGuard,
    SlurmContext,
    shard_configs,
    slurm_context,
)

__all__ = [
    "Check",
    "DeviceResolutionError",
    "EXIT_INCOMPLETE",
    "GpuContendedError",
    "InterruptGuard",
    "ResolvedDevice",
    "SlurmContext",
    "assert_exclusive_gpu",
    "assert_no_mig",
    "format_report",
    "gpu_processes",
    "power_limits",
    "resolve_device",
    "run_preflight",
    "shard_configs",
    "slurm_context",
]
