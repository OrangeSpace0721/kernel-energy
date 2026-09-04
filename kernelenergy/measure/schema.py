"""The dataset schema: one row per (kernel config, GPU) measurement.

Column names in the hardware block match ``standard_data`` in the run-level project, so
the two datasets join on ``gpu_key`` and a model fitted on one can be checked against the
other without a translation layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

__all__ = ["MeasurementRow", "SCHEMA_COLUMNS"]


@dataclass
class MeasurementRow:
    # -- identity ------------------------------------------------------------ #
    kernel_sig: str
    category: str
    dtype: str
    gpu_key: str
    source_model: str = ""
    source_module: str = ""
    op_name: str = ""

    # -- measured ------------------------------------------------------------ #
    latency_s: float = float("nan")  # per-iteration, launch overhead removed
    energy_j: float = float("nan")  # per-iteration board energy, overhead removed
    energy_raw_j: float = float("nan")  # before overhead subtraction
    power_avg_w: float = float("nan")
    latency_sd_rel: float = float("nan")  # spread across repeats
    energy_sd_rel: float = float("nan")

    # -- measurement conditions ---------------------------------------------- #
    n_iters: int = 0
    n_repeats: int = 0
    window_s: float = float("nan")
    idle_power_w: float = float("nan")  # measured on this card, this session
    sm_clock_mean_mhz: float = float("nan")
    sm_clock_median_mhz: float = float("nan")
    mem_clock_mean_mhz: float = float("nan")
    temperature_mean_c: float = float("nan")
    temperature_max_c: float = float("nan")
    frac_sw_power_cap: float = float("nan")
    frac_hw_slowdown: float = float("nan")
    frac_sw_thermal: float = float("nan")
    frac_hw_thermal: float = float("nan")
    power_limit_w: float = float("nan")  # what actually binds; may be below TDP
    power_limit_default_w: float = float("nan")  # the card's own default
    energy_counter_used: int = 1
    energy_check_ratio: float = float("nan")  # counter / integrated power
    n_buffers: int = 1

    # -- which physical card, and was it ours alone -------------------------- #
    # Under a scheduler the CUDA ordinal and the NVML ordinal differ, and the UUID is
    # the only identifier invariant under both CUDA_VISIBLE_DEVICES filtering and
    # CUDA_DEVICE_ORDER reordering. Recording all three makes a suspect row traceable to
    # a physical GPU months later; recording none makes an index-mapping error
    # undetectable.
    cuda_index: int = 0
    nvml_index: int = 0
    gpu_uuid: str = ""
    contended: int = 0  # another process was on the board: do not trust this row

    # -- hardware (spec sheet; mirrors standard_data) ------------------------ #
    gpu_name: str = ""
    architecture: str = ""
    compute_capability: float = float("nan")
    sms: int = 0
    ops_per_clk: int = 0
    tensor_clock_mhz: float = float("nan")
    boost_clock_mhz: float = float("nan")
    peak_tensor_flops: float = float("nan")
    mem_bandwidth_gbs: float = float("nan")
    l2_bandwidth_gbs: float = float("nan")
    l2_cache_mb: float = float("nan")
    tdp_w: float = float("nan")
    is_hbm: int = 0

    # -- analytical (filled by the feature stage) ---------------------------- #
    theoretical_time_s: float = float("nan")
    bottleneck: str = ""
    analytic_flops: float = float("nan")
    analytic_bytes_global: float = float("nan")

    # -- provenance ---------------------------------------------------------- #
    timestamp: str = ""
    host: str = ""
    slurm_job: str = ""  # job or array-task label, for tracing back to a log
    shard: str = ""  # "k/N" when the sweep was split across array tasks
    driver_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    kernelenergy_version: str = ""
    notes: str = ""

    # -- kernel parameters, flattened as p_* --------------------------------- #
    params: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_row(self) -> dict:
        d = asdict(self)
        params = d.pop("params", {}) or {}
        for k, v in params.items():
            d[f"p_{k}"] = v
        return d


SCHEMA_COLUMNS = [f.name for f in fields(MeasurementRow) if f.name != "params"]
