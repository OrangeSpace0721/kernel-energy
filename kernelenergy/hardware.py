"""Hardware descriptors for the fleet.

Two things live here.

1. A static table of spec-sheet parameters, one row per GPU. The columns are a superset
   of PipeWeave Table II (the parameters its analytical model needs) and of the hardware
   columns already used by ``standard_data`` in the run-level project, so the two
   datasets join on ``gpu_key``.

2. ``probe_local_gpu()``, which identifies which row the machine you are sitting on
   corresponds to, so a collection script does not have to be told.

On clocks
---------
The table carries **two** clocks and they are not interchangeable. ``boost_clock_mhz`` is
what the CSV and ``nvidia-smi`` report. ``tensor_clock_mhz`` is the frequency NVIDIA
quotes tensor throughput against, recovered as ``peak / (SMs * ops_per_clk)``. They differ
by up to 15.5% (H200 NVL), and the run-level work in this project found the tensor clock
strictly better for latency at every shrinkage level while the *energy* fold moved the
other way. That divergence is unresolved, so both are carried and
``peak_tensor_flops()`` takes the clock as an argument rather than choosing for you.

On ``ops_per_clk``
------------------
Tensor MMA output elements per clock per SM, for the precision named in
``ops_per_clk_precision``. Ada parts are ambiguous: 512 with FP32 accumulate, 1024 with
FP16 accumulate. PipeWeave Table VI lists the L40 at 512; the run-level work in this
project uses 1024. Both are carried; ``ops_per_clk`` follows the project convention and
``ops_per_clk_fp32acc`` holds the other.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

__all__ = [
    "GPU",
    "GPUS",
    "get_gpu",
    "probe_local_gpu",
    "hardware_frame",
    "load_hardware_csv",
]


@dataclass(frozen=True)
class GPU:
    """Spec-sheet descriptor for one GPU model."""

    gpu_key: str  # join key, matches standard_data
    name: str  # human label
    architecture: str  # Ampere | Ada | Hopper | Blackwell
    compute_capability: float

    # --- execution resources (PipeWeave Table II) ---
    sms: int
    ops_per_clk: int  # tensor MMA output elements / clk / SM
    ops_per_clk_fp32acc: int  # same, FP32 accumulate (Ada differs)
    ops_per_clk_precision: str
    fma_per_clk: int  # FP32 FMA lanes / SM
    xu_per_clk: int  # special-function lanes / SM

    # --- clocks ---
    boost_clock_mhz: float  # datasheet boost, what nvidia-smi reports
    tensor_clock_mhz: float  # frequency tensor throughput is quoted against
    max_sm_clock_mhz: float  # highest SM clock observed / allowed

    # --- memory hierarchy ---
    mem_bandwidth_gbs: float
    l2_bandwidth_gbs: float
    l2_cache_mb: float
    smem_bandwidth_b_per_clk_sm: int  # shared-memory bytes / clk / SM
    smem_size_kb_sm: float
    regfile_kb_sm: float
    mem_tech: str  # HBM | GDDR

    # --- power ---
    tdp_w: float
    idle_power_w: float  # measured at idle; falls back to tdp**0.64

    # --- identification ---
    nvml_patterns: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------ #

    @property
    def is_hbm(self) -> bool:
        return self.mem_tech.upper() == "HBM"

    def peak_tensor_flops(self, clock: str = "tensor") -> float:
        """Peak tensor throughput in FLOP/s.

        ``clock='tensor'`` reproduces the datasheet number exactly by construction;
        ``clock='boost'`` overstates it by 0-15.5% depending on the card.
        """
        mhz = self.tensor_clock_mhz if clock == "tensor" else self.boost_clock_mhz
        # ops_per_clk counts multiply-accumulates as output elements; x2 for FLOPs.
        return self.sms * self.ops_per_clk * mhz * 1e6 * 2.0

    def peak_fma_flops(self, clock: str = "boost") -> float:
        mhz = self.tensor_clock_mhz if clock == "tensor" else self.boost_clock_mhz
        return self.sms * self.fma_per_clk * mhz * 1e6 * 2.0

    def peak_xu_ops(self, clock: str = "boost") -> float:
        mhz = self.tensor_clock_mhz if clock == "tensor" else self.boost_clock_mhz
        return self.sms * self.xu_per_clk * mhz * 1e6

    def smem_bandwidth_gbs(self, clock: str = "boost") -> float:
        mhz = self.tensor_clock_mhz if clock == "tensor" else self.boost_clock_mhz
        return self.sms * self.smem_bandwidth_b_per_clk_sm * mhz * 1e6 / 1e9

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("nvml_patterns", None)
        d["peak_tensor_flops"] = self.peak_tensor_flops("tensor")
        d["peak_tensor_flops_boost"] = self.peak_tensor_flops("boost")
        d["peak_fma_flops"] = self.peak_fma_flops()
        d["is_hbm"] = int(self.is_hbm)
        return d


# --------------------------------------------------------------------------- #
# The fleet.
#
# Values are spec-sheet figures. ``idle_power_w`` is the one number that must be
# measured per card -- ``kernelenergy measure-idle`` writes it back. Until then it is
# the tdp**0.64 estimate used in the run-level model.
# --------------------------------------------------------------------------- #

GPUS: dict[str, GPU] = {
    "L4": GPU(
        gpu_key="L4",
        name="NVIDIA L4",
        architecture="Ada",
        compute_capability=8.9,
        sms=60,
        ops_per_clk=1024,
        ops_per_clk_fp32acc=512,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=2040.0,
        tensor_clock_mhz=1969.0,
        max_sm_clock_mhz=2010.0,
        mem_bandwidth_gbs=300.0,
        l2_bandwidth_gbs=2430.0,
        l2_cache_mb=48.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=100.0,
        regfile_kb_sm=256.0,
        mem_tech="GDDR",
        tdp_w=72.0,
        idle_power_w=15.0,
        nvml_patterns=(r"\bL4\b",),
    ),
    "L40S": GPU(
        gpu_key="L40S",
        name="NVIDIA L40S",
        architecture="Ada",
        compute_capability=8.9,
        sms=142,
        ops_per_clk=1024,
        ops_per_clk_fp32acc=512,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=2520.0,
        tensor_clock_mhz=2490.0,
        max_sm_clock_mhz=2520.0,
        mem_bandwidth_gbs=864.0,
        l2_bandwidth_gbs=2430.0,
        l2_cache_mb=96.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=100.0,
        regfile_kb_sm=256.0,
        mem_tech="GDDR",
        tdp_w=350.0,
        idle_power_w=32.0,
        nvml_patterns=(r"L40S",),
    ),
    "L40": GPU(
        # The L40S's quieter sibling: same AD102 die, same 142 SMs, near-identical
        # clocks -- and a 300 W board instead of 350 W, with NVIDIA quoting its tensor
        # throughput at FP32 accumulate (362 TFLOP/s) where the L40S is quoted at FP16
        # accumulate (733). The descriptor difference is therefore almost entirely
        # *power*, which makes the pair unusually informative for an energy model:
        # two cards at the same point in SM/clock space with different budgets is a
        # near-controlled experiment on what TDP alone does to achieved efficiency.
        # Note this does not answer the L40S extrapolation problem in
        # machine-descriptor-pipeweave.md §3 -- it duplicates the L40S's position
        # rather than bracketing it -- but it does make that position no longer a
        # single point.
        gpu_key="L40",
        name="NVIDIA L40",
        architecture="Ada",
        compute_capability=8.9,
        sms=142,
        ops_per_clk=1024,
        ops_per_clk_fp32acc=512,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=2490.0,
        tensor_clock_mhz=2490.0,
        max_sm_clock_mhz=2490.0,
        mem_bandwidth_gbs=864.0,
        l2_bandwidth_gbs=2430.0,
        l2_cache_mb=96.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=100.0,
        regfile_kb_sm=256.0,
        mem_tech="GDDR",
        tdp_w=300.0,
        idle_power_w=30.0,
        # Must not swallow "NVIDIA L40S" -- the negative lookahead is the whole point.
        nvml_patterns=(r"L40(?!S)",),
    ),
    "A100_PCIE": GPU(
        gpu_key="A100_PCIE",
        name="NVIDIA A100 80GB PCIe",
        architecture="Ampere",
        compute_capability=8.0,
        sms=108,
        ops_per_clk=2048,
        ops_per_clk_fp32acc=2048,
        ops_per_clk_precision="bf16",
        fma_per_clk=64,
        xu_per_clk=16,
        boost_clock_mhz=1410.0,
        tensor_clock_mhz=1411.0,
        max_sm_clock_mhz=1410.0,
        mem_bandwidth_gbs=1935.0,
        l2_bandwidth_gbs=5120.0,
        l2_cache_mb=40.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=164.0,
        regfile_kb_sm=256.0,
        mem_tech="HBM",
        tdp_w=250.0,
        idle_power_w=38.0,
        nvml_patterns=(r"A100.*PCI", r"A100 80GB PCIe"),
    ),
    "A100_SXM4": GPU(
        gpu_key="A100_SXM4",
        name="NVIDIA A100-SXM4-80GB",
        architecture="Ampere",
        compute_capability=8.0,
        sms=108,
        ops_per_clk=2048,
        ops_per_clk_fp32acc=2048,
        ops_per_clk_precision="bf16",
        fma_per_clk=64,
        xu_per_clk=16,
        boost_clock_mhz=1410.0,
        tensor_clock_mhz=1411.0,
        max_sm_clock_mhz=1410.0,
        mem_bandwidth_gbs=2039.0,
        l2_bandwidth_gbs=5120.0,
        l2_cache_mb=40.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=164.0,
        regfile_kb_sm=256.0,
        mem_tech="HBM",
        tdp_w=400.0,
        idle_power_w=52.0,
        nvml_patterns=(r"A100.*SXM",),
    ),
    "H100": GPU(
        gpu_key="H100",
        name="NVIDIA H100 SXM5 80GB",
        architecture="Hopper",
        compute_capability=9.0,
        sms=132,
        ops_per_clk=4096,
        ops_per_clk_fp32acc=4096,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=1980.0,
        tensor_clock_mhz=1830.0,
        max_sm_clock_mhz=1980.0,
        mem_bandwidth_gbs=3352.0,
        l2_bandwidth_gbs=8000.0,
        l2_cache_mb=50.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=228.0,
        regfile_kb_sm=256.0,
        mem_tech="HBM",
        tdp_w=700.0,
        idle_power_w=72.0,
        nvml_patterns=(r"H100",),
    ),
    "H200_NVL": GPU(
        gpu_key="H200_NVL",
        name="NVIDIA H200 NVL",
        architecture="Hopper",
        compute_capability=9.0,
        sms=132,
        ops_per_clk=4096,
        ops_per_clk_fp32acc=4096,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=1785.0,
        tensor_clock_mhz=1545.0,
        max_sm_clock_mhz=1785.0,
        mem_bandwidth_gbs=4800.0,
        l2_bandwidth_gbs=8000.0,
        l2_cache_mb=50.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=228.0,
        regfile_kb_sm=256.0,
        mem_tech="HBM",
        tdp_w=600.0,
        idle_power_w=68.0,
        # Anchored so it cannot claim an SXM H200. NVML reports the NVL part as
        # "NVIDIA H200 NVL" and the SXM part as "NVIDIA H200".
        nvml_patterns=(r"H200\s*NVL",),
    ),
    "H200_SXM": GPU(
        # Same GH100 die and the same quoted tensor throughput as the H100 SXM
        # (1979 TFLOP/s), so it shares the H100's 1830 MHz tensor clock -- what changes
        # is memory: 141 GB of HBM3e at 4.8 TB/s against the H100's 80 GB at 3.35 TB/s,
        # at the same 700 W.
        #
        # That makes the H100 SXM / H200 SXM pair the cleanest bandwidth contrast
        # available anywhere in the fleet: identical compute, identical power budget,
        # 43% more bandwidth. Every other bandwidth comparison in this project is
        # confounded with SM count, clock or memory technology at the same time.
        gpu_key="H200_SXM",
        name="NVIDIA H200",
        architecture="Hopper",
        compute_capability=9.0,
        sms=132,
        ops_per_clk=4096,
        ops_per_clk_fp32acc=4096,
        ops_per_clk_precision="bf16",
        fma_per_clk=128,
        xu_per_clk=16,
        boost_clock_mhz=1980.0,
        tensor_clock_mhz=1830.0,
        max_sm_clock_mhz=1980.0,
        mem_bandwidth_gbs=4800.0,
        l2_bandwidth_gbs=8000.0,
        l2_cache_mb=50.0,
        smem_bandwidth_b_per_clk_sm=128,
        smem_size_kb_sm=228.0,
        regfile_kb_sm=256.0,
        mem_tech="HBM",
        tdp_w=700.0,
        idle_power_w=75.0,
        nvml_patterns=(r"H200(?!\s*NVL)",),
    ),
}


def get_gpu(key: str) -> GPU:
    """Look up a GPU by key, case-insensitively, with a helpful error."""
    k = key.strip().upper().replace("-", "_").replace(" ", "_")
    if k in GPUS:
        return GPUS[k]
    for gk, gpu in GPUS.items():
        if gk.upper() == k or gpu.name.upper() == key.strip().upper():
            return gpu
    raise KeyError(f"unknown gpu {key!r}; known keys: {sorted(GPUS)}")


def probe_local_gpu(index: int = 0) -> GPU:
    """Identify the local GPU against the table by matching its NVML product name.

    Raises if the name matches nothing, rather than guessing -- a silently wrong
    hardware descriptor would poison every row the machine collects.
    """
    from kernelenergy.nvml import device_name  # local import, keeps import cheap

    name = device_name(index)
    for gpu in GPUS.values():
        for pat in gpu.nvml_patterns:
            if re.search(pat, name, flags=re.IGNORECASE):
                return gpu
    raise RuntimeError(
        f"GPU {name!r} is not in the hardware table. Add a GPU(...) entry in "
        f"kernelenergy/hardware.py with its spec-sheet values before collecting."
    )


def hardware_frame(gpus: Iterable[GPU] | None = None) -> pd.DataFrame:
    """The hardware table as a DataFrame, one row per GPU."""
    gpus = list(GPUS.values()) if gpus is None else list(gpus)
    return pd.DataFrame([g.as_row() for g in gpus]).set_index("gpu_key", drop=False)


def load_hardware_csv(path: str | Path) -> dict[str, GPU]:
    """Override the built-in table from a CSV.

    Lets you point at the hardware table already maintained in the run-level project
    instead of keeping two copies in step. Columns must match ``GPU`` field names;
    missing columns fall back to the built-in value for that ``gpu_key``.
    """
    df = pd.read_csv(path)
    if "gpu_key" not in df.columns:
        raise ValueError("hardware CSV needs a gpu_key column")
    fields = {f for f in GPU.__dataclass_fields__ if f != "nvml_patterns"}
    out: dict[str, GPU] = {}
    for _, row in df.iterrows():
        key = str(row["gpu_key"]).strip().upper()
        base = GPUS.get(key)
        vals = {} if base is None else {f: getattr(base, f) for f in fields}
        for col in df.columns:
            if col in fields and pd.notna(row[col]):
                vals[col] = row[col]
        missing = fields - vals.keys()
        if missing:
            raise ValueError(f"{key}: no built-in default and CSV lacks {sorted(missing)}")
        # dataclass wants exact types for the int fields
        for f in ("sms", "ops_per_clk", "ops_per_clk_fp32acc", "fma_per_clk",
                  "xu_per_clk", "smem_bandwidth_b_per_clk_sm"):
            vals[f] = int(vals[f])
        pats = base.nvml_patterns if base is not None else ()
        out[key] = GPU(nvml_patterns=pats, **vals)
    return out
