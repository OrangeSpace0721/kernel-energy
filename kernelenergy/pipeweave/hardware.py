"""Translate a fleet :class:`~kernelenergy.hardware.GPU` into PipeWeave's ``HardwareSpec``.

Their checkpoints were trained against a specific set of hardware constants. Feeding
the trunk a feature vector computed from *different* constants for the same card is the
quiet way to make transfer fail: nothing errors, the features are simply in the wrong
place and the trunk reads them as a different GPU. So for every card PipeWeave ships a
``hardware/*.json`` for, this module uses **their numbers verbatim**, not the fleet
table's.

Where the two disagree, theirs wins here and the fleet table is left alone -- the fleet
table serves the analytical floor in ``kernelenergy.model.features``, which is a
different computation with its own conventions (sparse peaks, tensor-vs-boost clock).
Mixing them would corrupt both.

Four cards in the fleet have no upstream JSON: **A100 PCIe, L40S, L4, H200 NVL**. Their
rows below are built in upstream's style from the same die, and every one of them is
annotated with what it was derived from. The L4's L2 bandwidth is the only number in
this file that is neither measured nor a datasheet figure.

On ``tcBf16``
-------------
Upstream's ``tcBf16`` is MMA output elements per clock per SM **with sparsity**, the
same convention as ``GPU.ops_per_clk``: A100 2048 gives ``108 * 2048 * 1410e6 * 2 =
623.7 TFLOP/s``, which is the sparse BF16 figure. The Ada FP32/FP16-accumulate
ambiguity survives into their table too, and they resolve it the same way this project
does: L40 at 512, RTX 6000 Ada at 1024. Two AD102 parts, same 142 SMs, differing by 2x
because NVIDIA quotes them against different accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelenergy.hardware import GPU, get_gpu
from kernelenergy.pipeweave.vendor import HardwareSpec

__all__ = ["PW_HARDWARE", "PWHardware", "to_hardware_spec", "gemm_calculator_for"]


@dataclass(frozen=True)
class PWHardware:
    """One row of PipeWeave's ``hardware/*.json``, plus where it came from."""

    name: str  # upstream's exact ``name`` field, or ours for a derived row
    architecture: str  # ampere | ada | hopper | blackwell -- drives GEMM dispatch
    num_sms: int
    tc_bf16: float
    tc_fp8: float
    xu_fp32: float
    fma_fp32: float
    sm_freq: float  # MHz
    mem_bandwidth: float  # GB/s
    l2_cache_bandwidth: float  # GB/s
    shared_memory_bandwidth: float  # bytes / clk / SM
    shared_memory_size: float  # KB / SM
    upstream: bool  # True = verbatim from their JSON
    note: str = ""

    def spec(self) -> HardwareSpec:
        return HardwareSpec(
            tc_bf16=self.tc_bf16,
            tc_fp8=self.tc_fp8,
            xu_fp32=self.xu_fp32,
            fma_fp32=self.fma_fp32,
            num_sms=self.num_sms,
            sm_freq=self.sm_freq,
            mem_bandwidth=self.mem_bandwidth,
            l2_cache_bandwidth=self.l2_cache_bandwidth,
            shared_memory_bandwidth=self.shared_memory_bandwidth,
            shared_memory_size=self.shared_memory_size,
        )


#: Keyed by ``gpu_key`` from :mod:`kernelenergy.hardware`.
PW_HARDWARE: dict[str, PWHardware] = {
    # ------------------------------------------------------------------ #
    # Verbatim from upstream hardware/*.json
    # ------------------------------------------------------------------ #
    "A100_SXM4": PWHardware(
        name="NVIDIA A100-SXM4-80GB",
        architecture="ampere",
        num_sms=108, tc_bf16=2048, tc_fp8=0, xu_fp32=16, fma_fp32=64,
        sm_freq=1410, mem_bandwidth=2039.04, l2_cache_bandwidth=3235,
        shared_memory_bandwidth=128, shared_memory_size=164,
        upstream=True,
        note="hardware/A100.json. Also a *training* GPU for every one of their "
             "checkpoints, so this card is the one place the transfer is pure "
             "interpolation.",
    ),
    "H100": PWHardware(
        name="NVIDIA H100",
        architecture="hopper",
        num_sms=132, tc_bf16=4096, tc_fp8=8192, xu_fp32=16, fma_fp32=128,
        sm_freq=1830, mem_bandwidth=3352.32, l2_cache_bandwidth=8820,
        shared_memory_bandwidth=128, shared_memory_size=228,
        upstream=True,
        note="hardware/H100.json. Held out of their GEMM training set (H800 is the "
             "near-identical stand-in that is in it).",
    ),
    "H200_SXM": PWHardware(
        name="NVIDIA H200",
        architecture="hopper",
        num_sms=132, tc_bf16=4096, tc_fp8=8192, xu_fp32=16, fma_fp32=128,
        sm_freq=1830, mem_bandwidth=4916.7, l2_cache_bandwidth=10403,
        shared_memory_bandwidth=128, shared_memory_size=228,
        upstream=True,
        note="hardware/H200.json -- the 141 GB SXM part.",
    ),
    "L40": PWHardware(
        name="NVIDIA L40",
        architecture="ada",
        num_sms=142, tc_bf16=512, tc_fp8=0, xu_fp32=16, fma_fp32=128,
        sm_freq=2490, mem_bandwidth=864.096, l2_cache_bandwidth=5647,
        shared_memory_bandwidth=128, shared_memory_size=100,
        upstream=True,
        note="hardware/L40.json. Independently confirms tc_bf16=512, the correction "
             "this project had to make after first writing 1024.",
    ),
    # ------------------------------------------------------------------ #
    # Derived. Same die as an upstream row wherever possible.
    # ------------------------------------------------------------------ #
    "A100_PCIE": PWHardware(
        name="NVIDIA A100-PCIE-80GB",
        architecture="ampere",
        num_sms=108, tc_bf16=2048, tc_fp8=0, xu_fp32=16, fma_fp32=64,
        sm_freq=1410, mem_bandwidth=1935.0, l2_cache_bandwidth=3235,
        shared_memory_bandwidth=128, shared_memory_size=164,
        upstream=False,
        note="GA100, identical to A100.json except HBM2e bandwidth: 1935 GB/s for the "
             "80 GB PCIe part against 2039 for SXM4. Everything on-die is the same "
             "silicon at the same clock, so L2 bandwidth carries over.",
    ),
    "L40S": PWHardware(
        name="NVIDIA L40S",
        architecture="ada",
        num_sms=142, tc_bf16=1024, tc_fp8=0, xu_fp32=16, fma_fp32=128,
        sm_freq=2520, mem_bandwidth=864.096, l2_cache_bandwidth=6326,
        shared_memory_bandwidth=128, shared_memory_size=100,
        upstream=False,
        note="AD102, same 142 SMs as L40 and RTX 6000 Ada. tc_bf16 follows RTX 6000 "
             "Ada (1024, FP16 accumulate) not L40 (512, FP32 accumulate) -- the L40S "
             "is quoted at 724 sparse BF16 TFLOP/s, exactly 2x the L40's 362. L2 "
             "bandwidth takes RTX 6000 Ada's 6326 rather than L40's 5647: those two "
             "are the same die and disagree by 12%, and the L40S's 2520 MHz sits "
             "closer to RTX 6000 Ada's 2505 than to L40's 2490.",
    ),
    "L4": PWHardware(
        name="NVIDIA L4",
        architecture="ada",
        num_sms=58, tc_bf16=1024, tc_fp8=0, xu_fp32=16, fma_fp32=128,
        sm_freq=2040, mem_bandwidth=300.0, l2_cache_bandwidth=2576,
        shared_memory_bandwidth=128, shared_memory_size=100,
        upstream=False,
        note="AD104. num_sms is 58, not the 60 in the fleet table -- 7424 CUDA cores "
             "at 128 per SM. The fleet table's 60 x 1969 MHz happens to reproduce the "
             "242 TFLOP/s datasheet peak, so the error is invisible in any aggregate "
             "and 3% wrong in every per-SM quantity. Confirm on the card with "
             "torch.cuda.get_device_properties().multi_processor_count. "
             "l2_cache_bandwidth is the ONLY estimated number in this file: AD104 has "
             "half AD102's L2 slices (48 vs 96 MB), so 6326 * (48/96) * (2040/2505) "
             "= 2576. Treat any L4 result that hinges on it as provisional.",
    ),
    "H200_NVL": PWHardware(
        name="NVIDIA H200 NVL",
        architecture="hopper",
        num_sms=132, tc_bf16=4096, tc_fp8=8192, xu_fp32=16, fma_fp32=128,
        sm_freq=1830, mem_bandwidth=4800.0, l2_cache_bandwidth=10403,
        shared_memory_bandwidth=128, shared_memory_size=228,
        upstream=False,
        note="GH100, identical to H200.json except the NVL part's 4800 GB/s against "
             "the SXM's 4916.7.",
    ),
}


def to_hardware_spec(gpu: GPU | str) -> HardwareSpec:
    """Fleet GPU -> PipeWeave ``HardwareSpec``.

    Raises rather than falling back to the fleet table's own constants. A card with no
    row here cannot be pushed through their checkpoints honestly, and inventing
    constants on the fly is how a transfer silently degrades into noise.
    """
    key = gpu if isinstance(gpu, str) else gpu.gpu_key
    key = str(key).upper()
    if key not in PW_HARDWARE:
        # Resolve aliases through the fleet table before giving up.
        try:
            key = get_gpu(key).gpu_key
        except Exception:
            pass
    try:
        return PW_HARDWARE[key].spec()
    except KeyError:
        raise KeyError(
            f"no PipeWeave hardware row for {key!r}. Add one to PW_HARDWARE with a "
            f"note saying where each constant came from -- do not fall back to the "
            f"fleet table, its conventions differ."
        ) from None


def gemm_calculator_for(gpu: GPU | str):
    """The GEMM calculator upstream's ``aggregator.py`` would pick for this card.

    ``hopper -> gemm9``, **everything else -> gemm8**, Blackwell included. That is the
    dispatch in their aggregator, and it is not what ``gemm_9_calculator``'s docstring
    ("SM90/100") implies. Following the docstring reproduces their own Blackwell rows
    with 99.4% error on ``tensor_sm_max_ops``; following the dispatch reproduces them
    to 4.8e-16.
    """
    from kernelenergy.pipeweave.vendor import gemm8_calculator, gemm9_calculator

    key = gpu if isinstance(gpu, str) else gpu.gpu_key
    key = str(key).upper()
    row = PW_HARDWARE.get(key)
    if row is None:
        try:
            row = PW_HARDWARE[get_gpu(key).gpu_key]
        except Exception:
            raise KeyError(f"no PipeWeave hardware row for {key!r}") from None
    return gemm9_calculator if row.architecture == "hopper" else gemm8_calculator
