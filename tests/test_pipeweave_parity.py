"""Prove the vendored analytical model reproduces PipeWeave's own datasets.

This is the test the entire transfer rests on. PipeWeave's checkpoints read an 11- or
15-element vector; if this project computes that vector even slightly differently, the
trunk reads noise and *nothing raises*. The failure looks like "transfer doesn't work",
which is indistinguishable from a real negative result. So it gets proved, not assumed.

The fixtures under ``tests/fixtures/`` are stratified samples of the CSVs PipeWeave
ships -- the same files their published weights were fitted on. Each row carries both
the problem description and the feature values their pipeline produced, so the check is
end to end: build their problem config from their columns, run the vendored calculator,
compare against the columns they recorded.

Tolerance is ``1e-9`` relative. Observed agreement is ~5e-16, i.e. float64 rounding,
because this is not an equivalent computation -- it is the same one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kernelenergy.pipeweave import vendor
from kernelenergy.pipeweave.vendor import (
    FaProblemConfig,
    GemmProblemConfig,
    HardwareSpec,
    RmsNormProblemConfig,
    SiluMulProblemConfig,
    calculate_fa2_params,
    calculate_fa3_params,
    gemm8_calculator,
    gemm9_calculator,
    rmsnorm_calculator,
    silu_mul_calculator,
)

FIXTURES = Path(__file__).parent / "fixtures"
RTOL = 1e-9

#: Upstream's ``hardware/*.json`` architecture field, needed for calculator dispatch.
#: Kept here rather than imported so the test fails loudly if the mapping drifts.
ARCH = {
    "NVIDIA A100-SXM4-80GB": "ampere",
    "NVIDIA A40": "ampere",
    "NVIDIA RTX A6000": "ampere",
    "NVIDIA H100": "hopper",
    "NVIDIA H20": "hopper",
    "NVIDIA H200": "hopper",
    "NVIDIA H800": "hopper",
    "NVIDIA L20": "ada",
    "NVIDIA L40": "ada",
    "NVIDIA RTX 6000 Ada Generation": "ada",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "blackwell",
}

#: Upstream's ``sharedMemorySize`` (KB/SM), verbatim from their ``hardware/*.json``.
#:
#: Only FA2/FA3 read it, and they are unforgiving about it: it sets
#: ``num_ctas_per_sm`` (2 when two CTAs' shared memory fits, else 1) and bounds
#: ``max_num_mma_kv_smem``, which fixes ``cta_tile_kv``. Getting it wrong shifts the
#: whole feature vector by a factor of two on the rows where occupancy flips, and by a
#: few percent everywhere else. The GA100/GA102 split is the trap: the A100 has 164 KB
#: per SM, every other Ampere part here has 100.
SMEM_KB = {
    "NVIDIA A100-SXM4-80GB": 164,
    "NVIDIA A40": 100,
    "NVIDIA RTX A6000": 100,
    "NVIDIA H100": 228, "NVIDIA H20": 228, "NVIDIA H200": 228, "NVIDIA H800": 228,
    "NVIDIA L20": 100, "NVIDIA L40": 100, "NVIDIA RTX 6000 Ada Generation": 100,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 100,
}

MEMORY = ["global_in_flight", "global_cycle", "local_cycle", "sm_max_in_flight",
          "sm_max_global_cycle", "sm_max_shared_cycle", "sm_max_local_cycle"]


def _spec(r) -> HardwareSpec:
    """Build the hardware spec from the row's own columns.

    Every constant comes off the CSV rather than out of a table, so this test checks
    the *calculators* and nothing else. ``kernelenergy/pipeweave/hardware.py`` is
    checked separately in ``test_fleet_hardware_matches_upstream``.
    """
    return HardwareSpec(
        tc_bf16=r.tc_bf16, tc_fp8=0.0, xu_fp32=r.xu_fp32, fma_fp32=r.fma_fp32,
        num_sms=int(r.num_sms), sm_freq=r.sm_freq, mem_bandwidth=r.mem_bandwidth,
        l2_cache_bandwidth=r.l2_cache_bandwidth,
        shared_memory_bandwidth=r.shared_memory_bandwidth,
        shared_memory_size=SMEM_KB.get(r.hardware),
    )


def _memory_values(mp):
    return [mp.global_in_flight, mp.global_cycle, mp.local_cycle, mp.sm_max_in_flight,
            mp.sm_max_global_cycle, mp.sm_max_shared_cycle, mp.sm_max_local_cycle]


def _pipe_values(p):
    return [p.all_ops, p.all_cycle, p.sm_max_ops, p.sm_max_cycle]


def _compare(got: dict[str, float], row, cols) -> tuple[float, str]:
    worst, which = 0.0, ""
    for c in cols:
        ref = float(row[c])
        rel = abs(got[c] - ref) / max(abs(ref), 1e-30)
        if rel > worst:
            worst, which = rel, c
    return worst, which


# --------------------------------------------------------------------------- #
# The vendored files are what they claim to be
# --------------------------------------------------------------------------- #


def test_vendored_files_match_their_checksums():
    """A silent edit to a vendored file voids every parity claim in NOTICE.md."""
    result = vendor.verify()
    assert result, "SHA256SUMS is empty -- the vendored tree is not intact"
    bad = [name for name, ok in result.items() if not ok]
    assert not bad, (
        f"vendored PipeWeave files no longer match their checksums: {bad}. These are "
        f"upstream's code and must stay byte-identical; revert the edit and put the "
        f"change in an adapter instead."
    )


# --------------------------------------------------------------------------- #
# GEMM
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hardware", sorted(ARCH))
def test_gemm_features_reproduce_upstream(hardware):
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz")
    g = df[df["hardware"] == hardware]
    assert len(g) > 0, f"no fixture rows for {hardware}"
    cols = ["tensor_all_ops", "tensor_all_cycle", "tensor_sm_max_ops",
            "tensor_sm_max_cycle"] + MEMORY

    calc = gemm9_calculator if ARCH[hardware] == "hopper" else gemm8_calculator
    worst, which, at = 0.0, "", None
    for _, r in g.iterrows():
        f = calc(
            GemmProblemConfig(
                m=int(r.M), n=int(r.N), k=int(r.K),
                tile_m=int(r.tile_M), tile_n=int(r.tile_N), tile_k=int(r.tile_K),
                cta_count=int(r.cta_count), is_split_k=bool(r.is_split_k),
                data_size_bytes=2,
            ),
            _spec(r),
        )
        got = dict(zip(cols, _pipe_values(f.tensor_pipe) + _memory_values(f.memory_pipe)))
        w, c = _compare(got, r, cols)
        if w > worst:
            worst, which, at = w, c, (int(r.M), int(r.N), int(r.K))
    assert worst < RTOL, (
        f"{hardware}: max relative error {worst:.3e} on {which} at MNK={at}. "
        f"Agreement should be float64 rounding (~5e-16)."
    )


def test_blackwell_dispatches_to_gemm8_not_gemm9():
    """The dispatch is by ``architecture == 'hopper'``, not by SM version.

    ``gemm_9_calculator``'s docstring says "SM90/100", and Blackwell is SM100 -- but
    ``aggregator.py`` sends everything that is not ``hopper`` to ``gemm8``, and their
    Blackwell data was generated that way. Reading the docstring instead of the
    dispatch is a 99% error that this test exists to keep caught.
    """
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz")
    g = df[df["hardware"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"]
    assert len(g) > 0

    def err(calc):
        worst = 0.0
        for _, r in g.iterrows():
            f = calc(
                GemmProblemConfig(
                    m=int(r.M), n=int(r.N), k=int(r.K),
                    tile_m=int(r.tile_M), tile_n=int(r.tile_N), tile_k=int(r.tile_K),
                    cta_count=int(r.cta_count), is_split_k=bool(r.is_split_k),
                    data_size_bytes=2,
                ),
                _spec(r),
            )
            ref = float(r["tensor_sm_max_ops"])
            worst = max(worst, abs(f.tensor_pipe.sm_max_ops - ref) / max(abs(ref), 1e-30))
        return worst

    assert err(gemm8_calculator) < RTOL
    assert err(gemm9_calculator) > 0.1, (
        "gemm9 now reproduces the Blackwell rows too -- if upstream changed the "
        "dispatch, gemm_calculator_for() must change with it."
    )


# --------------------------------------------------------------------------- #
# Norm / elementwise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture,calculator,make",
    [
        ("pipeweave_rmsnorm.csv.gz", rmsnorm_calculator,
         lambda r: RmsNormProblemConfig(batch_size=int(r.seq), dim=int(r.dim), dtype_size=2)),
        ("pipeweave_siluandmul.csv.gz", silu_mul_calculator,
         lambda r: SiluMulProblemConfig(seq_len=int(r.seq), dim=int(r.dim), dtype_size=2)),
    ],
)
def test_fma_operators_reproduce_upstream(fixture, calculator, make):
    df = pd.read_csv(FIXTURES / fixture)
    cols = (["fma_all_ops", "fma_all_cycle", "fma_sm_max_ops", "fma_sm_max_cycle",
             "xu_all_ops", "xu_all_cycle", "xu_sm_max_ops", "xu_sm_max_cycle"] + MEMORY)
    worst, which, at = 0.0, "", None
    for _, r in df.iterrows():
        f = calculator(make(r), _spec(r))
        got = dict(zip(
            cols,
            _pipe_values(f.fma_pipe) + _pipe_values(f.xu_pipe) + _memory_values(f.memory_pipe),
        ))
        w, c = _compare(got, r, cols)
        if w > worst:
            worst, which, at = w, c, (int(r.seq), int(r.dim), r.hardware)
    # dtype_size=2 is not a column in their CSVs; that this reproduces every row is
    # itself the evidence that bf16 is what they generated with.
    assert worst < RTOL, f"max relative error {worst:.3e} on {which} at {at}"


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


def test_attention_features_reproduce_upstream():
    df = pd.read_csv(FIXTURES / "pipeweave_attn.csv.gz")
    cols = (["tensor_all_ops", "tensor_all_cycle", "tensor_sm_max_ops", "tensor_sm_max_cycle",
             "xu_all_ops", "xu_all_cycle", "xu_sm_max_ops", "xu_sm_max_cycle"] + MEMORY)
    worst, which, at = 0.0, "", None
    for _, r in df.iterrows():
        t = r.attention_type
        f = (calculate_fa2_params if "fa2" in t else calculate_fa3_params)(
            FaProblemConfig(
                batch_size=int(r.bs),
                q_lengths=ast.literal_eval(r.q_lengths),
                kv_lengths=ast.literal_eval(r.kv_lengths),
                num_qo_heads=int(r.nh), num_kv_heads=int(r.nkv), head_dim=int(r.hd),
                layout="paged" if "paged" in t else "ragged",
                data_size_q=2, data_size_kv=2, data_size_o=2,
                # Hardcoded True in aggregator.py for every row they generated.
                causal=True,
            ),
            _spec(r),
        )
        got = dict(zip(
            cols,
            _pipe_values(f.tensor_pipe) + _pipe_values(f.xu_pipe) + _memory_values(f.memory_pipe),
        ))
        w, c = _compare(got, r, cols)
        if w > worst:
            worst, which, at = w, c, (t, r.hardware)
    assert worst < RTOL, f"max relative error {worst:.3e} on {which} at {at}"


def test_upstream_attention_corpus_is_entirely_causal():
    """Documents the single largest distribution gap in the transfer.

    Diffusion attention is non-causal. Their attention checkpoint has never seen a
    non-causal row, because ``aggregator.py`` passes ``causal=True`` unconditionally
    and there is no causal column in the dataset to vary. This is not a bug to fix --
    it is a limit to report, and ``features.emit`` attaches a note saying so.
    """
    df = pd.read_csv(FIXTURES / "pipeweave_attn.csv.gz")
    assert "causal" not in df.columns
    assert set(df["attention_type"]) <= {"fa2_paged", "fa2_ragged", "fa3_paged", "fa3_ragged"}


# --------------------------------------------------------------------------- #
# Our hardware rows against theirs
# --------------------------------------------------------------------------- #


def test_fleet_hardware_matches_upstream_where_they_overlap():
    """Cards upstream measured must carry upstream's constants, not the fleet table's.

    The fleet table in ``kernelenergy/hardware.py`` serves the analytical floor, which
    uses sparse peaks and a tensor clock. Their constants serve their checkpoints. The
    two disagree on several cards and both are right for their own purpose; what would
    be wrong is feeding one into the other.
    """
    from kernelenergy.pipeweave.hardware import PW_HARDWARE

    upstream = {
        "A100_SXM4": dict(num_sms=108, tc_bf16=2048, fma_fp32=64, sm_freq=1410,
                          mem_bandwidth=2039.04, l2_cache_bandwidth=3235),
        "H100": dict(num_sms=132, tc_bf16=4096, fma_fp32=128, sm_freq=1830,
                     mem_bandwidth=3352.32, l2_cache_bandwidth=8820),
        "H200_SXM": dict(num_sms=132, tc_bf16=4096, fma_fp32=128, sm_freq=1830,
                         mem_bandwidth=4916.7, l2_cache_bandwidth=10403),
        "L40": dict(num_sms=142, tc_bf16=512, fma_fp32=128, sm_freq=2490,
                    mem_bandwidth=864.096, l2_cache_bandwidth=5647),
    }
    for key, expected in upstream.items():
        row = PW_HARDWARE[key]
        assert row.upstream, f"{key} should be marked as taken verbatim from upstream"
        for field, value in expected.items():
            assert getattr(row, field) == value, (
                f"{key}.{field} is {getattr(row, field)}, upstream says {value}"
            )


def test_l40_tensor_throughput_is_half_the_l40s():
    """The correction that cost this project a 2x peak error, pinned from two sides.

    Same AD102 die, same 142 SMs, near-identical clocks -- and NVIDIA quotes the L40 at
    362 sparse BF16 TFLOP/s against the L40S's 724, because the L40 figure is against
    an FP32 accumulator. Upstream's ``hardware/L40.json`` says 512 independently.
    """
    from kernelenergy.pipeweave.hardware import PW_HARDWARE

    assert PW_HARDWARE["L40"].tc_bf16 == 512
    assert PW_HARDWARE["L40S"].tc_bf16 == 1024
    assert PW_HARDWARE["L40"].num_sms == PW_HARDWARE["L40S"].num_sms == 142
