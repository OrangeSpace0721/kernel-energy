"""Tests for the analytical stack: decomposition, scheduling, features.

These check identities that must hold exactly -- op counts against closed forms, task
counts against tile arithmetic, conservation across the scheduler. They do not check
accuracy against hardware, which needs hardware.
"""

from __future__ import annotations

import math

import pytest

from kernelenergy.hardware import GPUS, get_gpu
from kernelenergy.kernels import (
    AttentionKernel,
    ConvKernel,
    ElementwiseKernel,
    GemmKernel,
    KernelConfig,
    NormKernel,
    make_kernel,
)
from kernelenergy.model.features import FEATURE_COLUMNS, analyse
from kernelenergy.model.schedule import simulate


def cfg(category, dtype="bf16", **params):
    return KernelConfig(category=category, dtype=dtype, params=params)


# --------------------------------------------------------------------------- #
# GEMM
# --------------------------------------------------------------------------- #


def test_gemm_flops_closed_form():
    k = GemmKernel(cfg("gemm", m=1024, n=2048, k=512, bias=False))
    assert k.flops() == pytest.approx(2 * 1024 * 2048 * 512)


def test_gemm_task_count_matches_tiling():
    gpu = get_gpu("A100_PCIE")
    k = GemmKernel(cfg("gemm", m=4096, n=3072, k=3072))
    tile = k.tile(gpu)
    tasks = k.decompose(gpu)
    expected = math.ceil(4096 / tile.m) * math.ceil(3072 / tile.n)
    assert sum(t.count for t in tasks) == expected
    assert k.n_ctas(gpu) == expected


def test_gemm_total_mma_recovers_flops():
    """Summed task MMA ops must equal the matmul's multiply-accumulates.

    Tiles overhang when the dimensions do not divide, so the sum is >= and the excess is
    exactly the padding. Both facts are worth pinning: an implementation that quietly
    dropped the ragged edge would pass a looser test.
    """
    gpu = get_gpu("H100")
    m, n, kk = 4096, 3072, 3072
    k = GemmKernel(cfg("gemm", m=m, n=n, k=kk, bias=False))
    tile = k.tile(gpu)
    total_mma = sum(t.n_mma * t.count for t in k.decompose(gpu))
    padded_m = math.ceil(m / tile.m) * tile.m
    padded_n = math.ceil(n / tile.n) * tile.n
    assert total_mma == pytest.approx(2.0 * padded_m * padded_n * kk)
    assert total_mma >= 2.0 * m * n * kk


def test_gemm_l2_traffic_exceeds_global():
    """Tiling re-reads panels; those re-reads are L2 hits, not HBM traffic."""
    gpu = get_gpu("L40S")
    k = GemmKernel(cfg("gemm", m=8192, n=8192, k=1024))
    t = k.decompose(gpu)[0]
    assert t.bytes_l2 > t.bytes_global


def test_tile_choice_fills_the_machine():
    from kernelenergy.kernels.base import choose_gemm_tile

    for sms in (60, 108, 132, 142):
        tile = choose_gemm_tile(8192, 8192, 1024, sms)
        ctas = math.ceil(8192 / tile.m) * math.ceil(8192 / tile.n)
        assert ctas >= sms, "tile leaves SMs idle on a large GEMM"


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


def test_attention_flops_closed_form():
    a = AttentionKernel(cfg("attention", b=2, h=24, s_q=4096, s_kv=4096, d=128))
    assert a.flops() == pytest.approx(4 * 2 * 24 * 4096 * 4096 * 128)


def test_attention_causal_halves_the_work():
    base = dict(b=1, h=16, s_q=4096, s_kv=4096, d=64)
    full = AttentionKernel(cfg("attention", **base, causal=False))
    causal = AttentionKernel(cfg("attention", **base, causal=True))
    assert causal.flops() == pytest.approx(full.flops() / 2)


def test_attention_does_not_materialise_scores():
    """FlashAttention's global traffic is linear in sequence length, not quadratic."""
    small = AttentionKernel(cfg("attention", b=1, h=16, s_q=1024, s_kv=1024, d=64))
    big = AttentionKernel(cfg("attention", b=1, h=16, s_q=4096, s_kv=4096, d=64))
    ratio = big.bytes_global() / small.bytes_global()
    assert ratio == pytest.approx(4.0), "traffic should scale with S, not S^2"


def test_attention_gqa_reduces_global_kv_traffic():
    mha = AttentionKernel(cfg("attention", b=1, h=24, s_q=4096, s_kv=4096, d=128))
    gqa = AttentionKernel(cfg("attention", b=1, h=24, s_q=4096, s_kv=4096, d=128, h_kv=4))
    assert gqa.bytes_global() < mha.bytes_global()
    assert gqa.flops() == pytest.approx(mha.flops())


def test_attention_uses_the_xu_pipe():
    """Softmax exponentials are special-function work and must not land on FMA."""
    a = AttentionKernel(cfg("attention", b=1, h=8, s_q=1024, s_kv=1024, d=64))
    t = a.decompose(get_gpu("A100_PCIE"))[0]
    assert t.n_xu > 0
    assert t.n_mma > t.n_xu


# --------------------------------------------------------------------------- #
# Conv, norm, elementwise
# --------------------------------------------------------------------------- #


def test_conv_output_shape_and_implicit_gemm_dims():
    c = ConvKernel(cfg("conv", n=1, c_in=128, c_out=256, h=128, w=128, kh=3, kw=3,
                       stride=1, pad=1))
    assert (c.h_out, c.w_out) == (128, 128)
    assert c.gemm_m == 1 * 128 * 128
    assert c.gemm_n == 256
    assert c.gemm_k == 128 * 3 * 3


def test_conv_flops_match_direct_count():
    c = ConvKernel(cfg("conv", n=2, c_in=64, c_out=128, h=64, w=64, kh=3, kw=3,
                       bias=False))
    direct = 2.0 * 2 * 128 * 64 * 64 * 64 * 3 * 3
    assert c.flops() == pytest.approx(direct)


def test_norm_is_memory_bound_not_compute_bound():
    n = NormKernel(cfg("norm", rows=8192, dim=3072))
    assert n.arithmetic_intensity() < 1.0


def test_rmsnorm_does_less_work_than_layernorm():
    ln = NormKernel(cfg("norm", rows=4096, dim=3072, kind="layer"))
    rms = NormKernel(cfg("norm", rows=4096, dim=3072, kind="rms"))
    assert rms.flops() < ln.flops()


def test_elementwise_kind_drives_pipe_split():
    add = ElementwiseKernel(cfg("elementwise", n_elem=1 << 20, kind="add"))
    gelu = ElementwiseKernel(cfg("elementwise", n_elem=1 << 20, kind="gelu"))
    ta = add.decompose(get_gpu("L4"))[0]
    tg = gelu.decompose(get_gpu("L4"))[0]
    assert ta.n_xu == 0, "a plain add must not touch the special-function pipe"
    assert tg.n_xu > 0
    assert ta.bytes_global > tg.bytes_global, "add reads two tensors, gelu one"


def test_unknown_elementwise_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown elementwise kind"):
        ElementwiseKernel(cfg("elementwise", n_elem=1024, kind="not_a_real_op"))


def test_missing_params_are_rejected_at_construction():
    with pytest.raises(ValueError, match="missing"):
        GemmKernel(cfg("gemm", m=128, n=128))  # no k


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def test_scheduler_conserves_work():
    gpu = get_gpu("A100_PCIE")
    k = GemmKernel(cfg("gemm", m=4096, n=4096, k=1024))
    tasks = k.decompose(gpu)
    total = sum(t.n_mma * t.count for t in tasks)
    for sched in ("hardware", "software"):
        dist = simulate(tasks, gpu.sms, scheduler=sched)
        assert dist.total("n_mma") == pytest.approx(total)
        assert dist.n_tasks_total == sum(t.count for t in tasks)


def test_scheduler_spreads_across_all_sms():
    gpu = get_gpu("H100")
    k = GemmKernel(cfg("gemm", m=8192, n=8192, k=1024))
    dist = simulate(k.decompose(gpu), gpu.sms)
    assert dist.sm_occupancy == 1.0
    assert dist.imbalance("n_mma") < 1.2


def test_tiny_kernel_leaves_sms_idle():
    """A one-task kernel occupies one SM; the features must show that, not hide it."""
    gpu = get_gpu("H100")
    k = GemmKernel(cfg("gemm", m=64, n=64, k=64))
    dist = simulate(k.decompose(gpu), gpu.sms)
    assert dist.sm_occupancy < 0.05
    assert dist.imbalance("n_mma") > 10


def test_imbalance_is_one_when_tasks_divide_evenly():
    from kernelenergy.kernels.base import Task

    dist = simulate([Task(n_mma=1.0, count=264)], 132)
    assert dist.imbalance("n_mma") == pytest.approx(1.0)
    assert dist.tail_fraction == pytest.approx(0.0)


def test_ragged_tail_shows_up_as_imbalance():
    from kernelenergy.kernels.base import Task

    dist = simulate([Task(n_mma=1.0, count=133)], 132)
    assert dist.imbalance("n_mma") > 1.0


def test_bad_scheduler_name_is_rejected():
    from kernelenergy.kernels.base import Task

    with pytest.raises(ValueError, match="hardware.*software"):
        simulate([Task(n_mma=1.0, count=10)], 8, scheduler="magic")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gpu_key", sorted(GPUS))
@pytest.mark.parametrize(
    "config",
    [
        cfg("gemm", m=4096, n=3072, k=3072),
        cfg("attention", b=1, h=24, s_q=4608, s_kv=4608, d=128),
        cfg("conv", n=1, c_in=128, c_out=128, h=512, w=512, kh=3, kw=3),
        cfg("norm", rows=4608, dim=3072),
        cfg("elementwise", n_elem=4608 * 3072, kind="silu"),
    ],
    ids=["gemm", "attention", "conv", "norm", "elementwise"],
)
def test_features_are_finite_and_complete(gpu_key, config):
    res = analyse(make_kernel(config), GPUS[gpu_key])
    assert set(res.features) == set(FEATURE_COLUMNS)
    for name, v in res.features.items():
        assert v == v and abs(v) != float("inf"), f"{name} is not finite on {gpu_key}"
    assert res.theoretical_time_s > 0


def test_bottleneck_is_sensible_per_category():
    gpu = get_gpu("A100_PCIE")
    gemm = analyse(make_kernel(cfg("gemm", m=8192, n=8192, k=8192)), gpu)
    norm = analyse(make_kernel(cfg("norm", rows=65536, dim=4096)), gpu)
    assert gemm.bottleneck == "tensor", "a large square GEMM should be tensor-bound"
    assert norm.bottleneck == "mem_global", "a norm should be bandwidth-bound"


def test_theoretical_floor_ignores_estimated_bandwidths():
    """L2 and shared bandwidths are estimates and must not set the floor.

    If they did, an over-conservative L2 constant would produce eta > 1 and silently cap
    the sigmoid head.
    """
    from kernelenergy.model.features import MATH_PIPES

    gpu = get_gpu("H100")
    res = analyse(make_kernel(cfg("gemm", m=4096, n=3072, k=3072)), gpu)
    assert res.bottleneck in set(MATH_PIPES) | {"mem_global"}


def test_hopper_gemm_uses_persistent_scheduling():
    h = analyse(make_kernel(cfg("gemm", m=4096, n=4096, k=4096)), get_gpu("H100"))
    a = analyse(make_kernel(cfg("gemm", m=4096, n=4096, k=4096)), get_gpu("A100_PCIE"))
    assert h.features["is_persistent"] == 1.0
    assert a.features["is_persistent"] == 0.0


def test_power_features_separate_the_cards():
    """The energy extension must actually distinguish a 72 W card from a 700 W one."""
    c = cfg("gemm", m=4096, n=3072, k=3072)
    feats = {k: analyse(make_kernel(c), g).features for k, g in GPUS.items()}
    assert feats["L4"]["log_tdp"] < feats["H100"]["log_tdp"]

    # Every card must be distinguishable by the power-domain block *as a whole*. Not by
    # any single feature: the H100 and H200 SXM share a die, a clock and a 700 W budget,
    # so their watts-per-TFLOP is identical by construction and only bandwidth separates
    # them. That is the pair's value, not a defect -- but it does mean the invariant to
    # test is joint, not marginal.
    block = ("log_tdp", "log_idle_power", "idle_fraction", "log_watt_per_tflop",
             "log_watt_per_gbs", "is_hbm", "compute_capability")
    vecs = {k: tuple(round(f[c], 6) for c in block) for k, f in feats.items()}
    assert len(set(vecs.values())) == len(vecs), "two cards share a power descriptor"


def test_nominal_efficiency_is_not_monotone_in_tdp():
    """The L4 beats the H100 on paper W/TFLOP, and that is the point of the feature.

    72 W over 242 TFLOP/s is 0.30 W per TFLOP/s; 700 W over 1979 is 0.35. Yet the
    run-level data in this project has the L4 achieving the *lowest* utilisation of the
    fleet. Nominal efficiency and achieved efficiency point opposite ways here, which is
    precisely why the model needs both the spec-sheet ratio and the learned correction --
    a predictor built on either alone gets this card wrong.
    """
    l4, h100 = get_gpu("L4"), get_gpu("H100")
    l4_w_per_tflop = l4.tdp_w / (l4.peak_tensor_flops() / 1e12)
    h100_w_per_tflop = h100.tdp_w / (h100.peak_tensor_flops() / 1e12)
    assert l4_w_per_tflop < h100_w_per_tflop


# --------------------------------------------------------------------------- #
# Config identity
# --------------------------------------------------------------------------- #


def test_signature_survives_a_csv_round_trip(tmp_path):
    """The bug this guards against is silent and total.

    Params come back from pandas as NumPy scalars -- ``4608`` as ``numpy.int64``,
    ``False`` as ``numpy.bool_``, and any integer column with a NaN as ``4608.0``. If
    those hash differently from the Python originals, a saved catalogue reloads as a
    *different* set of kernels, every ``(kernel_sig, gpu_key)`` join misses, and the
    sweep re-measures everything it already has while the dataset build finds nothing.
    """
    import pandas as pd

    from kernelenergy.trace.catalogue import build_catalogue, load_catalogue, save_catalogue

    caps = {
        "flux1-dev": [
            cfg("gemm", m=4608, n=9216, k=3072, bias=True),
            cfg("attention", b=1, h=24, s_q=4608, s_kv=4608, d=128, causal=False),
            cfg("norm", rows=4608, dim=3072, kind="layer"),
            cfg("conv", n=1, c_in=128, c_out=128, h=512, w=512, kh=3, kw=3),
        ]
    }
    cat = build_catalogue(caps)
    path = tmp_path / "catalogue.csv"
    save_catalogue(cat, path)
    back = load_catalogue(path)
    assert {c.signature() for c in cat.configs()} == {c.signature() for c in back.configs()}


def test_numpy_scalars_hash_like_python_scalars():
    import numpy as np

    plain = cfg("gemm", m=4608, n=9216, k=3072, bias=True)
    numpyish = cfg("gemm", m=np.int64(4608), n=np.int32(9216), k=3072.0,
                   bias=np.bool_(True))
    assert plain.signature() == numpyish.signature()


def test_false_does_not_collapse_into_zero():
    """bool subclasses int, so a careless cast would make causal=False == causal=0."""
    a = cfg("attention", b=1, h=8, s_q=512, s_kv=512, d=64, causal=False)
    b = cfg("attention", b=1, h=8, s_q=512, s_kv=512, d=64, causal=True)
    assert a.signature() != b.signature()
    assert a.params["causal"] is False


def test_signature_ignores_provenance():
    """The same kernel captured from two pipelines is one measurement, not two."""
    from kernelenergy.kernels.base import KernelConfig

    a = KernelConfig("gemm", "bf16", {"m": 4608, "n": 9216, "k": 3072},
                     source_model="flux1-dev")
    b = KernelConfig("gemm", "bf16", {"m": 4608, "n": 9216, "k": 3072},
                     source_model="sd35-large")
    assert a.signature() == b.signature()


@pytest.mark.parametrize(
    "nvml_name,expected",
    [
        ("NVIDIA L4", "L4"),
        ("NVIDIA L40", "L40"),
        ("NVIDIA L40S", "L40S"),
        ("NVIDIA A100 80GB PCIe", "A100_PCIE"),
        ("NVIDIA A100-SXM4-80GB", "A100_SXM4"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H100 SXM5 80GB", "H100"),
        ("NVIDIA H200", "H200_SXM"),
        ("NVIDIA H200 NVL", "H200_NVL"),
    ],
)
def test_each_card_name_resolves_to_exactly_one_entry(nvml_name, expected):
    """Near-miss product names must not collide.

    Four pairs in this table differ by a suffix: L4/L40, L40/L40S, H200/H200 NVL,
    H100/H200. A pattern that swallows its neighbour would silently attribute every row
    from one card to another's descriptor -- wrong TDP, wrong bandwidth, wrong peak --
    and the sweep would run to completion looking fine.
    """
    import re

    hits = {
        key for key, gpu in GPUS.items()
        for pat in gpu.nvml_patterns if re.search(pat, nvml_name, re.IGNORECASE)
    }
    assert hits == {expected}, f"{nvml_name!r} matched {sorted(hits)}"


def test_l40_and_l40s_isolate_ops_per_clk():
    """The pair varies ops_per_clk alone, with SMs, clock and bandwidth held fixed.

    An earlier version of this test asserted the two cards had *equal* ops_per_clk,
    encoding the assumption that Ada means 1024 everywhere. It passed, because the table
    had been written from the same assumption -- which is how a 2x error in the L40's
    peak survived. The datasheet is the authority: BF16 sparse is 362 TFLOP/s for the
    L40 against 724 for the L40S, on the same die at the same clock.
    """
    a, b = get_gpu("L40"), get_gpu("L40S")
    assert a.sms == b.sms
    assert a.mem_bandwidth_gbs == b.mem_bandwidth_gbs
    assert abs(a.tensor_clock_mhz - b.tensor_clock_mhz) < 1e-9
    assert b.ops_per_clk == 2 * a.ops_per_clk
    assert b.peak_tensor_flops() == pytest.approx(2 * a.peak_tensor_flops())
    assert a.tdp_w < b.tdp_w


@pytest.mark.parametrize(
    "key,sparse_bf16_tflops",
    [
        ("L4", 242.0),
        ("L40", 362.0),
        ("L40S", 724.0),
        ("A100_PCIE", 624.0),
        ("A100_SXM4", 624.0),
        ("H100", 1979.0),
        ("H200_NVL", 1671.0),
        ("H200_SXM", 1979.0),
    ],
)
def test_peak_matches_the_published_bf16_figure(key, sparse_bf16_tflops):
    """Every row must reproduce its datasheet BF16-with-sparsity number to ~1%.

    This is the check that would have caught the L40 error immediately, and it is worth
    having because the failure mode is invisible downstream: a peak that is 2x too high
    just makes a card look half as efficient as it is, which is indistinguishable from a
    card that really is slow. Whatever convention the fleet uses -- this one is sparse
    BF16 throughout -- it has to be applied uniformly, and only a comparison against
    published figures can confirm that.
    """
    got = get_gpu(key).peak_tensor_flops("tensor") / 1e12
    assert got == pytest.approx(sparse_bf16_tflops, rel=0.01), (
        f"{key}: table gives {got:.1f} TFLOP/s, datasheet says {sparse_bf16_tflops}"
    )


def test_h100_and_h200_sxm_differ_only_in_bandwidth():
    """Same die, same power budget, 43% more bandwidth -- the one clean bandwidth
    contrast in the fleet, and only clean if nothing else moves."""
    a, b = get_gpu("H100"), get_gpu("H200_SXM")
    assert (a.sms, a.ops_per_clk, a.tdp_w) == (b.sms, b.ops_per_clk, b.tdp_w)
    assert a.tensor_clock_mhz == b.tensor_clock_mhz
    assert b.mem_bandwidth_gbs > a.mem_bandwidth_gbs * 1.4


def test_peak_reproduces_datasheet_from_tensor_clock():
    """peak = SMs x ops/clk x tensor clock, by construction (machine-descriptor sec 1)."""
    for gpu in GPUS.values():
        expected = gpu.sms * gpu.ops_per_clk * gpu.tensor_clock_mhz * 1e6 * 2
        assert gpu.peak_tensor_flops("tensor") == pytest.approx(expected)
        if gpu.boost_clock_mhz > gpu.tensor_clock_mhz:
            assert gpu.peak_tensor_flops("boost") > gpu.peak_tensor_flops("tensor")
