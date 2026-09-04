"""Tests for the HPC layer.

The device-resolution tests matter more than their size suggests. The failure they guard
against -- running a kernel on one card and reading the energy counter of another -- is
invisible in the output: it produces a full sweep of numbers that look like data. There
is no downstream check that would catch it, so it has to be caught here.

NVML is mocked, so these run anywhere.
"""

from __future__ import annotations

import os
import signal
import sys
import types

import pytest

from kernelenergy.hpc import device as dev_mod
from kernelenergy.hpc.device import (
    DeviceResolutionError,
    GpuContendedError,
    ResolvedDevice,
    assert_exclusive_gpu,
    assert_no_mig,
    resolve_device,
)
from kernelenergy.hpc.slurm import (
    InterruptGuard,
    estimate_cost_s,
    shard_configs,
    slurm_context,
)
from kernelenergy.kernels.base import KernelConfig


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

EIGHT_GPUS = [
    {
        "nvml_index": i,
        "uuid": f"GPU-0000000{i}-aaaa-bbbb-cccc-dddddddddddd",
        "pci_bus_id": f"00000000:{0x40 + i:02X}:00.0",
        "name": "NVIDIA H100 SXM5 80GB",
        "mig_enabled": False,
    }
    for i in range(8)
]


@pytest.fixture
def eight_gpu_node(monkeypatch):
    monkeypatch.setattr(dev_mod, "_nvml_inventory", lambda: list(EIGHT_GPUS))
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(var, raising=False)
    # No torch in the test environment; make sure that stays true for these tests so the
    # environment-variable fallbacks are the paths actually exercised.
    monkeypatch.setitem(sys.modules, "torch", None)
    yield


def _fake_torch(uuid: str = "", bus: str = ""):
    """A stand-in exposing just the properties resolve_device reads."""
    props = types.SimpleNamespace(uuid=uuid, pci_bus_id=bus)
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda i: props,
        )
    )


# --------------------------------------------------------------------------- #
# Device resolution -- the important ones
# --------------------------------------------------------------------------- #


def test_cuda_zero_is_not_nvml_zero_under_slurm(eight_gpu_node, monkeypatch):
    """The whole reason this module exists.

    Granted GPU 5 of an eight-GPU node, SLURM sets CUDA_VISIBLE_DEVICES=5. Torch calls it
    cuda:0. NVML is not filtered and still calls it 5. Reading nvml index 0 would sample
    an idle neighbour for the entire sweep, and nothing downstream would notice.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    d = resolve_device(0)
    assert d.cuda_index == 0
    assert d.nvml_index == 5


def test_uuid_beats_index_arithmetic(eight_gpu_node, monkeypatch):
    """With a torch that reports UUIDs, the env vars are not consulted at all."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(uuid=EIGHT_GPUS[6]["uuid"]))
    d = resolve_device(0)
    assert d.nvml_index == 6
    assert d.method == "torch-uuid"


def test_pci_bus_id_is_matched_across_padding_conventions(eight_gpu_node, monkeypatch):
    """NVML says 00000000:43:00.0 and CUDA says 0000:43:00.0 for the same slot."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(bus="0000:43:00.0"))
    assert resolve_device(0).nvml_index == 3


def test_uuid_matching_tolerates_the_gpu_prefix_and_case(eight_gpu_node, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    bare = EIGHT_GPUS[2]["uuid"].removeprefix("GPU-").upper()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(uuid=bare))
    assert resolve_device(0).nvml_index == 2


def test_uuid_form_of_cuda_visible_devices(eight_gpu_node, monkeypatch):
    """Some sites set CUDA_VISIBLE_DEVICES to UUIDs rather than indices."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", EIGHT_GPUS[4]["uuid"])
    d = resolve_device(0)
    assert d.nvml_index == 4
    assert d.method == "cvd-uuid"


def test_index_fallback_is_refused_without_pci_bus_order(eight_gpu_node, monkeypatch):
    """FASTEST_FIRST can reorder CUDA devices relative to PCI, so indices are not safe.

    Refusing is the right behaviour: the alternative is a plausible wrong answer.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    with pytest.raises(DeviceResolutionError, match="CUDA_DEVICE_ORDER"):
        resolve_device(0)


def test_ambiguous_multi_gpu_node_is_refused(eight_gpu_node):
    """No CUDA_VISIBLE_DEVICES, no torch UUID, eight cards: guessing would be reckless."""
    with pytest.raises(DeviceResolutionError, match="cannot be mapped"):
        resolve_device(0)


def test_single_gpu_node_needs_no_disambiguation(monkeypatch):
    monkeypatch.setattr(dev_mod, "_nvml_inventory", lambda: [EIGHT_GPUS[0]])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device(0).nvml_index == 0


def test_index_beyond_the_allocation_is_an_error(eight_gpu_node, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    with pytest.raises(DeviceResolutionError, match="lists only 1"):
        resolve_device(1)


def test_no_gpus_at_all(monkeypatch):
    monkeypatch.setattr(dev_mod, "_nvml_inventory", list)
    with pytest.raises(DeviceResolutionError, match="no GPUs"):
        resolve_device(0)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_mig_is_refused():
    """Board-level energy cannot be attributed to a MIG instance, and unlike the
    co-tenancy case there is no process list that would reveal it afterwards."""
    d = ResolvedDevice(0, 0, "u", "b", "NVIDIA A100", "test", mig_enabled=True)
    with pytest.raises(GpuContendedError, match="MIG"):
        assert_no_mig(d)
    assert_no_mig(ResolvedDevice(0, 0, "u", "b", "n", "test", mig_enabled=False))


def test_foreign_process_blocks_measurement(monkeypatch):
    monkeypatch.setattr(dev_mod, "gpu_processes",
                        lambda i: [{"pid": 999999, "used_memory": 1 << 30}])
    d = ResolvedDevice(0, 0, "u", "b", "NVIDIA H100", "test")
    with pytest.raises(GpuContendedError, match="999999"):
        assert_exclusive_gpu(d)


def test_our_own_process_does_not_count_as_contention(monkeypatch):
    monkeypatch.setattr(dev_mod, "gpu_processes",
                        lambda i: [{"pid": os.getpid(), "used_memory": 1 << 30}])
    d = ResolvedDevice(0, 0, "u", "b", "NVIDIA H100", "test")
    assert assert_exclusive_gpu(d) == []


def test_allow_shared_reports_rather_than_raises(monkeypatch):
    monkeypatch.setattr(dev_mod, "gpu_processes", lambda i: [{"pid": 4242}])
    d = ResolvedDevice(0, 0, "u", "b", "NVIDIA H100", "test")
    assert assert_exclusive_gpu(d, allow_shared=True) == [4242]


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #


def _configs(n: int) -> list[KernelConfig]:
    return [
        KernelConfig("gemm", "bf16", {"m": 256 * (i + 1), "n": 4096, "k": 4096})
        for i in range(n)
    ]


@pytest.mark.parametrize("n_tasks", [1, 2, 3, 4, 7, 16])
def test_shards_partition_exactly(n_tasks):
    cfgs = _configs(50)
    shards = [shard_configs(cfgs, i, n_tasks) for i in range(n_tasks)]
    sigs = [{c.signature() for c in s} for s in shards]
    assert sum(len(s) for s in shards) == len(cfgs), "work lost or duplicated"
    assert set().union(*sigs) == {c.signature() for c in cfgs}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            assert not (sigs[i] & sigs[j]), "a config landed in two shards"


def test_sharding_is_deterministic_and_order_independent():
    """A requeued task must reconstruct its own shard exactly.

    If it did not, it would re-measure work another task already did and skip work
    nobody did -- and the second half of that is silent.
    """
    cfgs = _configs(40)
    a = [c.signature() for c in shard_configs(cfgs, 2, 5)]
    b = [c.signature() for c in shard_configs(cfgs, 2, 5)]
    shuffled = list(reversed(cfgs))
    c = [x.signature() for x in shard_configs(shuffled, 2, 5)]
    assert a == b
    assert sorted(a) == sorted(c), "shard membership depends on input order"


def test_shards_are_cost_balanced():
    from kernelenergy.measure.replay import ReplayConfig

    rc = ReplayConfig()
    cfgs = _configs(60)
    loads = [
        sum(estimate_cost_s(c, rc) for c in shard_configs(cfgs, i, 6))
        for i in range(6)
    ]
    assert max(loads) / min(loads) < 1.15, f"unbalanced shards: {loads}"


def test_shard_bounds_are_checked():
    with pytest.raises(ValueError, match="outside"):
        shard_configs(_configs(10), 4, 4)


# --------------------------------------------------------------------------- #
# SLURM context
# --------------------------------------------------------------------------- #


def test_context_is_inert_outside_slurm(monkeypatch):
    for v in ("SLURM_JOB_ID", "SLURM_JOBID"):
        monkeypatch.delenv(v, raising=False)
    ctx = slurm_context()
    assert not ctx.in_slurm
    assert ctx.task_count == 1
    assert ctx.seconds_remaining() is None


def test_one_based_arrays_are_normalised(monkeypatch):
    """`--array=1-4` and `--array=0-3` must both give shard ids 0..3."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", "4")
    monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    ctx = slurm_context()
    assert (ctx.task_id, ctx.task_count) == (0, 4)


def test_walltime_is_read_from_the_environment(monkeypatch):
    import time

    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    rem = slurm_context().seconds_remaining()
    assert rem is not None and 3500 < rem <= 3600


# --------------------------------------------------------------------------- #
# Interruption
# --------------------------------------------------------------------------- #


def test_guard_records_the_signal_without_dying():
    with InterruptGuard(signals=(signal.SIGUSR1,)) as g:
        assert not g.interrupted
        os.kill(os.getpid(), signal.SIGUSR1)
        assert g.interrupted
        assert g.triggered_by == "SIGUSR1"


def test_guard_keeps_the_first_signal_and_restores_handlers():
    original = signal.getsignal(signal.SIGUSR1)
    with InterruptGuard(signals=(signal.SIGUSR1,)) as g:
        os.kill(os.getpid(), signal.SIGUSR1)
        os.kill(os.getpid(), signal.SIGUSR1)
        assert g.triggered_by == "SIGUSR1"
    assert signal.getsignal(signal.SIGUSR1) is original
