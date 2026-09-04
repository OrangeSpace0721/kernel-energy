"""Resolving which physical GPU you are actually measuring.

This module exists because of one bug, and it is the most dangerous bug in the whole
repo: **CUDA device indices and NVML device indices are not the same thing.**

On a workstation with one GPU they coincide and nothing goes wrong. Under a scheduler
they routinely do not. When SLURM grants you GPU 2 of an eight-GPU node it sets
``CUDA_VISIBLE_DEVICES=2``, so PyTorch's ``cuda:0`` is physical GPU 2 -- while NVML,
which is not filtered by that variable, still enumerates all eight and hands you physical
GPU 0 for index 0. The kernel runs on one card and the energy counter is read from
another. Nothing errors. Every row is wrong, and wrong in a way that looks like plausible
data: you get the *idle* energy of an unrelated card.

``CUDA_DEVICE_ORDER`` makes it worse. Its default is ``FASTEST_FIRST``, which orders CUDA
devices by capability rather than by PCI slot, so even the naive "index into
CUDA_VISIBLE_DEVICES" mapping can be wrong on a heterogeneous node.

The fix is to match on UUID, which is what the hardware calls itself and is invariant
under every reordering. The fallbacks below exist for older PyTorch builds, and the last
of them refuses rather than guesses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "ResolvedDevice",
    "resolve_device",
    "gpu_processes",
    "assert_exclusive_gpu",
    "assert_no_mig",
    "power_limits",
    "DeviceResolutionError",
    "GpuContendedError",
]


class DeviceResolutionError(RuntimeError):
    """Could not establish which NVML device corresponds to a CUDA device."""


class GpuContendedError(RuntimeError):
    """Another process is using the GPU, so its energy counter is not ours alone."""


@dataclass(frozen=True)
class ResolvedDevice:
    cuda_index: int  # what torch calls it
    nvml_index: int  # what NVML calls it
    uuid: str
    pci_bus_id: str
    name: str
    method: str  # how the mapping was established
    mig_enabled: bool = False

    def __str__(self) -> str:
        return (
            f"cuda:{self.cuda_index} -> nvml:{self.nvml_index} "
            f"({self.name}, {self.uuid}, via {self.method})"
        )


# --------------------------------------------------------------------------- #
# NVML side
# --------------------------------------------------------------------------- #


def _nvml_inventory() -> list[dict]:
    """Every GPU NVML can see, with the identifiers we might match on."""
    from kernelenergy.nvml import nvml_session

    out = []
    with nvml_session() as nv:
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)

            def _s(fn, *a):
                try:
                    v = fn(h, *a)
                    return v.decode() if isinstance(v, bytes) else str(v)
                except Exception:
                    return ""

            mig = False
            try:
                current, _pending = nv.nvmlDeviceGetMigMode(h)
                mig = bool(current)
            except Exception:
                pass

            out.append(
                {
                    "nvml_index": i,
                    "uuid": _s(nv.nvmlDeviceGetUUID),
                    "pci_bus_id": _s(lambda hh: nv.nvmlDeviceGetPciInfo(hh).busId),
                    "name": _s(nv.nvmlDeviceGetName),
                    "mig_enabled": mig,
                }
            )
    return out


def _norm_uuid(u: str) -> str:
    """``GPU-abc...``, ``abc...`` and mixed case all name the same card."""
    return re.sub(r"^GPU-", "", (u or "").strip(), flags=re.IGNORECASE).lower()


def _norm_bus(b: str) -> str:
    """NVML gives ``00000000:41:00.0``; CUDA gives ``0000:41:00.0``."""
    b = (b or "").strip().lower().rstrip("\x00")
    parts = b.split(":")
    if len(parts) == 3:
        parts[0] = parts[0].lstrip("0").rjust(4, "0") or "0000"
        return ":".join(parts)
    return b


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve_device(cuda_index: int = 0, strict: bool = True) -> ResolvedDevice:
    """Find the NVML index for a CUDA device, by UUID wherever possible.

    Strategies, in order of trustworthiness:

    1. **torch UUID.** ``torch.cuda.get_device_properties(i).uuid`` against NVML's UUID.
       Invariant under ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER`` both.
    2. **torch PCI bus id.** Same idea where the torch build exposes the bus id but not
       the UUID.
    3. **``CUDA_VISIBLE_DEVICES`` as UUIDs.** SLURM sets this form on some sites.
    4. **``CUDA_VISIBLE_DEVICES`` as indices.** Correct only when
       ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, which the job scripts set; refused otherwise
       under ``strict``.
    5. **Identity.** Only when ``CUDA_VISIBLE_DEVICES`` is unset and the node has one
       GPU.

    Raises rather than guessing. A wrong answer here is undetectable downstream.
    """
    inv = _nvml_inventory()
    if not inv:
        raise DeviceResolutionError("NVML reports no GPUs on this node")

    by_uuid = {_norm_uuid(d["uuid"]): d for d in inv}
    by_bus = {_norm_bus(d["pci_bus_id"]): d for d in inv}

    # -- 1 & 2: ask torch what card it is holding --------------------------- #
    torch_uuid = torch_bus = ""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(cuda_index)
            torch_uuid = _norm_uuid(str(getattr(props, "uuid", "")))
            torch_bus = _norm_bus(str(getattr(props, "pci_bus_id", "")))
            if not torch_bus:
                # Older builds expose it only through the runtime API.
                try:
                    torch_bus = _norm_bus(
                        torch.cuda.get_device_properties(cuda_index).pci_bus_id
                    )
                except Exception:
                    pass
    except Exception:
        pass

    if torch_uuid and torch_uuid in by_uuid:
        return _make(by_uuid[torch_uuid], cuda_index, "torch-uuid")
    if torch_bus and torch_bus in by_bus:
        return _make(by_bus[torch_bus], cuda_index, "torch-pci-bus-id")

    # -- 3 & 4: fall back to the environment variable ------------------------ #
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        entries = [e.strip() for e in cvd.split(",") if e.strip()]
        if cuda_index >= len(entries):
            raise DeviceResolutionError(
                f"cuda:{cuda_index} requested but CUDA_VISIBLE_DEVICES lists only "
                f"{len(entries)} device(s): {cvd!r}"
            )
        entry = entries[cuda_index]
        if _norm_uuid(entry) in by_uuid:
            return _make(by_uuid[_norm_uuid(entry)], cuda_index, "cvd-uuid")
        if entry.isdigit():
            order = os.environ.get("CUDA_DEVICE_ORDER", "").upper()
            if strict and order != "PCI_BUS_ID":
                raise DeviceResolutionError(
                    "CUDA_VISIBLE_DEVICES is a numeric list and CUDA_DEVICE_ORDER is "
                    f"{order or 'unset (defaults to FASTEST_FIRST)'}, under which CUDA "
                    "indices need not follow PCI order -- so the mapping to NVML is not "
                    "determinable. Export CUDA_DEVICE_ORDER=PCI_BUS_ID before the job "
                    "(the provided sbatch scripts do), or use a PyTorch build that "
                    "exposes device UUIDs."
                )
            idx = int(entry)
            match = next((d for d in inv if d["nvml_index"] == idx), None)
            if match is None:
                raise DeviceResolutionError(
                    f"CUDA_VISIBLE_DEVICES names GPU {idx}, which NVML does not see"
                )
            return _make(match, cuda_index, "cvd-index")
        raise DeviceResolutionError(
            f"cannot interpret CUDA_VISIBLE_DEVICES entry {entry!r}"
        )

    # -- 5: identity, and only when it is unambiguous ------------------------ #
    if len(inv) == 1:
        return _make(inv[0], cuda_index, "single-gpu-node")
    if strict:
        raise DeviceResolutionError(
            f"node has {len(inv)} GPUs, CUDA_VISIBLE_DEVICES is unset, and this "
            "PyTorch build does not expose device UUIDs, so cuda:"
            f"{cuda_index} cannot be mapped to an NVML index. Set CUDA_VISIBLE_DEVICES."
        )
    match = next((d for d in inv if d["nvml_index"] == cuda_index), inv[0])
    return _make(match, cuda_index, "identity-unchecked")


def _make(d: dict, cuda_index: int, method: str) -> ResolvedDevice:
    return ResolvedDevice(
        cuda_index=cuda_index,
        nvml_index=d["nvml_index"],
        uuid=d["uuid"],
        pci_bus_id=d["pci_bus_id"],
        name=d["name"],
        method=method,
        mig_enabled=d["mig_enabled"],
    )


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def assert_no_mig(dev: ResolvedDevice) -> None:
    """MIG makes board energy meaningless as a per-instance measurement.

    ``nvmlDeviceGetTotalEnergyConsumption`` reports for the whole board. With MIG
    enabled the board is shared between instances, so the counter includes work from
    every other instance -- and unlike the co-tenancy case there is no process list that
    would let you detect it after the fact.
    """
    if dev.mig_enabled:
        raise GpuContendedError(
            f"{dev.name} has MIG enabled. The NVML energy counter is board-level, so "
            "per-instance energy cannot be measured. Request a non-MIG GPU partition."
        )


def gpu_processes(nvml_index: int) -> list[dict]:
    """Compute processes NVML reports on this GPU."""
    from kernelenergy.nvml import nvml_session

    out = []
    with nvml_session() as nv:
        h = nv.nvmlDeviceGetHandleByIndex(nvml_index)
        for fn in ("nvmlDeviceGetComputeRunningProcesses_v3",
                   "nvmlDeviceGetComputeRunningProcesses_v2",
                   "nvmlDeviceGetComputeRunningProcesses"):
            try:
                procs = getattr(nv, fn)(h)
            except Exception:
                continue
            for p in procs:
                out.append({
                    "pid": int(p.pid),
                    "used_memory": getattr(p, "usedGpuMemory", None),
                })
            break
    return out


def assert_exclusive_gpu(dev: ResolvedDevice, allow_shared: bool = False) -> list[int]:
    """Refuse to measure when another process holds the GPU.

    On a shared node this is the difference between data and noise. The energy counter
    is board-level, so a co-tenant's kernels are added to yours with no way to separate
    them afterwards -- and the contamination is not random, it is a positive bias that
    varies with whatever the neighbour happens to be doing.

    Returns the list of foreign PIDs (empty on success, or populated when
    ``allow_shared`` lets the run proceed anyway).
    """
    ours = {os.getpid()}
    try:  # our own children, if the harness ever forks
        ours |= set(map(int, os.listdir(f"/proc/{os.getpid()}/task")))
    except Exception:
        pass

    foreign = [p["pid"] for p in gpu_processes(dev.nvml_index) if p["pid"] not in ours]
    if foreign and not allow_shared:
        raise GpuContendedError(
            f"{dev.name} (nvml:{dev.nvml_index}) already has compute process(es) "
            f"{foreign}. Board energy would include their work. Request an exclusive "
            "allocation, or pass --allow-shared-gpu to record anyway (every affected "
            "row is stamped contended=1 and should not be trusted)."
        )
    return foreign


def power_limits(nvml_index: int) -> dict:
    """Enforced, default and hardware power limits.

    HPC sites routinely cap cards below their datasheet TDP for facility power or
    cooling reasons. That matters here more than it would for a latency study: the
    power-fraction target is ``P_avg / TDP``, and if the card physically cannot reach
    TDP then the denominator is wrong and every ``pi`` on that site is compressed by the
    same unknown factor. Recording the enforced limit lets the target be computed
    against the limit that actually binds.
    """
    from kernelenergy.nvml import nvml_session

    with nvml_session() as nv:
        h = nv.nvmlDeviceGetHandleByIndex(nvml_index)

        def _try(fn, *a):
            try:
                return float(fn(h, *a)) / 1000.0
            except Exception:
                return float("nan")

        limits = {
            "enforced_w": _try(nv.nvmlDeviceGetEnforcedPowerLimit),
            "management_w": _try(nv.nvmlDeviceGetPowerManagementLimit),
            "default_w": _try(nv.nvmlDeviceGetPowerManagementDefaultLimit),
        }
        try:
            lo, hi = nv.nvmlDeviceGetPowerManagementLimitConstraints(h)
            limits["min_w"] = lo / 1000.0
            limits["max_w"] = hi / 1000.0
        except Exception:
            limits["min_w"] = limits["max_w"] = float("nan")
    return limits
