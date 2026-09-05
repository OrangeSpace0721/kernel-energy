"""Preflight: everything that can invalidate a run, checked before the run.

Worth its own module because of the economics. A measurement sweep holds an exclusive
node for hours. The failures that matter here are not crashes -- a crash is cheap, you
find out immediately -- but the silent ones: the wrong NVML device, a co-tenant process,
a card capped below its datasheet TDP, a missing model weight that only surfaces at the
third resolution. Each of those produces a full run of plausible-looking numbers that are
wrong, and you discover it days later when the folds do not make sense.

So this runs first, takes about a minute, and is cheap to submit as its own tiny job.

Severities:

* **fail**    -- data collected now would be invalid. Stop.
* **warn**    -- collectable, but something is not as assumed and belongs in the notes.
* **ok**      -- checked and fine.
* **skip**    -- not applicable to this stage.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Check", "run_preflight", "format_report"]


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def _ok(n, d="") -> Check:
    return Check(n, "ok", d)


def _warn(n, d) -> Check:
    return Check(n, "warn", d)


def _fail(n, d) -> Check:
    return Check(n, "fail", d)


def _skip(n, d="") -> Check:
    return Check(n, "skip", d)


# --------------------------------------------------------------------------- #


def run_preflight(
    cuda_index: int = 0,
    stage: str = "measure",
    catalogue: str | Path | None = None,
    out_dir: str | Path | None = None,
    allow_shared: bool = False,
    models: tuple[str, ...] = (),
) -> list[Check]:
    """Run every applicable check. ``stage`` is ``capture``, ``measure`` or ``all``."""
    checks: list[Check] = []
    want_measure = stage in ("measure", "all")
    want_capture = stage in ("capture", "all")

    # -- scheduler context --------------------------------------------------- #
    from kernelenergy.hpc.slurm import slurm_context

    ctx = slurm_context()
    checks.append(_ok("slurm.context", ctx.describe()))
    if ctx.in_slurm and ctx.end_time is None:
        checks.append(_warn(
            "slurm.walltime",
            "walltime expiry unknown (no SLURM_JOB_END_TIME and scontrol unavailable); "
            "the sweep cannot stop before the wall and may lose the kernel in flight. "
            "Pass --time-budget explicitly.",
        ))

    # -- NVML and device resolution ------------------------------------------ #
    try:
        from kernelenergy.hpc.device import (
            assert_exclusive_gpu,
            assert_no_mig,
            power_limits,
            resolve_device,
        )
        from kernelenergy.nvml import supports_energy_counter
    except Exception as e:
        return checks + [_fail("nvml.import", f"{type(e).__name__}: {e}")]

    try:
        dev = resolve_device(cuda_index, strict=True)
    except Exception as e:
        return checks + [_fail("device.resolve", str(e))]

    checks.append(Check(
        "device.resolve",
        "ok" if dev.method.startswith(("torch-", "cvd-uuid")) else "warn",
        f"{dev}"
        + ("" if dev.method.startswith(("torch-", "cvd-uuid"))
           else "  -- mapping inferred from indices rather than UUID; verify the card "
                "name matches what you requested"),
    ))

    try:
        assert_no_mig(dev)
        checks.append(_ok("device.mig", "MIG disabled"))
    except Exception as e:
        checks.append(_fail("device.mig", str(e)))

    if want_measure:
        try:
            foreign = assert_exclusive_gpu(dev, allow_shared=allow_shared)
            checks.append(
                _ok("device.exclusive", "no other compute processes on this GPU")
                if not foreign else
                _warn("device.exclusive",
                      f"foreign PIDs {foreign} present and --allow-shared-gpu is set; "
                      "board energy includes their work")
            )
        except Exception as e:
            checks.append(_fail("device.exclusive", str(e)))
    else:
        checks.append(_skip("device.exclusive", "capture does not measure energy"))

    # -- the instrument ------------------------------------------------------ #
    if supports_energy_counter(dev.nvml_index):
        checks.append(_ok("nvml.energy_counter",
                          "nvmlDeviceGetTotalEnergyConsumption available"))
    elif want_measure:
        checks.append(_warn(
            "nvml.energy_counter",
            "energy counter unavailable; falling back to integrating polled power, "
            "which is materially noisier. Acceptable but record it in --notes.",
        ))
    else:
        checks.append(_skip("nvml.energy_counter"))

    # -- power limits -------------------------------------------------------- #
    try:
        lim = power_limits(dev.nvml_index)
        enforced, default = lim["enforced_w"], lim["default_w"]
        if enforced == enforced and default == default and abs(enforced - default) > 1.0:
            checks.append(_warn(
                "device.power_limit",
                f"enforced limit {enforced:.0f} W differs from the card default "
                f"{default:.0f} W. The card cannot reach its datasheet TDP, so pi = "
                f"P_avg/TDP is compressed on this site. Every row records "
                f"power_limit_w; fit pi against the enforced limit "
                f"(add_targets(..., tdp_col='power_limit_w')).",
            ))
        else:
            checks.append(_ok("device.power_limit", f"enforced {enforced:.0f} W"))
    except Exception as e:
        checks.append(_warn("device.power_limit", f"could not read: {e}"))

    # -- the hardware table -------------------------------------------------- #
    try:
        from kernelenergy.hardware import probe_local_gpu

        gpu = probe_local_gpu(dev.nvml_index)
        detail = f"{dev.name} -> {gpu.gpu_key} (TDP {gpu.tdp_w:.0f} W)"
        if gpu.idle_power_w <= 0:
            checks.append(_warn("hardware.table", detail + "; idle power not measured"))
        else:
            checks.append(_ok("hardware.table", detail))
    except Exception as e:
        checks.append(_fail("hardware.table", str(e)))

    # -- torch --------------------------------------------------------------- #
    try:
        import torch

        if not torch.cuda.is_available():
            checks.append(_fail("torch.cuda", "torch.cuda.is_available() is False"))
        else:
            free, total = torch.cuda.mem_get_info(cuda_index)
            x = torch.zeros(1024, 1024, device=f"cuda:{cuda_index}")
            del x
            torch.cuda.empty_cache()
            checks.append(_ok(
                "torch.cuda",
                f"torch {torch.__version__} / CUDA {torch.version.cuda}, "
                f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free",
            ))
    except Exception as e:
        checks.append(_fail("torch.cuda", f"{type(e).__name__}: {e}"))

    # -- offline model cache (capture only) ---------------------------------- #
    if want_capture:
        checks.extend(_check_diffusers(models))
        checks.extend(_check_hf_cache(models))
    else:
        checks.append(_skip("hf.cache", "measure does not load the pipelines"))

    # -- filesystem ---------------------------------------------------------- #
    if catalogue is not None:
        p = Path(catalogue)
        checks.append(
            _ok("io.catalogue", f"{p} ({p.stat().st_size / 1024:.0f} KiB)")
            if p.exists() else
            _fail("io.catalogue", f"{p} does not exist -- run the capture stage first")
        )
    if out_dir is not None:
        p = Path(out_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / f".preflight_{os.getpid()}"
            probe.write_text("ok")
            probe.unlink()
            usage = shutil.disk_usage(p)
            checks.append(_ok(
                "io.out_dir", f"{p} writable, {usage.free / 2**30:.0f} GiB free"
            ))
        except Exception as e:
            checks.append(_fail("io.out_dir", f"{p}: {e}"))

    return checks


def _check_diffusers(models: tuple[str, ...]) -> list[Check]:
    """Is the pipeline class this model needs actually in the installed diffusers?

    Cheap to check and expensive to discover late. Qwen-Image needs diffusers >= 0.35;
    on an older build the capture job loads, allocates, and then dies on an AttributeError
    after however long the queue took. Worse in an array job, where the other two tasks
    succeed and the failure is one line in one log.
    """
    out: list[Check] = []
    try:
        import diffusers
    except ImportError as e:
        return [_fail("diffusers.import", f"{e}. Install with the [gpu] extra.")]

    from kernelenergy.trace.pipelines import PIPELINES

    out.append(_ok("diffusers.version", diffusers.__version__))

    for key in (models or tuple(PIPELINES)):
        spec = PIPELINES.get(key)
        if spec is None:
            out.append(_fail(f"diffusers.{key}", "unknown pipeline key"))
            continue
        if getattr(diffusers, spec.loader, None) is None:
            out.append(_fail(
                f"diffusers.{key}",
                f"{spec.loader} is not in diffusers {diffusers.__version__}. "
                f"Upgrade (pip install -U diffusers), or drop {key} from this run.",
            ))
        else:
            out.append(_ok(f"diffusers.{key}", spec.loader))
    return out


def _check_hf_cache(models: tuple[str, ...]) -> list[Check]:
    """Are the weights on disk, and is the offline switch consistent with that?

    Compute nodes without internet fail deep inside ``from_pretrained`` with a connection
    error that names a URL rather than a missing file, minutes into a job. Checking the
    cache up front turns that into one line.
    """
    from kernelenergy.trace.pipelines import PIPELINES, env_var_for, resolve_source

    out: list[Check] = []
    home = (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or ""
    )
    offline = os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True")

    wanted = models or tuple(PIPELINES)
    # A pipeline pointed at a local directory does not need the hub cache at all, so
    # only complain about HF_HOME if something still resolves through it.
    overridden = {k for k in wanted if os.environ.get(env_var_for(k), "").strip()}
    needs_hub = set(wanted) - overridden

    if not home and needs_hub:
        out.append(_warn(
            "hf.home",
            "HF_HOME is unset, so weights resolve to ~/.cache/huggingface -- usually a "
            "small home quota on HPC. Point it at shared scratch, or set "
            "KE_MODEL_<KEY> to a pipeline directory you already have.",
        ))
    elif home:
        out.append(_ok("hf.home", home))
    else:
        out.append(_skip("hf.home", "every pipeline is pointed at a local directory"))

    root = Path(home or Path.home() / ".cache" / "huggingface")
    hub = root / "hub" if (root / "hub").exists() else root

    missing, found = [], []
    for key in wanted:
        spec = PIPELINES.get(key)
        if spec is None:
            missing.append(f"{key} (unknown pipeline)")
            continue
        if key in overridden:
            # Validate the override itself -- resolve_source raises with a specific
            # message (wrong path, or a hub cache handed to the wrong variable).
            try:
                src, _how = resolve_source(key)
                found.append(f"{key} <- {src}")
            except Exception as e:
                missing.append(f"{key}: {e}")
            continue
        stub = "models--" + spec.repo.replace("/", "--")
        if (hub / stub).exists():
            found.append(f"{key} <- {hub / stub}")
        else:
            missing.append(f"{key} -> {spec.repo} (not in {hub})")

    if missing:
        out.append(_fail(
            "hf.cache",
            "cannot resolve: " + "; ".join(missing)
            + ".  Either run slurm/00_setup.sh on a login node, point HF_HOME at an "
            "existing hub cache, or set KE_MODEL_<KEY> to a pipeline directory.",
        ))
    else:
        detail = f"all {len(wanted)} pipeline(s) resolvable"
        if overridden:
            detail += f" ({len(overridden)} via KE_MODEL_* override)"
        out.append(_ok("hf.cache", detail))
        for line in found:
            out.append(_ok("hf.cache.path", line))

    out.append(
        _ok("hf.offline", "HF_HUB_OFFLINE=1")
        if offline else
        _warn("hf.offline",
              "HF_HUB_OFFLINE is not set. On a node without internet the hub client "
              "stalls on connection timeouts before falling back to the cache; setting "
              "it makes a missing weight fail immediately and clearly instead.")
    )
    return out


# --------------------------------------------------------------------------- #


_GLYPH = {"ok": "  ok  ", " warn ": " warn ", "warn": " WARN ", "fail": " FAIL ",
          "skip": " skip "}


def format_report(checks: list[Check]) -> str:
    width = max((len(c.name) for c in checks), default=10)
    lines = []
    for c in checks:
        detail = c.detail.replace("\n", " ")
        lines.append(f"[{_GLYPH.get(c.status, c.status):^6}] {c.name:<{width}}  {detail}")
    n_fail = sum(c.failed for c in checks)
    n_warn = sum(c.status == "warn" for c in checks)
    lines.append("")
    lines.append(
        f"{len(checks)} checks: {n_fail} fail, {n_warn} warn, "
        f"{sum(c.status == 'ok' for c in checks)} ok, "
        f"{sum(c.status == 'skip' for c in checks)} skipped"
    )
    if n_fail:
        lines.append("PREFLIGHT FAILED -- do not submit the sweep until these are fixed.")
    return "\n".join(lines)
