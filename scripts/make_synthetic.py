"""Generate a synthetic measurement set with the schema of a real one.

This exists so the modelling half can be exercised, tested and demoed before a single
GPU-hour is spent -- and so a change to the estimator or the folds can be regression
tested in CI where there is no hardware.

It is **not** a simulator and its numbers are not predictions. The generative law is
deliberately simple and known: efficiency and power fraction are smooth functions of a
few analytical features plus lognormal noise. A model that fits this well has
demonstrated that the plumbing works, and nothing more. Treat any MAPE produced from it
as a test fixture, never as a result.

    python scripts/make_synthetic.py --out data/synthetic_raw.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from kernelenergy.hardware import GPUS
from kernelenergy.kernels.base import KernelConfig
from kernelenergy.kernels.registry import make_kernel
from kernelenergy.measure.schema import MeasurementRow
from kernelenergy.model.features import analyse

MODELS = ["flux1-dev", "sd35-large", "qwen-image"]

# Hidden widths and head counts roughly matching the three transformers.
_MODEL_DIMS = {
    "flux1-dev": (3072, 24, 128),
    "sd35-large": (2432, 38, 64),
    "qwen-image": (3072, 24, 128),
}
_TOKENS = [1024, 2304, 4096, 4608, 9216]


def _configs() -> list[KernelConfig]:
    out: list[KernelConfig] = []
    for model in MODELS:
        dim, heads, hd = _MODEL_DIMS[model]
        for tok in _TOKENS:
            for mult, name in ((3, "qkv"), (1, "out"), (4, "fc1")):
                out.append(KernelConfig("gemm", "bf16",
                                        {"m": tok, "n": dim * mult, "k": dim},
                                        source_model=model, source_module=name))
            out.append(KernelConfig("gemm", "bf16",
                                    {"m": tok, "n": dim, "k": dim * 4},
                                    source_model=model, source_module="fc2"))
            out.append(KernelConfig("attention", "bf16",
                                    {"b": 1, "h": heads, "s_q": tok, "s_kv": tok,
                                     "d": hd},
                                    source_model=model, source_module="attn"))
            out.append(KernelConfig("norm", "bf16",
                                    {"rows": tok, "dim": dim, "kind": "layer"},
                                    source_model=model, source_module="norm"))
            for kind in ("silu", "scale_shift", "add"):
                out.append(KernelConfig("elementwise", "bf16",
                                        {"n_elem": tok * dim, "kind": kind},
                                        source_model=model, source_module=kind))
        for res, ch in ((256, 256), (512, 128), (1024, 64)):
            out.append(KernelConfig("conv", "bf16",
                                    {"n": 1, "c_in": ch, "c_out": ch, "h": res,
                                     "w": res, "kh": 3, "kw": 3},
                                    source_model=model, source_module="vae"))
    return out


def _true_eta(f: dict, rng) -> float:
    """A plausible efficiency law: better with more waves, worse when imbalanced."""
    z = (
        -0.55
        + 0.16 * np.tanh(f["log_n_waves"] / 3.0)
        + 0.30 * f["sm_occupancy"]
        - 0.22 * np.tanh(f["imbalance_mma"] - 1.0)
        - 0.08 * np.tanh(f["log_bytes_per_flop"] / 5.0)
        + 0.10 * f["is_persistent"]
    )
    return float(np.clip(1 / (1 + np.exp(-z)) * np.exp(rng.normal(0, 0.06)), 1e-3, 0.999))


def _true_pi(f: dict, rng) -> float:
    """Power fraction: high when the tensor pipe is busy, low when stalled on memory."""
    z = (
        0.35
        + 0.45 * f["uses_tensor_core"]
        + 0.30 * f["sm_occupancy"]
        - 0.25 * np.tanh(f["log_bytes_per_flop"] / 4.0)
        - 0.15 * f["idle_fraction"]
    )
    return float(np.clip(1 / (1 + np.exp(-z)) * np.exp(rng.normal(0, 0.05)), 1e-3, 0.999))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic_raw.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for config, (key, gpu) in itertools.product(_configs(), GPUS.items()):
        try:
            res = analyse(make_kernel(config), gpu)
        except Exception:
            continue
        eta = _true_eta(res.features, rng)
        pi = _true_pi(res.features, rng)
        lat = res.theoretical_time_s / eta
        energy = pi * gpu.tdp_w * lat

        row = MeasurementRow(
            kernel_sig=config.signature(),
            category=config.category,
            dtype=config.dtype,
            gpu_key=key,
            source_model=config.source_model,
            source_module=config.source_module,
            latency_s=lat,
            energy_j=energy,
            energy_raw_j=energy * 1.02,
            power_avg_w=energy / lat,
            latency_sd_rel=abs(rng.normal(0, 0.004)),
            energy_sd_rel=abs(rng.normal(0, 0.006)),
            n_iters=1000,
            n_repeats=3,
            window_s=3.0,
            idle_power_w=gpu.idle_power_w,
            sm_clock_median_mhz=gpu.boost_clock_mhz * rng.uniform(0.75, 1.0),
            frac_sw_power_cap=float(rng.uniform() < 0.7),
            energy_counter_used=1,
            energy_check_ratio=float(rng.normal(1.0, 0.02)),
            gpu_name=gpu.name,
            architecture=gpu.architecture,
            compute_capability=gpu.compute_capability,
            sms=gpu.sms,
            ops_per_clk=gpu.ops_per_clk,
            tensor_clock_mhz=gpu.tensor_clock_mhz,
            boost_clock_mhz=gpu.boost_clock_mhz,
            peak_tensor_flops=gpu.peak_tensor_flops(),
            mem_bandwidth_gbs=gpu.mem_bandwidth_gbs,
            l2_bandwidth_gbs=gpu.l2_bandwidth_gbs,
            l2_cache_mb=gpu.l2_cache_mb,
            tdp_w=gpu.tdp_w,
            is_hbm=int(gpu.is_hbm),
            notes="SYNTHETIC -- not measured",
            params=dict(config.params),
        )
        rows.append(row.as_row())

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} synthetic rows to {out}")
    print(df.groupby(["gpu_key", "category"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
