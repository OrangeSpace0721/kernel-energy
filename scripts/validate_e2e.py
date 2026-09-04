"""Does the sum of replayed kernel energies reconstruct a real generation?

This is the script that tells you how much the replay methodology costs you, and it
should be run before any kernel-level result is presented as meaning something about
whole pipelines.

The reconstruction is

    E_pipeline_hat = sum over catalogue of  calls_per_step * steps * E_kernel
                     + P_idle * t_measured

and it will **overestimate**. Three reasons, in roughly descending order of size:

1. A kernel replayed alone has the card's whole power budget. In the pipeline it shares
   that budget with whatever overlaps it, and on a power-capped card the sum of what
   several kernels would each draw alone exceeds what the card will deliver.
2. Replay runs at the steady-state clock for that one kernel. A mixed pipeline settles at
   a different clock.
3. The catalogue does not cover every kernel, so some of the gap is missing work pulling
   the other way.

A ratio near 1.0 would be surprising. What matters is that the ratio is *stable* across
GPUs and models: a consistent 1.25 is a calibration constant, while 1.05 on one card and
1.6 on another means the per-kernel numbers do not compose and the end-to-end claim has
to be dropped.

    python scripts/validate_e2e.py --dataset data/dataset.csv --model flux1-dev --steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def measure_e2e(pipe, spec, *, steps: int, height: int, width: int,
                device: int = 0, repeats: int = 3) -> dict:
    """Measure a whole generation with the same instrument the replay harness uses."""
    import torch

    from kernelenergy.nvml import PowerSampler, measure_idle

    def _gen():
        with torch.no_grad():
            pipe(prompt="a photograph of a city street", num_inference_steps=steps,
                 height=height, width=width, **spec.extra_call_kwargs)

    _gen()  # warm caches and autotuning
    torch.cuda.synchronize()

    lats, energies, summaries = [], [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        with PowerSampler(device, 0.005) as s:
            t0 = time.perf_counter()
            _gen()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        summ = s.summary()
        summaries.append(summ)
        e = summ.energy_counter_j if summ.energy_counter_j is not None \
            else summ.energy_integrated_j
        lats.append(elapsed)
        energies.append(e)
        time.sleep(1.0)

    idle = measure_idle(device, seconds=10.0)
    return {
        "latency_s": float(np.median(lats)),
        "energy_j": float(np.median(energies)),
        "power_avg_w": float(np.median(energies) / np.median(lats)),
        "idle_power_w": idle.power_w,
        "sm_clock_median_mhz": summaries[-1].sm_clock_median_mhz,
        "frac_sw_power_cap": summaries[-1].frac_sw_power_cap,
    }


def reconstruct(dataset: pd.DataFrame, catalogue: pd.DataFrame, *, gpu_key: str,
                model: str, steps: int, e2e: dict) -> dict:
    """Sum measured per-kernel energies weighted by how often the pipeline calls them."""
    calls_col = f"calls_{model}"
    if calls_col not in catalogue.columns:
        raise KeyError(
            f"catalogue has no {calls_col}; capture {model} before validating it"
        )

    d = dataset[dataset["gpu_key"] == gpu_key]
    joined = catalogue.merge(
        d[["kernel_sig", "latency_s", "energy_j"]], on="kernel_sig", how="left"
    )
    have = joined["energy_j"].notna()

    # Capture ran for one denoising step, so transformer kernels scale with `steps`
    # while the VAE decode does not. Convs in these pipelines are the VAE.
    per_step = joined["category"] != "conv"
    weight = joined[calls_col].fillna(0) * np.where(per_step, steps, 1)

    e_kernels = float((joined.loc[have, "energy_j"] * weight[have]).sum())
    t_kernels = float((joined.loc[have, "latency_s"] * weight[have]).sum())
    e_idle = e2e["idle_power_w"] * e2e["latency_s"]

    covered = float(weight[have].sum() / max(weight.sum(), 1))
    return {
        "gpu_key": gpu_key,
        "model": model,
        "steps": steps,
        "e2e_energy_j": e2e["energy_j"],
        "e2e_latency_s": e2e["latency_s"],
        "kernel_sum_energy_j": e_kernels,
        "kernel_sum_latency_s": t_kernels,
        "reconstructed_energy_j": e_kernels + e_idle,
        "energy_ratio": (e_kernels + e_idle) / max(e2e["energy_j"], 1e-12),
        "latency_ratio": t_kernels / max(e2e["latency_s"], 1e-12),
        "catalogue_call_coverage": covered,
        "n_configs_matched": int(have.sum()),
        "n_configs_total": int(len(joined)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--catalogue", default="data/catalogue.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="data/e2e_validation.csv")
    args = ap.parse_args()

    from kernelenergy.hardware import probe_local_gpu
    from kernelenergy.trace.pipelines import load_pipeline

    gpu = probe_local_gpu(args.device)
    dataset = pd.read_csv(args.dataset)
    catalogue = pd.read_csv(args.catalogue)

    pipe, spec = load_pipeline(args.model)
    e2e = measure_e2e(pipe, spec, steps=args.steps, height=args.height,
                      width=args.width, device=args.device)
    row = reconstruct(dataset, catalogue, gpu_key=gpu.gpu_key, model=args.model,
                      steps=args.steps, e2e=e2e)

    print(f"\n{gpu.gpu_key} / {args.model} / {args.steps} steps "
          f"@ {args.height}x{args.width}")
    print(f"  measured end to end     {e2e['energy_j']:9.2f} J   "
          f"{e2e['latency_s']:7.3f} s   {e2e['power_avg_w']:6.1f} W")
    print(f"  sum of replayed kernels {row['reconstructed_energy_j']:9.2f} J   "
          f"{row['kernel_sum_latency_s']:7.3f} s")
    print(f"  energy ratio            {row['energy_ratio']:9.3f}")
    print(f"  latency ratio           {row['latency_ratio']:9.3f}")
    print(f"  catalogue call coverage {row['catalogue_call_coverage']:9.1%} "
          f"({row['n_configs_matched']}/{row['n_configs_total']} configs measured)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{**row, **{f"e2e_{k}": v for k, v in e2e.items()}}])
    header = not out.exists()
    df.to_csv(out, mode="a", header=header, index=False)
    print(f"\nappended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
