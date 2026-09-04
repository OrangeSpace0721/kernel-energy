"""Assemble the model-ready dataset: measurements + analytical features + targets.

Kept separate from both the measurement and the modelling code because it is the join,
and the join is where the two halves have to agree. Measurements are collected once and
never recomputed; features are recomputed from the analytical stack every time, so a
change to a decomposer or a hardware constant flows through without touching a single
recorded joule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kernelenergy.hardware import GPUS, GPU, get_gpu
from kernelenergy.kernels.base import KernelConfig
from kernelenergy.kernels.registry import make_kernel
from kernelenergy.model.features import analyse
from kernelenergy.model.targets import add_targets, validate_efficiency

__all__ = ["build_dataset", "attach_features", "quality_report", "ANALYTIC_COLUMNS"]

#: Columns the schema reserves but the analytical stage owns. Recomputed every time.
ANALYTIC_COLUMNS = (
    "theoretical_time_s",
    "bottleneck",
    "analytic_flops",
    "analytic_bytes_global",
)


def attach_features(
    df: pd.DataFrame,
    clock: str = "tensor",
    hardware: dict[str, GPU] | None = None,
    on_error: str = "drop",
) -> pd.DataFrame:
    """Run the analytical stack for every row and append its features."""
    hardware = hardware or GPUS
    feats: list[dict] = []
    keep: list[int] = []
    failures: list[tuple[int, str]] = []

    for i, row in df.iterrows():
        try:
            gpu = hardware.get(str(row["gpu_key"]).upper()) or get_gpu(str(row["gpu_key"]))
            cfg = KernelConfig.from_row(row.to_dict())
            res = analyse(make_kernel(cfg), gpu, clock=clock)
            f = dict(res.features)
            f["theoretical_time_s"] = res.theoretical_time_s
            f["bottleneck"] = res.bottleneck
            f["analytic_flops"] = res.flops
            f["analytic_bytes_global"] = res.bytes_global
            feats.append(f)
            keep.append(i)
        except Exception as e:
            failures.append((i, f"{type(e).__name__}: {e}"))
            if on_error == "raise":
                raise

    if failures and on_error == "drop":
        print(f"attach_features: dropped {len(failures)} rows that failed analysis")
        for i, msg in failures[:5]:
            print(f"  row {i}: {msg}")

    base = df.loc[keep].reset_index(drop=True)
    fdf = pd.DataFrame(feats).reset_index(drop=True)

    # Measurement columns win on a name clash -- the analytical stack must never
    # overwrite something that was actually measured. The four names in
    # ANALYTIC_COLUMNS are the exception: the schema reserves them so a raw file is
    # self-describing, but they are *derived*, and the freshly computed value always
    # supersedes whatever a previous run wrote. Without this exception a change to a
    # decomposer would silently fail to propagate.
    overwrite = [c for c in ANALYTIC_COLUMNS if c in base.columns and c in fdf.columns]
    base = base.drop(columns=overwrite)
    dupes = [c for c in fdf.columns if c in base.columns]
    return pd.concat([base, fdf.drop(columns=dupes)], axis=1)


def build_dataset(
    measurements: str | Path | pd.DataFrame,
    clock: str = "tensor",
    hardware: dict[str, GPU] | None = None,
    drop_unreliable: bool = True,
    max_energy_sd_rel: float = 0.05,
    min_latency_s: float = 5e-6,
) -> pd.DataFrame:
    """Measurements -> a frame ready for :func:`kernelenergy.model.evaluate.evaluate`.

    ``drop_unreliable`` removes two classes of row that will otherwise dominate the error
    metric without carrying information:

    * **Unstable repeats.** An energy spread above ``max_energy_sd_rel`` across repeats
      means the card was not in a steady state -- another process, a thermal event, a
      window too short for the counter. The measurement is not wrong so much as it is not
      a measurement of what we think.

    * **Kernels below a few microseconds.** Launch overhead subtraction leaves a large
      relative residue there, and MAPE weights small values hardest, so a handful of
      trivial kernels can set the headline number. The run-level work in this project hit
      exactly this: calibrating on the cheapest configurations took the hardware fold
      from 9.66% to 37.28%.

    Both filters are off with ``drop_unreliable=False``, and both print what they removed.
    """
    if isinstance(measurements, (str, Path)):
        df = pd.read_csv(measurements)
    else:
        df = measurements.copy()

    n0 = len(df)
    if drop_unreliable:
        before = len(df)
        df = df[df["energy_sd_rel"].fillna(0) <= max_energy_sd_rel]
        n_unstable = before - len(df)
        before = len(df)
        df = df[df["latency_s"] > min_latency_s]
        n_tiny = before - len(df)
        if n_unstable or n_tiny:
            print(f"build_dataset: dropped {n_unstable} unstable and {n_tiny} sub-"
                  f"{min_latency_s * 1e6:.0f}us rows of {n0}")

    df = attach_features(df, clock=clock, hardware=hardware)
    df = add_targets(df)

    over = (df["eta"] > 1.0).sum()
    if over:
        print(
            f"build_dataset: {over} of {len(df)} rows have eta > 1 -- the analytical "
            f"floor exceeds the measured time. Inspect with validate_efficiency(); do "
            f"not train on these until it is understood."
        )
    return df


def quality_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A few tables worth reading before trusting any fold result."""
    out: dict[str, pd.DataFrame] = {}

    out["efficiency"] = validate_efficiency(df)

    out["coverage"] = (
        df.groupby(["gpu_key", "category"])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )

    # The counter and the integrated power should agree once the window is long enough.
    if "energy_check_ratio" in df:
        out["instrument_agreement"] = (
            df.groupby("gpu_key")["energy_check_ratio"]
            .agg(["median", "std", "min", "max"])
            .round(3)
        )

    # Two regimes, per the run-level finding: cards that sit against the power cap and
    # cards that do not. This is the table that tells you which is which.
    if "frac_sw_power_cap" in df:
        out["power_regime"] = (
            df.groupby("gpu_key")
            .agg(
                frac_capped=("frac_sw_power_cap", "mean"),
                frac_thermal=("frac_sw_thermal", "mean"),
                clock_frac=("sm_clock_median_mhz", "median"),
                power_frac=("pi", "median"),
                util=("eta", "median"),
            )
            .round(3)
        )

    out["targets"] = (
        df.groupby("category")[["eta", "pi", "power_avg_w"]]
        .describe()
        .round(3)
    )
    return out
