"""Target definitions -- where PipeWeave is re-aimed at energy.

PipeWeave's MLP emits one bounded scalar, the execution efficiency

    eta = t_theoretical / t_measured   in (0, 1]

and recovers latency by division, ``t = C / eta``. The reason that works is not that
efficiency is more interesting than latency; it is that efficiency varies over about one
order of magnitude while latency across a kernel catalogue varies over five or six, so
the network is asked to learn a small, well-conditioned thing and the analytical stage
supplies the large one. The run-level work in this project reached the same conclusion
independently: log utilisation has sd 0.192 across the fleet where log latency has 1.884.

Energy needs a second bounded scalar, and the natural one is the **power fraction**

    pi = P_avg / TDP   in (0, 1]

which gives

    E = pi * TDP * t = pi * TDP * C / eta

Both heads are ratios against a spec-sheet constant, both live in roughly the same
range, and both are recoverable from one measurement of a kernel replayed in isolation.
Predicting energy directly instead would ask the network to re-learn the entire dynamic
range that the analytical stage already accounts for.

Why not the dynamic fraction only
---------------------------------
``pi_dynamic = (P_avg - P_idle) / (TDP - P_idle)`` is arguably the more physical
quantity, since the idle floor is not attributable to the kernel. It is computed here
too, and it is the better target for *attribution* questions. It is the worse target for
*prediction*, because dividing by a small measured difference amplifies the error in
``P_idle`` -- on a 72 W L4 the floor is a fifth of the budget and a 2 W error in it moves
the target by 3%. Default to ``pi``; use ``pi_dynamic`` when the question is how much
energy the kernel itself is responsible for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TARGETS",
    "add_targets",
    "energy_from_heads",
    "latency_from_eta",
    "validate_efficiency",
]

#: Head name -> the column it is fitted against. Both are bounded in (0, 1].
TARGETS = {
    "eta": "eta",  # execution efficiency  -> latency
    "pi": "pi",  # power fraction        -> energy, given latency
}


def add_targets(
    df: pd.DataFrame,
    latency_col: str = "latency_s",
    energy_col: str = "energy_j",
    theory_col: str = "theoretical_time_s",
    tdp_col: str = "tdp_w",
    idle_col: str = "idle_power_w",
) -> pd.DataFrame:
    """Derive every target column from the measured columns. Returns a copy."""
    out = df.copy()
    lat = out[latency_col].astype(float)
    en = out[energy_col].astype(float)
    theory = out[theory_col].astype(float)
    tdp = out[tdp_col].astype(float)

    out["eta"] = theory / lat
    out["power_avg_w"] = en / lat
    out["pi"] = out["power_avg_w"] / tdp

    if idle_col in out.columns:
        idle = out[idle_col].astype(float)
        head = (tdp - idle).clip(lower=1e-6)
        out["pi_dynamic"] = ((out["power_avg_w"] - idle) / head).clip(lower=1e-6)
        out["energy_dynamic_j"] = (en - idle * lat).clip(lower=0.0)
    else:
        out["pi_dynamic"] = np.nan
        out["energy_dynamic_j"] = np.nan

    # Against the limit that actually binds, rather than the datasheet TDP.
    #
    # HPC sites routinely cap GPUs below their rated TDP for facility power or cooling
    # reasons, and a capped card physically cannot reach the denominator `pi` is defined
    # against -- so every `pi` from that site is compressed by a factor that is invisible
    # in the data and, worse, differs between sites. `pi_limit` is the version that
    # travels: it is bounded by 1 whatever the cap, and on an uncapped card it equals
    # `pi`. Prefer it whenever `power_limit_w` is present and differs from TDP; the
    # hardware fold in particular is otherwise comparing two conventions.
    if "power_limit_w" in out.columns:
        lim = out["power_limit_w"].astype(float)
        lim = lim.where(lim > 0).fillna(tdp)
        out["pi_limit"] = out["power_avg_w"] / lim
        out["power_cap_ratio"] = lim / tdp
    else:
        out["pi_limit"] = out["pi"]
        out["power_cap_ratio"] = 1.0

    out["log_energy"] = np.log(en.clip(lower=1e-12))
    out["log_latency"] = np.log(lat.clip(lower=1e-12))
    if "analytic_flops" in out.columns:
        f = out["analytic_flops"].astype(float).clip(lower=1.0)
        out["energy_per_flop_j"] = en / f
        out["pj_per_flop"] = en / f * 1e12
    if "analytic_bytes_global" in out.columns:
        b = out["analytic_bytes_global"].astype(float).clip(lower=1.0)
        out["pj_per_byte"] = en / b * 1e12
    return out


def energy_from_heads(
    eta: np.ndarray, pi: np.ndarray, theoretical_time_s: np.ndarray, tdp_w: np.ndarray
) -> np.ndarray:
    """E = pi * TDP * C / eta."""
    eta = np.clip(np.asarray(eta, float), 1e-6, None)
    return np.asarray(pi, float) * np.asarray(tdp_w, float) * np.asarray(
        theoretical_time_s, float
    ) / eta


def latency_from_eta(eta: np.ndarray, theoretical_time_s: np.ndarray) -> np.ndarray:
    eta = np.clip(np.asarray(eta, float), 1e-6, None)
    return np.asarray(theoretical_time_s, float) / eta


def validate_efficiency(
    df: pd.DataFrame, by: tuple[str, ...] = ("category", "gpu_key")
) -> pd.DataFrame:
    """Report where the analytical floor is above the measured time.

    Any ``eta > 1`` means the analytical model claims the kernel ran faster than
    physically possible, so something upstream is wrong -- usually a throughput constant
    for a precision the card handles differently, or a decomposition that undercounts
    work. This is worth checking every time the catalogue or the hardware table changes,
    because a sigmoid-bounded head will happily fit around such rows and hide them.
    """
    g = df.groupby(list(by))
    rep = g.agg(
        n=("eta", "size"),
        eta_median=("eta", "median"),
        eta_p95=("eta", lambda s: float(np.nanpercentile(s, 95))),
        eta_max=("eta", "max"),
        frac_over_1=("eta", lambda s: float((s > 1.0).mean())),
    )
    return rep.sort_values("frac_over_1", ascending=False)
