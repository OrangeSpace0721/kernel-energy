"""Reconcile a sum over kernels against a measured generation.

The accounting identity, and every term in it is measured except ``epsilon``::

    E_generation  =  sum_k  calls_k * E_k   +   P_idle * T_gap   +   epsilon
    T_gap         =  T_generation - T_busy

Reporting ``epsilon`` as a share of the total is the whole point. A reconstruction that
lands within a few percent with a small residual is a reconstruction; one that lands
within a few percent because a large positive residual cancels a large negative
attribution error is a coincidence, and only the decomposition tells them apart.

Latency is reconciled against ``T_busy``, not wall time
-------------------------------------------------------
A replayed kernel's latency should reconstruct the *device* time of the generation. The
gap is not something per-kernel measurement can be expected to predict -- it is
scheduler, Python and synchronisation. Comparing the kernel sum against wall time
charges the model for time no kernel was running, which is not its error to carry, and
it is why ``latency_ratio`` in the earlier script always looked worse than the model
deserved.

Two levels
----------
``A`` sums **measured** replay energies; ``B`` sums **model predictions**. Both are
weighted by the same actual call counts, so the difference between them is model error
and nothing else. Read together they localise the problem:

* A near 1, B far from 1 -- replay composes; the model is wrong.
* A far from 1, B similar to A -- the model is fine; replaying kernels in isolation
  does not represent them in a pipeline, and no amount of model work fixes it.
* both near 1 -- the per-kernel result means what it appears to mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["reconcile", "summarise", "step_model"]


def _sum_over_calls(calls: dict[str, int], table: pd.DataFrame, value_col: str):
    """Sum ``value_col`` weighted by call count. Returns (total, matched, unmatched)."""
    if not calls:
        return 0.0, 0, 0
    s = pd.Series(calls, name="calls")
    s.index.name = "kernel_sig"
    j = s.to_frame().join(table.set_index("kernel_sig")[value_col], how="left")
    have = j[value_col].notna()
    total = float((j.loc[have, value_col] * j.loc[have, "calls"]).sum())
    return total, int(j.loc[have, "calls"].sum()), int(j.loc[~have, "calls"].sum())


def reconcile(
    measurement,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    energy_pred_col: str = "energy_pred",
    latency_pred_col: str = "latency_pred",
) -> dict:
    """One measured generation against the sum over its kernels.

    ``dataset`` supplies measured per-kernel ``energy_j`` and ``latency_s`` (level A).
    ``predictions`` optionally supplies per-kernel predicted energy for the same card --
    the ``predictions`` frame from :func:`kernelenergy.model.evaluate.evaluate` or
    :func:`kernelenergy.pipeweave.evaluate.evaluate_transfer`, which are leave-one-GPU-out
    and so have never seen this card (level B).
    """
    m = measurement
    d = dataset[dataset["gpu_key"] == m.gpu_key]
    if not len(d):
        raise KeyError(f"dataset has no rows for {m.gpu_key}")

    e_a, matched_calls, missing_calls = _sum_over_calls(m.calls, d, "energy_j")
    t_a, _, _ = _sum_over_calls(m.calls, d, "latency_s")

    total_calls = max(sum(m.calls.values()), 1)
    gap_s = max(m.latency_s - m.device_time_s, 0.0)
    e_gap = m.idle_power_w * gap_s

    out = {
        "gpu_key": m.gpu_key, "model": m.model, "steps": m.steps,
        "height": m.height, "width": m.width,

        # ground truth
        "e2e_energy_j": m.energy_j,
        "e2e_latency_s": m.latency_s,
        "e2e_power_w": m.power_avg_w,

        # where the wall time went
        "device_time_s": m.device_time_s,
        "gap_s": gap_s,
        "busy_fraction": m.busy_fraction,
        "profiler_coverage": m.coverage_fraction,

        # how much of the generation the catalogue even names
        "call_coverage": matched_calls / total_calls,
        "calls_matched": matched_calls,
        "calls_missing": missing_calls,

        # level A -- measured replay energies
        "A_kernel_energy_j": e_a,
        "A_gap_energy_j": e_gap,
        "A_total_j": e_a + e_gap,
        "A_ratio": (e_a + e_gap) / max(m.energy_j, 1e-12),
        "A_residual_j": m.energy_j - (e_a + e_gap),
        "A_residual_frac": (m.energy_j - (e_a + e_gap)) / max(m.energy_j, 1e-12),

        # latency reconstructs DEVICE time, not wall time
        "A_kernel_latency_s": t_a,
        "A_latency_ratio": t_a / max(m.device_time_s, 1e-12),
    }

    if predictions is not None and len(predictions):
        p = predictions[predictions["gpu_key"] == m.gpu_key]
        if len(p) and energy_pred_col in p.columns:
            pe = p[["kernel_sig", energy_pred_col]].rename(
                columns={energy_pred_col: "energy_j"})
            e_b, mb, _ = _sum_over_calls(m.calls, pe, "energy_j")
            out.update({
                "B_kernel_energy_j": e_b,
                "B_total_j": e_b + e_gap,
                "B_ratio": (e_b + e_gap) / max(m.energy_j, 1e-12),
                "B_residual_frac": (m.energy_j - (e_b + e_gap)) / max(m.energy_j, 1e-12),
                "B_call_coverage": mb / total_calls,
            })
            if latency_pred_col in p.columns:
                pl = p[["kernel_sig", latency_pred_col]].rename(
                    columns={latency_pred_col: "latency_s"})
                t_b, _, _ = _sum_over_calls(m.calls, pl, "latency_s")
                out["B_kernel_latency_s"] = t_b
                out["B_latency_ratio"] = t_b / max(m.device_time_s, 1e-12)
    return out


def summarise(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """The table to actually look at, one row per measured generation."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    cols = [c for c in [
        "gpu_key", "model", "steps",
        "e2e_energy_j", "A_total_j", "B_total_j",
        "A_ratio", "B_ratio", "A_residual_frac",
        "busy_fraction", "call_coverage", "profiler_coverage",
        "A_latency_ratio", "B_latency_ratio",
    ] if c in df.columns]
    return df[cols].round(3)


def step_model(rows: list[dict] | pd.DataFrame, value: str = "e2e_energy_j") -> pd.DataFrame:
    """Fit ``value = a + b * steps`` per (gpu, model).

    A single ratio has several causes at once. The intercept is what runs once per image
    -- text encoding and the VAE decode -- and the slope is the per-step transformer
    cost. Fitting both sides and comparing them separately says *where* a reconstruction
    goes wrong: a right slope with a wrong intercept is the VAE, and the reverse is the
    denoiser. Needs at least three step counts per cell to be worth reading.
    """
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    out = []
    for (gpu, model), g in df.groupby(["gpu_key", "model"]):
        g = g.sort_values("steps")
        if len(g) < 3:
            continue
        rec = {"gpu_key": gpu, "model": model, "n_points": len(g)}
        for col, tag in [(value, "measured"), ("A_total_j", "A"), ("B_total_j", "B")]:
            if col not in g.columns or g[col].isna().any():
                continue
            b, a = np.polyfit(g["steps"].to_numpy(float), g[col].to_numpy(float), 1)
            resid = g[col].to_numpy(float) - (a + b * g["steps"].to_numpy(float))
            ss = float(np.sum(resid ** 2))
            tot = float(np.sum((g[col] - g[col].mean()) ** 2))
            rec[f"{tag}_fixed_j"] = a          # once per image: text encode + VAE
            rec[f"{tag}_per_step_j"] = b       # the transformer
            rec[f"{tag}_r2"] = 1 - ss / tot if tot > 0 else np.nan
        if "measured_per_step_j" in rec:
            for tag in ("A", "B"):
                if f"{tag}_per_step_j" in rec:
                    rec[f"{tag}_slope_ratio"] = (
                        rec[f"{tag}_per_step_j"] / rec["measured_per_step_j"])
                    rec[f"{tag}_fixed_ratio"] = (
                        rec[f"{tag}_fixed_j"] / rec["measured_fixed_j"]
                        if abs(rec["measured_fixed_j"]) > 1e-9 else np.nan)
        out.append(rec)
    return pd.DataFrame(out).round(3)
