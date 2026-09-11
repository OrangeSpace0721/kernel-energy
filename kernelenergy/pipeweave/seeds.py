"""Repeat a transfer evaluation over seeds, so a difference can be told from noise.

One run of ``evaluate_transfer`` gives one number per (GPU, operator) cell, and several
of those cells hold seventeen rows. At that size, an initialisation draw moves the
result by more than most of the effects worth arguing about. The A/B that motivated this
module -- power features on against off -- improved GEMM on four cards out of five at
n~205 a cell, and moved the seventeen-row rmsnorm cells in both directions at once. The
first is a result; the second is a coin.

Two things make the comparison sharp rather than merely repeated:

**Pairing.** Both arms run under the *same* seed, which fixes the validation split, the
shuffling and every random initialisation the two arms share. So a per-seed difference
isolates the mechanism instead of averaging over a nuisance that could have been held
constant. Comparing two independent sets of runs would need far more seeds to see the
same effect.

**A sign test, not a t-test.** Across ``n`` paired seeds, count how many favoured the
treatment. Under a null of no effect that count is Binomial(n, 0.5), which needs no
assumption about the shape of the error distribution -- and APE across folds is skewed
enough that assuming normality would be doing real work unearned. With five seeds a
clean sweep is p = 0.031 one-sided; four of five is p = 0.19 and settles nothing. Five
seeds is therefore the least that can produce a publishable claim, and it is what the
default runs.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from kernelenergy.pipeweave.evaluate import evaluate_transfer
from kernelenergy.pipeweave.transfer import TransferConfig

__all__ = ["run_seeds", "seed_summary", "paired_compare", "sign_test_p"]

#: Metrics worth tracking across seeds. Mean APE is the project's reporting convention;
#: the median is carried alongside because on a target recovered by division the mean is
#: set by whichever row had the smallest eta.
METRICS = ("zeroshot", "finetuned", "scratch", "hybrid", "ft_med", "pi_ft", "eta_ft")


def sign_test_p(wins: int, n: int) -> float:
    """One-sided binomial p for ``wins`` of ``n`` paired comparisons favouring one arm.

    P(X >= wins) under X ~ Binomial(n, 0.5). Reported rather than compared against a
    threshold: with five seeds the smallest attainable p is 0.031, so the number is
    better read as "how surprising is this run of results" than as a verdict.
    """
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def run_seeds(df, models_root, config: TransferConfig | None = None,
              seeds: int = 5, base_seed: int = 0, label: str = "",
              verbose: bool = True, jobs: int = 1) -> pd.DataFrame:
    """``evaluate_transfer`` over ``seeds`` seeds. Returns every cell of every run.

    ``jobs`` is passed straight down to the fold map inside each run rather than used
    to run whole seeds side by side. Both would parallelise the same total work, but
    the fold map has four times the tasks to play with and they finish at uneven times,
    so it packs cores better; running seeds in parallel would also multiply peak memory
    by the seed count for no gain.
    """
    cfg = config or TransferConfig()
    frames = []
    for i in range(seeds):
        s = base_seed + i
        if verbose:
            tag = f"{label} " if label else ""
            print(f"  {tag}seed {s} ({i + 1}/{seeds})...", flush=True)
        tab, _ = evaluate_transfer(df, models_root, dataclasses.replace(cfg, seed=s),
                                   jobs=jobs, verbose=verbose)
        t = tab.reset_index()
        t["seed"] = s
        t["arm"] = label or "run"
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def seed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Median and spread per cell. Spread is what says whether a cell means anything."""
    cols = [m for m in METRICS if m in runs.columns]
    g = runs.groupby(["gpu", "operator"])
    out = pd.DataFrame({"n_test": g["n_test"].first(), "seeds": g["seed"].nunique()})
    for m in cols:
        out[m] = g[m].median()
        # Half the min-max range, as a plain statement of how far the seeds disagreed.
        out[f"{m}_pm"] = (g[m].max() - g[m].min()) / 2.0
    return out.round(2)


def paired_compare(a: pd.DataFrame, b: pd.DataFrame, metric: str = "hybrid",
                   a_label: str = "A", b_label: str = "B") -> tuple[pd.DataFrame, dict]:
    """Compare two arms seed by seed on the same cells.

    Returns a per-cell table and a pooled verdict. Negative ``delta`` means ``b`` scored
    lower -- better, since these are errors.
    """
    key = ["gpu", "operator", "seed"]
    m = a[key + [metric, "n_test"]].merge(
        b[key + [metric]], on=key, suffixes=("_a", "_b"))
    if not len(m):
        raise ValueError("the two arms share no (gpu, operator, seed) cells")
    m["delta"] = m[f"{metric}_b"] - m[f"{metric}_a"]

    per_cell = m[m["operator"] != "-"].groupby(["gpu", "operator"]).agg(
        n_test=("n_test", "first"),
        seeds=("seed", "nunique"),
        **{a_label: (f"{metric}_a", "median"), b_label: (f"{metric}_b", "median")},
        delta=("delta", "median"),
        b_wins=("delta", lambda d: int((d < 0).sum())),
    )
    per_cell["p_sign"] = [
        sign_test_p(int(w), int(n)) for w, n in zip(per_cell["b_wins"], per_cell["seeds"])
    ]

    pooled = m[m["operator"] == "-"]
    verdict: dict = {"metric": metric, "a": a_label, "b": b_label}
    if len(pooled):
        wins = int((pooled["delta"] < 0).sum())
        n = len(pooled)
        verdict.update({
            "seeds": n,
            f"{a_label}_median": float(pooled[f"{metric}_a"].median()),
            f"{b_label}_median": float(pooled[f"{metric}_b"].median()),
            f"{a_label}_range": (float(pooled[f"{metric}_a"].min()),
                                 float(pooled[f"{metric}_a"].max())),
            f"{b_label}_range": (float(pooled[f"{metric}_b"].min()),
                                 float(pooled[f"{metric}_b"].max())),
            "delta_median": float(pooled["delta"].median()),
            "b_wins": wins,
            "p_sign": sign_test_p(wins, n),
        })
    return per_cell.round(3), verdict


def format_verdict(v: dict) -> str:
    a, b = v["a"], v["b"]
    if "seeds" not in v:
        return "no pooled row to compare"
    lo_a, hi_a = v[f"{a}_range"]
    lo_b, hi_b = v[f"{b}_range"]
    lines = [
        f"    {a:14s} {v[f'{a}_median']:7.2f}   (over {v['seeds']} seeds: "
        f"{lo_a:.2f} to {hi_a:.2f})",
        f"    {b:14s} {v[f'{b}_median']:7.2f}   (over {v['seeds']} seeds: "
        f"{lo_b:.2f} to {hi_b:.2f})",
        f"    delta          {v['delta_median']:+7.2f}   {b} won {v['b_wins']}/"
        f"{v['seeds']} seeds, sign-test p = {v['p_sign']:.3f}",
    ]
    if v["b_wins"] == v["seeds"] and v["seeds"] >= 5:
        lines.append(f"    Every seed favoured {b}. With {v['seeds']} seeds that is the "
                     f"strongest this design can show.")
    elif v["p_sign"] > 0.2:
        lines.append(f"    Not separable at {v['seeds']} seeds -- the arms overlap. "
                     f"Treat them as equivalent until more seeds say otherwise.")
    return "\n".join(lines)
