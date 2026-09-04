"""Leave-one-group-out evaluation.

Reporting follows the convention already established in this project: **every table is
per held-out group**, with ``MEAN`` and ``POOLED`` columns. Averages hide folds, and the
hardware fold is exactly where they hide the most -- one card extrapolating badly can sit
invisible behind five that interpolate.

Three folds:

* **hardware** -- hold out a GPU. Tests whether the model transfers to a card it has
  never seen, which is PipeWeave's headline claim and the one that matters for planning.
* **architecture** -- hold out a diffusion model. Tests whether the kernel catalogue
  learned from FLUX and Qwen predicts SD3.5's kernels.
* **category** -- hold out a kernel category. The strictest test, and the one PipeWeave
  never runs: it trains a *separate* MLP per kernel category, so it cannot ask whether
  anything transfers between them. Worth knowing, because if nothing does, every new
  kernel class needs its own measurement campaign.

Three baselines, all of which the model has to beat to be worth its complexity:

* **roofline** -- eta = 1, pi = 1: the kernel runs at its analytical floor drawing full
  TDP. Crude, but it is the honest zero-parameter prediction.
* **constant** -- the training-set median eta and pi. This is the one that actually
  matters. If a fitted constant is close to the MLP, the features are not carrying
  information and the analytical stage is doing all the work.
* **ridge** -- linear regression on the same features, in log space. Separates "the
  features help" from "the nonlinearity helps".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from kernelenergy.model.estimator import MLP, TrainConfig, mape
from kernelenergy.model.features import FEATURE_COLUMNS
from kernelenergy.model.targets import energy_from_heads, latency_from_eta

__all__ = ["FOLDS", "evaluate", "FoldResult", "report_table"]

FOLDS = {
    "hardware": "gpu_key",
    "architecture": "source_model",
    "category": "category",
}


@dataclass
class FoldResult:
    fold: str
    group: str
    n_train: int
    n_test: int
    energy_mape: float
    latency_mape: float
    eta_mape: float
    pi_mape: float
    baseline_roofline: float
    baseline_constant: float
    baseline_ridge: float
    predictions: pd.DataFrame


def _prepare(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=float)
    Y = df[["eta", "pi"]].to_numpy(dtype=float)
    return X, Y


def _fit_ridge(Xtr, df_tr, Xte, alpha: float = 1.0):
    """Ridge on log energy, as a linear stand-in for the MLP."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    r = Ridge(alpha=alpha).fit((Xtr - mu) / sd, np.log(df_tr["energy_j"].to_numpy(float)))
    return np.exp(r.predict((Xte - mu) / sd))


def evaluate(
    df: pd.DataFrame,
    fold: str = "hardware",
    feature_cols: list[str] | None = None,
    config: TrainConfig | None = None,
    min_group_rows: int = 10,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Leave-one-group-out over ``fold``. Returns (table, per-fold results)."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {sorted(FOLDS)}, got {fold!r}")
    group_col = FOLDS[fold]
    if group_col not in df.columns:
        raise KeyError(f"dataframe has no {group_col!r} column for the {fold} fold")

    feature_cols = feature_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
    needed = {"eta", "pi", "energy_j", "latency_s", "theoretical_time_s", "tdp_w"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"dataframe is missing {sorted(missing)}; run add_targets first")

    df = df.dropna(subset=feature_cols + ["eta", "pi", "energy_j"]).copy()
    results: list[FoldResult] = []

    for group, te in df.groupby(group_col):
        tr = df[df[group_col] != group]
        if len(te) < min_group_rows or len(tr) < min_group_rows:
            continue

        Xtr, Ytr = _prepare(tr, feature_cols)
        Xte, Yte = _prepare(te, feature_cols)

        model = MLP(Xtr.shape[1], n_heads=2, config=config or TrainConfig())
        model.fit(Xtr, Ytr)
        pred = model.predict(Xte)
        eta_hat, pi_hat = pred[:, 0], pred[:, 1]

        theory = te["theoretical_time_s"].to_numpy(float)
        tdp = te["tdp_w"].to_numpy(float)
        e_true = te["energy_j"].to_numpy(float)
        t_true = te["latency_s"].to_numpy(float)

        e_hat = energy_from_heads(eta_hat, pi_hat, theory, tdp)
        t_hat = latency_from_eta(eta_hat, theory)

        ones = np.ones(len(te))
        e_roof = energy_from_heads(ones, ones, theory, tdp)
        e_const = energy_from_heads(
            ones * float(np.median(tr["eta"])), ones * float(np.median(tr["pi"])),
            theory, tdp,
        )
        e_ridge = _fit_ridge(Xtr, tr, Xte)

        # dict.fromkeys keeps order and drops the duplicate when group_col is one of
        # these -- selecting it twice yields a frame with two identically named columns,
        # which then breaks any comparison against it.
        id_cols = [c for c in dict.fromkeys(
            [group_col, "category", "gpu_key", "source_model", "kernel_sig"]
        ) if c in te.columns]
        preds = te[id_cols].copy()
        preds["energy_true"] = e_true
        preds["energy_pred"] = e_hat
        preds["latency_true"] = t_true
        preds["latency_pred"] = t_hat
        preds["eta_true"], preds["eta_pred"] = Yte[:, 0], eta_hat
        preds["pi_true"], preds["pi_pred"] = Yte[:, 1], pi_hat

        results.append(
            FoldResult(
                fold=fold,
                group=str(group),
                n_train=len(tr),
                n_test=len(te),
                energy_mape=mape(e_true, e_hat),
                latency_mape=mape(t_true, t_hat),
                eta_mape=mape(Yte[:, 0], eta_hat),
                pi_mape=mape(Yte[:, 1], pi_hat),
                baseline_roofline=mape(e_true, e_roof),
                baseline_constant=mape(e_true, e_const),
                baseline_ridge=mape(e_true, e_ridge),
                predictions=preds,
            )
        )

    if not results:
        raise RuntimeError(
            f"no {fold} fold had at least {min_group_rows} rows on both sides"
        )
    return report_table(results), results


def report_table(results: list[FoldResult]) -> pd.DataFrame:
    """Per-group rows plus MEAN and POOLED.

    POOLED is recomputed over the concatenated predictions rather than averaged from the
    per-group numbers, so a large group is weighted like a large group. MEAN weights every
    held-out group equally. They differ, and which one is right depends on the question --
    both are reported for that reason.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "group": r.group,
                "n_test": r.n_test,
                "energy": r.energy_mape,
                "latency": r.latency_mape,
                "eta": r.eta_mape,
                "pi": r.pi_mape,
                "roofline": r.baseline_roofline,
                "constant": r.baseline_constant,
                "ridge": r.baseline_ridge,
            }
        )
    tab = pd.DataFrame(rows).set_index("group")

    mean_row = tab.drop(columns=["n_test"]).mean()
    mean_row["n_test"] = tab["n_test"].sum()

    allp = pd.concat([r.predictions for r in results], ignore_index=True)
    pooled = {
        "n_test": len(allp),
        "energy": mape(allp["energy_true"], allp["energy_pred"]),
        "latency": mape(allp["latency_true"], allp["latency_pred"]),
        "eta": mape(allp["eta_true"], allp["eta_pred"]),
        "pi": mape(allp["pi_true"], allp["pi_pred"]),
        # Baselines are per-fold MAPEs, so their pooled value is the row-count-weighted
        # average -- exact for MAPE, which is itself a mean over rows.
        "roofline": float(np.average([r.baseline_roofline for r in results],
                                     weights=[r.n_test for r in results])),
        "constant": float(np.average([r.baseline_constant for r in results],
                                     weights=[r.n_test for r in results])),
        "ridge": float(np.average([r.baseline_ridge for r in results],
                                  weights=[r.n_test for r in results])),
    }

    tab.loc["MEAN"] = mean_row
    tab.loc["POOLED"] = pd.Series(pooled)
    tab["n_test"] = tab["n_test"].astype(int)
    return tab.round(2)
