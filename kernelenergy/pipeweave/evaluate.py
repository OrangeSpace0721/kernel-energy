"""Leave-one-GPU-out evaluation of the transferred model, against what it must beat.

The question this answers is not "does the transferred model predict energy". It is
"**was the transfer worth doing**" -- and that needs three numbers side by side on the
same folds:

* **zero-shot** -- their checkpoint untouched, ``pi`` set to the training median. No
  fitting of any kind on diffusion data. This is the honest floor: whatever it scores,
  the transfer contributed nothing below it.
* **fine-tuned** -- their trunk and efficiency head, fine-tuned, plus a fitted power
  head. The thing being argued for.
* **scratch** -- the same architecture on the same features with random initialisation,
  fitted on the same rows. If this matches fine-tuned, half a million of their samples
  bought nothing and the pretrained weights are decoration.

Everything is reported per held-out GPU, never as a single average, for the reason
stated in :mod:`kernelenergy.model.evaluate`: one card extrapolating badly hides behind
five that interpolate, and the hardware fold is exactly where that happens.

Per operator, always
--------------------
PipeWeave trains a separate network per operator, so the transfer inherits that: four
checkpoints, four models, four fine-tunings per fold. There is no "one model over all
kernels" version of this -- their weights do not exist in that form. Categories with too
few training rows to fine-tune are reported as such rather than silently folded into
another operator's model.

The floor changes with the model
--------------------------------
``eta`` here is measured against **their** analytical floor
(``reference_cycles / sm_freq``), not this project's. Their efficiency head was fitted
against theirs; composing its output with ours would put the ratio on a different scale
and the mismatch would present as model error. That is why ``prepare`` recomputes
``eta_pw`` rather than reusing the ``eta`` column, and why the two are worth comparing
directly -- ``compare_floors`` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kernelenergy.model.estimator import mape
from kernelenergy.pipeweave.features import PIPEWEAVE_FEATURES
from kernelenergy.pipeweave.hardware import PW_HARDWARE
from kernelenergy.pipeweave.transfer import (
    TransferConfig,
    TransferModel,
    find_checkpoint,
    reference_cycles,
)

__all__ = ["prepare", "evaluate_transfer", "compare_floors", "range_report", "OperatorResult"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``theory_pw_s`` and ``eta_pw`` to a frame that already carries ``pw_*``.

    Run :func:`kernelenergy.pipeweave.features.emit_frame` first.
    """
    need = {"pw_operator", "latency_s", "gpu_key"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"missing {sorted(missing)}; run emit_frame() and add_targets() first")

    out = df.copy()
    theory = np.full(len(out), np.nan)
    for op, g in out.groupby("pw_operator"):
        names = PIPEWEAVE_FEATURES[op]
        cols = [f"pw_{n}" for n in names]
        if not set(cols) <= set(out.columns):
            continue
        cycles = reference_cycles(op, g[cols].to_numpy(float), names)
        freq = g["gpu_key"].map(lambda k: PW_HARDWARE[str(k).upper()].sm_freq).to_numpy(float)
        theory[out.index.get_indexer(g.index)] = cycles / (freq * 1e6)
    out["theory_pw_s"] = theory
    out["eta_pw"] = out["theory_pw_s"] / out["latency_s"]
    return out


@dataclass
class OperatorResult:
    gpu: str
    operator: str
    n_train: int
    n_test: int
    energy_zeroshot: float
    energy_finetuned: float
    energy_scratch: float
    energy_finetuned_median: float
    energy_hybrid: float
    frac_saturated: float
    eta_zeroshot: float
    eta_finetuned: float
    pi_finetuned: float
    #: Fraction of test rows whose true eta falls below the training split's minimum --
    #: rows where the model is extrapolating downward into the division's danger zone.
    frac_eta_below_train: float
    predictions: pd.DataFrame
    #: {"finetuned": TrainHistory, "scratch": TrainHistory}. The test curve is the
    #: held-out card, recorded as a diagnostic; early stopping reads validation only.
    histories: dict = field(default_factory=dict)


def _energy(eta, pi, theory, tdp, eta_floor=1e-6):
    """``E = pi * TDP * C / eta``, with eta held off zero.

    Dividing by a sigmoid is the one genuinely dangerous step in this composition. A
    head that outputs 1e-8 -- entirely possible, since a sigmoid saturates -- turns a
    small efficiency error into an energy prediction eight orders of magnitude out, and
    because MAPE averages ratios, a single such row sets the headline number for its
    whole fold. ``eta_floor`` is not a fudge: it is the statement that the model may not
    predict an efficiency lower than anything it was trained on, which is a bound the
    caller supplies from the training split rather than a constant invented here.
    """
    return pi * tdp * theory / np.maximum(eta, eta_floor)


def _ape(y_true, y_pred):
    """Mean and median absolute percentage error.

    Both, always. MAPE is the project's reporting convention and is what the run-level
    numbers use, but on a target recovered by division it is dominated by whichever row
    had the smallest eta. When mean and median diverge by more than about 3x, the mean
    is describing a handful of rows and the median is describing the model.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12) * 100.0
    return float(np.mean(ape)), float(np.median(ape))


def evaluate_transfer(
    df: pd.DataFrame,
    models_root: str | Path,
    config: TransferConfig | None = None,
    min_rows: int = 40,
    operators: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[OperatorResult]]:
    """Leave-one-GPU-out, one model per (fold, operator).

    ``df`` must have been through :func:`prepare`. Returns a per-(GPU, operator) table
    plus the raw results.
    """
    models_root = Path(models_root)
    cfg = config or TransferConfig()

    # Path first. It is the likeliest thing to be wrong, and a bad one otherwise
    # surfaces much later as an empty result that looks like a data problem.
    if not models_root.is_dir():
        raise FileNotFoundError(
            f"--models points at {models_root}, which does not exist.\n"
            f"\n"
            f"This must be the 'mlp_models' directory of a PipeWeave checkout -- the "
            f"one holding gemm/, attn/, rmsnorm/ and siluandmul/. Clone it with\n"
            f"    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/zksainx/pipeweave\n"
            f"(the LFS skip matters: gemm_train.csv is a 131 MB object this code does "
            f"not need)."
        )

    needed = {"eta_pw", "pi", "theory_pw_s", "tdp_w", "energy_j", "pw_operator", "gpu_key"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"missing {sorted(missing)}")

    df = df.dropna(subset=["eta_pw", "pi", "theory_pw_s", "energy_j"]).copy()
    results: list[OperatorResult] = []
    skipped: list[str] = []
    missing_checkpoints: list[str] = []

    for op, sub in df.groupby("pw_operator"):
        if operators and op not in operators:
            continue
        names = list(PIPEWEAVE_FEATURES[op])
        cols = [f"pw_{n}" for n in names]
        sub = sub.dropna(subset=cols)
        try:
            ckpt, _ = find_checkpoint(models_root, op)
        except FileNotFoundError as e:
            missing_checkpoints.append(f"{op}: {e}")
            continue

        for gpu, te in sub.groupby("gpu_key"):
            tr = sub[sub["gpu_key"] != gpu]
            if len(tr) < min_rows or len(te) < 10:
                skipped.append(
                    f"{op}/{gpu}: {len(tr)} train, {len(te)} test -- below threshold"
                )
                continue

            Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
            Ytr = tr[["eta_pw", "pi"]].to_numpy(float)
            eta_true = te["eta_pw"].to_numpy(float)
            pi_true = te["pi"].to_numpy(float)
            theory = te["theory_pw_s"].to_numpy(float)
            tdp = te["tdp_w"].to_numpy(float)
            e_true = te["energy_j"].to_numpy(float)

            # The model may not claim an efficiency below anything the training split
            # contained. Half the training minimum leaves room to extrapolate a little
            # without letting the division run away.
            eta_floor = max(float(Ytr[:, 0].min()) * 0.5, 1e-8)

            # --- zero-shot: their weights, untouched; pi = training median -------
            zs = TransferModel.from_checkpoint(ckpt, op, cfg)
            eta_zs = zs.predict(Xte)[:, 0]
            pi_zs = np.full(len(te), float(np.median(Ytr[:, 1])))
            e_zs = _energy(eta_zs, pi_zs, theory, tdp, eta_floor)

            # --- fine-tuned ------------------------------------------------------
            Yte_m = te[["eta_pw", "pi"]].to_numpy(float)
            ft = TransferModel.from_checkpoint(ckpt, op, cfg)
            ft.fit(Xtr, Ytr,
                   theory=tr["theory_pw_s"].to_numpy(float),
                   tdp=tr["tdp_w"].to_numpy(float),
                   energy=tr["energy_j"].to_numpy(float),
                   monitor=(Xte, Yte_m))
            ft.history.label = f"{gpu}/{op}/finetuned"
            p_ft = ft.predict(Xte)
            e_ft = _energy(p_ft[:, 0], p_ft[:, 1], theory, tdp, eta_floor)

            # --- from scratch: same architecture, random init ---------------------
            sc = TransferModel(len(names), cfg)
            sc._loaded = True  # deliberate: this is the control, not a transfer
            sc.fit(Xtr, Ytr,
                   theory=tr["theory_pw_s"].to_numpy(float),
                   tdp=tr["tdp_w"].to_numpy(float),
                   energy=tr["energy_j"].to_numpy(float),
                   monitor=(Xte, Yte_m))
            sc.history.label = f"{gpu}/{op}/scratch"
            p_sc = sc.predict(Xte)
            e_sc = _energy(p_sc[:, 0], p_sc[:, 1], theory, tdp, eta_floor)

            # --- hybrid: transfer where it is speaking, scratch where it is not ----
            # A saturated efficiency head has stopped carrying information -- the L4
            # norm case returns a logit of -71.6, i.e. an efficiency of 1e-31, on a
            # kernel whose features are every one of them inside their training range.
            # Composing C/eta from that is not a wrong prediction, it is not a
            # prediction. Where it happens, use the model that was fitted on data
            # resembling the row.
            sat = ft.saturated(Xte)
            e_hy = np.where(sat, e_sc, e_ft)

            preds = te[[c for c in ("gpu_key", "category", "source_model", "kernel_sig")
                        if c in te.columns]].copy()
            preds["operator"] = op
            preds["energy_true"] = e_true
            preds["energy_zeroshot"] = e_zs
            preds["energy_finetuned"] = e_ft
            preds["energy_scratch"] = e_sc
            preds["energy_hybrid"] = e_hy
            preds["saturated"] = sat
            preds["eta_true"], preds["eta_zeroshot"] = eta_true, eta_zs
            preds["eta_finetuned"] = p_ft[:, 0]
            preds["pi_true"], preds["pi_finetuned"] = pi_true, p_ft[:, 1]

            ft_mean, ft_med = _ape(e_true, e_ft)
            results.append(OperatorResult(
                gpu=str(gpu), operator=op, n_train=len(tr), n_test=len(te),
                energy_zeroshot=_ape(e_true, e_zs)[0],
                energy_finetuned=ft_mean,
                energy_scratch=_ape(e_true, e_sc)[0],
                energy_finetuned_median=ft_med,
                energy_hybrid=_ape(e_true, e_hy)[0],
                frac_saturated=float(sat.mean()),
                eta_zeroshot=_ape(eta_true, eta_zs)[1],
                eta_finetuned=_ape(eta_true, p_ft[:, 0])[1],
                pi_finetuned=_ape(pi_true, p_ft[:, 1])[1],
                frac_eta_below_train=float((eta_true < Ytr[:, 0].min()).mean()),
                predictions=preds,
                histories={"finetuned": ft.history, "scratch": sc.history},
            ))

    if missing_checkpoints:
        print(f"evaluate_transfer: no checkpoint for {len(missing_checkpoints)} operator(s)")
        for s in missing_checkpoints:
            print(f"  {s}")
    if skipped:
        print(f"evaluate_transfer: skipped {len(skipped)} (fold, operator) pairs "
              f"for want of rows")
        for s in skipped[:8]:
            print(f"  {s}")

    if not results:
        # Distinguish the two ways this ends up empty. They have completely different
        # fixes, and reporting the wrong one sends you looking at your data when the
        # actual problem is a path.
        if missing_checkpoints:
            raise RuntimeError(
                f"no operator had a checkpoint under {models_root}. Nothing was "
                f"evaluated because there was nothing to transfer from -- check the "
                f"--models path, not the dataset."
            )
        raise RuntimeError(
            f"checkpoints were found, but no (fold, operator) pair had at least "
            f"{min_rows} training rows and 10 test rows. Either the dataset is too "
            f"small to hold a GPU out, or emit_frame dropped the rows -- check its "
            f"failure summary above."
        )
    return _table(results), results


def _table(results: list[OperatorResult]) -> pd.DataFrame:
    """Per (held-out GPU, operator), plus a pooled row.

    ``zeroshot`` / ``finetuned`` / ``scratch`` are mean APE on energy -- the project's
    convention, comparable to the run-level tables. ``ft_med`` is the median for the
    fine-tuned model; when it sits far below ``finetuned``, the mean is being set by a
    few rows with tiny eta rather than by the model. ``oob`` is the fraction of test
    rows whose true efficiency falls below anything in the training split, which is
    where that happens.
    """
    rows = [{
        "gpu": r.gpu, "operator": r.operator, "n_test": r.n_test,
        "zeroshot": r.energy_zeroshot, "finetuned": r.energy_finetuned,
        "scratch": r.energy_scratch, "hybrid": r.energy_hybrid,
        "ft_med": r.energy_finetuned_median, "sat": r.frac_saturated,
        "eta_zs": r.eta_zeroshot, "eta_ft": r.eta_finetuned, "pi_ft": r.pi_finetuned,
        "oob": r.frac_eta_below_train,
    } for r in results]
    tab = pd.DataFrame(rows).set_index(["gpu", "operator"])

    allp = pd.concat([r.predictions for r in results], ignore_index=True)
    pooled = pd.Series({
        "n_test": len(allp),
        "zeroshot": _ape(allp["energy_true"], allp["energy_zeroshot"])[0],
        "finetuned": _ape(allp["energy_true"], allp["energy_finetuned"])[0],
        "scratch": _ape(allp["energy_true"], allp["energy_scratch"])[0],
        "hybrid": _ape(allp["energy_true"], allp["energy_hybrid"])[0],
        "sat": float(np.average([r.frac_saturated for r in results],
                                weights=[r.n_test for r in results])),
        "ft_med": _ape(allp["energy_true"], allp["energy_finetuned"])[1],
        "eta_zs": _ape(allp["eta_true"], allp["eta_zeroshot"])[1],
        "eta_ft": _ape(allp["eta_true"], allp["eta_finetuned"])[1],
        "pi_ft": _ape(allp["pi_true"], allp["pi_finetuned"])[1],
        "oob": float(np.average([r.frac_eta_below_train for r in results],
                                weights=[r.n_test for r in results])),
    })
    tab.loc[("POOLED", "-"), :] = pooled
    tab["n_test"] = tab["n_test"].astype(int)
    return tab.round(2)


def range_report(df: pd.DataFrame, models_root: str | Path) -> pd.DataFrame:
    """How far outside each checkpoint's training range this project's kernels sit.

    Every ``metadata.json`` records per-feature ``min``/``max``/``mean``/``std`` over
    the training split. A transferred model asked for a prediction outside those bounds
    is extrapolating, and the honest thing is to know by how much before reading any
    fold result -- a checkpoint that has never seen a feature value cannot be blamed for
    what it does with one.

    Returns, per (operator, feature): the fraction of rows below their min, above their
    max, and how many standard deviations the median sits from their mean.
    """
    import json

    models_root = Path(models_root)
    rows = []
    for op, sub in df.groupby("pw_operator"):
        try:
            _, meta = find_checkpoint(models_root, op)
        except FileNotFoundError:
            continue
        ranges = meta.get("feature_ranges", {})
        for name in PIPEWEAVE_FEATURES[op]:
            col, r = f"pw_{name}", ranges.get(name)
            if col not in sub or not r:
                continue
            v = sub[col].dropna().to_numpy(float)
            if not len(v):
                continue
            rows.append({
                "operator": op, "feature": name, "n": len(v),
                "frac_below_min": float((v < r["min"]).mean()),
                "frac_above_max": float((v > r["max"]).mean()),
                "median_z": float((np.median(v) - r["mean"]) / max(r["std"], 1e-30)),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index(["operator", "feature"]).round(3)
    return out


def compare_floors(df: pd.DataFrame) -> pd.DataFrame:
    """This project's analytical floor against PipeWeave's, per GPU and category.

    Worth running before anything else. Two floors that disagree by a large factor mean
    the two ``eta`` columns are not comparable, and any narrative that moves between
    them is confused. A ratio far from 1 on one card and near 1 on another points at a
    hardware constant, not at a kernel model.

    ``eta > 1`` is the diagnostic to watch: it says the floor exceeds the measured time,
    which is impossible, so the floor is wrong somewhere.
    """
    if "theoretical_time_s" not in df or "theory_pw_s" not in df:
        raise KeyError("need both theoretical_time_s (ours) and theory_pw_s (theirs)")
    d = df.dropna(subset=["theoretical_time_s", "theory_pw_s"]).copy()
    d["floor_ratio"] = d["theory_pw_s"] / d["theoretical_time_s"]
    g = d.groupby(["gpu_key", "category"])
    return pd.DataFrame({
        "n": g.size(),
        "ratio_median": g["floor_ratio"].median(),
        "ratio_p10": g["floor_ratio"].quantile(0.10),
        "ratio_p90": g["floor_ratio"].quantile(0.90),
        "eta_ours_over_1": g.apply(
            lambda x: float((x["eta"] > 1).mean()) if "eta" in x else np.nan,
            include_groups=False),
        "eta_pw_over_1": g.apply(
            lambda x: float((x["eta_pw"] > 1).mean()) if "eta_pw" in x else np.nan,
            include_groups=False),
    }).round(3)
