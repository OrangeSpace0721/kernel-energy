"""Is this kernel one their checkpoint can speak to?

The per-feature range check in :func:`kernelenergy.pipeweave.evaluate.range_report` is
not enough, and the way it failed is worth stating precisely, because it is the standard
way an out-of-distribution check gives false comfort.

An L4 LayerNorm at 4096x3072 emits fifteen features that are **every one of them inside**
PipeWeave's per-feature training range -- each sits at about 1% of that feature's
maximum, with z-scores around -0.25. The checkpoint returns an efficiency of
7.8e-32. Its logit is -71.6, against -4.75 for the identical kernel on an A100.

Marginally in range, jointly impossible. The L4 moves 300 GB/s at 2040 MHz; the most
bandwidth-starved card PipeWeave ever trained on is the A40 at 696 GB/s and 1740 MHz. So
for the same kernel the L4's ``global_cycle`` is 9.3x the A100's where their worst
training case reaches 3.4x, and the ratio ``global_cycle / fma_all_cycle`` is 321 against
their 32. No single feature is unusual. The *arithmetic intensity* is one they have never
seen, and that is a property of the combination.

The fix is a joint check: Mahalanobis distance in the same ``log1p`` space the model
reads, against the mean and covariance of the operator's own training split.

    d(x) = sqrt( (x - mu)' P (x - mu) )        P = inv(Cov)

``data/training_moments.json`` carries ``mu`` and ``P`` per operator, computed over all
307,229 rows of the splits PipeWeave ships -- 19 KB for what would otherwise be 260 MB of
CSV. The covariance is ridged at 1e-6 of its mean eigenvalue before inversion because
several of their features are near-collinear by construction (each ``sm_max_*`` differs
from its ``all_*`` counterpart by a fixed hardware factor), which makes the raw
covariance singular and its inverse meaningless.

Reference distances from their own training data:

======  ======  ====  ====  ====  =====
op      n       p50   p95   p99   max
======  ======  ====  ====  ====  =====
gemm    118800  2.57  4.21  5.46  16.39
attn     71969  2.78  5.10  7.78  33.57
rmsnorm  44592  2.62  6.00  10.17 54.04
silu     71868  2.61  4.57  9.47  61.28
======  ======  ====  ====  ====  =====

A kernel past the operator's ``p99`` is somewhere its checkpoint has essentially no
evidence. That is not automatically wrong -- extrapolation is what a model is for -- but
it is where a prediction should stop being believed without a second opinion, and it is
where :mod:`kernelenergy.pipeweave.evaluate` falls back to the from-scratch model.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from kernelenergy.pipeweave.features import PIPEWEAVE_FEATURES

__all__ = ["MOMENTS_PATH", "load_moments", "mahalanobis", "ood_report", "SATURATED_LOGIT"]

MOMENTS_PATH = Path(__file__).resolve().parent / "data" / "training_moments.json"

#: A logit past this is a saturated sigmoid, and the prediction carries no information.
#: ``sigmoid(-12)`` is 6e-6; the L4 norm case reached -71.6. Cheap to check and needs no
#: reference data, so it is worth reporting even when the moments file is absent.
SATURATED_LOGIT = 12.0


@functools.lru_cache(maxsize=1)
def load_moments(path: str | Path | None = None) -> dict:
    p = MOMENTS_PATH if path is None else Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"training moments missing from {p}. Rebuild from a PipeWeave checkout, or "
            f"check that a 'data/' .gitignore rule has not swallowed it again."
        )
    return json.loads(p.read_text())


def mahalanobis(operator: str, X: np.ndarray) -> np.ndarray:
    """Distance of each row of ``X`` from the operator's training distribution.

    ``X`` is raw PipeWeave features in :data:`PIPEWEAVE_FEATURES` order -- the ``log1p``
    is applied here, because the distance must be measured in the space the model
    actually reads.
    """
    # GroupNorm shares RMSNorm's feature vector, so its distance is measured against
    # the RMSNorm distribution -- which is the point: that distance is how we know it
    # does not belong there.
    m = load_moments()["rmsnorm" if operator == "groupnorm" else operator]
    names = list(PIPEWEAVE_FEATURES[operator])
    if list(m["features"]) != names:
        raise ValueError(
            f"{operator}: moments were built for {m['features']}, features are {names}"
        )
    X = np.log1p(np.atleast_2d(np.asarray(X, float)))
    d = X - np.asarray(m["mean"])
    P = np.asarray(m["precision"])
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, P, d), 0.0))


def ood_report(df, models_root=None, quantile: str = "md_p99"):
    """Per (operator, GPU): how far outside the training distribution these kernels are.

    Reports the fraction past the operator's training ``p99`` -- the threshold beyond
    which the checkpoint has almost no evidence -- alongside the median distance, so a
    fold that scores badly can be read as "the model is wrong here" or "the model was
    never asked a question like this" rather than the two being conflated.
    """
    import pandas as pd

    rows = []
    for (op, gpu), g in df.groupby(["pw_operator", "gpu_key"]):
        names = list(PIPEWEAVE_FEATURES[op])
        cols = [f"pw_{n}" for n in names]
        if not set(cols) <= set(g.columns):
            continue
        sub = g.dropna(subset=cols)
        if not len(sub):
            continue
        d = mahalanobis(op, sub[cols].to_numpy(float))
        ref = load_moments()["rmsnorm" if op == "groupnorm" else op]
        rows.append({
            "operator": op, "gpu_key": gpu, "n": len(sub),
            "md_median": float(np.median(d)),
            "md_p95": float(np.quantile(d, 0.95)),
            "train_p99": ref["md_p99"],
            "frac_beyond_train_p99": float((d > ref[quantile]).mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index(["operator", "gpu_key"]).sort_values(
            "frac_beyond_train_p99", ascending=False).round(3)
    return out
