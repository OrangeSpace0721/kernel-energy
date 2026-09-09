"""The transfer path itself: read their weights, run them, fine-tune them.

``test_pipeweave_parity.py`` proves the *features* are theirs. This file proves the
*model* is theirs -- that the forward pass reproduces their published numbers, which is
the only way to know the weights were loaded into the right layer order with the right
preprocessing. A transposed weight or a BatchNorm on the wrong side of the ReLU still
runs, still produces values in (0, 1), and is simply wrong.

Requires a PipeWeave checkout for the checkpoints. Set ``PIPEWEAVE_ROOT`` or place one
at ``../pipeweave``; the tests skip rather than fail without it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kernelenergy.pipeweave.checkpoint import load_metadata, load_state_dict
from kernelenergy.pipeweave.features import PIPEWEAVE_FEATURES
from kernelenergy.pipeweave.transfer import (
    REFERENCE_CYCLE_COLUMNS,
    TransferConfig,
    TransferModel,
    find_checkpoint,
    reference_cycles,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _root() -> Path | None:
    for cand in [os.environ.get("PIPEWEAVE_ROOT"),
                 Path(__file__).resolve().parents[2] / "pipeweave"]:
        if cand and Path(cand).is_dir() and (Path(cand) / "mlp_models").is_dir():
            return Path(cand)
    return None


ROOT = _root()
needs_upstream = pytest.mark.skipif(
    ROOT is None,
    reason="no PipeWeave checkout found; set PIPEWEAVE_ROOT to enable",
)

OPERATORS = ["gemm", "attn", "rmsnorm", "siluandmul"]


# --------------------------------------------------------------------------- #
# Their target is our eta
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("operator", OPERATORS)
def test_overall_perf_is_theoretical_over_measured(operator):
    """``overall_perf`` is an efficiency ratio, which is what makes transfer possible.

    Their paper does not spell out the denominator. Their data does: for every row of
    all four datasets,

        overall_perf * avg_duration_us * sm_freq_MHz == reference_cycles

    with ``reference_cycles`` the tensor pipe for GEMM and attention, and the sum of the
    FMA and XU pipes for the two elementwise-shaped operators. So their scalar is
    ``t_theoretical / t_measured`` against their own analytical floor -- the same
    quantity this project calls eta, differing only in whose floor is in the numerator.

    If this ever fails, the transferred efficiency head is predicting something other
    than eta and composing it into an energy is meaningless.
    """
    df = pd.read_csv(FIXTURES / f"pipeweave_{operator}.csv.gz")
    cycles = sum(df[c].to_numpy(float) for c in REFERENCE_CYCLE_COLUMNS[operator])
    implied = cycles / (df["avg_duration"].to_numpy(float) * df["sm_freq"].to_numpy(float))
    rel = np.abs(implied - df["overall_perf"].to_numpy(float)) / df["overall_perf"].to_numpy(float)
    assert rel.max() < 1e-9, f"max deviation {rel.max():.3e}"


def test_reference_cycles_accepts_both_dict_and_array():
    names = PIPEWEAVE_FEATURES["rmsnorm"]
    vec = np.arange(len(names), dtype=float)
    as_dict = dict(zip(names, vec))
    assert reference_cycles("rmsnorm", as_dict) == pytest.approx(
        reference_cycles("rmsnorm", vec, names)
    )


# --------------------------------------------------------------------------- #
# Reading their checkpoints without torch
# --------------------------------------------------------------------------- #


@needs_upstream
@pytest.mark.parametrize("operator", OPERATORS)
def test_checkpoint_loads_and_has_the_expected_shape(operator):
    path, meta = find_checkpoint(ROOT / "mlp_models", operator)
    state = load_state_dict(path)
    n_in = state["network.0.weight"].shape[1]
    assert n_in == len(PIPEWEAVE_FEATURES[operator]), (
        f"{operator}: checkpoint wants {n_in} features, PIPEWEAVE_FEATURES lists "
        f"{len(PIPEWEAVE_FEATURES[operator])}"
    )
    if meta:
        assert list(meta["model_info"]["features"]) == list(PIPEWEAVE_FEATURES[operator]), (
            f"{operator}: feature *order* differs from the checkpoint's metadata. The "
            f"order is load-bearing -- these go straight into a Linear layer."
        )
    # Indices 0,2 / 4,6 / 8,10 / 12: Linear, BatchNorm, ..., output Linear.
    for key in ["network.0.weight", "network.2.running_mean", "network.4.weight",
                "network.6.running_var", "network.8.weight", "network.10.weight",
                "network.12.weight"]:
        assert key in state, f"{operator}: missing {key}"
    assert all(np.isfinite(v).all() for v in state.values())
    assert load_metadata(path).get("epoch", 0) > 0


@needs_upstream
@pytest.mark.parametrize("operator", OPERATORS)
def test_forward_reproduces_upstream_metrics(operator):
    """Run their weights on their data and match the numbers in their metadata.

    This is the end-to-end check on the whole reading path: unpickling, the (out, in)
    to (in, out) weight transpose, log1p with no standardisation, Linear -> ReLU ->
    BatchNorm -> Dropout ordering, BatchNorm in inference mode with eps 1e-5, and the
    sigmoid output. Any one of those wrong moves R2 by far more than the tolerance here.

    The fixtures are stratified samples rather than their full test split, so the
    tolerance is loose enough to absorb sampling: 0.03 on R2. Run against their full
    ``dataset/*_test.csv`` and the agreement is to five decimal places.
    """
    path, meta = find_checkpoint(ROOT / "mlp_models", operator)
    if not meta:
        pytest.skip("checkpoint has no metadata.json to compare against")
    model = TransferModel.from_checkpoint(path, operator)

    df = pd.read_csv(FIXTURES / f"pipeweave_{operator}.csv.gz")
    X = df[list(PIPEWEAVE_FEATURES[operator])].to_numpy(float)
    y = df["overall_perf"].to_numpy(float)
    p = model.predict_overall_perf(X)

    r2 = 1.0 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    theirs = meta["evaluation_results"]["overall"]["r2"]
    assert r2 == pytest.approx(theirs, abs=0.03), (
        f"{operator}: R2 {r2:.5f} against their published {theirs:.5f}. The weights "
        f"are being read or applied differently than they were trained."
    )
    assert p.min() > 0.0 and p.max() < 1.0


@needs_upstream
def test_wrong_feature_count_is_refused_not_absorbed():
    """The failure this whole file exists to prevent: a silently wrong feature vector."""
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    with pytest.raises(ValueError, match="features"):
        TransferModel.from_checkpoint(path, "rmsnorm")  # 15-feature operator, 11-in weights


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #


@needs_upstream
def test_finetuning_moves_pi_without_destroying_eta():
    """The transfer's core requirement, stated as a test.

    Fine-tune on a synthetic energy target built from their own rows: eta is their
    recorded ``overall_perf``, pi is a smooth function of the features. Afterwards the
    pi head must have learned something, and the efficiency head must still be roughly
    as good as it started -- a transfer that forgets what it transferred is worthless.
    """
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz").sample(n=900, random_state=3)
    names = list(PIPEWEAVE_FEATURES["gemm"])
    X = df[names].to_numpy(float)
    eta = df["overall_perf"].to_numpy(float)

    # A plausible pi: memory-bound kernels draw less than compute-bound ones.
    ratio = df["global_cycle"].to_numpy(float) / np.maximum(
        df["tensor_all_cycle"].to_numpy(float), 1.0)
    pi = np.clip(0.75 - 0.15 * np.tanh(np.log1p(ratio)), 0.05, 0.99)
    Y = np.column_stack([eta, pi])

    before = TransferModel.from_checkpoint(path, "gemm")
    eta_before = before.predict(X)[:, 0]
    pi_before = before.predict(X)[:, 1]

    model = TransferModel.from_checkpoint(
        path, "gemm",
        TransferConfig(warmup_epochs=40, max_epochs=60, patience=30, seed=1),
    )
    model.fit(X, Y)
    eta_after, pi_after = model.predict(X)[:, 0], model.predict(X)[:, 1]

    def err(pred, true):
        return float(np.mean(np.abs(pred - true) / true))

    assert err(pi_after, pi) < err(pi_before, pi) * 0.6, (
        "the pi head learned little: "
        f"{err(pi_before, pi):.4f} -> {err(pi_after, pi):.4f}"
    )
    assert err(eta_after, eta) < err(eta_before, eta) * 1.5 + 0.02, (
        "fine-tuning degraded the transferred efficiency head: "
        f"{err(eta_before, eta):.4f} -> {err(eta_after, eta):.4f}. Lower "
        "trunk_lr_scale, or check that freeze_bn is on."
    )


@needs_upstream
def test_fit_refuses_to_train_random_weights():
    model = TransferModel(11)
    with pytest.raises(RuntimeError, match="from_checkpoint"):
        model.fit(np.ones((8, 11)), np.full((8, 2), 0.5))


@needs_upstream
def test_frozen_batchnorm_keeps_running_statistics():
    """With a few hundred rows, BatchNorm statistics are the thing most easily ruined."""
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz").sample(n=400, random_state=5)
    X = df[list(PIPEWEAVE_FEATURES["gemm"])].to_numpy(float)
    Y = np.column_stack([df["overall_perf"].to_numpy(float), np.full(len(df), 0.4)])

    model = TransferModel.from_checkpoint(
        path, "gemm", TransferConfig(warmup_epochs=5, max_epochs=10, seed=2))
    original = model.bn[0].run_mean.copy()
    model.fit(X, Y)
    np.testing.assert_allclose(model.bn[0].run_mean, original, rtol=0, atol=0)
