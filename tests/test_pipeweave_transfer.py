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
def test_mape_collapses_on_an_unfittable_target_and_log_does_not():
    """The failure that made a real fold score worse than doing nothing.

    MAPE is asymmetric: over-predicting costs ``pred/true - 1``, unbounded; under-
    predicting costs at most 1. So when the target carries variance the features cannot
    explain -- the normal condition when transferring to a different kernel population
    -- the loss is minimised by hedging *downward*, and on a target spanning three
    orders of magnitude the prediction goes most of the way to zero. It presents as an
    eta APE of almost exactly 100%, which reads like a head that learned nothing rather
    than one that learned to hedge.

    It never bit PipeWeave: their features explain their targets at R^2 0.97, so there
    is nothing to hedge against. It bites immediately on transfer.

    The log loss is symmetric in ratio and has no such fixed point. This test pins the
    property that matters in practice: **fine-tuning must not end up predicting further
    from the truth than the untouched checkpoint did.**
    """
    path, _ = find_checkpoint(ROOT / "mlp_models", "rmsnorm")
    df = pd.read_csv(FIXTURES / "pipeweave_rmsnorm.csv.gz")
    names = list(PIPEWEAVE_FEATURES["rmsnorm"])
    X = df[names].to_numpy(float)

    rng = np.random.default_rng(0)
    # Heavy lognormal noise: the features still describe the centre, nothing describes
    # the spread.
    eta = np.clip(df["overall_perf"].to_numpy(float)
                  * np.exp(1.5 * rng.standard_normal(len(df))), 1e-6, 0.999)
    pi = np.clip(0.45 + 0.02 * rng.standard_normal(len(df)), 0.05, 0.95)
    Y = np.column_stack([eta, pi])
    tr = (df["hardware"] != "NVIDIA H200").to_numpy()
    te = ~tr

    zero_shot = TransferModel.from_checkpoint(path, "rmsnorm").predict(X[te])[:, 0]

    fitted = {}
    for loss in ("mape", "log"):
        m = TransferModel.from_checkpoint(
            path, "rmsnorm", TransferConfig(seed=0, loss=loss))
        m.fit(X[tr], Y[tr])
        fitted[loss] = m.predict(X[te])[:, 0]

    # MAPE hedges the prediction downward, away from where the checkpoint had it.
    assert fitted["mape"].mean() < zero_shot.mean() * 0.75, (
        "MAPE no longer collapses -- if the loss or the schedule changed such that this "
        "is genuinely fixed, delete this test rather than loosening it."
    )
    # The log loss stays with the checkpoint.
    assert fitted["log"].mean() > zero_shot.mean() * 0.75, (
        f"the log loss collapsed too: {fitted['log'].mean():.3e} against a zero-shot "
        f"{zero_shot.mean():.3e}. Fine-tuning must not move the prediction further from "
        f"the truth than leaving the checkpoint alone would."
    )


@needs_upstream
def test_gradient_clipping_bounds_the_update():
    """Present in their train_mlp.py, absent from the first version of this module."""
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    model = TransferModel.from_checkpoint(
        path, "gemm", TransferConfig(seed=0, grad_clip=1.0))
    X = np.log1p(pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz")
                 [list(PIPEWEAVE_FEATURES["gemm"])].to_numpy(float)[:64])
    yhat = model._forward(X, training=True)
    # A deliberately enormous incoming gradient.
    model._backward(np.full_like(yhat, 1e6))
    model._clip_gradients()
    grads = [gf() for _, gf, _ in model._trunk_params() + model._head_params("both")]
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    assert total <= 1.0 + 1e-6, f"global gradient norm {total:.3f} after clipping to 1.0"


@needs_upstream
def test_warmup_only_leaves_the_efficiency_head_exactly_as_released():
    """``--warmup-only``: fit the power head, touch nothing they trained."""
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz").sample(n=300, random_state=4)
    X = df[list(PIPEWEAVE_FEATURES["gemm"])].to_numpy(float)
    Y = np.column_stack([df["overall_perf"].to_numpy(float), np.full(len(df), 0.4)])

    model = TransferModel.from_checkpoint(
        path, "gemm", TransferConfig(seed=0, warmup_epochs=30, max_epochs=0))
    eta_before = model.predict(X)[:, 0]
    w_before = model.dense[0].w.copy()
    model.fit(X, Y)

    np.testing.assert_allclose(model.predict(X)[:, 0], eta_before, rtol=1e-9)
    np.testing.assert_allclose(model.dense[0].w, w_before, rtol=0, atol=0)


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


# --------------------------------------------------------------------------- #
# Knowing when the checkpoint has nothing to say
# --------------------------------------------------------------------------- #


@needs_upstream
def test_l4_layernorm_saturates_the_efficiency_head():
    """The failure that made rmsnorm score 568% and set the pooled headline.

    An L4 LayerNorm at 4096x3072 emits fifteen features that are every one of them
    inside PipeWeave's per-feature training range -- each about 1% of that feature's
    maximum, z-scores near -0.25. The checkpoint returns an efficiency of 7.8e-32.

    Marginally in range, jointly impossible. The L4 moves 300 GB/s at 2040 MHz; their
    most bandwidth-starved training card is the A40 at 696 GB/s and 1740 MHz. For the
    same kernel the L4's ``global_cycle`` is 9.3x the A100's, where their worst training
    case reaches 3.4x. No feature is unusual; the arithmetic intensity is.

    This pins the detection, not the failure -- the failure is a property of their
    training set and cannot be fixed from here.
    """
    from kernelenergy.kernels.base import KernelConfig
    from kernelenergy.pipeweave.features import emit

    path, meta = find_checkpoint(ROOT / "mlp_models", "rmsnorm")
    model = TransferModel.from_checkpoint(path, "rmsnorm")
    cfg = KernelConfig(category="norm", dtype="bf16",
                       params={"rows": 4096, "dim": 3072, "kind": "layer"})

    l4 = emit(cfg, "L4").features[None, :]
    a100 = emit(cfg, "A100_PCIE").features[None, :]

    # Every L4 feature is inside their range -- this is what made the marginal check
    # useless, and it must stay true or the test is no longer about what it says.
    ranges = meta["feature_ranges"]
    for name, value in zip(PIPEWEAVE_FEATURES["rmsnorm"], l4[0]):
        r = ranges[name]
        assert r["min"] <= value <= r["max"], f"{name} left their range; rewrite this test"

    assert model.saturated(l4)[0], "the L4 norm case no longer saturates"
    assert not model.saturated(a100)[0], "the A100 norm case should be fine"
    assert model.logits(l4)[0, 0] < -30
    assert model.logits(a100)[0, 0] > -10


def test_mahalanobis_ranks_l4_furthest_from_their_training_data():
    """The joint check, which sees what the marginal one cannot.

    It is a diagnostic, not a gate: L4 comes out 2.5x further from the training centre
    than any other card, which is the right signal, but it does not clear rmsnorm's
    training p99 of 10.17 because their own rmsnorm tail runs to 54. The reliable gate
    is the saturated logit; this explains *why*.
    """
    from kernelenergy.kernels.base import KernelConfig
    from kernelenergy.pipeweave.features import emit
    from kernelenergy.pipeweave.ood import mahalanobis

    cfg = KernelConfig(category="norm", dtype="bf16",
                       params={"rows": 4096, "dim": 3072, "kind": "layer"})
    d = {k: float(mahalanobis("rmsnorm", emit(cfg, k).features[None, :])[0])
         for k in ("A100_PCIE", "H100", "L40S", "L4")}
    assert d["L4"] == max(d.values())
    assert d["L4"] > 2.0 * max(v for k, v in d.items() if k != "L4")


def test_training_moments_cover_every_operator():
    from kernelenergy.pipeweave.ood import load_moments

    m = load_moments()
    for op, names in PIPEWEAVE_FEATURES.items():
        assert op in m, f"no training moments for {op}"
        assert list(m[op]["features"]) == list(names)
        assert len(m[op]["mean"]) == len(names)
        assert np.array(m[op]["precision"]).shape == (len(names), len(names))
        assert m[op]["md_p50"] < m[op]["md_p99"] < m[op]["md_max"]


@needs_upstream
def test_the_warmup_state_is_kept_when_unfreezing_makes_things_worse():
    """Best-epoch selection spans both stages, and that is load-bearing.

    ``fit`` used to reset the incumbent validation loss when the trunk was unfrozen.
    Stage 2's first epoch then always beat an infinite incumbent and overwrote the
    saved weights, so the warmup state could never be recovered -- even when it was
    strictly better. On real data that showed up as fine-tuning scoring worse than
    leaving the checkpoint alone, and ``--warmup-only`` had to be added by hand to get
    back the configuration this loop should have chosen by itself.

    Both stages share a validation split, a loss and a metric, so they are directly
    comparable and one global best is the correct rule. Given a target the features
    cannot explain, unfreezing cannot help, and the selected epoch must land in the
    warmup.
    """
    path, _ = find_checkpoint(ROOT / "mlp_models", "rmsnorm")
    df = pd.read_csv(FIXTURES / "pipeweave_rmsnorm.csv.gz")
    X = df[list(PIPEWEAVE_FEATURES["rmsnorm"])].to_numpy(float)

    rng = np.random.default_rng(0)
    eta = np.clip(df["overall_perf"].to_numpy(float)
                  * np.exp(2.0 * rng.standard_normal(len(df))), 1e-6, 0.999)
    pi = np.clip(0.45 + 0.02 * rng.standard_normal(len(df)), 0.05, 0.95)
    tr = (df["hardware"] != "NVIDIA H200").to_numpy()

    warmup = 60
    m = TransferModel.from_checkpoint(
        path, "rmsnorm",
        TransferConfig(warmup_epochs=warmup, max_epochs=200, seed=0))
    m.fit(X[tr], np.column_stack([eta, pi])[tr])

    assert m.history.stage_starts == [0, warmup]
    assert m.history.best_epoch < warmup, (
        f"best epoch {m.history.best_epoch} is in stage 2, but unfreezing the trunk "
        f"cannot help on a target the features do not explain. The cross-stage best "
        f"is being reset again."
    )
    # And the returned weights really are that epoch's, not the last one's.
    assert m.history.best_val <= min(m.history.val_loss) + 1e-9


@needs_upstream
def test_best_epoch_is_the_lowest_validation_loss_and_nothing_else():
    """Not train, not test, not the last epoch."""
    path, _ = find_checkpoint(ROOT / "mlp_models", "gemm")
    df = pd.read_csv(FIXTURES / "pipeweave_gemm.csv.gz").sample(n=600, random_state=11)
    X = df[list(PIPEWEAVE_FEATURES["gemm"])].to_numpy(float)
    Y = np.column_stack([df["overall_perf"].to_numpy(float), np.full(len(df), 0.4)])

    m = TransferModel.from_checkpoint(
        path, "gemm", TransferConfig(warmup_epochs=20, max_epochs=60, seed=3))
    m.fit(X[:500], Y[:500], monitor=(X[500:], Y[500:]))
    h = m.history
    assert h.val_loss[h.best_epoch] == pytest.approx(min(h.val_loss), abs=1e-9)
    assert h.best_val == pytest.approx(min(h.val_loss), abs=1e-9)
