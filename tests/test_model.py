"""Tests for the modelling half: targets, estimator, folds.

Runs on a small synthetic set generated in-process, so it needs no GPU and no data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kernelenergy.hardware import GPUS
from kernelenergy.kernels.base import KernelConfig
from kernelenergy.kernels.registry import make_kernel
from kernelenergy.model.estimator import MLP, TrainConfig, mape
from kernelenergy.model.evaluate import evaluate
from kernelenergy.model.features import FEATURE_COLUMNS, analyse
from kernelenergy.model.targets import (
    add_targets,
    energy_from_heads,
    latency_from_eta,
    validate_efficiency,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = ["flux1-dev", "sd35-large", "qwen-image"]
    rows = []
    for mi, model in enumerate(models):
        dim = [3072, 2432, 3072][mi]
        for tok in (1024, 2304, 4096, 4608):
            cfgs = [
                KernelConfig("gemm", "bf16", {"m": tok, "n": dim * 3, "k": dim}),
                KernelConfig("gemm", "bf16", {"m": tok, "n": dim, "k": dim * 4}),
                KernelConfig("attention", "bf16",
                             {"b": 1, "h": 24, "s_q": tok, "s_kv": tok, "d": 128}),
                KernelConfig("norm", "bf16", {"rows": tok, "dim": dim}),
                KernelConfig("elementwise", "bf16",
                             {"n_elem": tok * dim, "kind": "silu"}),
            ]
            for cfg in cfgs:
                for key, gpu in GPUS.items():
                    res = analyse(make_kernel(cfg), gpu)
                    f = res.features
                    eta = float(np.clip(
                        1 / (1 + np.exp(-(-0.5 + 0.3 * f["sm_occupancy"]
                                          + 0.15 * np.tanh(f["log_n_waves"] / 3))))
                        * np.exp(rng.normal(0, 0.04)), 1e-3, 0.99))
                    pi = float(np.clip(
                        1 / (1 + np.exp(-(0.3 + 0.4 * f["uses_tensor_core"]
                                          - 0.2 * np.tanh(f["log_bytes_per_flop"] / 4))))
                        * np.exp(rng.normal(0, 0.04)), 1e-3, 0.99))
                    lat = res.theoretical_time_s / eta
                    r = dict(f)
                    r.update(
                        kernel_sig=cfg.signature(), category=cfg.category,
                        dtype=cfg.dtype, gpu_key=key, source_model=model,
                        latency_s=lat, energy_j=pi * gpu.tdp_w * lat,
                        theoretical_time_s=res.theoretical_time_s,
                        analytic_flops=res.flops,
                        analytic_bytes_global=res.bytes_global,
                        tdp_w=gpu.tdp_w, idle_power_w=gpu.idle_power_w,
                    )
                    rows.append(r)
    return add_targets(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def frame():
    return _synthetic_frame()


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def test_targets_round_trip_exactly(frame):
    """E = pi * TDP * C / eta must invert the definitions with no drift."""
    e = energy_from_heads(frame["eta"], frame["pi"], frame["theoretical_time_s"],
                          frame["tdp_w"])
    np.testing.assert_allclose(e, frame["energy_j"], rtol=1e-9)
    t = latency_from_eta(frame["eta"], frame["theoretical_time_s"])
    np.testing.assert_allclose(t, frame["latency_s"], rtol=1e-9)


def test_targets_are_bounded(frame):
    assert (frame["eta"] > 0).all() and (frame["eta"] <= 1.0).all()
    assert (frame["pi"] > 0).all() and (frame["pi"] <= 1.0).all()


def test_power_fraction_is_energy_over_time_over_tdp(frame):
    np.testing.assert_allclose(
        frame["pi"], frame["energy_j"] / frame["latency_s"] / frame["tdp_w"], rtol=1e-9
    )


def test_dynamic_fraction_amplifies_idle_error():
    """The stated reason for defaulting to pi over pi_dynamic, made a test.

    A 2 W error in the idle estimate must move pi_dynamic more than pi -- most sharply
    on the smallest card, where the floor is the largest share of the budget.
    """
    base = pd.DataFrame({
        "latency_s": [1e-3], "energy_j": [0.04], "theoretical_time_s": [4e-4],
        "tdp_w": [72.0], "idle_power_w": [15.0],
    })
    shifted = base.assign(idle_power_w=17.0)
    a, b = add_targets(base), add_targets(shifted)
    d_pi = abs(float(a["pi"].iloc[0] - b["pi"].iloc[0]))
    d_dyn = abs(float(a["pi_dynamic"].iloc[0] - b["pi_dynamic"].iloc[0]))
    assert d_pi == 0.0, "pi does not depend on the idle estimate at all"
    assert d_dyn > 0.01


def test_validate_efficiency_flags_impossible_rows():
    df = pd.DataFrame({
        "eta": [0.4, 0.5, 1.7], "category": ["gemm"] * 3, "gpu_key": ["H100"] * 3,
    })
    rep = validate_efficiency(df)
    assert rep["frac_over_1"].iloc[0] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #


def test_mlp_learns_a_known_function():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3])))
    Y = np.column_stack([y, np.clip(y * 0.8 + 0.1, 1e-3, 0.99)])
    m = MLP(6, 2, TrainConfig(max_epochs=250, patience=40, seed=0)).fit(X, Y)
    pred = m.predict(X)
    assert mape(Y[:, 0], pred[:, 0]) < 8.0
    assert mape(Y[:, 1], pred[:, 1]) < 8.0


def test_mlp_output_is_bounded():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5)) * 50  # deliberately extreme inputs
    Y = rng.uniform(0.1, 0.9, size=(200, 2))
    m = MLP(5, 2, TrainConfig(max_epochs=40, seed=0)).fit(X, Y)
    p = m.predict(rng.normal(size=(50, 5)) * 500)
    assert ((p > 0) & (p < 1)).all(), "a sigmoid head must never leave (0, 1)"


def test_mlp_rejects_nonpositive_targets():
    X = np.random.default_rng(0).normal(size=(50, 3))
    Y = np.zeros((50, 2))
    with pytest.raises(ValueError, match="strictly positive"):
        MLP(3, 2, TrainConfig(max_epochs=5)).fit(X, Y)


def test_mlp_handles_constant_features():
    """A pipeline no kernel in the fold touches gives a zero-variance column."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[:, 2] = 7.0
    Y = np.clip(rng.uniform(0.2, 0.8, size=(200, 2)), 1e-3, 0.99)
    m = MLP(4, 2, TrainConfig(max_epochs=20, seed=0)).fit(X, Y)
    assert np.isfinite(m.predict(X)).all()


def test_early_stopping_restores_the_best_epoch():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    Y = np.clip(rng.uniform(0.2, 0.8, size=(300, 2)), 1e-3, 0.99)
    m = MLP(4, 2, TrainConfig(max_epochs=120, patience=10, seed=0)).fit(X, Y)
    assert m.history.best_epoch >= 0
    assert m.history.best_val == pytest.approx(min(m.history.val_loss))


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fold", ["hardware", "architecture", "category"])
def test_folds_run_and_report_both_summaries(frame, fold):
    cfg = TrainConfig(max_epochs=60, patience=15, seed=0)
    tab, results = evaluate(frame, fold=fold, config=cfg)
    assert "MEAN" in tab.index and "POOLED" in tab.index
    assert (tab.loc["POOLED", "n_test"] == sum(r.n_test for r in results))
    assert tab["energy"].notna().all()


def test_held_out_group_is_absent_from_training(frame):
    """The whole point of the fold. A leak here would make every number meaningless."""
    _, results = evaluate(frame, fold="hardware",
                          config=TrainConfig(max_epochs=20, seed=0))
    for r in results:
        assert r.n_train + r.n_test == len(frame)
        assert (r.predictions["gpu_key"] == r.group).all()


def test_model_beats_the_roofline_baseline(frame):
    tab, _ = evaluate(frame, fold="hardware",
                      config=TrainConfig(max_epochs=120, patience=25, seed=0))
    pooled = tab.loc["POOLED"]
    assert pooled["energy"] < pooled["roofline"], (
        "the learned model must beat eta=pi=1; if it does not, the features are "
        "actively misleading rather than merely uninformative"
    )


def test_evaluate_rejects_a_frame_without_targets():
    df = pd.DataFrame({"gpu_key": ["H100"] * 20, "category": ["gemm"] * 20})
    with pytest.raises(KeyError, match="missing"):
        evaluate(df, fold="hardware")


def test_unknown_fold_is_rejected(frame):
    with pytest.raises(ValueError, match="fold must be one of"):
        evaluate(frame, fold="phase-of-the-moon")
