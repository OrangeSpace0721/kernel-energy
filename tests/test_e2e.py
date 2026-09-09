"""The end-to-end reconciliation, checked against an exactly-known case.

The whole value of this stage is that the accounting identity is arithmetic, not a fit:
if the per-kernel numbers compose, the residual is zero, and any nonzero residual is a
real quantity with a real cause. So the tests build a generation whose answer is known
by construction and check that the identity closes on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kernelenergy.e2e.measure import E2EMeasurement
from kernelenergy.e2e.reconcile import reconcile, step_model, summarise


def _measurement(**kw):
    base = dict(
        gpu_key="L40S", model="flux1-dev", steps=20, height=1024, width=1024,
        latency_s=10.0, energy_j=3000.0, power_avg_w=300.0,
        latency_sd_rel=0.01, energy_sd_rel=0.01, n_repeats=3,
        energy_counter_used=True, idle_power_w=30.0,
        sm_clock_median_mhz=2100.0, frac_sw_power_cap=0.6, frac_sw_thermal=0.0,
        temperature_max_c=70.0,
        device_time_s=9.0, device_time_covered_s=8.9, profiled_latency_s=12.0,
        calls={"k1": 100, "k2": 50},
    )
    base.update(kw)
    return E2EMeasurement(**base)


def _dataset(e1=20.0, e2=20.0, t1=0.05, t2=0.08):
    return pd.DataFrame({
        "kernel_sig": ["k1", "k2"],
        "gpu_key": ["L40S", "L40S"],
        "energy_j": [e1, e2],
        "latency_s": [t1, t2],
    })


def test_identity_closes_exactly_when_the_parts_add_up():
    """E = sum(calls x E_k) + P_idle x T_gap, with no residual, when it should be so.

    100 x 20 J + 50 x 20 J = 3000 J of kernels, plus 30 W over a 1.0 s gap = 30 J,
    against a measured 3030 J. Ratio 1.0, residual 0.
    """
    m = _measurement(energy_j=3030.0)
    r = reconcile(m, _dataset())
    assert r["A_kernel_energy_j"] == pytest.approx(3000.0)
    assert r["A_gap_energy_j"] == pytest.approx(30.0)     # 30 W x (10.0 - 9.0) s
    assert r["A_ratio"] == pytest.approx(1.0)
    assert r["A_residual_j"] == pytest.approx(0.0, abs=1e-9)


def test_idle_is_charged_to_the_gap_not_to_the_whole_run():
    """The bug in the first version of this, stated as a test.

    Measured per-kernel energies already contain the card's baseline draw during those
    kernels. Charging ``P_idle x T_wall`` on top counts the idle share of the busy time
    twice -- here that would add 300 J to a 3030 J generation, a 10% error invented by
    the accounting rather than found in the data.
    """
    m = _measurement(energy_j=3030.0)
    r = reconcile(m, _dataset())
    wrong = 3000.0 + m.idle_power_w * m.latency_s     # the old identity
    assert r["A_total_j"] == pytest.approx(3030.0)
    assert wrong == pytest.approx(3300.0)
    assert abs(r["A_total_j"] - m.energy_j) < abs(wrong - m.energy_j)


def test_latency_reconciles_against_device_time_not_wall_time():
    """A kernel sum should reconstruct GPU-busy time; the gap is not its to explain.

    100 x 0.05 + 50 x 0.08 = 9.0 s of kernels against 9.0 s of device time: ratio 1.0.
    Against the 10.0 s wall clock it would read 0.90 and look like a 10% shortfall that
    is really the scheduler, Python and synchronisation.
    """
    r = reconcile(_measurement(), _dataset())
    assert r["A_kernel_latency_s"] == pytest.approx(9.0)
    assert r["A_latency_ratio"] == pytest.approx(1.0)
    assert r["A_kernel_latency_s"] / _measurement().latency_s == pytest.approx(0.9)


def test_unmatched_calls_are_reported_not_silently_dropped():
    """A catalogue that names 60% of the invocations must say so.

    Otherwise a reconstruction that is low because it is *missing kernels* is
    indistinguishable from one that is low because the model under-predicts, and the
    two have opposite fixes.
    """
    # The unmatched kernels really did run and really did cost something: the measured
    # generation is 5030 J, of which 2000 J belongs to a kernel the catalogue cannot name.
    m = _measurement(calls={"k1": 100, "k2": 50, "unknown": 100}, energy_j=5030.0)
    r = reconcile(m, _dataset())
    assert r["calls_matched"] == 150
    assert r["calls_missing"] == 100
    assert r["call_coverage"] == pytest.approx(0.6)
    # The sum can only cover what it can match, so it falls short by exactly the
    # missing work -- and ``call_coverage`` is what tells you that is the reason.
    assert r["A_kernel_energy_j"] == pytest.approx(3000.0)
    assert r["A_ratio"] == pytest.approx(3030.0 / 5030.0)
    assert r["A_residual_j"] == pytest.approx(2000.0)


def test_level_b_uses_predictions_and_differs_from_level_a_only_by_the_model():
    """Same counts, same gap term; the gap between A and B is model error alone."""
    m = _measurement(energy_j=3030.0)
    preds = pd.DataFrame({
        "kernel_sig": ["k1", "k2"],
        "gpu_key": ["L40S", "L40S"],
        "energy_pred": [22.0, 18.0],      # 100x22 + 50x18 = 3100
        "latency_pred": [0.05, 0.08],
    })
    r = reconcile(m, _dataset(), preds)
    assert r["B_kernel_energy_j"] == pytest.approx(3100.0)
    assert r["B_total_j"] == pytest.approx(3130.0)
    assert r["B_ratio"] == pytest.approx(3130.0 / 3030.0)
    # Identical gap term on both sides, so A and B are directly comparable.
    assert r["A_total_j"] - r["A_kernel_energy_j"] == pytest.approx(
        r["B_total_j"] - r["B_kernel_energy_j"])


def test_negative_gap_is_clamped_rather_than_crediting_energy():
    """Device time above wall time means a measurement problem, not free energy.

    It should not silently subtract from the reconstruction.
    """
    m = _measurement(device_time_s=11.0, latency_s=10.0)
    r = reconcile(m, _dataset())
    assert r["gap_s"] == 0.0
    assert r["A_gap_energy_j"] == 0.0


def test_step_model_separates_fixed_cost_from_per_step_cost():
    """The point of sweeping steps: an intercept and a slope, not one ratio.

    Built from E = 500 + 100 x steps, so the fit must recover exactly that. A
    reconstruction with the right slope and the wrong intercept is a VAE or text-encoder
    problem; the reverse is a denoiser problem. One ratio cannot say which.
    """
    rows = [{
        "gpu_key": "L40S", "model": "flux1-dev", "steps": s,
        "e2e_energy_j": 500.0 + 100.0 * s,
        "A_total_j": 480.0 + 102.0 * s,
    } for s in (4, 12, 20, 28)]
    sm = step_model(rows)
    assert len(sm) == 1
    row = sm.iloc[0]
    assert row["measured_fixed_j"] == pytest.approx(500.0, abs=1e-6)
    assert row["measured_per_step_j"] == pytest.approx(100.0, abs=1e-6)
    assert row["A_fixed_j"] == pytest.approx(480.0, abs=1e-6)
    assert row["A_per_step_j"] == pytest.approx(102.0, abs=1e-6)
    assert row["A_slope_ratio"] == pytest.approx(1.02, abs=1e-6)
    assert row["A_fixed_ratio"] == pytest.approx(0.96, abs=1e-6)


def test_step_model_needs_three_points():
    rows = [{"gpu_key": "L4", "model": "m", "steps": s, "e2e_energy_j": 10.0 * s}
            for s in (4, 20)]
    assert len(step_model(rows)) == 0


def test_summarise_survives_a_missing_level_b():
    r = reconcile(_measurement(), _dataset())
    tab = summarise([r])
    assert "A_ratio" in tab.columns
    assert "B_ratio" not in tab.columns
    assert len(tab) == 1


def test_busy_and_coverage_fractions_are_what_they_claim():
    m = _measurement()
    assert m.busy_fraction == pytest.approx(0.9)          # 9.0 s of 10.0 s
    assert m.coverage_fraction == pytest.approx(8.9 / 9.0)
    assert m.gap_s == pytest.approx(1.0)
