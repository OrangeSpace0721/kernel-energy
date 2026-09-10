"""Multi-seed comparison: the arithmetic, and the pairing that makes it worth doing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kernelenergy.pipeweave.seeds import paired_compare, seed_summary, sign_test_p


def test_sign_test_matches_the_binomial_tail():
    """P(X >= wins) for X ~ Binomial(n, 1/2), by hand."""
    assert sign_test_p(5, 5) == pytest.approx(1 / 32)         # 0.03125
    assert sign_test_p(4, 5) == pytest.approx(6 / 32)         # 0.1875
    assert sign_test_p(3, 5) == pytest.approx(16 / 32)        # 0.5
    assert sign_test_p(0, 5) == pytest.approx(1.0)
    assert sign_test_p(10, 10) == pytest.approx(1 / 1024)


def test_five_seeds_is_the_minimum_that_can_show_anything():
    """Documents why the default is five and not three.

    A clean sweep of three seeds gives p = 0.125 -- nothing. Five gives 0.031. So four
    of five (p = 0.19) settles nothing either, and a run that is not unanimous at five
    seeds should be reported as unresolved rather than as a small win.
    """
    assert sign_test_p(3, 3) > 0.1
    assert sign_test_p(5, 5) < 0.05
    assert sign_test_p(4, 5) > 0.1


def _runs(arm, per_seed, seeds=5, operator="gemm", gpu="H100"):
    per_seed = dict(enumerate(per_seed)) if not isinstance(per_seed, dict) else per_seed
    return pd.DataFrame([
        {"gpu": gpu, "operator": operator, "seed": s, "n_test": 100,
         "hybrid": per_seed[s], "arm": arm}
        for s in range(seeds)
    ] + [
        {"gpu": "POOLED", "operator": "-", "seed": s, "n_test": 500,
         "hybrid": per_seed[s], "arm": arm}
        for s in range(seeds)
    ])


def test_paired_compare_is_paired_by_seed():
    """A consistent small win must beat a large-but-inconsistent one.

    Arm B is better on every seed by 1.0. If the comparison were unpaired -- comparing
    the two distributions -- the overlap would swamp it. Paired, it is unanimous.
    """
    a = _runs("a", [50.0, 70.0, 40.0, 90.0, 60.0])
    b = _runs("b", [49.0, 69.0, 39.0, 89.0, 59.0])
    cells, v = paired_compare(a, b, "hybrid", "a", "b")
    assert v["b_wins"] == 5
    assert v["p_sign"] == pytest.approx(1 / 32)
    assert v["delta_median"] == pytest.approx(-1.0)
    # As unpaired samples the two arms overlap almost completely -- 39-89 against
    # 40-90 -- so an unpaired test would see nothing. Pairing is what resolves them.
    assert min(b["hybrid"]) < max(a["hybrid"]) and min(a["hybrid"]) < max(b["hybrid"])


def test_a_coin_flip_reports_as_a_coin_flip():
    a = _runs("a", [50.0, 50.0, 50.0, 50.0, 50.0])
    b = _runs("b", [48.0, 53.0, 47.0, 54.0, 49.0])
    _, v = paired_compare(a, b, "hybrid", "a", "b")
    assert v["b_wins"] == 3
    assert v["p_sign"] > 0.2


def test_seed_summary_reports_spread_not_just_a_median():
    """A cell whose spread rivals its value is describing the seed, not the model."""
    runs = pd.concat([_runs("x", [10.0, 90.0, 50.0, 20.0, 80.0])])
    s = seed_summary(runs)
    row = s.loc[("H100", "gemm")]
    assert row["hybrid"] == pytest.approx(50.0)
    assert row["hybrid_pm"] == pytest.approx(40.0)      # (90 - 10) / 2
    assert row["seeds"] == 5


def test_mismatched_arms_are_refused():
    """Arms with no seed in common cannot be paired, and silently returning an empty
    table would read as 'no difference' rather than 'no comparison'."""
    a = _runs("a", [1.0, 1.0], seeds=2)
    b = _runs("b", [1.0, 1.0], seeds=2)
    b["seed"] = b["seed"] + 100          # no seed in common
    with pytest.raises(ValueError, match="share no"):
        paired_compare(a, b, "hybrid", "a", "b")
