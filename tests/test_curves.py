"""The training-curve page: it must render, and it must not lie about the test split."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from kernelenergy.model.curves import curves_from_results, histories_frame, render_curves
from kernelenergy.model.estimator import MLP, TrainConfig, TrainHistory


def _fake_results():
    class R:
        pass

    out = []
    for g, n in [("L4", 40), ("H100", 25)]:
        r = R()
        r.group, r.fold, r.n_test = g, "hardware", 100
        h = TrainHistory()
        h.train_loss = list(np.linspace(0.4, 0.05, n))
        h.val_loss = list(np.linspace(0.38, 0.06, n))
        h.test_loss = list(np.linspace(0.36, 0.20, n))
        h.best_epoch = n - 3
        h.stage_starts = [0, n // 2]
        r.histories = {"all": h}
        out.append(r)
    return out


def test_monitor_records_a_test_curve_and_never_changes_the_fit():
    """The peeking guarantee, as a test: the same seed must give the same weights.

    If the held-out loss ever reached the stopping rule, passing it would change where
    training stopped -- and every fold number in the project would be optimistic by an
    unknown amount. Fitting twice, once with ``monitor`` and once without, must produce
    identical predictions.
    """
    rng = np.random.default_rng(0)
    X, Y = rng.random((200, 6)), np.clip(rng.random((200, 2)) * 0.6 + 0.2, 0.01, 0.99)
    Xte, Yte = rng.random((50, 6)), np.clip(rng.random((50, 2)) * 0.6 + 0.2, 0.01, 0.99)
    cfg = TrainConfig(max_epochs=30, patience=100, seed=7)

    plain = MLP(6, 2, cfg).fit(X, Y)
    watched = MLP(6, 2, cfg).fit(X, Y, monitor=(Xte, Yte))

    np.testing.assert_allclose(plain.predict(Xte), watched.predict(Xte), rtol=0, atol=0)
    assert plain.history.test_loss == []
    assert len(watched.history.test_loss) == len(watched.history.train_loss)
    assert plain.history.best_epoch == watched.history.best_epoch


def test_histories_frame_is_tidy_and_complete():
    f = histories_frame(_fake_results())
    assert set(f["split"]) == {"train", "val", "test"}
    assert set(f["group"]) == {"L4", "H100"}
    assert len(f) == (40 + 25) * 3
    assert f["loss"].notna().all()


def test_page_renders_with_every_required_element(tmp_path):
    sections = curves_from_results({"hardware": _fake_results()})
    p = render_curves(sections, tmp_path / "c.html")
    html = p.read_text()

    assert "<svg" in html and html.count("polyline") >= 6      # 3 series x 2 panels
    # A legend for >=2 series, and direct labels so identity is never colour alone.
    for tag in ("Train", "Validation", "Test"):
        assert tag in html
    assert ">tr<" in html and ">va<" in html and ">te<" in html
    # Dark mode declared under both scopes: the OS setting and the explicit toggle.
    assert "prefers-color-scheme:dark" in html
    assert '[data-theme="dark"]' in html
    # A table view -- the relief the light-mode contrast WARN on aqua obliges.
    assert "<table" in html and "Test@best" in html
    # Self-contained: nothing fetched. Iridis compute nodes have no network.
    assert not re.search(r'<(script|link)[^>]+\b(src|href)=', html)


def test_end_labels_are_distinguishable_from_one_another():
    """Train and Test both began with 'T'; two identical glyphs is worse than none."""
    from kernelenergy.model.curves import _SERIES

    tags = [t for _k, _l, t, _a, _b in _SERIES]
    assert len(set(tags)) == len(tags)


def test_shared_axes_across_panels_in_a_section(tmp_path):
    """Per-panel autoscaling makes different folds look identical. It must not happen."""
    res = _fake_results()
    res[1].histories["all"].train_loss = [x * 10 for x in res[1].histories["all"].train_loss]
    sections = curves_from_results({"hardware": res})
    html = render_curves(sections, tmp_path / "c.html").read_text()
    ticks = re.findall(r'class="tick"[^>]*>([\d.]+)</text>', html)
    # Four y ticks per panel; both panels must carry the same four values.
    ys = [t for t in ticks if "." in t]
    assert len(set(ys[:4])) == len(set(ys[4:8])) and set(ys[:4]) == set(ys[4:8])


def test_log_scale_engages_on_a_wide_range(tmp_path):
    res = _fake_results()
    res[0].histories["all"].train_loss = list(np.geomspace(1.0, 0.001, 40))
    sections = curves_from_results({"hardware": res})
    html = render_curves(sections, tmp_path / "c.html").read_text()
    assert "logarithmic" in html


def test_empty_histories_produce_no_section():
    class R:
        group, fold, n_test, histories = "L4", "hardware", 10, {}

    assert curves_from_results({"hardware": [R()]}) == []
