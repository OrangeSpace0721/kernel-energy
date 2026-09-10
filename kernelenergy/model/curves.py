"""Training curves per fold, as a self-contained HTML page.

What the picture is for
-----------------------
A fold table gives one number per held-out group and cannot say *why* that number is
what it is. Three curves per fold can:

* train falling while **test rises** -- overfitting the remaining cards, and the fold
  number is a function of where early stopping happened to land;
* train and val falling together while **test sits flat and high** -- the model
  generalises within the training cards and does not transfer to the held-out one,
  which is a hardware-extrapolation failure and not a capacity problem;
* all three flat from the first epoch -- the features carry nothing, and no amount of
  training changes that;
* val far below test -- the validation split is drawn from the training cards, so this
  gap *is* the generalisation gap the fold exists to measure.

On the test curve
-----------------
It is a **diagnostic and nothing else**. Early stopping reads the validation split
alone; ``MLP.fit`` and ``TransferModel.fit`` take the held-out fold as ``monitor`` and
only append its loss to the history. That separation is structural rather than a
promise -- nothing in either stopping rule references ``history.test_loss``. If it ever
did, the held-out card would be leaking into model selection and every fold number in
the project would be optimistic.

Design notes
------------
Small multiples, one panel per held-out group, because the question is "which fold
behaves differently" and that is a comparison across panels. **Axes are shared within a
section** so the panels are actually comparable -- per-panel autoscaling is the classic
way to make five different pictures look identical.

Three series is exactly the cap for a small-multiples form, so the palette is slots 1-3
of the reference categorical theme, validated in both modes. Aqua fails the 3:1 contrast
check on the light surface, which obliges relief: lines are directly labelled at their
right end and a table view carries every value.

No external dependencies. Iridis compute nodes have no internet and the login node is
not somewhere to be fetching a plotting library from, so the SVG is generated here and
the whole page is one file.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

__all__ = ["histories_frame", "render_curves", "curves_from_results"]

# Reference categorical palette, slots 1-3. Validated: light passes with a contrast
# WARN on aqua (relief supplied); dark passes clean.
_SERIES = [
    # The short tag is the direct end-label. Two of these used to be the first letter
    # of their name, which rendered Train and Test as the same glyph -- direct labels
    # that cannot be told apart are worse than none, because they look decisive.
    ("train", "Train", "tr", "#2a78d6", "#3987e5"),
    ("val", "Validation", "va", "#eb6834", "#d95926"),
    ("test", "Test (held out)", "te", "#1baf7a", "#199e70"),
]


def histories_frame(results, fold: str = ""):
    """Fold results -> a tidy frame of (fold, group, model, epoch, split, loss)."""
    import pandas as pd

    rows = []
    for r in results:
        for name, h in (r.histories or {}).items():
            n = len(h.train_loss)
            for epoch in range(n):
                base = {
                    "fold": fold or getattr(r, "fold", ""),
                    "group": r.group,
                    "model": name,
                    "epoch": epoch,
                    "best_epoch": h.best_epoch,
                }
                rows.append({**base, "split": "train", "loss": h.train_loss[epoch]})
                rows.append({**base, "split": "val", "loss": h.val_loss[epoch]})
                if epoch < len(h.test_loss):
                    rows.append({**base, "split": "test", "loss": h.test_loss[epoch]})
    return pd.DataFrame(rows)


def curves_from_results(results_by_fold: dict) -> list[dict]:
    """``{fold_name: [FoldResult, ...]}`` -> the panel structures the renderer wants."""
    sections = []
    for fold, results in results_by_fold.items():
        panels = []
        for r in results:
            for name, h in (r.histories or {}).items():
                title = r.group if name == "all" else f"{r.group} · {name}"
                panels.append({
                    "title": title,
                    "train": [float(x) for x in h.train_loss],
                    "val": [float(x) for x in h.val_loss],
                    "test": [float(x) for x in h.test_loss],
                    "best_epoch": int(h.best_epoch),
                    "stage_starts": [int(s) for s in getattr(h, "stage_starts", [])],
                    "n_test": int(getattr(r, "n_test", 0)),
                })
        if panels:
            sections.append({"fold": fold, "panels": panels})
    return sections


# --------------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------------- #

_W, _H = 340, 200
_M = {"t": 12, "r": 46, "b": 30, "l": 44}   # right margin holds the direct labels


def _nice(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _panel_svg(p: dict, xmax: int, ymin: float, ymax: float, idx: int,
               log_y: bool = False) -> str:
    iw = _W - _M["l"] - _M["r"]
    ih = _H - _M["t"] - _M["b"]
    span = max(ymax - ymin, 1e-12)
    lo, hi = math.log10(max(ymin, 1e-12)), math.log10(max(ymax, 1e-11))
    lspan = max(hi - lo, 1e-12)

    def X(e):
        return _M["l"] + (e / max(xmax, 1)) * iw

    def Y(v):
        if log_y:
            v = max(float(v), ymin)
            return _M["t"] + ih - ((math.log10(v) - lo) / lspan) * ih
        return _M["t"] + ih - ((v - ymin) / span) * ih

    out = [f'<svg viewBox="0 0 {_W} {_H}" role="img" class="panel-svg" '
           f'data-panel="{idx}" aria-label="{html.escape(p["title"])} loss curves">']

    # Recessive grid: four horizontal rules, no verticals.
    for f in (0, 1 / 3, 2 / 3, 1):
        v = 10 ** (lo + f * lspan) if log_y else ymin + f * span
        y = Y(v)
        out.append(f'<line class="grid" x1="{_M["l"]}" x2="{_M["l"] + iw}" '
                   f'y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{_M["l"] - 6}" y="{y + 3.2:.1f}" '
                   f'text-anchor="end">{_nice(v)}</text>')
    for e in (0, xmax):
        out.append(f'<text class="tick" x="{X(e):.1f}" y="{_H - 10}" '
                   f'text-anchor="{"start" if e == 0 else "end"}">{e}</text>')
    out.append(f'<text class="axis-title" x="{_M["l"] + iw / 2:.1f}" y="{_H - 1}" '
               f'text-anchor="middle">epoch</text>')

    # Stage boundaries first, so the data draws over them.
    for s in p["stage_starts"]:
        if 0 < s <= xmax:
            out.append(f'<line class="stage" x1="{X(s):.1f}" x2="{X(s):.1f}" '
                       f'y1="{_M["t"]}" y2="{_M["t"] + ih}"/>')
    if 0 <= p["best_epoch"] <= xmax:
        bx = X(p["best_epoch"])
        out.append(f'<line class="best" x1="{bx:.1f}" x2="{bx:.1f}" '
                   f'y1="{_M["t"]}" y2="{_M["t"] + ih}"/>')

    ends = []
    for key, label, tag, _l, _d in _SERIES:
        ys = p.get(key) or []
        if not ys:
            continue
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in enumerate(ys))
        out.append(f'<polyline class="line s-{key}" points="{pts}"/>')
        ends.append([Y(ys[-1]), X(len(ys) - 1), key, tag])

    # Direct labels at the right end: the relief the light-mode contrast WARN
    # requires, and identity that is not colour alone. Curves converge in late
    # training, so nudge the labels apart or they overprint into a smudge.
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 9.0:
            ends[i][0] = ends[i - 1][0] + 9.0
    for y, x, key, tag in ends:
        out.append(f'<text class="endlab s-{key}" x="{x + 4:.1f}" '
                   f'y="{y + 3:.1f}">{tag}</text>')

    out.append(f'<line class="cross" x1="0" x2="0" y1="{_M["t"]}" '
               f'y2="{_M["t"] + ih}" style="display:none"/>')
    out.append(f'<rect class="hit" x="{_M["l"]}" y="{_M["t"]}" width="{iw}" '
               f'height="{ih}" fill="transparent"/>')
    out.append("</svg>")
    return "".join(out)


def _table(sections) -> str:
    rows = []
    for sec in sections:
        for p in sec["panels"]:
            n = len(p["train"])
            be = p["best_epoch"]
            def at(key, i):
                v = p.get(key) or []
                return _nice(v[i]) if 0 <= i < len(v) else "—"
            rows.append(
                f"<tr><td>{html.escape(sec['fold'])}</td>"
                f"<td>{html.escape(p['title'])}</td><td>{n}</td><td>{be}</td>"
                f"<td>{at('train', be)}</td><td>{at('val', be)}</td>"
                f"<td>{at('test', be)}</td>"
                f"<td>{at('train', n - 1)}</td><td>{at('val', n - 1)}</td>"
                f"<td>{at('test', n - 1)}</td></tr>"
            )
    return (
        "<table><thead><tr><th>Fold</th><th>Held out</th><th>Epochs</th>"
        "<th>Best</th><th>Train@best</th><th>Val@best</th><th>Test@best</th>"
        "<th>Train@last</th><th>Val@last</th><th>Test@last</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


_CSS = """
:root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f4f2;--border:#e3e2de;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#84837d;
--s-train:#2a78d6;--s-val:#eb6834;--s-test:#1baf7a;--rule:#b9b8b2;--stage:#cfcec8;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#232322;--border:#3a3a38;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d85;
--s-train:#3987e5;--s-val:#d95926;--s-test:#199e70;--rule:#5a5a56;--stage:#3f3f3c;}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#232322;
--border:#3a3a38;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d85;
--s-train:#3987e5;--s-val:#d95926;--s-test:#199e70;--rule:#5a5a56;--stage:#3f3f3c;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:21px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:15px;margin:34px 0 4px;letter-spacing:.02em;text-transform:uppercase;
color:var(--text-secondary)}
p.lede{color:var(--text-secondary);margin:0 0 4px;max-width:74ch}
p.note{color:var(--text-muted);font-size:12.5px;margin:6px 0 0;max-width:78ch}
.legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:18px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:7px;color:var(--text-secondary);
font-size:13px}
.legend code{color:var(--text-muted);font-size:11px;font-family:ui-monospace,monospace}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
.grid-panels{display:grid;gap:14px;margin-top:14px;
grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.panel{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;
padding:10px 10px 4px;position:relative}
.panel h3{margin:0 0 2px;font-size:13px;font-weight:600}
.panel .sub{color:var(--text-muted);font-size:11.5px;margin:0 0 2px}
.panel-svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--border);stroke-width:1}
.tick{fill:var(--text-muted);font-size:9px}
.axis-title{fill:var(--text-muted);font-size:9px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.s-train{stroke:var(--s-train)} .s-val{stroke:var(--s-val)} .s-test{stroke:var(--s-test)}
text.s-train{fill:var(--s-train);stroke:none}
text.s-val{fill:var(--s-val);stroke:none}
text.s-test{fill:var(--s-test);stroke:none}
.endlab{font-size:9.5px;font-weight:700}
.best{stroke:var(--rule);stroke-width:1;stroke-dasharray:3 3}
.stage{stroke:var(--stage);stroke-width:1}
.cross{stroke:var(--rule);stroke-width:1}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);
border:1px solid var(--border);border-radius:7px;padding:6px 9px;font-size:11.5px;
box-shadow:0 3px 10px rgba(0,0,0,.13);display:none;z-index:5;white-space:nowrap}
.tip b{font-weight:600}
.tip i{display:inline-block;width:9px;height:3px;border-radius:2px;
margin-right:5px;font-style:normal}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:12.5px}
th,td{text-align:right;padding:5px 9px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--text-secondary);font-weight:600}
details{margin-top:28px} summary{cursor:pointer;color:var(--text-secondary)}
"""

_JS = """
const DATA = __DATA__;
document.querySelectorAll('.panel').forEach(pel=>{
  const svg=pel.querySelector('svg'), i=+svg.dataset.panel, d=DATA[i];
  const tip=pel.querySelector('.tip'), cross=svg.querySelector('.cross');
  const hit=svg.querySelector('.hit');
  const L=__L__, R=__R__, W=__W__, n=Math.max(d.train.length-1,1), xmax=d.xmax;
  const iw=W-L-R;
  const fmt=v=> v==null?'—':(Math.abs(v)>=1?v.toFixed(3):v.toFixed(4));
  function move(ev){
    const r=svg.getBoundingClientRect();
    const px=(ev.clientX-r.left)/r.width*W;
    let e=Math.round((px-L)/iw*xmax);
    e=Math.max(0,Math.min(d.train.length-1,e));
    const x=L+(e/Math.max(xmax,1))*iw;
    cross.setAttribute('x1',x); cross.setAttribute('x2',x);
    cross.style.display='';
    tip.innerHTML='<b>epoch '+e+'</b><br>'+
      [['train','Train','--s-train'],['val','Validation','--s-val'],
       ['test','Test','--s-test']]
      .filter(([k])=>d[k]&&d[k].length>e)
      .map(([k,lab,v])=>'<i style="background:var('+v+')"></i>'+lab+' '+fmt(d[k][e]))
      .join('<br>');
    tip.style.display='block';
    const pr=pel.getBoundingClientRect();
    let tx=ev.clientX-pr.left+12;
    if(tx+tip.offsetWidth>pr.width-6) tx=ev.clientX-pr.left-tip.offsetWidth-12;
    tip.style.left=tx+'px';
    tip.style.top=Math.max(4,ev.clientY-pr.top-tip.offsetHeight-8)+'px';
  }
  hit.addEventListener('mousemove',move);
  hit.addEventListener('mouseleave',()=>{tip.style.display='none';
    cross.style.display='none';});
});
"""


def render_curves(sections: list[dict], path: str | Path,
                  title: str = "Training curves by fold") -> Path:
    """Write the self-contained page. ``sections`` comes from :func:`curves_from_results`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat, body = [], []
    for sec in sections:
        panels = sec["panels"]
        # Shared scales within a section. Per-panel autoscaling would make five
        # different pictures look the same, which is the opposite of the point.
        xmax = max(len(p["train"]) - 1 for p in panels)
        vals = [v for p in panels for k in ("train", "val", "test") for v in (p.get(k) or [])]
        ymin, ymax = (min(vals), max(vals)) if vals else (0.0, 1.0)

        # Loss over a full run falls by one to two orders of magnitude. On a linear
        # axis that squashes everything after the first few epochs into the baseline,
        # which is precisely where overfitting and the train/test gap become visible.
        # Log whenever the range is wide enough to matter.
        pos = [v for v in vals if v > 0]
        # 8x is low for a log threshold by general charting standards and right for
        # this one. The first few epochs of any loss curve cover most of the range, so
        # on a linear axis the remaining hundred epochs -- where overfitting and the
        # train/test gap actually appear -- compress into the bottom few pixels.
        log_y = bool(pos) and (max(pos) / min(pos)) > 8.0
        if log_y:
            ymin, ymax = min(pos) / 1.3, max(pos) * 1.3
        else:
            # Pad by a share of the range, but never drop the floor far below the data:
            # a 6%-of-range pad on a series whose minimum is small leaves the curves
            # crammed into the top half of an otherwise empty panel.
            pad = (ymax - ymin) * 0.06 or 0.01
            ymin = max(0.0, min(ymin - pad, ymin * 0.98) if ymin > 0 else ymin - pad)
            ymax = ymax + pad

        cards = []
        for p in panels:
            idx = len(flat)
            flat.append({"train": p["train"], "val": p["val"], "test": p["test"],
                         "xmax": xmax})
            sub = f'{len(p["train"])} epochs · best {p["best_epoch"]}'
            if p["n_test"]:
                sub += f' · {p["n_test"]} test rows'
            cards.append(
                f'<div class="panel"><h3>{html.escape(p["title"])}</h3>'
                f'<p class="sub">{sub}</p>'
                f'{_panel_svg(p, xmax, ymin, ymax, idx, log_y)}'
                f'<div class="tip"></div></div>'
            )
        scale_note = (
            " &mdash; y is logarithmic, because loss spans more than an order of "
            "magnitude and a linear axis flattens everything after the first few "
            "epochs, which is exactly where the train/test gap appears"
            if log_y else ""
        )
        body.append(
            f'<h2>{html.escape(sec["fold"])} fold</h2>'
            f'<p class="note">Axes shared across the panels below, so they compare '
            f'directly{scale_note}. Dashed rule marks the epoch early stopping '
            f'selected; a solid faint rule marks a change of optimisation stage.</p>'
            f'<div class="grid-panels">{"".join(cards)}</div>'
        )

    legend = "".join(
        f'<span><i class="swatch" style="background:var(--s-{k})"></i>{lab} '
        f'<code>{tag}</code></span>'
        for k, lab, tag, _a, _b in _SERIES
    )
    js = (_JS.replace("__DATA__", json.dumps(flat))
             .replace("__L__", str(_M["l"])).replace("__R__", str(_M["r"]))
             .replace("__W__", str(_W)))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="lede">One panel per held-out group. Train and validation come from the
cards or models the fold trained on; <b>test is the held-out group</b> and is a
diagnostic only &mdash; early stopping reads validation alone, so the test curve never
influenced the fit.</p>
<p class="note">The gap between validation and test is the quantity the fold exists to
measure: validation is drawn from the training groups, test is not. Train falling while
test rises is overfitting to the remaining groups; test flat and high while both others
fall is a failure to transfer, which more capacity will not fix.</p>
<p class="note"><b>Train usually sits above the other two, and that is not
underfitting.</b> Train loss is accumulated over the epoch with dropout active and
BatchNorm in batch mode; validation and test are evaluated afterwards in inference
mode, with dropout off. The offset is the regularisation, not the model. What matters
is the <i>shape</i> of each curve and the gap between validation and test.</p>
<div class="legend">{legend}</div>
{''.join(body)}
<details><summary>Table view &mdash; every panel at its selected and final epoch</summary>
{_table(sections)}</details>
</div><script>{js}</script></body></html>"""
    path.write_text(page, encoding="utf-8")
    return path
