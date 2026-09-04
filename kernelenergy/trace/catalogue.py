"""The kernel catalogue: the set of configs to measure.

Built by merging capture passes across pipelines and resolutions, deduplicating by
signature, and adding back the elementwise work that operator syntax hides from the
capture layer.

Deduplication matters more than it sounds. FLUX, SD3.5 and Qwen-Image share a great many
GEMM shapes -- they are all MMDiT models with similar hidden widths and identical token
counts at a given resolution. Measuring by pipeline would measure the same kernel three
times and then, worse, let it appear on both sides of the architecture fold and quietly
inflate the score. One config, one measurement, all three provenance tags.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from kernelenergy.kernels.base import KernelConfig
from kernelenergy.kernels.registry import make_kernel

__all__ = [
    "Catalogue",
    "build_catalogue",
    "save_catalogue",
    "load_catalogue",
    "synthesise_elementwise",
]


class Catalogue:
    """A deduplicated set of kernel configs with provenance and call counts."""

    def __init__(self) -> None:
        self._by_sig: dict[str, KernelConfig] = {}
        self._models: dict[str, set[str]] = defaultdict(set)
        self._calls: dict[str, dict[str, int]] = defaultdict(dict)

    def add(self, config: KernelConfig, model: str, calls: int = 1) -> None:
        sig = config.signature()
        if sig not in self._by_sig:
            self._by_sig[sig] = config
        self._models[sig].add(model)
        self._calls[sig][model] = self._calls[sig].get(model, 0) + calls

    def __len__(self) -> int:
        return len(self._by_sig)

    def configs(self) -> list[KernelConfig]:
        return list(self._by_sig.values())

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for sig, cfg in self._by_sig.items():
            r = cfg.as_row()
            models = sorted(self._models[sig])
            r["source_model"] = "|".join(models)
            r["n_models"] = len(models)
            r["calls_total"] = sum(self._calls[sig].values())
            for m, c in self._calls[sig].items():
                r[f"calls_{m}"] = c
            rows.append(r)
        df = pd.DataFrame(rows)
        return df.sort_values(["category", "calls_total"], ascending=[True, False])

    def filter(self, *, categories=None, max_working_set_gb: float | None = None,
               min_calls: int = 1) -> "Catalogue":
        """Narrow the catalogue. Returns a new one; the original is untouched."""
        out = Catalogue()
        for sig, cfg in self._by_sig.items():
            if categories and cfg.category not in categories:
                continue
            if sum(self._calls[sig].values()) < min_calls:
                continue
            if max_working_set_gb is not None:
                try:
                    ws = make_kernel(cfg).working_set_bytes() / 2**30
                except Exception:
                    continue
                if ws > max_working_set_gb:
                    continue
            for m, c in self._calls[sig].items():
                out.add(cfg, m, c)
        return out


# --------------------------------------------------------------------------- #
# Elementwise recovery
# --------------------------------------------------------------------------- #


def synthesise_elementwise(cat: Catalogue, kinds=("scale_shift", "gate_residual", "add")
                           ) -> Catalogue:
    """Add the elementwise kernels that operator syntax hides.

    A DiT block applies AdaLN modulation (scale, shift) and a gated residual around each
    of its two sub-blocks. Those are written as ``x * (1 + scale) + shift`` and
    ``x + gate * branch`` in Python, so they never pass through a patched
    ``torch.nn.functional`` entry point -- but they are launched kernels moving the full
    activation tensor, and at DiT widths that is a meaningful share of the bytes.

    They are reconstructed at the ``(rows, dim)`` sizes the norms were observed at, since
    modulation is applied to exactly the tensors the norms produce. Six per block:
    scale/shift and a gate for each of the attention and MLP paths.
    """
    out = Catalogue()
    for sig, cfg in cat._by_sig.items():
        for m, c in cat._calls[sig].items():
            out.add(cfg, m, c)

    seen: set[tuple] = set()
    for sig, cfg in list(cat._by_sig.items()):
        if cfg.category != "norm" or cfg.params.get("kind") == "group":
            continue
        n_elem = int(cfg.params["rows"]) * int(cfg.params["dim"])
        for kind in kinds:
            key = (n_elem, kind, cfg.dtype)
            if key in seen:
                continue
            seen.add(key)
            new = KernelConfig(
                category="elementwise",
                dtype=cfg.dtype,
                params={"n_elem": n_elem, "kind": kind},
                op_name=f"synth::{kind}",
            )
            for m, c in cat._calls[sig].items():
                # Two modulations and one gate per sub-block, two sub-blocks per layer.
                out.add(new, m, c * 2)
    return out


# --------------------------------------------------------------------------- #
# Build / persist
# --------------------------------------------------------------------------- #


def build_catalogue(captures: dict[str, list], add_elementwise: bool = True,
                    max_working_set_gb: float | None = 40.0) -> Catalogue:
    """Merge capture results into one catalogue.

    ``captures`` maps model key -> list of ``CapturedOp`` (or ``KernelConfig``), typically
    one list per (model, resolution) accumulated by the capture script.
    """
    cat = Catalogue()
    for model, ops in captures.items():
        for op in ops:
            if isinstance(op, KernelConfig):
                cfg, calls = op, 1
            else:
                cfg, calls = op.to_config(model), op.count
            cat.add(cfg, model, calls)
    if add_elementwise:
        cat = synthesise_elementwise(cat)
    if max_working_set_gb is not None:
        cat = cat.filter(max_working_set_gb=max_working_set_gb)
    return cat


def save_catalogue(cat: Catalogue, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cat.to_frame().to_csv(path, index=False)
    return path


def load_catalogue(path: str | Path) -> Catalogue:
    df = pd.read_csv(path)
    cat = Catalogue()
    for _, row in df.iterrows():
        d = {k: v for k, v in row.items() if pd.notna(v)}
        cfg = KernelConfig.from_row(d)
        models = str(d.get("source_model", "")).split("|") or [""]
        total = int(d.get("calls_total", 1))
        per = max(total // max(len(models), 1), 1)
        for m in models:
            cat.add(cfg, m, per)
    return cat


def summarise(cat: Catalogue) -> pd.DataFrame:
    """Configs and call counts per category -- the shape of the measurement campaign."""
    df = cat.to_frame()
    return (
        df.groupby("category")
        .agg(n_configs=("kernel_sig", "nunique"), calls=("calls_total", "sum"))
        .assign(share_of_configs=lambda d: d["n_configs"] / d["n_configs"].sum())
        .sort_values("n_configs", ascending=False)
    )
