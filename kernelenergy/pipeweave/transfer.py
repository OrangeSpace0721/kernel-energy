"""Fine-tune PipeWeave's published checkpoints for energy.

The argument for transfer, in one line: their GEMM checkpoint was fitted on 494,463
measured kernels across six GPUs, and this project has 1,753 rows. A trunk that has
already learned how execution efficiency responds to tile shape, wave quantisation and
bandwidth pressure is worth far more than anything 1,753 rows can discover unaided.

Their model predicts one bounded scalar, ``overall_perf``. Reverse-engineered from their
own datasets -- exactly, on every row of all four -- that scalar is

    overall_perf = reference_cycles / (duration_us * sm_freq_MHz)

where ``reference_cycles`` is ``tensor_all_cycle`` for GEMM and attention, and
``fma_all_cycle + xu_all_cycle`` for the norm and elementwise operators. Which is to say
it is ``t_theoretical / t_measured``: **their target is this project's eta**, computed
against their analytical floor rather than ours. That equivalence is what makes the
transfer more than an analogy, and it is checked in ``tests/test_pipeweave_transfer.py``.

So the re-targeting is narrow. Keep the trunk, keep the efficiency head, bolt on a
second head for the power fraction:

    eta = sigmoid(W_eta . h)      <- their weights, fine-tuned
    pi  = sigmoid(W_pi  . h)      <- new
    E   = pi * TDP * C_pipeweave / eta

No torch
--------
Their checkpoints are plain float state dicts, and their architecture is
Linear/ReLU/BatchNorm/Dropout. Both are reproduced here in numpy, which keeps the
project torch-free and -- more usefully -- keeps this code testable off a GPU node.
Running their published weights through this forward pass reproduces their published
metrics to five decimal places on all four of their test sets (R2 0.96184 against their
0.96184 on 118,800 GEMM rows). See ``test_forward_reproduces_upstream_metrics``.

Their layer order is Linear -> ReLU -> BatchNorm -> Dropout -- BatchNorm *after* the
activation. That is not the order in :mod:`kernelenergy.model.estimator`, which puts it
before, so their weights cannot be loaded into that class and this one exists instead.

Fine-tuning schedule
--------------------
Three things about 1,753 rows make the naive approach fail:

* **The pi head starts random.** Its gradients would tear through a trunk that took
  half a million samples to fit. So stage one trains ``pi`` alone with everything else
  frozen, and only then is the trunk released.
* **The trunk should move slowly when it moves.** ``trunk_lr_scale`` defaults to 0.1.
* **BatchNorm running statistics are the fastest way to destroy a transferred model.**
  A few hundred rows will happily overwrite statistics estimated from half a million,
  and the damage shows up as a fold that is worse than training from scratch.
  ``freeze_bn`` defaults to True: the normalisation stays exactly as their training
  left it, and only the affine parameters can move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kernelenergy.model.estimator import _AdamW, _BatchNorm, _Dense, _Dropout, mape
from kernelenergy.pipeweave.checkpoint import load_metadata, load_state_dict
from kernelenergy.pipeweave.features import PIPEWEAVE_FEATURES

__all__ = [
    "TransferConfig",
    "TransferModel",
    "find_checkpoint",
    "reference_cycles",
    "theoretical_time_s",
    "REFERENCE_CYCLE_COLUMNS",
]

_EPS = 1e-12

#: Which cycle features define ``t_theoretical`` for each operator, recovered from
#: upstream's own data: ``overall_perf * duration_us * sm_freq_MHz`` equals their sum,
#: to 1e-13, on every row of all four datasets.
REFERENCE_CYCLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "gemm": ("tensor_all_cycle",),
    "attn": ("tensor_all_cycle",),
    "rmsnorm": ("fma_all_cycle", "xu_all_cycle"),
    "siluandmul": ("fma_all_cycle", "xu_all_cycle"),
}


def reference_cycles(operator: str, features: dict[str, float] | np.ndarray,
                     names: tuple[str, ...] | None = None) -> float | np.ndarray:
    """The cycle count upstream divides by to get ``overall_perf``."""
    cols = REFERENCE_CYCLE_COLUMNS[operator]
    if isinstance(features, dict):
        return float(sum(features[c] for c in cols))
    names = names or PIPEWEAVE_FEATURES[operator]
    idx = [names.index(c) for c in cols]
    return np.asarray(features)[..., idx].sum(axis=-1)


def theoretical_time_s(operator, features, sm_freq_mhz, names=None):
    """Their analytical floor, in seconds.

    This replaces ``theoretical_time_s`` from :mod:`kernelenergy.model.features` when
    the prediction comes from a transferred head: the head was fitted against *their*
    floor, so composing it with ours would put the ratio on a different scale and the
    error would look like model error.
    """
    return reference_cycles(operator, features, names) / (np.asarray(sm_freq_mhz) * 1e6)


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #


def find_checkpoint(root: str | Path, operator: str) -> tuple[Path, dict]:
    """Newest checkpoint for ``operator`` under a ``mlp_models/`` tree, and its metadata.

    Upstream ships several timestamped runs per operator. The newest is not always the
    best -- ``metadata.json`` carries the evaluation results, so check them rather than
    trusting the date.
    """
    root = Path(root)
    base = root / operator
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} does not exist. Point --pipeweave-models at the 'mlp_models' "
            f"directory of a PipeWeave checkout."
        )
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no timestamped runs under {base}")
    run = runs[-1]
    ckpt = run / f"{operator}_mlp_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing")
    meta_path = run / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ckpt, meta


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TransferConfig:
    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1

    # Stage 1: pi head only, trunk and eta head frozen.
    warmup_epochs: int = 150
    warmup_lr: float = 3e-3

    # Stage 2: everything trainable, trunk at a fraction of the head learning rate.
    max_epochs: int = 400
    lr: float = 3e-4
    trunk_lr_scale: float = 0.1
    weight_decay: float = 1e-2

    batch_size: int = 128
    patience: int = 60
    val_fraction: float = 0.15
    seed: int = 0

    #: Keep BatchNorm in inference mode: running statistics stay as upstream left them.
    freeze_bn: bool = True
    #: Relative weight of the eta and pi MAPE terms.
    head_weights: tuple[float, float] = (1.0, 1.0)
    #: Extra weight on the composed log-energy error. Zero reproduces the per-head
    #: objective the from-scratch model uses, which is what makes the two comparable.
    energy_weight: float = 0.0
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class TransferModel:
    """PipeWeave's trunk and efficiency head, plus a power-fraction head.

    Inputs are the operator's PipeWeave feature vector in its published order,
    untransformed -- ``log1p`` is applied here, because it is part of their model, not
    part of the caller's preprocessing.
    """

    def __init__(self, n_features: int, config: TransferConfig | None = None):
        self.cfg = config or TransferConfig()
        self.n_features = n_features
        self._rng = np.random.default_rng(self.cfg.seed)

        dims = (n_features,) + tuple(self.cfg.hidden)
        self.dense = [_Dense(dims[i], dims[i + 1], self._rng) for i in range(len(dims) - 1)]
        self.bn = [_BatchNorm(d) for d in self.cfg.hidden]
        self.drop = [_Dropout(self.cfg.dropout) for _ in self.cfg.hidden]
        self.eta_head = _Dense(self.cfg.hidden[-1], 1, self._rng)
        self.pi_head = _Dense(self.cfg.hidden[-1], 1, self._rng)
        # A random sigmoid head starts near 0.5. Power fractions cluster well below
        # that, so biasing the head towards a plausible value costs nothing and saves
        # the warmup from spending its first epochs travelling.
        self.pi_head.b[:] = -1.0
        self.provenance: dict = {}
        self.history: dict[str, list[float]] = {"train": [], "val": []}
        self._loaded = False

    # -- construction from a checkpoint -------------------------------------- #

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, operator: str, config: TransferConfig | None = None
    ) -> "TransferModel":
        state = load_state_dict(path)
        n_in = int(state["network.0.weight"].shape[1])
        expected = len(PIPEWEAVE_FEATURES[operator])
        if n_in != expected:
            raise ValueError(
                f"{path} takes {n_in} features but operator {operator!r} emits "
                f"{expected}. These must match exactly -- a checkpoint fed the wrong "
                f"feature vector produces plausible numbers and no error."
            )
        model = cls(n_in, config)
        model.load_state(state)
        model.provenance = {
            "checkpoint": str(path),
            "operator": operator,
            "upstream_training": load_metadata(path),
        }
        return model

    def load_state(self, state: dict[str, np.ndarray]) -> "TransferModel":
        """Copy upstream's weights in. Their indices are ``0,2 / 4,6 / 8,10 / 12``."""
        for i, (dense, bn) in enumerate(zip(self.dense, self.bn)):
            li, bi = 4 * i, 4 * i + 2
            # torch Linear stores (out, in); this project stores (in, out).
            dense.w[:] = state[f"network.{li}.weight"].astype(float).T
            dense.b[:] = state[f"network.{li}.bias"].astype(float)
            bn.gamma[:] = state[f"network.{bi}.weight"].astype(float)
            bn.beta[:] = state[f"network.{bi}.bias"].astype(float)
            bn.run_mean[:] = state[f"network.{bi}.running_mean"].astype(float)
            bn.run_var[:] = state[f"network.{bi}.running_var"].astype(float)
        out = 4 * len(self.dense)
        self.eta_head.w[:] = state[f"network.{out}.weight"].astype(float).T
        self.eta_head.b[:] = state[f"network.{out}.bias"].astype(float)
        self._loaded = True
        return self

    # -- forward / backward --------------------------------------------------- #

    @staticmethod
    def _transform(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if np.any(X < 0):
            raise ValueError(
                "PipeWeave features are counts and cycle estimates and must be "
                "non-negative; log1p of a negative value is not what their model saw."
            )
        return np.log1p(X)

    def _forward(self, X: np.ndarray, training: bool) -> np.ndarray:
        """Their layer order: Linear -> ReLU -> BatchNorm -> Dropout."""
        h = X
        self._relu_masks = []
        bn_training = training and not self.cfg.freeze_bn
        for dense, bn, dr in zip(self.dense, self.bn, self.drop):
            h = dense.forward(h)
            mask = h > 0
            self._relu_masks.append(mask)
            h = h * mask
            h = bn.forward(h, bn_training)
            h = dr.forward(h, training, self._rng)
        self._h = h
        z_eta = self.eta_head.forward(h)
        z_pi = self.pi_head.forward(h)
        self._z = np.hstack([z_eta, z_pi])
        return _sigmoid(self._z)

    def _backward(self, dyhat: np.ndarray) -> None:
        sig = _sigmoid(self._z)
        dz = dyhat * sig * (1.0 - sig)
        dh = self.eta_head.backward(dz[:, :1]) + self.pi_head.backward(dz[:, 1:])
        for dense, bn, dr, mask in zip(
            reversed(self.dense), reversed(self.bn),
            reversed(self.drop), reversed(self._relu_masks),
        ):
            dh = dr.backward(dh)
            dh = bn.backward(dh)
            dh = dh * mask
            dh = dense.backward(dh)

    # -- parameter groups ----------------------------------------------------- #

    def _trunk_params(self):
        ps = []
        for dense, bn in zip(self.dense, self.bn):
            ps += dense.params() + bn.params()
        return ps

    def _head_params(self, which: str = "both"):
        ps = []
        if which in ("both", "eta"):
            ps += self.eta_head.params()
        if which in ("both", "pi"):
            ps += self.pi_head.params()
        return ps

    # -- loss ------------------------------------------------------------------ #

    def _loss_and_grad(self, Y, Yhat, theory=None, tdp=None, energy=None):
        """MAPE per head, optionally plus a log-error term on the composed energy.

        The composed term matters because ``eta`` and ``pi`` are not independently
        interesting: an over-prediction of one cancels an under-prediction of the other
        in ``E = pi * TDP * C / eta``. Supervising only the heads leaves that
        cancellation unrewarded. It is off by default so the transferred and
        from-scratch models are trained against the same objective and can be compared.
        """
        w = np.asarray(self.cfg.head_weights, float)
        w = w / w.sum()
        denom = np.maximum(np.abs(Y), _EPS)
        resid = Yhat - Y
        loss = float(np.mean(np.abs(resid) / denom * w))
        grad = np.sign(resid) / denom * w / Y.shape[0]

        if self.cfg.energy_weight > 0 and energy is not None:
            eta, pi = np.maximum(Yhat[:, 0], _EPS), np.maximum(Yhat[:, 1], _EPS)
            e_hat = pi * tdp * theory / eta
            r = np.log(np.maximum(e_hat, _EPS)) - np.log(np.maximum(energy, _EPS))
            loss += self.cfg.energy_weight * float(np.mean(np.abs(r)))
            s = np.sign(r) * self.cfg.energy_weight / Y.shape[0]
            grad[:, 0] += -s / eta   # d log E / d eta = -1/eta
            grad[:, 1] += s / pi     # d log E / d pi  = +1/pi
        return loss, grad

    # -- fit -------------------------------------------------------------------- #

    def fit(self, X, Y, theory=None, tdp=None, energy=None) -> "TransferModel":
        """Fine-tune on ``Y = [eta, pi]``, both in (0, 1].

        ``theory``, ``tdp`` and ``energy`` are only needed when
        ``config.energy_weight > 0``.
        """
        if not self._loaded:
            raise RuntimeError(
                "fit() on a TransferModel with random weights defeats the purpose. "
                "Build it with from_checkpoint(), or use "
                "kernelenergy.model.estimator.MLP if training from scratch is what "
                "you meant."
            )
        X = self._transform(X)
        Y = np.asarray(Y, float)
        if Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError(f"expected Y with two columns [eta, pi], got {Y.shape}")
        if np.any(Y <= 0):
            raise ValueError("eta and pi must be strictly positive; MAPE is undefined at 0")

        n = len(X)
        idx = self._rng.permutation(n)
        n_val = max(1, int(round(self.cfg.val_fraction * n)))
        val, tr = idx[:n_val], idx[n_val:]
        extras = dict(theory=theory, tdp=tdp, energy=energy)

        def slice_extras(sel):
            return {k: (None if v is None else np.asarray(v, float)[sel])
                    for k, v in extras.items()}

        best, best_epoch, best_state = np.inf, -1, None

        for stage, (params, lr, epochs) in enumerate([
            (self._head_params("pi"), self.cfg.warmup_lr, self.cfg.warmup_epochs),
            (
                # Two learning rates in one optimiser: the trunk sees lr * scale.
                None, self.cfg.lr, self.cfg.max_epochs,
            ),
        ]):
            if epochs <= 0:
                continue
            if stage == 0:
                opt = _AdamW(params, lr, self.cfg.weight_decay)
                opts = [opt]
            else:
                opts = [
                    _AdamW(self._trunk_params(), lr * self.cfg.trunk_lr_scale,
                           self.cfg.weight_decay),
                    _AdamW(self._head_params("both"), lr, self.cfg.weight_decay),
                ]
                best, best_epoch = np.inf, -1  # early stopping applies to stage 2

            for epoch in range(epochs):
                order = self._rng.permutation(len(tr))
                for s in range(0, len(order), self.cfg.batch_size):
                    sel = tr[order[s:s + self.cfg.batch_size]]
                    if len(sel) < 2:
                        continue
                    yhat = self._forward(X[sel], training=True)
                    _, g = self._loss_and_grad(Y[sel], yhat, **slice_extras(sel))
                    self._backward(g)
                    for o in opts:
                        o.step()

                yhat_val = self._forward(X[val], training=False)
                vloss, _ = self._loss_and_grad(Y[val], yhat_val, **slice_extras(val))
                self.history["val"].append(vloss)
                if vloss < best - 1e-9:
                    best, best_epoch, best_state = vloss, epoch, self._snapshot()
                elif stage == 1 and epoch - best_epoch >= self.cfg.patience:
                    break
                if self.cfg.verbose and epoch % 25 == 0:
                    print(f"  stage {stage} epoch {epoch:4d}  val {vloss:.5f}")

        if best_state is not None:
            self._restore(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns ``[eta, pi]`` per row."""
        return self._forward(self._transform(X), training=False)

    def predict_overall_perf(self, X: np.ndarray) -> np.ndarray:
        """Just the efficiency head -- upstream's own output, for comparison."""
        return self.predict(X)[:, 0]

    # -- snapshotting ----------------------------------------------------------- #

    def _snapshot(self):
        return (
            [(d.w.copy(), d.b.copy()) for d in self.dense],
            [(b.gamma.copy(), b.beta.copy(), b.run_mean.copy(), b.run_var.copy())
             for b in self.bn],
            (self.eta_head.w.copy(), self.eta_head.b.copy()),
            (self.pi_head.w.copy(), self.pi_head.b.copy()),
        )

    def _restore(self, snap):
        dens, bns, eta, pi = snap
        for d, (w, b) in zip(self.dense, dens):
            d.w[:], d.b[:] = w, b
        for bn, (g, be, m, v) in zip(self.bn, bns):
            bn.gamma[:], bn.beta[:], bn.run_mean[:], bn.run_var[:] = g, be, m, v
        self.eta_head.w[:], self.eta_head.b[:] = eta
        self.pi_head.w[:], self.pi_head.b[:] = pi


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
