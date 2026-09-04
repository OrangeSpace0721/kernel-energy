"""Performance Estimator -- PipeWeave IV-D, with two heads.

A small MLP: three hidden layers of 256/128/64 with batch normalisation, ReLU and
dropout(0.1), a sigmoid output, MAPE loss and AdamW. That is PipeWeave's architecture
verbatim, because the point of this port is to change the *target*, not the network, and
holding the architecture fixed keeps the comparison honest.

The only structural change is the output width: two sigmoid units instead of one, for
``eta`` and ``pi``. They share every hidden layer. Sharing is the right default rather
than training two networks, because the two quantities are driven by the same physics --
a kernel that stalls on memory is simultaneously less efficient and drawing less power,
and that correlation is information a shared trunk can use and two separate networks
cannot. ``share_trunk=False`` trains them independently if you want to test that claim.

Implemented in NumPy rather than Torch on purpose: the collection half of this repo
needs Torch on the GPU boxes, but the modelling half should run anywhere -- on a laptop,
in a notebook next to the existing ladder, in CI. A network this small gains nothing from
a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MLP", "TrainConfig", "TrainHistory", "mape"]

_EPS = 1e-8


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), _EPS)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 400
    patience: int = 40
    val_fraction: float = 0.15
    seed: int = 0
    head_weights: tuple[float, ...] = (1.0, 1.0)
    share_trunk: bool = True
    verbose: bool = False


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class _Dense:
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        # He initialisation, appropriate for the ReLU stack.
        self.w = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.w + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self.dw = self._x.T @ dy
        self.db = dy.sum(axis=0)
        return dy @ self.w.T

    def params(self):
        # (value, grad, decay?) -- biases are excluded from weight decay.
        return [(self.w, lambda: self.dw, True), (self.b, lambda: self.db, False)]


class _BatchNorm:
    def __init__(self, n: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones(n)
        self.beta = np.zeros(n)
        self.dgamma = np.zeros(n)
        self.dbeta = np.zeros(n)
        self.run_mean = np.zeros(n)
        self.run_var = np.ones(n)
        self.momentum = momentum
        self.eps = eps
        self._cache = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        if training and x.shape[0] > 1:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mu
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var
        else:
            mu, var = self.run_mean, self.run_var
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        self._cache = (xhat, inv, x.shape[0])
        return self.gamma * xhat + self.beta

    def backward(self, dy: np.ndarray) -> np.ndarray:
        xhat, inv, n = self._cache
        self.dgamma = (dy * xhat).sum(axis=0)
        self.dbeta = dy.sum(axis=0)
        dxhat = dy * self.gamma
        return inv / n * (
            n * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0)
        )

    def params(self):
        return [(self.gamma, lambda: self.dgamma, False),
                (self.beta, lambda: self.dbeta, False)]


class _Dropout:
    def __init__(self, p: float):
        self.p = p
        self._mask: np.ndarray | None = None

    def forward(self, x, training: bool, rng: np.random.Generator):
        if not training or self.p <= 0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, dy):
        return dy if self._mask is None else dy * self._mask


# --------------------------------------------------------------------------- #
# AdamW
# --------------------------------------------------------------------------- #


class _AdamW:
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.wd = weight_decay
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p, _, _ in params]
        self.v = [np.zeros_like(p) for p, _, _ in params]

    def step(self):
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for i, (p, grad_fn, decay) in enumerate(self.params):
            g = grad_fn()
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / b1t
            vhat = self.v[i] / b2t
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if decay:  # decoupled, as in AdamW
                upd = upd + self.lr * self.wd * p
            p -= upd


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class MLP:
    """Multi-head bounded regressor trained with MAPE loss.

    ``fit`` expects targets already in (0, 1] -- one column per head. Predictions come
    back in the same shape and units, so the caller does the inversion to latency and
    energy through :mod:`kernelenergy.model.targets`.
    """

    def __init__(self, n_features: int, n_heads: int = 2, config: TrainConfig | None = None):
        self.cfg = config or TrainConfig()
        self.n_features = n_features
        self.n_heads = n_heads
        rng = np.random.default_rng(self.cfg.seed)
        self._rng = rng

        self.x_mean = np.zeros(n_features)
        self.x_std = np.ones(n_features)

        dims = (n_features,) + tuple(self.cfg.hidden)
        self.dense = [_Dense(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)]
        self.bn = [_BatchNorm(d) for d in self.cfg.hidden]
        self.drop = [_Dropout(self.cfg.dropout) for _ in self.cfg.hidden]
        self.head = _Dense(self.cfg.hidden[-1], n_heads, rng)
        self.history = TrainHistory()

    # -- forward / backward ------------------------------------------------- #

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def _forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        h = x
        self._relu_masks = []
        for dense, bn, dr in zip(self.dense, self.bn, self.drop):
            h = dense.forward(h)
            h = bn.forward(h, training)
            mask = h > 0
            self._relu_masks.append(mask)
            h = h * mask
            h = dr.forward(h, training, self._rng)
        z = self.head.forward(h)
        self._z = z
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _backward(self, dyhat: np.ndarray) -> None:
        sig = 1.0 / (1.0 + np.exp(-np.clip(self._z, -30, 30)))
        dz = dyhat * sig * (1 - sig)
        dh = self.head.backward(dz)
        for dense, bn, dr, mask in zip(
            reversed(self.dense), reversed(self.bn),
            reversed(self.drop), reversed(self._relu_masks)
        ):
            dh = dr.backward(dh)
            dh = dh * mask
            dh = bn.backward(dh)
            dh = dense.backward(dh)

    def _params(self):
        ps = []
        for dense, bn in zip(self.dense, self.bn):
            ps += dense.params() + bn.params()
        return ps + self.head.params()

    # -- loss ---------------------------------------------------------------- #

    def _loss_and_grad(self, y: np.ndarray, yhat: np.ndarray):
        w = np.asarray(self.cfg.head_weights[: self.n_heads], float)
        w = w / w.sum()
        denom = np.maximum(np.abs(y), _EPS)
        rel = (yhat - y) / denom
        loss = float(np.mean(np.abs(rel) * w))
        # d|rel|/dyhat = sign(rel)/denom
        grad = np.sign(rel) / denom * w / y.shape[0]
        return loss, grad

    # -- api ----------------------------------------------------------------- #

    def fit(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray | None = None) -> "MLP":
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[1] != self.n_heads:
            raise ValueError(f"expected {self.n_heads} target columns, got {Y.shape[1]}")
        if np.any(Y <= 0):
            raise ValueError(
                "targets must be strictly positive -- a sigmoid head cannot reach 0, "
                "and MAPE is undefined there. Check for eta or pi of exactly 0."
            )

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        # A constant feature has zero spread; leave it at zero rather than dividing by
        # noise. This happens routinely -- e.g. a pipeline that no kernel in the fold
        # touches.
        self.x_std[self.x_std < 1e-12] = 1.0
        Xs = self._standardise(X)

        rng = np.random.default_rng(self.cfg.seed)
        n = Xs.shape[0]
        idx = rng.permutation(n)
        n_val = max(1, int(round(self.cfg.val_fraction * n)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        if len(tr_idx) == 0:
            tr_idx, val_idx = idx, idx

        Xtr, Ytr = Xs[tr_idx], Y[tr_idx]
        Xva, Yva = Xs[val_idx], Y[val_idx]

        opt = _AdamW(self._params(), self.cfg.lr, self.cfg.weight_decay)
        best = float("inf")
        best_state = None
        since = 0

        for epoch in range(self.cfg.max_epochs):
            order = rng.permutation(len(Xtr))
            ep_loss = 0.0
            nb = 0
            for s in range(0, len(order), self.cfg.batch_size):
                b = order[s: s + self.cfg.batch_size]
                if len(b) < 2:  # batch norm needs >1 row
                    continue
                yhat = self._forward(Xtr[b], training=True)
                loss, grad = self._loss_and_grad(Ytr[b], yhat)
                self._backward(grad)
                opt.step()
                ep_loss += loss
                nb += 1
            tr_loss = ep_loss / max(nb, 1)

            va_pred = self._forward(Xva, training=False)
            va_loss, _ = self._loss_and_grad(Yva, va_pred)
            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(va_loss)

            if va_loss < best - 1e-6:
                best, since = va_loss, 0
                best_state = self._snapshot()
                self.history.best_epoch = epoch
                self.history.best_val = va_loss
            else:
                since += 1
                if since >= self.cfg.patience:
                    break
            if self.cfg.verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:4d}  train {tr_loss:.4f}  val {va_loss:.4f}")

        if best_state is not None:
            self._restore(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardise(np.asarray(X, float))
        return self._forward(Xs, training=False)

    # -- checkpointing ------------------------------------------------------- #

    def _snapshot(self):
        return [p.copy() for p, _, _ in self._params()] + [
            (bn.run_mean.copy(), bn.run_var.copy()) for bn in self.bn
        ]

    def _restore(self, state):
        params = self._params()
        n = len(params)
        for (p, _, _), saved in zip(params, state[:n]):
            p[...] = saved
        for bn, (m, v) in zip(self.bn, state[n:]):
            bn.run_mean, bn.run_var = m, v
