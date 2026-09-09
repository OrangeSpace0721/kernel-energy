"""Read a PyTorch ``.pth`` state dict into numpy arrays, without importing torch.

Nothing in this project depends on torch. The model in
:mod:`kernelenergy.model.estimator` is numpy from top to bottom, which is what lets the
whole modelling stack run and be tested anywhere -- including in environments where the
torch wheel cannot be fetched at all. Adding a torch dependency solely to read four
190 KB files would be a poor trade, and would make the fine-tuning code untestable
outside a GPU node.

A ``torch.save`` archive is a zip containing a pickle (``data.pkl``) plus one raw byte
blob per storage under ``data/<key>``. The pickle refers to those blobs through
persistent ids, and rebuilds tensors with ``torch._utils._rebuild_tensor_v2``. Both are
stable, documented-by-usage formats; this module implements exactly the subset a plain
``state_dict`` of dense float tensors uses, and refuses anything else rather than
guessing.

What it deliberately does not support: sparse tensors, quantised tensors, nested
modules pickled by value, ``torch.save`` of a whole ``nn.Module``, and the legacy
(pre-zip) format. All of those would need real torch semantics, and none of them appear
in PipeWeave's checkpoints -- every one is an ``OrderedDict`` of ``network.N.*`` float
tensors.
"""

from __future__ import annotations

import collections
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["load_state_dict", "load_metadata", "describe"]

#: torch storage class name -> numpy dtype. Only the ones a float state dict uses.
_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": None,  # handled specially: no numpy equivalent
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


class _Storage:
    """A lazy handle to one ``data/<key>`` blob."""

    __slots__ = ("key", "dtype", "numel", "_zf", "_prefix")

    def __init__(self, key, dtype, numel, zf, prefix):
        self.key, self.dtype, self.numel = key, dtype, numel
        self._zf, self._prefix = zf, prefix

    def array(self) -> np.ndarray:
        raw = self._zf.read(f"{self._prefix}/data/{self.key}")
        if self.dtype is None:  # bfloat16: widen to float32 by zero-filling the mantissa
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype("<u4") << 16
            return u32.view("<f4")
        return np.frombuffer(raw, dtype=self.dtype)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_):
    """The numpy equivalent of ``torch._utils._rebuild_tensor_v2``.

    Strides are in *elements*, and a state dict's tensors are contiguous, so
    ``as_strided`` with byte strides reproduces them exactly. Non-contiguous tensors
    would still work; they simply never occur here.
    """
    flat = storage.array()
    size = tuple(int(s) for s in size)
    if not size:  # 0-d, e.g. num_batches_tracked
        return np.array(flat[int(storage_offset)])
    itemsize = flat.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        flat[int(storage_offset):],
        shape=size,
        strides=tuple(int(s) * itemsize for s in stride),
    ).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, fileobj, zf, prefix):
        super().__init__(fileobj)
        self._zf, self._prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return name  # the storage *class* is only ever used as a dtype tag
        if module == "collections" and name == "OrderedDict":
            # The real class, not ``dict``: the pickle drives it through BUILD, which
            # touches ``__dict__``, and a plain dict has none.
            return collections.OrderedDict
        if module == "numpy.core.multiarray" and name == "scalar":
            return np.core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(
            f"refusing to unpickle {module}.{name}. This reader supports plain float "
            f"state dicts only; a checkpoint needing this class must be loaded with "
            f"torch and converted."
        )

    def persistent_load(self, pid):
        kind = pid[0]
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {kind!r}")
        _, storage_type, key, _location, numel = pid
        name = storage_type if isinstance(storage_type, str) else storage_type.__name__
        if name not in _DTYPES:
            raise pickle.UnpicklingError(f"unsupported storage type {name!r}")
        return _Storage(key, _DTYPES[name], int(numel), self._zf, self._prefix)


#: Keys a training checkpoint wraps its weights in. PipeWeave saves the whole training
#: state -- weights, epoch, train_loss, val_loss -- not a bare state dict.
_NESTED_KEYS = ("model_state_dict", "state_dict", "model")


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Load a ``torch.save``'d state dict as ``{name: ndarray}``.

    Unwraps a training checkpoint if it finds one: PipeWeave's ``.pth`` files are
    ``{'model_state_dict': ..., 'epoch': ..., 'train_loss': ..., 'val_loss': ...}``, and
    only the first is wanted. ``load_metadata`` returns the rest.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a zip archive. Only the modern torch.save format is "
            f"supported; re-save the checkpoint with a current torch."
        )
    with zipfile.ZipFile(path) as zf:
        pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
        if pkl is None:
            raise ValueError(f"{path} has no data.pkl -- not a torch archive")
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as fh:
            obj = _Unpickler(fh, zf, prefix).load()
    if not isinstance(obj, dict):
        raise ValueError(
            f"{path} unpickled to {type(obj).__name__}, not a state dict. This is "
            f"probably a whole saved module, which needs torch to reconstruct."
        )
    for key in _NESTED_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            obj = inner
            break
    bad = {k: type(v).__name__ for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if bad:
        raise ValueError(
            f"{path}: these entries are not tensors: {bad}. If the checkpoint nests "
            f"its weights under a key other than {_NESTED_KEYS}, add it there."
        )
    return dict(obj)


def load_metadata(path: str | Path) -> dict:
    """Everything in the checkpoint that is *not* weights: epoch, losses, and so on.

    Useful for the audit trail -- which training run a set of weights came from is worth
    recording next to any result they produce.
    """
    with zipfile.ZipFile(Path(path)) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl) as fh:
            obj = _Unpickler(fh, zf, pkl.rsplit("/", 1)[0]).load()
    if not isinstance(obj, dict):
        return {}
    return {
        k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v)
        for k, v in obj.items()
        if not isinstance(v, dict) and not isinstance(v, np.ndarray) or
        (isinstance(v, np.ndarray) and v.ndim == 0)
    }


def describe(state: dict[str, np.ndarray]) -> str:
    return "\n".join(
        f"  {k:34s} {tuple(v.shape)!s:>14s}  {v.dtype}" for k, v in state.items()
    )
