"""Transfer from PipeWeave's published latency models to energy.

The point of this package is that PipeWeave released weights, and weights trained on
611,000 GEMMs across six GPUs are worth more than anything 1,753 rows of diffusion-kernel
energy can learn from scratch. Everything here exists to make those weights usable on
this project's kernels:

``vendor/``     their analytical model, byte-identical, Apache 2.0
``hardware.py`` fleet GPU -> their ``HardwareSpec``, using their constants
``tiles.py``    the GEMM launch geometry their features need but ``(M,N,K)`` lacks
``features.py`` a captured diffusion kernel -> the vector their checkpoint expects
``transfer.py`` load a checkpoint, keep the efficiency head, add a power head

The load-bearing claim is that ``features.py`` emits the same numbers their own
pipeline does. It is checked against their shipped datasets in
``tests/test_pipeweave_parity.py``, to float64 rounding, over all 118,800 rows of
``gemm_test.csv``. Break that and every result downstream is meaningless in a way that
produces no error message.
"""

from __future__ import annotations

from kernelenergy.pipeweave.features import (
    CATEGORY_TO_OPERATOR,
    PIPEWEAVE_FEATURES,
    Emission,
    emit,
    emit_frame,
)
from kernelenergy.pipeweave.hardware import PW_HARDWARE, to_hardware_spec
from kernelenergy.pipeweave.tiles import GemmLaunch, choose_launch

__all__ = [
    "PIPEWEAVE_FEATURES",
    "CATEGORY_TO_OPERATOR",
    "Emission",
    "emit",
    "emit_frame",
    "PW_HARDWARE",
    "to_hardware_spec",
    "GemmLaunch",
    "choose_launch",
]
