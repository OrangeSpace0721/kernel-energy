"""Byte-identical copy of PipeWeave's ``analytical_model/`` (Apache 2.0).

See ``NOTICE.md`` for provenance, the upstream commit, and the parity evidence.

The vendored files import each other flatly (``from pipes import ...``) because
they were written to be run with ``analytical_model/`` on the path. Rather than
rewrite those imports -- which would invalidate the checksums that make the
parity claim checkable -- this module puts the directory on ``sys.path`` and
imports them there.

Import side effects are confined to this one ``sys.path`` insert. The names it
exposes (``pipes``, ``utils``, ``gemm_8_calculator``, ...) are generic enough to
collide with something else on the path, so nothing outside this package should
import them by bare name; use the re-exports below.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipes import (  # noqa: E402
    FaFeatures,
    FaProblemConfig,
    FmaPipe,
    GemmFeatures,
    GemmProblemConfig,
    HardwareSpec,
    MemoryPipe,
    RmsNormFeatures,
    RmsNormProblemConfig,
    SiluMulFeatures,
    SiluMulProblemConfig,
    TensorPipe,
    XuPipe,
)
from fa2_calculator import calculate_fa2_params  # noqa: E402
from fa3_calculator import calculate_fa3_params  # noqa: E402
from gemm_8_calculator import gemm8_calculator  # noqa: E402
from gemm_9_calculator import gemm9_calculator  # noqa: E402
from rmsnorm_calculator import rmsnorm_calculator  # noqa: E402
from silumul_calculator import silu_mul_calculator  # noqa: E402

__all__ = [
    "HardwareSpec",
    "GemmProblemConfig",
    "GemmFeatures",
    "FaProblemConfig",
    "FaFeatures",
    "RmsNormProblemConfig",
    "RmsNormFeatures",
    "SiluMulProblemConfig",
    "SiluMulFeatures",
    "TensorPipe",
    "XuPipe",
    "FmaPipe",
    "MemoryPipe",
    "gemm8_calculator",
    "gemm9_calculator",
    "calculate_fa2_params",
    "calculate_fa3_params",
    "rmsnorm_calculator",
    "silu_mul_calculator",
    "verify",
    "UPSTREAM_COMMIT",
]

#: Upstream commit these files were taken from.
UPSTREAM_COMMIT = "6cac920915b59947150f1d6d5732d29ef2948fe6"


def verify() -> dict[str, bool]:
    """Re-check every vendored file against ``SHA256SUMS``.

    Returns ``{filename: ok}``. A ``False`` here means someone edited a vendored
    file, and every parity guarantee in ``NOTICE.md`` is void until it is
    reverted -- which is exactly the failure mode worth catching in CI rather
    than in a fold result six steps later.
    """
    sums = _HERE / "SHA256SUMS"
    out: dict[str, bool] = {}
    for line in sums.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        path = _HERE / name.strip()
        out[name.strip()] = (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        )
    return out
