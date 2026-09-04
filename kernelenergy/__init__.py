"""Kernel-level energy measurement and prediction for diffusion models.

A re-targeting of PipeWeave (Zhang et al., "PipeWeave: Synergizing Analytical and
Learning Models for Unified GPU Performance Prediction", ISCA 2026, arXiv 2601.14910)
from *latency* to *energy*.

The four PipeWeave stages are preserved:

    kernel + input  ->  Kernel Decomposer   ->  tasks
    tasks + hardware->  Scheduling Simulator->  task distribution over SMs
    distribution    ->  Feature Analyzer    ->  per-pipeline demands, theoretical cycles
    features        ->  Performance Estimator (MLP)

The fourth stage is where the re-targeting happens. PipeWeave's MLP emits one bounded
scalar, the *execution efficiency* eta in [0,1], and recovers latency by division. We
emit two bounded scalars from the same feature vector:

    eta = C_theoretical / t          execution efficiency  (PipeWeave, unchanged)
    pi  = P_avg / TDP                power fraction        (new)

and recover energy as E = pi * TDP * t = pi * TDP * C / eta.

Both heads predict a ratio confined to roughly one order of magnitude rather than a raw
quantity spanning several, which is the same argument that made utilisation the right
target in the run-level model in this project.
"""

__version__ = "0.1.0"

from kernelenergy.hardware import GPU, GPUS, get_gpu, probe_local_gpu  # noqa: F401

__all__ = ["GPU", "GPUS", "get_gpu", "probe_local_gpu", "__version__"]
