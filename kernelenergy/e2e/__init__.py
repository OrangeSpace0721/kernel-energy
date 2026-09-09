"""End-to-end validation: does a sum over kernels reconstruct a real generation?

Every result in this project so far is per-kernel, measured by replaying one kernel
configuration alone in a tight loop. That is the only way to attribute energy to a
kernel, and it is not what a pipeline does. A replayed kernel has the card's whole power
budget and reaches a steady clock; in a real generation it shares the budget with
whatever overlaps it, runs at whatever clock the mixed workload settles at, and is
separated from its neighbours by gaps the replay never sees.

So the per-kernel numbers mean nothing about pipelines until this is measured, and the
measurement has to be a decomposition rather than a ratio -- a single "we are 25% out"
cannot distinguish four quite different problems.

The identity
------------
::

    E_generation  =  sum over kernels  E_k  +  P_idle * T_gap  +  epsilon

    T_gap = T_generation - T_busy

``T_busy`` is the profiler's total device time, not a prediction. Charging idle over the
*whole* wall time -- as the first version of this did -- double-counts it, because the
measured per-kernel energies already contain the card's baseline draw during those
kernels.

Three levels, and the differences are the diagnosis
---------------------------------------------------
======  ==========================================  ============================
level   sum over                                    isolates
======  ==========================================  ============================
A       **measured** replay energy x actual counts  whether replay composes
B       **predicted** energy x actual counts        + model error
C       predicted energy x predicted counts         + catalogue coverage
======  ==========================================  ============================

A is the methodology check and B is the deliverable. Read them together: A at 1.25 with
B at 1.28 says the model is fine and replay does not compose; A at 1.02 with B at 1.60
says the opposite. One number cannot tell you which.

Counts are not extrapolated
---------------------------
Call counts come from running capture at the *same* step count and resolution as the
measured generation, so there is no ``calls x steps`` guess and no "conv means VAE so do
not scale it" rule. Those assumptions are exactly the kind that hold until a pipeline
puts a convolution in its transformer.

Why a step sweep
----------------
A single ratio is one number with several causes. Fitting

    E(steps) = a + b * steps

separates the fixed cost -- text encoding and VAE decode, which run once -- from the
per-step transformer cost, and both sides of the comparison have a predicted ``a`` and
``b`` to be checked against. A reconstruction whose slope is right and whose intercept is
wrong is a VAE problem; the reverse is a transformer problem.
"""

from kernelenergy.e2e.measure import E2EMeasurement, measure_generation
from kernelenergy.e2e.reconcile import reconcile, step_model, summarise

__all__ = [
    "E2EMeasurement",
    "measure_generation",
    "reconcile",
    "step_model",
    "summarise",
]
