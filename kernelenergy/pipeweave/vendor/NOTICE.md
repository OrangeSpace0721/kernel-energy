# Vendored PipeWeave analytical model

These eight files are copied **byte-for-byte** from the PipeWeave reference
implementation and are licensed Apache 2.0 (see `LICENSE`).

    upstream: https://github.com/zksainx/pipeweave
    commit:   6cac920915b59947150f1d6d5732d29ef2948fe6
    path:     analytical_model/

`SHA256SUMS` pins them. `kernelenergy.pipeweave.vendor.verify()` re-checks the
digests at import time in the test suite, so a silent edit fails a test rather
than a fold.

## Why byte-identical and not a reimplementation

The whole point of the transfer is that PipeWeave's published checkpoints read a
feature vector these files produce. A reimplementation that is 99% right produces
a trunk reading noise — the failure is silent and looks like "the model doesn't
transfer" rather than "the features are wrong". Copying removes that class of bug
by construction.

The files are unmodified, which means their imports are flat (`from pipes import
...`). `kernelenergy/pipeweave/vendor/__init__.py` puts this directory on
`sys.path` rather than rewriting the imports, so the digests stay valid.

## Which files, and why these

| file | used for |
|---|---|
| `pipes.py` | the dataclasses every calculator returns |
| `utils.py` | `ceil_div`, `gcd` |
| `gemm_8_calculator.py` | GEMM on Ampere, **Ada and Blackwell** |
| `gemm_9_calculator.py` | GEMM on Hopper only |
| `fa2_calculator.py` | attention on Ampere / Ada |
| `fa3_calculator.py` | attention on Hopper |
| `rmsnorm_calculator.py` | normalisation |
| `silumul_calculator.py` | elementwise activation |

Not vendored: `gemm_fp8_calculator.py`, `triton_moe_calculator.py`,
`fa_ck_calculator.py`, `fa_cutlass_calculator.py`. Nothing in a BF16 diffusion
pipeline reaches them.

**The GEMM dispatch is `hopper -> gemm9, everything else -> gemm8`.** That is what
`aggregator.py` does, and it is not what `gemm_9_calculator.py`'s own docstring
says ("SM90/100"). Blackwell is SM100 and still goes through `gemm8`. Following
the docstring instead of the dispatch reproduces the Blackwell rows of
`gemm_test.csv` with a 99.4% error on `tensor_sm_max_ops`; following the dispatch
reproduces them to 4.8e-16. See `tests/test_pipeweave_parity.py`.

## Verified against upstream's own datasets

Every calculator was checked against the CSVs PipeWeave ships, which are the
files their checkpoints were trained on. Agreement is float64 rounding — these
are the same computation, not a close one.

| operator | rows checked | GPUs | max relative error |
|---|---|---|---|
| gemm | 118,800 (all of `gemm_test.csv`) | 11 | 6.0e-16 |
| rmsnorm | 800 sampled (train + test) | 6 | 1.2e-14 |
| siluandmul | 800 sampled (train + test) | 6 | 4.0e-16 |
| attn | 60 sampled (train) | 6 | 3.6e-16 |

`gemm_train.csv` (494,463 rows, 131 MB) is a Git-LFS object and is **not**
fetchable through this environment's proxy. It is not needed for parity — the
test split covers all eleven GPUs on the same generator — but it is the file
upstream's `aggregator.py` reads for GEMM tile lookup, so `tiles.py` falls back
to `gemm_test.csv`.

## Two facts the upstream datasets fix for us

* `hardware/L40.json` gives `tcBf16 = 512`, independently confirming the
  correction in `kernelenergy/hardware.py`. The first version of that table had
  1024 and overstated the L40's peak by 2x.
* Their L2 bandwidths are measured, not derived: A100 3235, H100 8820, H200
  10403, L40 5647, A40 2430 GB/s. `pipeweave/hardware.py` uses them verbatim for
  the cards they cover.

## One thing their data does not cover

`aggregator.py` hardcodes `causal=True` for every attention row. Their attention
checkpoint has never seen a non-causal attention, and diffusion attention is
non-causal. The calculator takes the flag and handles it correctly; the
*checkpoint* is extrapolating. This is the single largest known distribution gap
in the transfer, and `transfer.py` reports attention separately for that reason.
