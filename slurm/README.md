# Running on the cluster

Four things about a scheduler change how this has to work, and three of them are
correctness problems rather than convenience ones.

## 1. CUDA ordinals are not NVML ordinals

**This is the one to understand before anything else runs.**

Granted GPU 5 of an eight-GPU node, SLURM sets `CUDA_VISIBLE_DEVICES=5`. PyTorch then
calls that card `cuda:0`. NVML is *not* filtered by that variable and still calls it 5 —
so `nvmlDeviceGetHandleByIndex(0)` hands you a completely different, probably idle, GPU.

The kernel runs on one card and the energy counter is read from another. **Nothing
errors.** You get a full sweep of plausible numbers that are the idle draw of somebody
else's card, and no downstream check would catch it.

`kernelenergy.hpc.device.resolve_device()` fixes this by matching on GPU UUID, which is
invariant under both `CUDA_VISIBLE_DEVICES` filtering and `CUDA_DEVICE_ORDER` reordering.
Where the PyTorch build is too old to expose UUIDs it falls back to the environment
variable, and where even that is ambiguous it **raises rather than guessing**. Every row
records `cuda_index`, `nvml_index` and `gpu_uuid`, so a suspect row is traceable to a
physical card months later.

`job_common.sh` exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` for the same reason. The default,
`FASTEST_FIRST`, orders CUDA devices by capability rather than PCI slot, which breaks the
index fallback on heterogeneous nodes.

## 2. The energy counter is per board — which is narrower than it sounds

`nvmlDeviceGetTotalEnergyConsumption` reports joules for **one GPU**, not one node. So a
job on a *different* GPU of the same node does not contaminate your counter, and
`--gres=gpu:1` — which already gives you sole use of that GPU — is enough for the
measurement to be attributable. `--exclusive` is not required for correctness.

What a co-tenant *on the same GPU* would do is add their joules to yours with no way to
separate them, so the harness refuses to measure a GPU carrying a foreign compute process,
re-checking before every kernel rather than only at job start. This is the guard that
matters if a site ever enables MPS or GPU sharding. `--allow-shared-gpu` overrides it and
stamps `contended=1` on every affected row; `merge_results` then prefers a clean
measurement of the same kernel wherever one exists.

What node sharing *does* affect is **thermal coupling**: neighbours in the same chassis
raise inlet temperature and can push your card into thermal throttling, which changes its
energy at fixed work. Real, but second-order, and every row records `temperature_mean_c`
and `frac_hw_thermal` so it is visible. `submit_measure.sh` therefore sets `--exclusive`
per card from `KE_GPU_SPEC` — on for small nodes where it is cheap, off for 8-GPU nodes
where demanding all eight to use one means never running. If you run shared, check
`frac_hw_thermal` in the quality report afterwards and re-measure exclusively any card
where it is materially non-zero.

MIG is refused outright. Board energy cannot be attributed to a MIG instance, and unlike
co-tenancy there is no process list that would reveal the problem after the fact.

## 3. Jobs get killed

`--signal=B:USR1@420` warns seven minutes before walltime. The sweep catches it, finishes
the kernel in flight, flushes, and exits **64** meaning "clean stop, work remains". Iridis disables `--requeue`, so
recovery is a manual resubmit — one command, and it skips everything already measured. Starting a 12-second measurement with 4 seconds
left is worse than skipping it, because a truncated window would be written as if it were
real, so the sweep also stops when the next kernel does not fit in the remaining walltime.

Array tasks split the catalogue. The split is balanced by estimated cost
(longest-processing-time-first, so the array finishes when the last task does rather than
when the unlucky one does) and deterministic, so a requeued task reconstructs exactly its
own shard. Each task writes its own file: concurrent appends to one file on Lustre or GPFS
interleave at unpredictable boundaries and corrupt rows.

Resume scans *every* shard file for the card, not just its own, so you can resubmit a
4-way array as an 8-way one and nothing gets measured twice.

## 4. Sites cap power

HPC sites routinely cap GPUs below their datasheet TDP for facility power or cooling
reasons. The power-fraction target is `pi = P_avg / TDP`, so on a capped card the
denominator is unreachable and every `pi` is compressed by a factor that is invisible in
the data. Preflight warns when the enforced limit differs from the card default; every row
records both; and `add_targets` emits `pi_limit`, computed against the limit that actually
binds. **Use `pi_limit` if `power_cap_ratio` is not 1.0 across the fleet** — otherwise the
hardware fold is comparing two conventions.

---

## Setup

Edit `slurm/config.sh`. It is the only file that should need changing: paths, conda
location, and the partition/constraint/gres for each card. `sinfo -o "%P %G %f"` lists what
your site actually calls them.

Then, **on a login node** (it needs internet):

```bash
bash slurm/00_setup.sh
```

This builds the conda environment and downloads all three pipelines into a shared HF cache.
Expect well over 100 GB — check your scratch quota first. FLUX.1-dev and SD3.5-Large are
gated, so accept their licences in a browser and run `huggingface-cli login` beforehand,
or you get a 401 that reads like a network error.

## Running

```bash
# 1. Capture the kernel catalogue -- one array task per model.
#    Does not need exclusivity: it records shapes and relative times, not energy.
sbatch --partition=$KE_CAPTURE_PARTITION --gres=gpu:1 slurm/01_capture.sbatch

# 2. Merge the captures (login node, seconds)
kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv

# 3. Sweep. One array job per card, sharded across tasks.
bash slurm/submit_measure.sh               # whole fleet
bash slurm/submit_measure.sh H100 L40S     # or named cards
KE_MEASURE_ARRAY=0-7 bash slurm/submit_measure.sh H100    # wider split

# 4. Merge, fit, evaluate. No GPU needed.
sbatch --partition=$KE_CPU_PARTITION slurm/03_dataset.sbatch

# 5. Do the replayed energies reconstruct a real generation? Per card.
KE_GPU_KEY=H100 sbatch --partition=gpu --gres=gpu:h100:1 slurm/04_validate_e2e.sbatch
```

## Preflight

```bash
kernelenergy preflight --stage measure --catalogue $KE_DATA/catalogue.csv --out $KE_DATA/raw
```

Runs in about a minute and is already the first thing both measurement sbatch scripts do.
It is worth understanding why it exists: the failures that matter on a cluster are not
crashes — a crash is cheap, you find out immediately — but the silent ones. Wrong NVML
device, a co-tenant, MIG, a capped card, a missing weight that only surfaces at the third
resolution. Each produces hours of plausible, wrong numbers that you discover days later
when the folds stop making sense.

Exit status is non-zero if anything would invalidate the run.

## Sizing

At default settings each config costs about `warmup + repeats x window + setup` ≈ 12 s.
A catalogue of ~400 configs is therefore ~80 minutes per card, and a 4-way array brings
that to ~20 minutes of walltime — comfortably inside a 4-hour allocation with room for the
idle measurement and overhead pricing.

`--window` and `--repeats` are the knobs. Shortening the window below ~2 s starts to matter:
the whole argument for replay is that the measurement window is long relative to the power
sensor's update interval, and `energy_check_ratio` (counter over integrated power) is the
diagnostic — if it drifts from 1.0, the window is too short.

## Monitoring

```bash
squeue -u $USER -o '%.18i %.12j %.8T %.10M %R'
tail -f $KE_LOGS/ke-H100-*.out
grep -c CONTENDED $KE_LOGS/*.out          # should be zero on exclusive nodes
```

After the sweep, the tables printed by `03_dataset.sbatch` are the ones to read first:

- **`efficiency`** — any `frac_over_1` above zero means the analytical floor exceeds a
  measured time, which is a bug upstream, not a modelling problem.
- **`instrument_agreement`** — the energy counter against integrated power. Should sit near
  1.0 with small spread; drift means the windows are too short.
- **`power_regime`** — which cards sit against their cap and which do not. This is the
  kernel-level version of the two-regime split the run-level work found.

## When something goes wrong

**`DeviceResolutionError: CUDA_DEVICE_ORDER is unset`** — you ran outside `job_common.sh`.
Export `CUDA_DEVICE_ORDER=PCI_BUS_ID`, or use a PyTorch build exposing device UUIDs.

**`GpuContendedError`** — the allocation was not exclusive, or a stale process is holding
the card. `nvidia-smi` names the PID. Do not reach for `--allow-shared-gpu` to make it go
away; the resulting rows are not usable.

**`hf.cache: not in the cache`** — `00_setup.sh` did not finish, or `HF_HOME` differs
between the login node and the job. It is set in `config.sh` and must be on shared scratch.

**Jobs exit 64 repeatedly** — the shard does not fit the walltime. Widen the array
(`KE_MEASURE_ARRAY=0-7`) or raise `KE_MEASURE_TIME`. Progress is not lost either way.

**A whole array task produced no rows** — check `nvml_index` and `gpu_uuid` in the CSVs it
did write, against `nvidia-smi -L` on that node.
