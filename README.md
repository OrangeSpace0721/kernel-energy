# kernelenergy

Kernel-level energy measurement and prediction for diffusion models, re-targeting
**PipeWeave** (Zhang et al., *Synergizing Analytical and Learning Models for Unified GPU
Performance Prediction*, ISCA 2026, arXiv 2601.14910) from latency to energy.

Collects a sample of per-kernel energy from FLUX.1-dev, SD3.5-Large and Qwen-Image across
the L4 / A100 / L40S / H100 / H200 fleet, and fits PipeWeave's analytical-plus-MLP stack
to predict it.

---

## What is ported, and what is changed

PipeWeave has four stages. Three are preserved as they are; the fourth is where the
re-targeting happens.

| Stage | PipeWeave | Here |
|---|---|---|
| Kernel Decomposer | kernel → tasks with per-pipeline demands | unchanged in structure; new decomposers for diffusion kernels |
| Scheduling Simulator | tasks → SMs, round-robin or persistent | unchanged |
| Feature Analyzer | demands + theoretical cycles, GPU and worst-SM | unchanged, **plus a power-domain block** |
| Performance Estimator | one MLP, sigmoid, MAPE loss → efficiency | **two heads**: efficiency *and* power fraction |

### The two heads

PipeWeave predicts one bounded scalar, execution efficiency `eta = C_theory / t`, and
recovers latency by division. Energy needs a second, the **power fraction**:

```
eta = C_theory / t         in (0,1]     execution efficiency
pi  = P_avg / TDP          in (0,1]     power fraction

E   = pi * TDP * t = pi * TDP * C_theory / eta
```

Both are ratios against a spec-sheet constant. Neither asks the network to learn the
five-orders-of-magnitude dynamic range that the analytical stage has already accounted
for. This is the same argument that made utilisation, rather than latency, the right
target in the run-level model in this project — `log u` has sd 0.192 across the fleet
where `log t` has 1.884.

The heads share a trunk by default, because the two quantities are driven by the same
physics: a kernel stalling on memory is simultaneously less efficient *and* drawing less
power. `share_trunk=False` trains them independently if you want to test that.

### The power-domain features

PipeWeave's feature vector is hardware-aware only through the throughputs in its cycle
denominators — an A100 differs from an H100 because its cycles differ. That is enough
when the target is time. It is not enough when the target is energy: two cards can have
identical theoretical cycles and very different power draw, because TDP is not a function
of throughput. So the analyser adds TDP, the enforced limit, the idle floor, memory
technology, and the energy-relevant ratios — watts per TFLOP/s, watts per GB/s, bytes
moved per flop.

Bytes-per-flop is the clearest case. It is a weak latency feature, where a memory-bound
and a compute-bound kernel of equal duration cost the same. It is a strong energy
feature, where moving a byte off-chip costs orders of magnitude more than the arithmetic
consuming it.

---

## Why replay, and what it costs

The obvious approach — run the pipeline, capture the kernel timeline, attribute energy by
timestamp — does not work. A kernel runs for tens to hundreds of microseconds. Board power
telemetry updates on the order of tens of milliseconds and is itself filtered. Two or
three orders of magnitude separate them, so any per-kernel number recovered that way is a
deconvolution of a heavily smoothed signal against a rapidly switching one, dominated by
the assumptions of the deconvolution rather than by the data.

Replaying a kernel in a tight loop inverts the problem. The loop runs for seconds; the
driver's own `nvmlDeviceGetTotalEnergyConsumption` counter integrates exactly over the
window. What you measure is unambiguous.

Three costs, all real, none fatal, all worth stating up front rather than discovering
later:

1. **Cache state.** A kernel replayed back-to-back finds its inputs warm in L2.
   `n_buffers` (default 4) rotates through several input copies so the working set
   exceeds L2.
2. **No overlap.** Replayed alone, a kernel has the whole power budget. In the pipeline it
   shares it. This is the main reason the sum of replayed kernel energies **will exceed**
   a measured end-to-end run — `scripts/validate_e2e.py` quantifies the gap rather than
   assuming it away.
3. **Steady-state clocks.** Seconds of one kernel drives the card to whatever clock that
   kernel sustains, which is not the clock the mixture sustains. Every row records
   `sm_clock_mean_mhz` and `frac_sw_power_cap` so this is visible, not assumed.

**Run `validate_e2e.py` before presenting any kernel-level number as a claim about whole
pipelines.** A ratio near 1.0 would be surprising. What matters is whether the ratio is
*stable* across cards: a consistent 1.25 is a calibration constant; 1.05 on one card and
1.6 on another means the per-kernel numbers do not compose.

---

## Running on a cluster

**If you are running this under SLURM, read [`slurm/README.md`](slurm/README.md) first.**
The short version of why:

Granted GPU 5 of a node, SLURM sets `CUDA_VISIBLE_DEVICES=5`, so PyTorch calls that card
`cuda:0` — while NVML, which is not filtered by that variable, still calls it 5. Read the
energy counter at index 0 and you get a different, idle card. Nothing errors, and you
collect a full sweep of plausible numbers that are somebody else's idle draw.

`kernelenergy.hpc` handles that (by UUID, refusing to guess when it cannot be sure), plus
the three other things a scheduler changes: board-level energy counters make GPU
exclusivity a correctness requirement rather than a nicety, jobs get killed so the sweep
has to stop cleanly and resume, and sites cap cards below their datasheet TDP so `pi`
needs a denominator that reflects the limit that actually binds.

```bash
bash slurm/00_setup.sh                          # login node: env + stage weights
sbatch --partition=gpu --gres=gpu:1 slurm/01_capture.sbatch
kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv
bash slurm/submit_measure.sh                    # one sharded array job per card
sbatch --partition=cpu slurm/03_dataset.sbatch  # merge, fit, evaluate
```

Edit `slurm/config.sh` — paths, conda, and the partition/constraint/gres per card — and
nothing else should need changing.

## Install

```bash
pip install -e .              # modelling only, no GPU needed
pip install -e '.[gpu]'       # adds torch + diffusers, for capture and measurement
pip install -e '.[dev]'       # pytest
```

The modelling half is pure NumPy/pandas/scikit-learn on purpose, so it runs in a
notebook next to the existing ladder, in CI, and on a CPU partition without holding a GPU
allocation to fit a model.

## Use directly

These are the commands the sbatch scripts wrap; run them by hand on a workstation, or to
debug an interactive allocation (`salloc --exclusive --gres=gpu:1`).

```bash
# 1. What kernels do these pipelines actually run?  (one box, needs the models)
kernelenergy capture --model flux1-dev   --out data/catalogue --profile
kernelenergy capture --model sd35-large  --out data/catalogue --profile
kernelenergy capture --model qwen-image  --out data/catalogue --profile

# 2. Merge and deduplicate into one measurement plan
kernelenergy catalogue --in data/catalogue --out data/catalogue.csv

# 3. On each card in the fleet, with nothing else running
kernelenergy idle
kernelenergy measure --catalogue data/catalogue.csv --out data/raw

# 4. Anywhere: join measurements with freshly recomputed analytical features
kernelenergy dataset --raw data/raw --out data/dataset.csv

# 5. Leave-one-group-out
kernelenergy evaluate --dataset data/dataset.csv --fold all
```

`scripts/run_fleet.sh` wraps steps 1–5 for a per-node run. The `measure` stage is
resumable: rerun after a preemption and it skips `(kernel_sig, gpu_key)` pairs already
recorded.

Before any hardware is available, the whole modelling path can be exercised:

```bash
python scripts/make_synthetic.py --out data/synthetic_raw.csv
kernelenergy dataset  --raw data/synthetic_raw.csv --out data/synthetic_dataset.csv
kernelenergy evaluate --dataset data/synthetic_dataset.csv --fold all
```

The synthetic generator is **not** a simulator. Its law is simple and known, so a model
that fits it has demonstrated that the plumbing works and nothing more. Any MAPE from it
is a test fixture, never a result.

---

## Three folds

| Fold | Held out | Question |
|---|---|---|
| `hardware` | one GPU | does this transfer to a card never seen? PipeWeave's headline claim |
| `architecture` | one diffusion model | do FLUX and Qwen kernels predict SD3.5's? |
| `category` | one kernel category | does anything transfer *between* kernel classes? |

The category fold is not in PipeWeave, which trains a separate MLP per kernel category
and therefore cannot ask the question. It is worth knowing the answer: if nothing
transfers, every new kernel class needs its own measurement campaign, and that is a
planning fact.

Reporting follows this project's convention — every table is per held-out group, with
both `MEAN` and `POOLED`. Averages hide folds, and the hardware fold is where they hide
most.

Three baselines the model must beat: **roofline** (`eta = pi = 1`), **constant** (the
training-set median of each head), and **ridge** on the same features in log space. The
constant baseline is the one that matters — if a fitted constant is close to the MLP, the
features carry no information and the analytical stage is doing all the work.

---

## Notes on the hardware table

`hardware.py` carries **two** clocks per card and they are not interchangeable.
`boost_clock_mhz` is what `nvidia-smi` reports; `tensor_clock_mhz` is the frequency NVIDIA
quotes tensor throughput against, recovered as `peak / (SMs * ops_per_clk)`. They differ
by up to 15.5% on the H200 NVL. The run-level work in this project found the tensor clock
strictly better for latency at every shrinkage level, while the *energy* fold moved the
other way — unresolved, so both are carried and `peak_tensor_flops()` takes the clock as
an argument rather than choosing for you. `--clock boost` re-runs the whole dataset build
on the other convention, which makes that comparison one command.

Ada parts are ambiguous on `ops_per_clk`: 512 with FP32 accumulate, 1024 with FP16.
PipeWeave Table VI lists the L40 at 512; this project uses 1024. Both are carried.

L2 and shared-memory bandwidths are **estimates**, unlike tensor throughput and HBM
bandwidth which are spec-sheet exact. They therefore appear as features but are excluded
from the theoretical floor — an overstated floor gives `eta > 1`, which a sigmoid head
cannot represent and would silently cap. `validate_efficiency()` reports any such rows and
should be read every time the catalogue or the hardware table changes.

One number per card has to be measured rather than looked up: `idle_power_w`. Run
`kernelenergy idle` on each and paste the result back. For a short kernel on a 700 W card
the idle floor can be most of the joules, and a model fit to unsubtracted energy on one
card and subtracted on another is comparing two different quantities.

`load_hardware_csv()` overrides the built-in table from a CSV, so you can point it at the
hardware table already maintained in the run-level project rather than keeping two copies
in step. The hardware columns use `standard_data`'s names and both datasets join on
`gpu_key`.

---

## A thing worth noticing early

The L4 has a *better* nominal energy efficiency than the H100 — 0.30 W per TFLOP/s against
0.35 — yet the run-level data in this project has it achieving the lowest utilisation of
the fleet (0.29), and its cause is still unexplained: not the clock, and not extrapolation
leverage alone. Nominal and achieved efficiency point in opposite directions on that card.

Kernel-level data is the natural place to find out why, because it can say *which kernels*
underperform there rather than only that the aggregate does. That is arguably the strongest
reason to collect this dataset at all, over and above fitting a better predictor.

---

## Layout

```
kernelenergy/
  hardware.py          fleet descriptors, two clocks, CSV override
  nvml.py              energy counter, power sampling, clocks, throttle reasons
  dataset.py           the join: measurements + recomputed features + targets
  cli.py               preflight | capture | catalogue | idle | measure | dataset | evaluate

  hpc/                 what a scheduler changes
    device.py          CUDA ordinal -> NVML ordinal by UUID; MIG and co-tenancy guards
    slurm.py           job context, cost-balanced sharding, walltime, clean interruption
    preflight.py       everything that could invalidate a run, checked before it

  kernels/             one KernelSpec per category: build, run, decompose
    base.py            Task, KernelConfig, tile heuristics
    gemm.py  attention.py  conv.py  pointwise.py

  trace/
    capture.py         patches torch.nn.functional to record real shapes
    pipelines.py       loading FLUX / SD3.5 / Qwen-Image, the resolution sweep
    profile.py         profiler pass: which kernels cost the runtime
    catalogue.py       dedupe, elementwise recovery, the measurement plan

  measure/
    replay.py          the replay harness
    schema.py          one row per (kernel config, GPU)
    writer.py          resumable, append-safe sweep driver

  model/
    schedule.py        Scheduling Simulator (IV-B)
    features.py        Feature Analyzer (IV-C) + power-domain block
    targets.py         eta / pi definitions and their inverses
    estimator.py       the two-head MLP, NumPy
    evaluate.py        three folds, three baselines, MEAN and POOLED

scripts/
  make_synthetic.py    exercise the modelling path with no GPU
  validate_e2e.py      do replayed kernel energies reconstruct a real generation?
  run_fleet.sh         per-node driver, for running without a scheduler

slurm/
  config.sh            THE ONLY FILE TO EDIT: paths, conda, partitions per card
  job_common.sh        sourced by every job; sets CUDA_DEVICE_ORDER and offline mode
  environment.yml      the conda environment
  00_setup.sh          login node: build the env, stage 100+ GB of weights
  01_capture.sbatch    one array task per pipeline; does not need exclusivity
  02_measure.sbatch    exclusive, signal-aware, requeueing sweep
  submit_measure.sh    fills in partition/constraint/gres per card
  03_dataset.sbatch    merge, fit, evaluate -- CPU only
  04_validate_e2e.sbatch
  README.md            why each of the above is shaped the way it is
```

## Extending

**A new kernel category.** Subclass `KernelSpec`, implement `build`/`run` (replay),
`decompose` (tasks), `flops`/`bytes_global`, and register it. Roughly 10–50 lines, as in
PipeWeave. Run `kernelenergy capture --profile` first: `top_uncovered()` ranks the kernels
no decomposer handles, so the effort goes where the runtime is.

**A new GPU.** Add a `GPU(...)` entry with its spec-sheet values, measure its idle power,
and give it an `nvml_patterns` entry. `probe_local_gpu()` raises rather than guessing on an
unknown card — a silently wrong hardware descriptor would poison every row that machine
collects.

**A new pipeline.** One entry in `PIPELINES`. The capture layer patches
`torch.nn.functional` rather than hooking modules, so it needs no knowledge of how the
model is structured.

## Tests

```bash
python -m pytest tests/ -q
```

108 tests, no GPU required. They check identities that must hold exactly — op counts
against closed forms, task counts against tile arithmetic, work conservation across the
scheduler, exact round-tripping of the target definitions, and that no held-out group
leaks into training. The HPC tests mock NVML to check the CUDA-to-NVML mapping under every
SLURM environment shape, because that failure is invisible in the output. They do not check
accuracy against hardware, which needs hardware.
