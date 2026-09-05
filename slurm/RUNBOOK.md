# Runbook: from nothing to a leave-one-GPU-out energy model

Every command in order, with what to check before moving on. Assumes `slurm/config.sh`
is already filled in (it is, for Iridis X).

Total: a few minutes of your attention, a few hours of queue.

---

## 0. Sync

```bash
# Windows, in the repo folder
git add -A && git commit -m "sync" && git push
```

```bash
# Iridis
cd ~/kernel-energy && git pull
source slurm/job_common.sh          # activates the env; every step below needs this
```

`source slurm/job_common.sh` is required in **every new shell**. It activates conda, sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, and puts the offline flags in place.

---

## 1. Capture — what kernels do these pipelines run?

Once, on any GPU. Shapes do not depend on hardware.

```bash
bash slurm/submit_capture.sh
squeue -u $USER -n ke-capture
```

~90 seconds per model once it starts. When all three finish:

```bash
grep -A12 "runtime coverage" $KE_LOGS/ke-capture-*.out
grep "wrote .* rows" $KE_LOGS/ke-capture-*.out
```

**Check before continuing:**
- `__covered__` share ≥ 0.95. Below that, a kernel category is missing — send
  `top_uncovered` output and get a decomposer added first.
- Each model wrote **several hundred rows**, not 0. Zero means the capture wrappers
  failed; the error is in the log next to `failed at 512x512`.

---

## 2. Catalogue — deduplicate into a measurement plan

Login node, seconds.

```bash
kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv
```

Prints configs per category. Expect roughly 500-1000 unique configs total. The three
models share many GEMM shapes, so this is well below the sum of the capture rows — that
is the deduplication working, and it is what stops the same kernel appearing on both
sides of the architecture fold.

---

## 3. Measure — replay every config on every card

The long pole. One sharded array job per GPU.

```bash
bash slurm/submit_measure.sh -n     # dry run: check partitions and gres first
bash slurm/submit_measure.sh        # all six cards
```

If the main GPU partitions are queueing badly, use the preemptible ones instead — the
sweep survives eviction by design (finishes the kernel in flight, flushes, requeues), so
this is the faster route, not a compromise:

```bash
KE_USE_SCAVENGER=1 KE_MEASURE_ARRAY=0-7 bash slurm/submit_measure.sh
```

Monitor:

```bash
bash slurm/status.sh
```

Shows rows collected per card as a percentage of the catalogue, plus any CONTENDED / OOM
/ FAILED warnings in the logs. Re-run it whenever; nothing is lost by closing the
terminal.

**Check before continuing:** every card near 100%. A card stuck well below it has either
run out of walltime (resubmit — it resumes) or is failing on something the log will name.

---

## 4. Dataset — join measurements with analytical features

No GPU. Login node or a CPU partition.

```bash
kernelenergy dataset --raw $KE_DATA/raw --out $KE_DATA/dataset.csv
```

This recomputes every analytical feature from scratch, derives `eta` and `pi`, and prints
four diagnostic tables. **Read these before trusting any MAPE:**

| table | what to look for |
|---|---|
| `efficiency` | `frac_over_1` must be 0. Anything above means the analytical floor exceeds a measured time — a bug upstream, not a modelling problem. |
| `instrument_agreement` | median near 1.0. Drift means the measurement windows were too short. |
| `power_regime` | `power_cap_ratio` — if not 1.0 across the fleet, fit `pi_limit`, not `pi` (see step 5). |
| `coverage` | rows per (card, category). Gaps mean a card missed a category entirely. |

---

## 5. Evaluate — the leave-one-GPU-out model

```bash
kernelenergy evaluate \
  --dataset $KE_DATA/dataset.csv \
  --fold hardware \
  --predictions $KE_DATA/predictions \
  --out $KE_DATA/results_hardware.txt
```

Trains the two-head MLP six times, each time holding out one GPU entirely, and reports
MAPE per held-out card with `MEAN` and `POOLED`, against three baselines.

**How to read it.** The column that matters is not `energy` on its own but `energy`
against `constant`. `roofline` (eta = pi = 1) is the zero-parameter prediction and should
lose badly. `constant` is the training-set median of each head — if the MLP is close to
it, the features carry no information and the analytical stage is doing all the work.
`ridge` separates "the features help" from "the nonlinearity helps".

All three folds at once:

```bash
kernelenergy evaluate --dataset $KE_DATA/dataset.csv --fold all
```

Or as a batch job that does steps 4 and 5 together, plus the boost-clock comparison:

```bash
sbatch --partition=$KE_CPU_PARTITION slurm/03_dataset.sbatch
```

### If the fleet is power-capped

`power_cap_ratio` not 1.0 everywhere means `pi = P_avg/TDP` has an unreachable
denominator on some cards, and the hardware fold is comparing two conventions. Every row
records `power_limit_w`, so refit against the limit that binds:

```python
import pandas as pd
from kernelenergy.model.targets import add_targets
from kernelenergy.model.evaluate import evaluate

df = pd.read_csv(f"{KE_DATA}/dataset.csv")
df = add_targets(df, tdp_col="power_limit_w")     # pi now against the enforced limit
tab, results = evaluate(df, fold="hardware")
print(tab.to_string())
```

---

## 6. Validate before claiming anything about whole pipelines

```bash
KE_GPU_KEY=H100 sbatch --partition=swarm_h100 --gres=gpu:h100swarm:1 \
  slurm/04_validate_e2e.sbatch
```

Measures a real generation and compares it against the sum of the replayed per-kernel
energies. **The sum will exceed the measurement** — a replayed kernel has the whole power
budget and a warm cache. What matters is whether the ratio is *stable across cards*: a
consistent 1.25 is a calibration constant; 1.05 on one card and 1.6 on another means the
per-kernel numbers do not compose and the end-to-end claim has to be dropped.

---

## Trying the modelling half with no data

Steps 4 and 5 run today, on synthetic measurements, so you can see the output format
before any hardware:

```bash
python scripts/make_synthetic.py --out /tmp/syn_raw.csv
kernelenergy dataset  --raw /tmp/syn_raw.csv --out /tmp/syn.csv
kernelenergy evaluate --dataset /tmp/syn.csv --fold hardware
```

The generative law is simple and known, so any MAPE from it is a test fixture, never a
result. The *shape* of the table is what you are looking at.
