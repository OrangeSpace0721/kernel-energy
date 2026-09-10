"""Command-line entry points.

The pipeline, in order::

    kernelenergy capture   --model flux1-dev --out data/catalogue     # per GPU box, once
    kernelenergy catalogue --in data/catalogue --out data/catalogue.csv
    kernelenergy idle                                                 # per card
    kernelenergy measure   --catalogue data/catalogue.csv --out data/raw
    kernelenergy dataset   --raw data/raw --out data/dataset.csv      # anywhere
    kernelenergy evaluate  --dataset data/dataset.csv --fold hardware

Only ``capture`` and ``measure`` need a GPU; the rest run anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


def cmd_capture(args) -> int:
    from kernelenergy.trace.capture import capture_pipeline
    from kernelenergy.trace.pipelines import DEFAULT_SHAPE_SWEEP, load_pipeline
    from kernelenergy.trace.profile import coverage_report, profile_pipeline, top_uncovered

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shapes = DEFAULT_SHAPE_SWEEP
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]

    pipe, spec = load_pipeline(args.model, enable_cpu_offload=args.cpu_offload)
    all_ops = []
    for h, w in shapes:
        print(f"capturing {args.model} at {h}x{w}")
        try:
            configs, counts = capture_pipeline(
                pipe, model=args.model, steps=args.steps, height=h, width=w,
                **spec.extra_call_kwargs,
            )
        except Exception as e:
            print(f"  failed at {h}x{w}: {type(e).__name__}: {e}")
            continue
        print(f"  {len(configs)} distinct configs; {dict(counts)}")
        for c in configs:
            r = c.as_row()
            r["height"], r["width"] = h, w
            all_ops.append(r)

    df = pd.DataFrame(all_ops)
    path = out / f"capture__{args.model}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows to {path}")

    if args.profile:
        prof = profile_pipeline(pipe, steps=2, height=1024, width=1024,
                                **spec.extra_call_kwargs)
        ppath = out / f"profile__{args.model}.csv"
        prof.to_csv(ppath, index=False)
        cov = coverage_report(prof)
        print("\nruntime coverage:")
        print(cov.to_string())
        print("\ntop uncovered kernels:")
        print(top_uncovered(prof).to_string())
        print(f"\nwrote {ppath}")
    return 0


# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #


def cmd_catalogue(args) -> int:
    from kernelenergy.kernels.base import KernelConfig
    from kernelenergy.trace.catalogue import build_catalogue, save_catalogue, summarise

    src = Path(args.inp)
    files = sorted(src.glob("capture__*.csv")) if src.is_dir() else [src]
    if not files:
        print(f"no capture__*.csv under {src}", file=sys.stderr)
        return 1

    captures: dict[str, list] = {}
    for f in files:
        df = pd.read_csv(f)
        for model, sub in df.groupby("source_model"):
            cfgs = [KernelConfig.from_row(r.to_dict()) for _, r in sub.iterrows()]
            captures.setdefault(str(model), []).extend(cfgs)

    cat = build_catalogue(captures, add_elementwise=not args.no_elementwise,
                          max_working_set_gb=args.max_working_set_gb)
    save_catalogue(cat, args.out)
    print(f"catalogue: {len(cat)} unique configs -> {args.out}\n")
    print(summarise(cat).to_string())
    return 0


# --------------------------------------------------------------------------- #
# idle / measure
# --------------------------------------------------------------------------- #


def cmd_idle(args) -> int:
    from kernelenergy.hardware import probe_local_gpu
    from kernelenergy.nvml import measure_idle, supports_energy_counter

    gpu = probe_local_gpu(args.device)
    print(f"{gpu.name}  (key {gpu.gpu_key}, TDP {gpu.tdp_w:.0f} W)")
    print(f"energy counter available: {supports_energy_counter(args.device)}")
    r = measure_idle(args.device, seconds=args.seconds)
    print(f"idle power  {r.power_w:.2f} W  (sd {r.power_sd_w:.2f}, "
          f"{r.samples} samples over {r.seconds:.1f} s)")
    print(f"idle clock  {r.sm_clock_mhz:.0f} MHz   temp {r.temperature_c:.0f} C")
    print(f"\nput this in hardware.py as {gpu.gpu_key}.idle_power_w = {r.power_w:.1f}")
    return 0


def cmd_preflight(args) -> int:
    from kernelenergy.hpc.preflight import format_report, run_preflight

    models = tuple(m for m in args.models.split(",") if m) if args.models else ()
    checks = run_preflight(
        cuda_index=args.device,
        stage=args.stage,
        catalogue=args.catalogue or None,
        out_dir=args.out or None,
        allow_shared=args.allow_shared_gpu,
        models=models,
    )
    print(format_report(checks))
    return 1 if any(c.failed for c in checks) else 0


def cmd_measure(args) -> int:
    from kernelenergy.hardware import probe_local_gpu
    from kernelenergy.hpc.device import assert_no_mig, resolve_device
    from kernelenergy.hpc.slurm import EXIT_INCOMPLETE, InterruptGuard, shard_configs, \
        slurm_context
    from kernelenergy.measure.replay import ReplayConfig
    from kernelenergy.measure.writer import run_sweep
    from kernelenergy.trace.catalogue import load_catalogue

    ctx = slurm_context()
    print(ctx.describe())

    # Resolve the physical card before anything else. Under a scheduler the CUDA ordinal
    # and the NVML ordinal are different numbers for the same GPU, and getting this wrong
    # reads an unrelated card's energy counter for the whole sweep without erroring.
    dev = resolve_device(args.device, strict=True)
    assert_no_mig(dev)
    print(f"device: {dev}")

    gpu = probe_local_gpu(dev.nvml_index)
    print(f"measuring on {gpu.name} ({gpu.gpu_key})")

    cat = load_catalogue(args.catalogue)
    if args.categories:
        cat = cat.filter(categories=tuple(args.categories.split(",")))
    configs = cat.configs()
    if args.limit:
        configs = configs[: args.limit]

    cfg = ReplayConfig(
        target_window_s=args.window,
        warmup_s=args.warmup,
        repeats=args.repeats,
        n_buffers=args.buffers,
        device_index=args.device,
        nvml_index=dev.nvml_index,
        allow_shared_gpu=args.allow_shared_gpu,
        notes=args.notes,
    )

    # Shard from the CLI if given, otherwise from the array task the scheduler assigned.
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        shard = (k, n)
    elif ctx.task_count > 1:
        shard = (ctx.task_id, ctx.task_count)
    else:
        shard = None

    total = len(configs)
    if shard:
        configs = shard_configs(configs, shard[0], shard[1], cfg)
        print(f"shard {shard[0]}/{shard[1]}: {len(configs)} of {total} configs")
    else:
        print(f"{total} configs to measure")

    budget = args.time_budget
    if budget <= 0:
        remaining = ctx.seconds_remaining()
        # Leave a margin for the idle measurement, overhead pricing and a clean flush.
        budget = max(remaining - args.reserve, 0.0) if remaining else None
    if budget:
        print(f"time budget: {budget / 60:.0f} min")

    with InterruptGuard() as guard:
        res = run_sweep(
            configs, gpu, args.out, cfg,
            resume=not args.no_resume,
            shard=shard,
            time_budget_s=budget,
            interrupt=guard,
        )

    print(f"\n{res.summary()}")
    print(f"output: {args.out}")
    if res.incomplete:
        print(
            f"\nexiting {EXIT_INCOMPLETE} (incomplete). The sweep is resumable: "
            "resubmitting this task skips everything already recorded."
        )
        return EXIT_INCOMPLETE
    return 0


# --------------------------------------------------------------------------- #
# dataset / evaluate
# --------------------------------------------------------------------------- #


def cmd_dataset(args) -> int:
    from kernelenergy.dataset import build_dataset, quality_report
    from kernelenergy.measure.writer import merge_results

    raw = Path(args.raw)
    df = merge_results(raw) if raw.is_dir() else pd.read_csv(raw)
    print(f"{len(df)} measured rows")

    ds = build_dataset(df, clock=args.clock, drop_unreliable=not args.keep_all)
    ds.to_csv(args.out, index=False)
    print(f"wrote {len(ds)} rows x {ds.shape[1]} columns to {args.out}\n")

    for name, tab in quality_report(ds).items():
        print(f"--- {name} ---")
        print(tab.to_string())
        print()
    return 0


def cmd_evaluate(args) -> int:
    from kernelenergy.model.estimator import TrainConfig
    from kernelenergy.model.evaluate import FOLDS, evaluate

    ds = pd.read_csv(args.dataset)
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    cfg = TrainConfig(seed=args.seed, max_epochs=args.epochs, verbose=args.verbose)

    tables = {}
    results_by_fold = {}
    for fold in folds:
        try:
            tab, results = evaluate(ds, fold=fold, config=cfg,
                                per_category=args.per_category)
        except (KeyError, RuntimeError) as e:
            print(f"[{fold}] skipped: {e}")
            continue
        tables[fold] = tab
        results_by_fold[fold] = results
        print(f"\n=== {fold} fold: MAPE (%) by held-out group ===")
        print(tab.to_string())
        if args.predictions:
            out = Path(args.predictions)
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([r.predictions for r in results]).to_csv(
                out / f"predictions__{fold}.csv", index=False
            )

    if args.curves and results_by_fold:
        from kernelenergy.model.curves import curves_from_results, render_curves

        sections = curves_from_results(results_by_fold)
        if sections:
            path = render_curves(sections, args.curves)
            n = sum(len(sec["panels"]) for sec in sections)
            print(f"\nwrote {n} training-curve panels to {path}")
            print("  train/val come from the training groups; test is the held-out "
                  "group and never touched early stopping")

    if args.out and tables:
        with pd.ExcelWriter(args.out) if str(args.out).endswith(".xlsx") else open(
            args.out, "w"
        ) as fh:
            if str(args.out).endswith(".xlsx"):
                for fold, tab in tables.items():
                    tab.to_excel(fh, sheet_name=fold[:28])
            else:
                for fold, tab in tables.items():
                    fh.write(f"=== {fold} ===\n{tab.to_string()}\n\n")
        print(f"\nwrote {args.out}")
    return 0


def cmd_transfer(args) -> int:
    """Fine-tune PipeWeave's published checkpoints on this project's energy data.

    Prints four tables, and the order matters -- read them top to bottom, because each
    one decides whether the next is worth believing:

    1. **range** -- how far outside each checkpoint's training data our kernels fall.
       A model asked to extrapolate cannot be blamed for extrapolating.
    2. **floors** -- their analytical floor against ours. If these disagree wildly for a
       category, ``eta`` and ``eta_pw`` are different quantities and nothing that mixes
       them means anything.
    3. **transfer** -- zero-shot, fine-tuned and from-scratch energy error per held-out
       GPU and operator. The comparison the whole exercise exists for.
    4. **verdict** -- the one-line reading of table 3.
    """
    from kernelenergy.pipeweave.evaluate import (
        compare_floors, evaluate_transfer, prepare, range_report,
    )
    from kernelenergy.pipeweave.features import emit_frame
    from kernelenergy.pipeweave.hardware import POWER_FEATURES
    from kernelenergy.pipeweave.transfer import TransferConfig

    ds = pd.read_csv(args.dataset)
    print(f"{len(ds)} rows in")

    feats, blocks = emit_frame(ds, tile_mode=args.tile_mode)
    print(f"emitted PipeWeave features for {feats['pw_operator'].notna().sum()} rows: "
          + ", ".join(f"{k} ({len(v)})" for k, v in sorted(blocks.items())))
    ds = prepare(feats)
    if args.features_out:
        ds.to_csv(args.features_out, index=False)
        print(f"wrote features to {args.features_out}")

    print("\n=== how far outside their training range our kernels sit ===")
    rr = range_report(ds, args.models)
    if len(rr):
        oob = (rr["frac_below_min"] + rr["frac_above_max"]).rename("out_of_range")
        worst = rr.assign(out_of_range=oob).sort_values("out_of_range", ascending=False)
        print(worst.head(args.top).to_string())
        if float(oob.max()) < 0.01:
            print("  -> every feature inside their training range; not extrapolating")

    print("\n=== joint distance from their training distribution ===")
    print("    per-feature range checks are blind to this: an L4 LayerNorm can have all")
    print("    15 features inside their range and still be a kernel they never saw,")
    print("    because the *ratio* between the pipes is one no training card produced.")
    try:
        from kernelenergy.pipeweave.ood import ood_report
        oo = ood_report(ds)
        if len(oo):
            print(oo.head(args.top if args.top else 10).to_string())
    except FileNotFoundError as e:
        print(f"    skipped: {e}")

    if "theoretical_time_s" in ds:
        print("\n=== their analytical floor vs ours (ratio; 1.0 = identical) ===")
        print(compare_floors(ds).to_string())

    cfg = TransferConfig(
        seed=args.seed,
        # --warmup-only stops after stage 1: the trunk and their efficiency head stay
        # exactly as released, and only the new power head is fitted. On these kernels
        # that has so far been the strongest configuration, and it is the one worth
        # reaching for first when full fine-tuning underperforms zero-shot.
        max_epochs=0 if args.warmup_only else args.epochs,
        warmup_epochs=args.warmup,
        trunk_lr_scale=args.trunk_lr_scale, freeze_bn=not args.train_bn,
        energy_weight=args.energy_weight, loss=args.loss,
        grad_clip=args.grad_clip, verbose=args.verbose,
        power_features=(POWER_FEATURES if args.power_features else ()),
        power_into=args.power_into,
        pi_floor_from_idle=args.pi_floor,
    )
    if args.power_features:
        print(f"power features -> {args.power_into}: {', '.join(POWER_FEATURES)}")
        print("  new weight columns start at zero, so epoch 0 is identical to the "
              "plain transfer")
    tab, results = evaluate_transfer(ds, args.models, cfg, operators=args.operators or None)

    print("\n=== energy APE (%) by held-out GPU and operator ===")
    print("    zeroshot = their weights untouched, pi = training median")
    print("    finetuned = their trunk + eta head fine-tuned, pi head fitted")
    print("    scratch   = same architecture, random init, same rows")
    print("    hybrid    = finetuned, falling back to scratch where the efficiency head")
    print("                saturated (|logit| > 12); sat = fraction of rows that hit")
    print("    ft_med    = median APE of finetuned; oob = frac. of rows below the "
          "training eta range")
    print(tab.to_string())

    pooled = tab.loc[("POOLED", "-")]
    print("\n=== verdict ===")
    zs, ft, sc = pooled["zeroshot"], pooled["finetuned"], pooled["scratch"]
    print(f"    zero-shot {zs:.1f}%   fine-tuned {ft:.1f}%   from scratch {sc:.1f}%")

    # Read against zero-shot first. Beating random initialisation is a low bar that a
    # broken fine-tune clears easily -- the pretrained weights are still in there
    # underneath -- so reporting "pretraining helped" while fine-tuning is actively
    # damaging the model states two true things that add up to a false impression.
    hy = pooled["hybrid"]
    print(f"    hybrid {hy:.1f}%  (falls back to scratch on {pooled['sat'] * 100:.0f}% "
          f"of rows where the transferred head saturated)")
    best = min(zs, ft, hy)
    if ft > zs * 1.05:
        print(f"    FINE-TUNING IS DAMAGING THE MODEL. Their untouched weights with a "
              f"constant pi score {zs:.1f}%; fine-tuning takes that to {ft:.1f}%.")
        print(f"    Best configuration measured here is zero-shot eta with a fitted pi "
              f"head, which this table does not isolate -- run with "
              f"--warmup-only to get it.")
        print(f"    Look for an eta APE near 100% in the table: that is a head "
              f"collapsing toward zero, not a head that learned nothing. --loss mape "
              f"causes it (asymmetric penalty on an unfittable target); the default "
              f"--loss log does not.")
    elif ft < zs * 0.95:
        print(f"    Fine-tuning helped: {zs:.1f}% -> {ft:.1f}%, "
              f"{(1 - ft / zs) * 100:.0f}% better than their weights untouched.")
    else:
        print(f"    Fine-tuning changed nothing measurable against zero-shot "
              f"({zs:.1f}% vs {ft:.1f}%). The transferred weights are carrying the "
              f"result; the diffusion energy data is not adding to them.")

    if hy < min(zs, ft) * 0.95:
        print(f"    The hybrid is the best of these. Where the transferred head still "
              f"speaks it is worth using; where it saturates it is worth ignoring, and "
              f"the sat column says which rows those are.")

    if sc < best * 0.95:
        print(f"    Note: from-scratch ({sc:.1f}%) beats both. The pretrained weights "
              f"are a liability here, not an asset -- check the range table.")
    elif best < sc * 0.85:
        print(f"    Pretraining is doing real work: {best:.1f}% against {sc:.1f}% "
              f"from random initialisation on identical rows.")
    else:
        print(f"    Pretraining is not clearly contributing: best transferred "
              f"{best:.1f}% against {sc:.1f}% from random initialisation.")

    if args.curves:
        from kernelenergy.model.curves import render_curves

        # One section per operator: their checkpoints are per-operator, so a shared
        # axis across operators would be comparing losses on different targets --
        # rmsnorm eta lives near 0.003 and gemm eta near 0.5.
        sections = []
        for op in sorted({r.operator for r in results}):
            panels = []
            for r in [x for x in results if x.operator == op]:
                for name, h in (r.histories or {}).items():
                    panels.append({
                        "title": f"{r.gpu} · {name}",
                        "train": [float(x) for x in h.train_loss],
                        "val": [float(x) for x in h.val_loss],
                        "test": [float(x) for x in h.test_loss],
                        "best_epoch": int(h.best_epoch),
                        "stage_starts": [int(s) for s in h.stage_starts],
                        "n_test": int(r.n_test),
                    })
            if panels:
                sections.append({"fold": f"{op} (hardware held out)", "panels": panels})
        if sections:
            path = render_curves(sections, args.curves,
                                 title="Transfer fine-tuning curves by operator")
            n = sum(len(s["panels"]) for s in sections)
            print(f"\nwrote {n} training-curve panels to {path}")
            print("  faint vertical rule = the warmup ended and the trunk was unfrozen")

    if args.predictions:
        out = Path(args.predictions)
        out.mkdir(parents=True, exist_ok=True)
        pd.concat([r.predictions for r in results]).to_csv(
            out / "predictions__transfer.csv", index=False)
        print(f"\nwrote predictions to {out}")
    if args.out:
        Path(args.out).write_text(tab.to_string())
        print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


def cmd_e2e(args) -> int:
    """Measure real generations. Needs a GPU and the model weights."""
    import json

    from kernelenergy.e2e.measure import calls_frame, measure_generation
    from kernelenergy.hardware import probe_local_gpu
    from kernelenergy.hpc.device import resolve_device
    from kernelenergy.trace.pipelines import load_pipeline

    dev = resolve_device(args.device, strict=True)
    gpu = probe_local_gpu(dev.nvml_index)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in str(args.steps).split(",") if x]
    print(f"{gpu.name} ({gpu.gpu_key}); {args.model} at "
          f"{args.height}x{args.width}, steps {steps}")

    pipe, spec = load_pipeline(args.model, enable_cpu_offload=args.cpu_offload)
    rows, calls = [], []
    for n in steps:
        print(f"\n--- {n} steps ---")
        m = measure_generation(
            pipe, spec, model=args.model, gpu_key=gpu.gpu_key, steps=n,
            height=args.height, width=args.width, device=args.device,
            repeats=args.repeats, prompt=args.prompt,
        )
        print(f"  {m.energy_j:9.2f} J   {m.latency_s:7.3f} s   {m.power_avg_w:6.1f} W"
              f"   (energy sd {m.energy_sd_rel:.1%})")
        print(f"  device busy {m.busy_fraction:6.1%} of wall clock"
              f"   gap {m.gap_s:.3f} s")
        print(f"  profiler names {m.coverage_fraction:6.1%} of device time as a "
              f"modelled category")
        print(f"  {len(m.calls)} configs, {sum(m.calls.values()):,} invocations")
        rows.append(m.as_row())
        calls.append(calls_frame(m))

    tag = f"{gpu.gpu_key}__{args.model}"
    rp = out / f"e2e__{tag}.csv"
    cp = out / f"e2e_calls__{tag}.csv"
    pd.DataFrame(rows).to_csv(rp, index=False)
    pd.concat(calls, ignore_index=True).to_csv(cp, index=False)
    print(f"\nwrote {rp}\n      {cp}")
    return 0


def cmd_reconcile(args) -> int:
    """Compare the sum over kernels against the measured generations. No GPU needed."""
    from kernelenergy.e2e.measure import E2EMeasurement
    from kernelenergy.e2e.reconcile import reconcile, step_model, summarise

    src = Path(args.e2e)
    files = sorted(src.glob("e2e__*.csv")) if src.is_dir() else [src]
    if not files:
        print(f"no e2e__*.csv under {src}", file=sys.stderr)
        return 1

    dataset = pd.read_csv(args.dataset)
    preds = pd.read_csv(args.predictions) if args.predictions else None

    rows = []
    for f in files:
        cf = f.parent / f.name.replace("e2e__", "e2e_calls__")
        if not cf.exists():
            print(f"  skipping {f.name}: no matching {cf.name}")
            continue
        calls_all = pd.read_csv(cf)
        for _, r in pd.read_csv(f).iterrows():
            sel = calls_all[(calls_all["steps"] == r["steps"])
                            & (calls_all["model"] == r["model"])
                            & (calls_all["gpu_key"] == r["gpu_key"])]
            m = E2EMeasurement(
                gpu_key=r["gpu_key"], model=r["model"], steps=int(r["steps"]),
                height=int(r["height"]), width=int(r["width"]),
                latency_s=r["latency_s"], energy_j=r["energy_j"],
                power_avg_w=r["power_avg_w"], latency_sd_rel=r["latency_sd_rel"],
                energy_sd_rel=r["energy_sd_rel"], n_repeats=int(r["n_repeats"]),
                energy_counter_used=bool(r["energy_counter_used"]),
                idle_power_w=r["idle_power_w"],
                sm_clock_median_mhz=r["sm_clock_median_mhz"],
                frac_sw_power_cap=r["frac_sw_power_cap"],
                frac_sw_thermal=r.get("frac_sw_thermal", 0.0),
                temperature_max_c=r.get("temperature_max_c", 0.0),
                device_time_s=r["device_time_s"],
                device_time_covered_s=r["device_time_covered_s"],
                profiled_latency_s=r["profiled_latency_s"],
                calls=dict(zip(sel["kernel_sig"], sel["calls"].astype(int))),
            )
            try:
                rows.append(reconcile(m, dataset, preds))
            except KeyError as e:
                print(f"  skipping {m.gpu_key}/{m.model}/{m.steps}: {e}")

    if not rows:
        print("nothing reconciled", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)

    print("\n=== where the wall clock went ===")
    print("    busy_fraction   share of wall time with a kernel resident")
    print("    call_coverage   share of invocations the catalogue has a measurement for")
    print("    profiler_cov    share of device time in a kernel category we model")
    print(df[["gpu_key", "model", "steps", "e2e_latency_s", "device_time_s", "gap_s",
              "busy_fraction", "call_coverage", "profiler_coverage"]].round(3).to_string(index=False))

    print("\n=== E_generation = sum(calls x E_kernel) + P_idle x T_gap + residual ===")
    print("    A = measured replay energies;  B = model predictions")
    print("    ratio 1.0 is exact; residual_frac is what the identity fails to explain")
    print(summarise(df).to_string(index=False))

    sm = step_model(df)
    if len(sm):
        print("\n=== E(steps) = fixed + per_step x steps ===")
        print("    fixed  = text encode + VAE decode, once per image")
        print("    slope  = the denoising transformer, per step")
        print("    a right slope with a wrong intercept is a VAE problem, and vice versa")
        print(sm.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


def cmd_info(args) -> int:
    from kernelenergy.hardware import hardware_frame
    from kernelenergy.model.features import FEATURE_COLUMNS

    print(hardware_frame()[
        ["name", "architecture", "sms", "ops_per_clk", "tensor_clock_mhz",
         "peak_tensor_flops", "mem_bandwidth_gbs", "tdp_w"]
    ].to_string())
    print(f"\n{len(FEATURE_COLUMNS)} analytical features:")
    for c in FEATURE_COLUMNS:
        print(f"  {c}")
    return 0


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kernelenergy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="record the kernel configs a pipeline runs")
    c.add_argument("--model", required=True, help="flux1-dev | sd35-large | qwen-image")
    c.add_argument("--out", default="data/catalogue")
    c.add_argument("--steps", type=int, default=1)
    c.add_argument("--shapes", default="", help="e.g. 512x512,1024x1024")
    c.add_argument("--cpu-offload", action="store_true")
    c.add_argument("--profile", action="store_true", help="also run a profiler pass")
    c.set_defaults(func=cmd_capture)

    c = sub.add_parser("catalogue", help="merge captures into one deduplicated catalogue")
    c.add_argument("--in", dest="inp", default="data/catalogue")
    c.add_argument("--out", default="data/catalogue.csv")
    c.add_argument("--no-elementwise", action="store_true")
    c.add_argument("--max-working-set-gb", type=float, default=40.0)
    c.set_defaults(func=cmd_catalogue)

    c = sub.add_parser("idle", help="measure this card's idle power")
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--seconds", type=float, default=20.0)
    c.set_defaults(func=cmd_idle)

    c = sub.add_parser("preflight",
                       help="check everything that could invalidate a run, before it")
    c.add_argument("--device", type=int, default=0, help="CUDA ordinal")
    c.add_argument("--stage", default="measure",
                   choices=["capture", "measure", "all"])
    c.add_argument("--catalogue", default="")
    c.add_argument("--out", default="")
    c.add_argument("--models", default="", help="comma-separated, for the capture stage")
    c.add_argument("--allow-shared-gpu", action="store_true")
    c.set_defaults(func=cmd_preflight)

    c = sub.add_parser("measure", help="replay every config and record its energy")
    c.add_argument("--catalogue", default="data/catalogue.csv")
    c.add_argument("--out", default="data/raw")
    c.add_argument("--device", type=int, default=0, help="CUDA ordinal, not NVML")
    c.add_argument("--window", type=float, default=3.0)
    c.add_argument("--warmup", type=float, default=2.0)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--buffers", type=int, default=4)
    c.add_argument("--categories", default="")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--no-resume", action="store_true")
    c.add_argument("--notes", default="")
    c.add_argument("--shard", default="",
                   help="k/N; defaults to the SLURM array task when in an array")
    c.add_argument("--time-budget", type=float, default=0.0,
                   help="seconds; 0 means derive it from the job's remaining walltime")
    c.add_argument("--reserve", type=float, default=420.0,
                   help="seconds of walltime to hold back for setup and a clean flush")
    c.add_argument("--allow-shared-gpu", action="store_true",
                   help="measure anyway when another process holds the GPU; rows are "
                        "stamped contended=1 and should not be trusted")
    c.set_defaults(func=cmd_measure)

    c = sub.add_parser("dataset", help="join measurements with analytical features")
    c.add_argument("--raw", default="data/raw")
    c.add_argument("--out", default="data/dataset.csv")
    c.add_argument("--clock", default="tensor", choices=["tensor", "boost"])
    c.add_argument("--keep-all", action="store_true", help="skip the quality filters")
    c.set_defaults(func=cmd_dataset)

    c = sub.add_parser("evaluate", help="leave-one-group-out folds")
    c.add_argument("--dataset", default="data/dataset.csv")
    c.add_argument("--fold", default="all",
                   choices=["all", "hardware", "architecture", "category"])
    c.add_argument("--epochs", type=int, default=400)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="")
    c.add_argument("--predictions", default="")
    c.add_argument("--per-category", action="store_true",
                   help="fit a separate network per kernel category, as PipeWeave does, "
                        "rather than one across all of them")
    c.add_argument("--curves", default="",
                   help="write per-fold train/validation/test loss curves to this HTML "
                        "file. The test curve is the held-out group, recorded as a "
                        "diagnostic and never used for early stopping")
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_evaluate)

    c = sub.add_parser(
        "transfer",
        help="fine-tune PipeWeave's released checkpoints on the measured energy data",
    )
    c.add_argument("--dataset", default="data/dataset.csv")
    c.add_argument("--models", required=True,
                   help="path to a PipeWeave checkout's mlp_models/ directory")
    c.add_argument("--tile-mode", default="structural",
                   choices=["structural", "upstream"],
                   help="structural derives cta_count from the architecture; upstream "
                        "copies it off the nearest training neighbour, as their "
                        "aggregator.py does")
    c.add_argument("--operators", nargs="*", default=[],
                   choices=["gemm", "attn", "rmsnorm", "siluandmul"])
    c.add_argument("--epochs", type=int, default=400)
    c.add_argument("--warmup", type=int, default=150,
                   help="epochs training the power head alone before the trunk is "
                        "unfrozen")
    c.add_argument("--trunk-lr-scale", type=float, default=0.1)
    c.add_argument("--power-features", action="store_true",
                   help="give the model per-card power descriptors PipeWeave's features "
                        "lack (idle fraction, W/TFLOP, W per GB/s). New weight columns "
                        "are zero-initialised, so this cannot change the model at "
                        "epoch 0 -- it can only be learned into")
    c.add_argument("--power-into", default="pi", choices=["pi", "trunk", "both"],
                   help="pi = side channel into the power head only, leaving the "
                        "efficiency path provably untouched (default and recommended); "
                        "trunk = widen the first Linear, more expressive but it can "
                        "perturb eta")
    c.add_argument("--pi-floor", action="store_true",
                   help="reparameterise pi = idle_frac + (1-idle_frac)*sigmoid(z), so "
                        "the prediction cannot fall below the card's idle draw. Refuses "
                        "if the training data violates that bound")
    c.add_argument("--warmup-only", action="store_true",
                   help="fit ONLY the new power head; leave their trunk and efficiency "
                        "head exactly as released. The strongest configuration when "
                        "full fine-tuning scores worse than zero-shot")
    c.add_argument("--loss", default="log", choices=["log", "mape"],
                   help="log = |log(pred) - log(true)|, symmetric in ratio. mape "
                        "reproduces PipeWeave's own objective, but its asymmetry "
                        "collapses a head toward zero when the target is not fully "
                        "explained by the features -- which shows up as an eta APE of "
                        "almost exactly 100%%")
    c.add_argument("--grad-clip", type=float, default=1.0,
                   help="global gradient-norm clip, as in their train_mlp.py; 0 to "
                        "disable")
    c.add_argument("--energy-weight", type=float, default=0.0,
                   help="weight on the composed log-energy loss; 0 keeps the objective "
                        "identical to the from-scratch model so the two compare")
    c.add_argument("--train-bn", action="store_true",
                   help="let BatchNorm running statistics update. Off by default: a few "
                        "hundred rows will overwrite statistics fitted on half a million")
    c.add_argument("--top", type=int, default=12,
                   help="rows of the out-of-range table to print")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--curves", default="",
                   help="write per-operator fine-tuning loss curves to this HTML file, "
                        "one panel per (held-out GPU, model). The faint rule marks "
                        "where the warmup ended and the trunk was unfrozen")
    c.add_argument("--features-out", default="",
                   help="write the dataset with pw_* feature columns here")
    c.add_argument("--predictions", default="")
    c.add_argument("--out", default="")
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_transfer)

    c = sub.add_parser("e2e", help="measure real generations end to end (needs a GPU)")
    c.add_argument("--model", required=True)
    c.add_argument("--steps", default="4,12,20,28",
                   help="comma-separated step counts. Three or more lets reconcile "
                        "separate the once-per-image cost from the per-step cost")
    c.add_argument("--height", type=int, default=1024)
    c.add_argument("--width", type=int, default=1024)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--cpu-offload", action="store_true")
    c.add_argument("--prompt", default="a photograph of a city street")
    c.add_argument("--out", default="data/e2e")
    c.set_defaults(func=cmd_e2e)

    c = sub.add_parser("reconcile",
                       help="sum over kernels vs the measured generations (no GPU)")
    c.add_argument("--e2e", default="data/e2e")
    c.add_argument("--dataset", default="data/dataset.csv")
    c.add_argument("--predictions", default="",
                   help="a predictions CSV from evaluate or transfer, giving level B. "
                        "Those are leave-one-GPU-out, so the predictions for a card "
                        "were made by a model that never saw it")
    c.add_argument("--out", default="data/e2e_reconciliation.csv")
    c.set_defaults(func=cmd_reconcile)

    c = sub.add_parser("info", help="show the hardware table and feature list")
    c.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
