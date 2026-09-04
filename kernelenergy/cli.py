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
    for fold in folds:
        try:
            tab, results = evaluate(ds, fold=fold, config=cfg)
        except (KeyError, RuntimeError) as e:
            print(f"[{fold}] skipped: {e}")
            continue
        tables[fold] = tab
        print(f"\n=== {fold} fold: MAPE (%) by held-out group ===")
        print(tab.to_string())
        if args.predictions:
            out = Path(args.predictions)
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([r.predictions for r in results]).to_csv(
                out / f"predictions__{fold}.csv", index=False
            )

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
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("info", help="show the hardware table and feature list")
    c.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
