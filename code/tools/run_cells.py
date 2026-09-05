"""Parallel entry point for the reviewer experiments.

Runs the same experiment functions as run_reviewer_experiments.py, but schedules
independent cells across processes instead of executing them one at a time.
Scientific definitions are untouched; only placement and bookkeeping change.

Usage
-----
    python Code/run_cells.py --list-pending
    python Code/run_cells.py --dry-run --cpu-workers 3
    python Code/run_cells.py --benchmark-scheduler
    python Code/run_cells.py --cpu-workers 3 --gpu-workers 2
    python Code/run_cells.py --aggregate-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue as ThreadQueue

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import cell_scheduler as cs  # noqa: E402

DEFAULT_DATASETS = ["WUSTLEHMS2020", "TONIOT", "CSECICIDS2018"]
DEFAULT_SEEDS = [42, 7, 123, 31, 99]
DEFAULT_ANALYSES = ["selection", "resampling", "gbm", "hsj_budget"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip cells already completed and validated (default on)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--cpu-workers", type=int, default=2,
                   help="concurrent CPU-class cells (default 2)")
    p.add_argument("--gpu-workers", type=int, default=1,
                   help="concurrent CUDA-heavy cells (default 1)")
    p.add_argument("--only", type=str, default=",".join(DEFAULT_ANALYSES),
                   help="comma-separated analyses")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--out", type=str, default="results_v2")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without running anything")
    p.add_argument("--list-pending", action="store_true",
                   help="list cells that still need to run, then exit")
    p.add_argument("--benchmark-scheduler", action="store_true",
                   help="time representative cells under several worker counts")
    p.add_argument("--aggregate-only", action="store_true",
                   help="rebuild the per-analysis summaries from existing cells")
    return p.parse_args(argv)


ALIASES = {"cse_cic": "CSECICIDS2018", "cic": "CSECICIDS2018",
           "toniot": "TONIOT", "ton_iot": "TONIOT",
           "wustl": "WUSTLEHMS2020", "wustlehms2020": "WUSTLEHMS2020",
           "cse_cic_temporal": "CSECICIDS2018_TEMPORAL"}


def norm_datasets(raw: str) -> list[str]:
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        out.append(ALIASES.get(t.lower(), t))
    return out


def describe_plan(specs, args) -> None:
    parts = cs.partition(specs)
    pending, skipped = [], []
    for s in specs:
        status, detail = cs.cell_status(s)
        (skipped if status in ("done", "adopt") and args.resume else pending).append(
            (s, status, detail))

    print("=" * 78)
    print(f"PLAN  {len(specs)} cells   cpu={len(parts['cpu'])} "
          f"gpu={len(parts['gpu'])} exclusive={len(parts['exclusive'])}")
    print(f"      cpu-workers={args.cpu_workers}  gpu-workers={args.gpu_workers}  "
          f"resume={'on' if args.resume else 'off'}")
    print("=" * 78)

    print(f"\nSKIPPED ({len(skipped)}):")
    for s, status, detail in skipped[:8]:
        print(f"  {s.cell_id:44} {status:6} {detail}")
    if len(skipped) > 8:
        print(f"  ... and {len(skipped) - 8} more")

    print(f"\nPENDING ({len(pending)}):")
    for s, _, detail in pending:
        print(f"  [{s.resource:9}] {s.cell_id:44} hash={s.config_hash}  ({detail})")
        print(f"              -> {s.result_path}")
    print()


def run_group(specs, workers: int, label: str) -> list[dict]:
    """Run a group of cells across `workers` processes."""
    if not specs:
        return []
    """Run `specs` with at most `workers` concurrent, each in its own killable
    process. Uses a thread pool purely to bound concurrency and collect
    results; the actual compute for each cell still runs in a dedicated OS
    process via cs.execute_cell_with_timeout, so a hang in one cell can be
    killed without touching any sibling in flight (see cell_scheduler.py for
    why a shared ProcessPoolExecutor cannot do this safely).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    workers = max(1, min(workers, len(specs)))
    print(f"\n--- {label}: {len(specs)} cell(s), {workers} worker(s), "
          f"timeout={cs.DEFAULT_CELL_TIMEOUT_S/60:.0f} min/cell ---", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cs.execute_cell_with_timeout, dict(
            analysis=s.analysis, dataset=s.dataset, seed=s.seed, full=s.full,
            out_dir=s.out_dir, config_hash=s.config_hash)): s for s in specs}
        done_n = 0
        timed_out = []
        for fut in as_completed(futures):
            r = fut.result()
            done_n += 1
            results.append(r)
            if r["status"] == "timeout":
                timed_out.append(r["cell_id"])
            print(f"  [{done_n}/{len(specs)}] {r['status']:8} {r['cell_id']:44} "
                  f"{r['duration_s']/60:6.1f} min  {r.get('detail','')}", flush=True)
    print(f"--- {label} finished in {(time.time()-t0)/60:.1f} min ---", flush=True)
    if timed_out:
        print(f"  !! {len(timed_out)} cell(s) hit the wall-clock timeout and were "
              f"killed, not silently retried with different settings:", flush=True)
        for cid in timed_out:
            print(f"       {cid}", flush=True)
    return results


def aggregate_all(analyses, datasets, seeds, out_dir) -> None:
    import run_reviewer_experiments as rex
    for a in analyses:
        for d in datasets:
            try:
                rex.aggregate(d, seeds, a, out_dir)
            except Exception as e:
                print(f"  !! aggregate {a} {d}: {e}")


def main(argv=None) -> int:
    args = parse_args(argv)
    analyses = [a.strip() for a in args.only.split(",") if a.strip()]
    datasets = norm_datasets(args.datasets)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.aggregate_only:
        aggregate_all(analyses, datasets, seeds, args.out)
        return 0

    specs = cs.build_plan(analyses, datasets, seeds, args.out, full=True)

    if args.list_pending or args.dry_run:
        describe_plan(specs, args)
        return 0

    if args.benchmark_scheduler:
        from bench_scheduler import run_benchmark
        return run_benchmark(args)

    return _run_locked(specs, analyses, datasets, seeds, args)


def _run_locked(specs, analyses, datasets, seeds, args) -> int:
    # Refuses to start if another scheduler/benchmark process already holds the
    # lock for this out_dir, which is what prevents two independent process
    # trees from touching the GPU concurrently (see cell_scheduler.RunLock).
    with cs.RunLock(args.out, label="run_cells"):
        return _run_body(specs, analyses, datasets, seeds, args)


def _run_body(specs, analyses, datasets, seeds, args) -> int:
    pending = [s for s in specs
               if not (args.resume and cs.cell_status(s)[0] in ("done", "adopt"))]
    # adopt pre-existing results so their markers exist for later runs
    for s in specs:
        if cs.cell_status(s)[0] == "adopt":
            cs.write_marker(s, 0.0, adopted=True)

    if not pending:
        print("Nothing to do: every requested cell is complete.")
        aggregate_all(analyses, datasets, seeds, args.out)
        return 0

    parts = cs.partition(pending)
    print(f"pending: {len(pending)} cell(s)  "
          f"cpu={len(parts['cpu'])} gpu={len(parts['gpu'])} "
          f"exclusive={len(parts['exclusive'])}")

    results = []
    # CPU-class and CUDA-heavy groups can overlap; exclusive cells run alone so
    # their native thread counts, and therefore their numerics, stay unchanged.
    results += run_group(parts["cpu"], args.cpu_workers, "cpu queue")
    results += run_group(parts["gpu"], args.gpu_workers, "gpu queue")
    results += run_group(parts["exclusive"], 1, "exclusive queue")

    aggregate_all(analyses, datasets, seeds, args.out)

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    timed_out = [r for r in results if r["status"] == "timeout"]
    print("\n" + "=" * 78)
    print(f"completed {len(ok)}   skipped {len([r for r in results if r['status']=='skipped'])}"
          f"   failed {len(failed)}   timed out {len(timed_out)}")
    for r in failed:
        print(f"  FAILED  {r['cell_id']}: {r['detail']}")
    for r in timed_out:
        print(f"  TIMEOUT {r['cell_id']}: {r['detail']}")
    if failed or timed_out:
        print("Re-run the same command to retry failed cells; a claim released by a "
              "timeout is retried automatically. A cell that times out twice needs "
              "investigation before a third automated attempt, not a longer timeout.")
    print("=" * 78)
    return 1 if (failed or timed_out) else 0


if __name__ == "__main__":
    # required on Windows: workers re-import this module under spawn
    sys.exit(main())
